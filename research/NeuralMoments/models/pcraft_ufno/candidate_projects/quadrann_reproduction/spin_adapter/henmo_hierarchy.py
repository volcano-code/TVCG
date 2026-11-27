"""Sparse, material-faithful HENMO hierarchy and deterministic reference path.

The tree changes *where work is executed*, not the finite-cut semantics. Its
full/empty certificate uses a node bounding sphere and the complete response
support. Actual contact events are not forced to have zero hard-cut derivative.
``from_material`` aggregates analytic RLE runs and mesh partial-cell moments;
it never expands all native voxels merely to construct a 512/1024 hierarchy.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .henmo_cubature import CompressedCubature, compress_positive_measure, moment_basis
from .v21_material import PreparedMaterial, validate_full_cell_rule


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def _readonly(value, dtype=None):
    array = np.ascontiguousarray(np.asarray(value,dtype=dtype))
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def _readonly_storage(array):
    """Immutable byte backing, except existing genuinely read-only mmap data."""
    value = array
    while isinstance(value,np.ndarray):
        if isinstance(value,np.memmap) and value.mode == 'r':
            array.flags.writeable = False
            return array
        value = value.base
    if isinstance(value,bytes) and not array.flags.writeable:
        return array
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(),dtype=contiguous.dtype).reshape(contiguous.shape)


def _array_state(array):
    if array.flags.writeable:
        raise ValueError('hierarchy material storage became writable; rebuild static hierarchy')
    mapped = None
    value = array
    while isinstance(value,np.ndarray):
        if value.flags.writeable:
            raise ValueError('hierarchy material has writable underlying storage')
        if isinstance(value,np.memmap):
            stat = Path(value.filename).stat()
            mapped = (str(value.filename),stat.st_dev,stat.st_ino,stat.st_size,
                      stat.st_mtime_ns,stat.st_ctime_ns,value.mode)
        value = value.base
    return id(array),array.shape,array.dtype.str,array.ctypes.data,mapped


def _morton(coordinates, depth):
    code = np.zeros(len(coordinates), dtype=np.uint64)
    for bit in range(depth):
        for axis in range(3):
            code |= ((coordinates[:, axis].astype(np.uint64) >> bit) & 1) << (3*bit + axis)
    return code


@dataclass(frozen=True)
class HierarchyNode:
    index: int
    depth: int
    grid: int
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    radius: float
    moments: np.ndarray
    children: tuple[int, ...]
    leaf_start: int
    leaf_end: int


@dataclass(frozen=True)
class HierarchySelection:
    retained_nodes: tuple[int, ...]
    discarded_nodes: tuple[int, ...]
    boundary_nodes: tuple[int, ...]
    retained_moments: np.ndarray
    visited_nodes: int
    distance_queries: int
    stop_grid: int


class CoarseCubatureUnavailable(RuntimeError):
    """Static compression failed; refine/stream rather than drop material."""
    def __init__(self, node_id, reason, *, source_count, working_atoms, requested_atoms):
        self.node_id = int(node_id)
        self.reason = str(reason)
        self.source_count = int(source_count)
        self.working_atoms = int(working_atoms)
        self.requested_atoms = int(requested_atoms)
        super().__init__(f'node {node_id}: coarse cubature unavailable ({reason}); '
                         f'{working_atoms} working atoms, {source_count} source atoms visited')


class SparseMaterialHierarchy:
    """An immutable geometry index with byte-bounded static atom caches.

    Query traversal hashes neither all material atoms nor all leaf blocks.
    Construction freezes ordinary source arrays with immutable byte backing;
    read-only mapped packs remain mapped and carry O(1) file-stat witnesses.
    Source replacement or file modification requires rebuilding the index.
    """
    def __init__(self, *, coordinates, moments, origin, pitch, side, leaf_grid,
                 identity, material=None, blocks=None, cache_bytes=64*1024**2):
        leaf_grid = _positive_integer(leaf_grid, 'leaf_grid')
        if leaf_grid & (leaf_grid - 1) or leaf_grid > side or side % leaf_grid:
            raise ValueError('leaf_grid must be a power of two dividing native side')
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('cache_bytes must be a nonnegative integer')
        ijk = np.asarray(coordinates, dtype=np.int64)
        values = np.asarray(moments, dtype=np.float64)
        if (ijk.ndim != 2 or ijk.shape[1] != 3 or not len(ijk)
                or values.shape != (len(ijk), 10) or not np.isfinite(values).all()
                or np.any(values[:, 0] <= 0) or np.any(ijk < 0) or np.any(ijk >= leaf_grid)
                or len(np.unique(ijk, axis=0)) != len(ijk)):
            raise ValueError('distinct occupied leaf coordinates and finite positive moments required')
        self.origin = _readonly(origin, np.float64)
        self.pitch, self.side, self.leaf_grid = float(pitch), int(side), leaf_grid
        if (self.origin.shape != (3,) or not np.isfinite(self.origin).all()
                or not np.isfinite(self.pitch) or self.pitch <= 0):
            raise ValueError('invalid physical grid')
        self.material, self.blocks = material, blocks
        self.material_identity = str(identity)
        self._cache, self._cache_used, self.cache_bytes = OrderedDict(), 0, cache_bytes
        self._features, self._feature_cache_used = OrderedDict(), 0
        # This changes only ownership/write permissions, never material values.
        # Copying a writable array onto immutable bytes avoids a writable base
        # silently changing cached moments while a child view looks read-only.
        if self.blocks is not None:
            for name in ('nodes','ids','offsets','moments','centers','origin'):
                setattr(self.blocks,name,_readonly_storage(getattr(self.blocks,name)))
        if self.material is not None:
            for name in ('full_runs','partial_cells','boundary_nodes','origin'):
                setattr(self.material,name,_readonly_storage(getattr(self.material,name)))
        self._source_signature = self._source_state()
        depth = leaf_grid.bit_length() - 1
        if depth > 20:
            raise ValueError('Morton index supports at most twenty octree levels')
        code = _morton(ijk, depth)
        order = np.argsort(code, kind='stable')
        self.leaf_order = _readonly(order, np.int64)
        sorted_code, sorted_ijk, sorted_values = code[order], ijk[order], values[order]
        levels, records = [], []
        length = self.side * self.pitch
        for level in range(depth + 1):
            prefix = sorted_code >> (3*(depth-level))
            keys, start, counts = np.unique(prefix, return_index=True, return_counts=True)
            ids = np.arange(len(records), len(records) + len(keys), dtype=np.int64)
            reduced = np.add.reduceat(sorted_values, start, axis=0)
            coordinates_at_level = sorted_ijk[start] >> (depth-level)
            edge = length / (2**level)
            for key, begin, count, row, coord, node_id in zip(keys, start, counts, reduced, coordinates_at_level, ids):
                low = self.origin + coord * edge
                high = low + edge
                records.append(dict(index=int(node_id), depth=level, grid=2**level,
                    lower=_readonly(low), upper=_readonly(high), center=_readonly((low+high)*.5),
                    radius=float(np.sqrt(3.)*edge*.5), moments=_readonly(row), children=(),
                    leaf_start=int(begin), leaf_end=int(begin+count)))
            levels.append((keys, ids))
        for level in range(depth):
            keys, ids = levels[level]
            next_keys, next_ids = levels[level+1]
            parents = next_keys >> 3
            left = np.searchsorted(parents, keys, side='left')
            right = np.searchsorted(parents, keys, side='right')
            for node_id, a, b in zip(ids, left, right):
                records[int(node_id)]['children'] = tuple(map(int, next_ids[a:b]))
        self.nodes = tuple(HierarchyNode(**record) for record in records)
        self.root = 0
        self.leaf_nodes = tuple(map(int, levels[-1][1]))
        self.total_moments = self.nodes[0].moments
        binding = hashlib.sha256(json.dumps(dict(schema='henmo.sparse-hierarchy.v1',
            material=self.material_identity, side=self.side, pitch=self.pitch,
            origin=self.origin.tolist(), leaf_grid=self.leaf_grid), sort_keys=True).encode())
        # One construction-only digest of compact occupied leaf metadata. The
        # full original material identity remains the geometry authority; no
        # full atom hash is added to the query path.
        binding.update(np.ascontiguousarray(ijk).tobytes())
        binding.update(np.ascontiguousarray(values).tobytes())
        self.identity = binding.hexdigest()

    @classmethod
    def from_blocks(cls, blocks, *, cache_bytes=64*1024**2):
        grid = blocks.side // blocks.block_voxels
        ijk = np.stack(np.unravel_index(blocks.ids, (grid,)*3), axis=1)
        return cls(coordinates=ijk, moments=blocks.moments, origin=blocks.origin,
                   pitch=blocks.pitch, side=blocks.side, leaf_grid=grid, identity=blocks.identity,
                   material=blocks.material, blocks=blocks, cache_bytes=cache_bytes)

    @classmethod
    def from_material(cls, material, *, leaf_grid=64, cache_bytes=64*1024**2):
        if not isinstance(material, PreparedMaterial):
            raise ValueError('expected original PreparedMaterial with material provenance')
        leaf_grid = _positive_integer(leaf_grid, 'leaf_grid')
        if material.side % leaf_grid or leaf_grid > material.side:
            raise ValueError('leaf_grid must divide native material side')
        ijk, moments = material.block_moments(material.side // leaf_grid)
        return cls(coordinates=ijk, moments=moments, origin=material.origin, pitch=material.pitch,
                   side=material.side, leaf_grid=leaf_grid, identity=material.identity,
                   material=material, cache_bytes=cache_bytes)

    def _source_state(self):
        source = self.blocks if self.blocks is not None else self.material
        if source is None:
            raise ValueError('source material required')
        names = ('nodes','ids','offsets','moments','centers','origin') if self.blocks is not None else ('full_runs','partial_cells','boundary_nodes','origin')
        states = []
        for name in names:
            array = getattr(source, name)
            states.append((name,_array_state(array)))
        provenance = None
        if self.blocks is not None and self.blocks.material is not self.material:
            raise ValueError('hierarchy material provenance was replaced')
        if self.material is not None:
            material = self.material
            provenance = (id(material),material.identity,material.side,material.pitch,material.density,
                material.source,material.partial_rule,tuple(_array_state(getattr(material,name))
                for name in ('full_runs','partial_cells','boundary_nodes','origin')))
        return (source.identity,source.side,source.pitch,tuple(np.asarray(source.origin)),tuple(states),
                provenance,getattr(source,'block_voxels',None),getattr(source,'radius',None))

    def assert_source_unchanged(self):
        if self._source_state() != self._source_signature:
            raise ValueError('hierarchy source material was replaced; rebuild static hierarchy')

    def __len__(self):
        return len(self.nodes)

    def node(self, node_id):
        if type(node_id) not in (int, np.int64, np.int32) or not 0 <= node_id < len(self.nodes):
            raise ValueError('invalid hierarchy node index')
        return self.nodes[int(node_id)]

    def children(self, node_id):
        return self.node(node_id).children

    def leaf_indices(self, node_id):
        node = self.node(node_id)
        return self.leaf_order[node.leaf_start:node.leaf_end]

    def classify_nodes(self, surface, node_ids, bandwidth, operation='intersection'):
        """Classify against the entire finite operand, including end caps.

        Strict separation from radius + support width ensures both value and
        derivative of this compact-band response are constant on that branch.
        It makes no assertion that exact hard-cut derivatives vanish at contact.
        """
        if (not np.isfinite(bandwidth) or bandwidth <= 0
                or operation not in ('intersection', 'difference')):
            raise ValueError('positive bandwidth and registered operation required')
        ids = tuple(node_ids)
        if not ids:
            return np.empty(0, np.int8)
        if hasattr(surface, 'check_tensor_source'):
            surface.check_tensor_source()
        centers = np.array([self.node(i).center for i in ids])
        radii = np.array([self.nodes[int(i)].radius for i in ids])
        distance = np.asarray(surface.signed_distance(centers), dtype=np.float64)
        if distance.shape != (len(ids),) or not np.isfinite(distance).all():
            raise ValueError('invalid finite-operand signed distances')
        safety = 64*np.finfo(float).eps*(1+np.max(np.abs(centers), axis=1))
        threshold = radii + float(bandwidth) + safety + float(getattr(surface,'tolerance',0.))
        sign = 1 if operation == 'intersection' else -1
        keep = sign*distance > threshold
        reject = sign*distance < -threshold
        return np.where(keep, 1, np.where(reject, -1, 0)).astype(np.int8)

    def select(self, surface, bandwidth, operation='intersection', *, stop_grid=None):
        self.assert_source_unchanged()
        stop = self.leaf_grid if stop_grid is None else _positive_integer(stop_grid, 'stop_grid')
        if stop & (stop-1) or stop > self.leaf_grid:
            raise ValueError('stop_grid must be a power of two no larger than leaf_grid')
        pending, retained, discarded, boundary = [self.root], [], [], []
        visited = 0
        while pending:
            status = self.classify_nodes(surface, pending, bandwidth, operation)
            visited += len(pending)
            following = []
            for node_id, kind in zip(pending, status):
                node = self.nodes[node_id]
                if kind == 1:
                    retained.append(node_id)
                elif kind == -1:
                    discarded.append(node_id)
                elif node.grid >= stop or not node.children:
                    boundary.append(node_id)
                else:
                    following.extend(node.children)
            pending = following
        total = np.zeros(10, dtype=np.float64)
        for index in retained:
            total += self.nodes[index].moments
        return HierarchySelection(tuple(retained), tuple(discarded), tuple(boundary),
                                  _readonly(total), visited, visited, stop)

    def chunks(self, node_id, max_nodes=65536, *, full_cell_rule='tetra4'):
        self.assert_source_unchanged()
        max_nodes = _positive_integer(max_nodes, 'max_nodes')
        rule = validate_full_cell_rule(full_cell_rule)
        node = self.node(node_id)
        if self.blocks is not None and rule == 'tetra4':
            # Read only this subtree's occupied leaves. No global mask scan.
            for index in self.leaf_indices(node_id):
                begin, end = self.blocks.offsets[index:index+2]
                for a in range(int(begin), int(end), max_nodes):
                    rows = self.blocks.nodes[a:min(a+max_nodes,int(end))]
                    yield rows[:,:3], rows[:,3]
            return
        if self.material is None:
            raise ValueError('refined cubature requires original material provenance')
        yield from self.material.iter_chunks(chunk_size=max_nodes,
            box=(node.lower, node.upper), full_cell_rule=rule)

    def coarse_chunks(self, node_id, max_nodes=65536):
        """Positive four-point *run-cuboid* rule plus original partial atoms.

        A long full RLE interval is integrated as one cuboid without expanding
        each native voxel. These points remain inside known material and match
        all complete moments. Their partial-cut response is an approximation,
        not the native-cell rule returned by ``chunks``.
        """
        self.assert_source_unchanged()
        max_nodes = _positive_integer(max_nodes, 'max_nodes')
        if self.material is None:
            yield from self.chunks(node_id,max_nodes)
            return
        material = self.material
        node = self.node(node_id)
        bounds = material._box((node.lower,node.upper))
        runs = material._selected_runs(bounds)
        signs = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=np.float64)/np.sqrt(12.)
        # One run consumes four nodes; slice the final flat arrays to respect
        # even tiny caller capacities without an unbounded run expansion.
        for begin in range(0,len(runs),max(1,max_nodes//4)):
            part = runs[begin:begin+max(1,max_nodes//4)]
            extent = np.broadcast_to(material.pitch,(len(part),3)).copy()
            extent[:,2] *= part[:,3]-part[:,2]
            center = material.origin + material.pitch*np.column_stack((part[:,0]+.5,
                part[:,1]+.5,(part[:,2].astype(np.float64)+part[:,3])*.5))
            xyz = (center[:,None,:]+signs[None,:,:]*extent[:,None,:]).reshape(-1,3)
            weights = np.repeat(material.density*np.prod(extent,axis=1)/4,4)
            for a in range(0,len(weights),max_nodes):
                yield xyz[a:a+max_nodes],weights[a:a+max_nodes]
        unit = material.density*material.pitch**3
        for rows in material._boundary_chunks(bounds,max_nodes):
            yield material.origin+material.pitch*rows[:,:3],unit*rows[:,3]

    def _reserve_cache(self,size):
        """One shared atom/descriptor budget with constant-time accounting."""
        if size>self.cache_bytes:
            return False
        while self._cache_used+self._feature_cache_used+size>self.cache_bytes:
            if self._cache and (not self._features or self._cache_used>=self._feature_cache_used):
                _,(_,old_size)=self._cache.popitem(last=False)
                self._cache_used-=old_size
            elif self._features:
                _,old=self._features.popitem(last=False)
                self._feature_cache_used-=old.nbytes
            else:
                raise RuntimeError('static cache byte accounting became inconsistent')
        return True

    def measure(self, node_id, *, max_atoms=None, full_cell_rule='tetra4', chunk_size=65536,
                candidate_rule='auto'):
        """Static local measure; compression does not certify partial-cut error."""
        self.assert_source_unchanged()
        rule = validate_full_cell_rule(full_cell_rule)
        if max_atoms is not None and (type(max_atoms) is not int or max_atoms < 10):
            raise ValueError('max_atoms must be None or an integer at least ten')
        if candidate_rule not in ('auto','run4','native_cells'):
            raise ValueError('candidate_rule must be auto, run4, or native_cells')
        run_rule = candidate_rule == 'run4' or (
            candidate_rule == 'auto' and max_atoms is not None and self.material is not None)
        if run_rule and self.material is None:
            raise ValueError('run4 requires original RLE/partial-cell material provenance')
        source_rule = 'analytic_full_runs_tetra4_and_original_partial_atoms' if run_rule else rule
        key = (int(node_id), max_atoms, rule, source_rule)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        if max_atoms is not None:
            # Bound temporary storage even when a caller asks for the root of a
            # 1024 material. Repeated failed numerical compression is reported
            # to the adaptive caller, never converted to an unbounded atom list.
            local_capacity = max(1024,4*max_atoms)
            _positive_integer(chunk_size,'chunk_size')
            # Compression's static reduction schedule is part of this module's
            # semantics. Query execution chunk size must not silently change
            # the selected atoms or make cache results depend on call order.
            read_size = local_capacity
            capacity = 4*local_capacity
            chunks = self.coarse_chunks(node_id,read_size) if run_rule else self.chunks(
                node_id,read_size,full_cell_rule=rule)
            node = self.node(node_id)
            center,extent = node.center,float((node.upper-node.lower).max())
            total = np.zeros(10)
            local_total = np.zeros(10)
            source_count = 0
            xyz = np.empty((0,3)); weight = np.empty(0); indices = np.empty(0,np.int64)
            last_reason = 'NO_SOURCE_ATOMS'
            for new_xyz,new_weight in chunks:
                count = len(new_weight)
                if len(weight)+count > capacity:
                    raise CoarseCubatureUnavailable(node_id,last_reason,
                        source_count=source_count,working_atoms=len(weight)+count,requested_atoms=max_atoms)
                total += moment_basis(new_xyz).T @ new_weight
                local_total += moment_basis((new_xyz-center)/extent).T @ new_weight
                mapping = np.r_[indices,np.arange(source_count,source_count+count,dtype=np.int64)]
                source_count += count
                reduced = compress_positive_measure(np.concatenate((xyz,new_xyz)),
                    np.concatenate((weight,new_weight)),max_atoms)
                xyz,weight = reduced.nodes,reduced.weights
                indices = mapping[reduced.source_indices]
                last_reason = reduced.reason
            if not source_count:
                raise ValueError('occupied hierarchy node produced no material atoms')
            error = float(np.max(np.abs(moment_basis((xyz-center)/extent).T @ weight-local_total))/local_total[0])
            if len(weight)>max_atoms or error>2e-11:
                raise CoarseCubatureUnavailable(node_id,'STREAM_MOMENT_OR_BUDGET_CHECK_FAILED',
                    source_count=source_count,working_atoms=len(weight),requested_atoms=max_atoms)
            result = CompressedCubature(_readonly(xyz),_readonly(weight),_readonly(indices),
                _readonly(total),source_count,error,len(weight)<source_count,
                'STREAMED_POSITIVE_SOURCE_SUPPORT_MOMENT_MATCH' if len(weight)<source_count else 'ALREADY_WITHIN_BUDGET')
        else:
            points,weights=[],[]
            chunks = self.coarse_chunks(node_id,chunk_size) if run_rule else self.chunks(
                node_id,chunk_size,full_cell_rule=rule)
            for xyz,w in chunks:
                points.append(np.array(xyz,copy=True)); weights.append(np.array(w,copy=True))
            xyz,weight = np.concatenate(points),np.concatenate(weights)
            source = np.arange(len(xyz), dtype=np.int64)
            total = (moment_basis(xyz)*weight[:,None]).sum(0)
            result = CompressedCubature(_readonly(xyz),_readonly(weight),_readonly(source),
                _readonly(total),len(xyz),0.,False,'FULL_SOURCE_RULE')
        result = replace(result,source_rule=source_rule)
        size = sum(a.nbytes for a in (result.nodes,result.weights,result.source_indices,result.total_moments))
        if self._reserve_cache(size):
            self._cache[key] = result,size
            self._cache_used += size
        return result

    def spatial_features(self, node_id, *, grid=2):
        """Static local subregion moments, retaining more than ten block moments.

        Returns [grid**3,10], scaled by node mass and local extent. These are
        descriptors, not a unique encoding of arbitrary fine material geometry.
        """
        self.assert_source_unchanged()
        grid = _positive_integer(grid, 'feature grid')
        key = int(node_id), grid
        if key in self._features:
            self._features.move_to_end(key)
            return self._features[key]
        node = self.node(node_id)
        edge = node.upper-node.lower
        features = np.zeros((grid**3,10), dtype=np.float64)
        width = self.side // node.grid
        if self.material is not None and width % grid == 0:
            center = (node.lower+node.upper)*.5
            for flat, index in enumerate(np.ndindex((grid,)*3)):
                lo = node.lower+np.asarray(index)*edge/grid
                hi = lo+edge/grid
                raw = self.material.moments(box=(lo,hi))
                local = np.zeros(10)
                local[0] = raw[0]
                local[1:4] = (raw[1:4]-center*raw[0])/edge
                local[4:7] = (raw[4:7]-2*center*raw[1:4]+center**2*raw[0])/edge**2
                for k,(a,b) in enumerate(((1,2),(0,2),(0,1)),start=7):
                    local[k] = (raw[k]-center[a]*raw[1+b]-center[b]*raw[1+a]
                                +center[a]*center[b]*raw[0])/(edge[a]*edge[b])
                features[flat] = local/node.moments[0]
        else:
            for xyz, weight in self.chunks(node_id):
                local = (xyz-node.lower)/edge
                ijk = np.floor(local*grid).astype(np.int64).clip(0,grid-1)
                index = np.ravel_multi_index(ijk.T,(grid,)*3)
                basis = moment_basis(local-.5)*weight[:,None]/node.moments[0]
                for j in range(10):
                    features[:,j] += np.bincount(index,weights=basis[:,j],minlength=grid**3)
        features=_readonly(features)
        if self._reserve_cache(features.nbytes):
            self._features[key] = features
            self._feature_cache_used += features.nbytes
        return features

    def clear_cache(self):
        self._cache.clear(); self._cache_used = 0
        self._features.clear(); self._feature_cache_used = 0

    @property
    def diagnostics(self):
        return dict(schema='henmo.sparse-hierarchy.v1', material_identity=self.material_identity,
            hierarchy_identity=self.identity, native_side=self.side, leaf_grid=self.leaf_grid,
            nodes=len(self.nodes), occupied_leaves=len(self.leaf_nodes),
            cache_bytes=self._cache_used+self._feature_cache_used,
            atom_cache_bytes=self._cache_used,feature_cache_bytes=self._feature_cache_used,
            dense_native_grid=False, material_source=getattr(self.material,'source','legacy_positive_pack'))


@dataclass(frozen=True)
class DeterministicPrediction:
    moments: object
    jacobian: object | None
    diagnostics: dict


def evaluate_deterministic(hierarchy, surface, bandwidth, operation='intersection', fixed_moments=None,
                           vertex_velocity=None, *, full_cell_rule='gl3', stop_grid=None,
                           chunk_size=65536):
    """No learned weights, no atom compression; same finite-cut smoothing.

    A higher-order cell rule is a strong deterministic *smoothed-response*
    baseline, not a certified original-mesh hard-Boolean oracle. Refinement of
    full cells cannot remove already-present partial-material approximation.
    """
    import torch
    from .v21_response import prepare_measure, evaluate_response
    rule = validate_full_cell_rule(full_cell_rule)
    fixed = np.zeros(10) if fixed_moments is None else np.asarray(fixed_moments,dtype=np.float64)
    if fixed.shape != (10,) or not np.isfinite(fixed).all():
        raise ValueError('fixed material requires ten finite moments')
    before = time.perf_counter()
    selected = hierarchy.select(surface, bandwidth, operation, stop_grid=stop_grid)
    value = torch.as_tensor(fixed+selected.retained_moments,dtype=torch.float64)
    jacobian = None
    if vertex_velocity is not None:
        velocity = np.asarray(vertex_velocity,dtype=np.float64)
        if velocity.ndim != 3 or velocity.shape[:2] != (len(surface.vertices),3) or not np.isfinite(velocity).all():
            raise ValueError('vertex motion must match the finite operand')
        jacobian = torch.zeros((10,velocity.shape[2]),dtype=torch.float64)
    count = active = ambiguous = chunks = 0
    for index in selected.boundary_nodes:
        for xyz, weight in hierarchy.chunks(index,chunk_size,full_cell_rule=rule):
            out = evaluate_response(prepare_measure(xyz,weight),surface,bandwidth=bandwidth,
                operation=operation,vertex_velocity=vertex_velocity,chunk_size=chunk_size)
            value = value+out.moments
            if jacobian is not None:
                jacobian = None if out.jacobian is None else jacobian+out.jacobian
            count += len(weight); active += out.diagnostics['active_nodes']
            ambiguous += out.diagnostics['ambiguous_nodes']; chunks += 1
    return DeterministicPrediction(value,jacobian,dict(method='hierarchical_deterministic',
        full_cell_rule=rule, visited_nodes=selected.visited_nodes, boundary_nodes=len(selected.boundary_nodes),
        retained_nodes=len(selected.retained_nodes), distance_queries=selected.distance_queries,
        quadrature_nodes=count, active_nodes=active, ambiguous_nodes=ambiguous,chunks=chunks,
        elapsed_seconds=time.perf_counter()-before, exact_hard_mesh_oracle=False,
        solver_status='PROPOSAL_REQUIRES_NATIVE_ACCEPTANCE'))


def compare_deterministic_rules(hierarchy, surface, bandwidth, operation='intersection',
                                fixed_moments=None, vertex_velocity=None, *,
                                rules=('tetra4','gl2','gl3'), moment_scales=None,
                                parameter_scales=None, tolerance=1e-4, **kwargs):
    """Increasing-rule comparison with empirical M/J discrepancies.

    The last available rule is returned. ``converged`` only describes this
    discrete rule comparison; it is explicitly not an original-mesh error bound.
    """
    if len(rules) < 2 or len(set(rules)) != len(rules):
        raise ValueError('at least two distinct deterministic quadrature rules required')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive comparison tolerance required')
    scales = np.ones(10) if moment_scales is None else np.asarray(moment_scales,dtype=np.float64)
    if scales.shape != (10,) or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('ten positive moment scales required')
    previous, records, output = None, [], None
    for rule in rules:
        output = evaluate_deterministic(hierarchy,surface,bandwidth,operation,fixed_moments,
            vertex_velocity,full_cell_rule=rule,**kwargs)
        row = dict(rule=rule, **output.diagnostics)
        if previous is not None:
            delta = (output.moments-previous.moments).detach().cpu().numpy()/scales
            row['moment_discrepancy'] = float(np.max(np.abs(delta)))
            row['jacobian_discrepancy'] = None
            if output.jacobian is not None and previous.jacobian is not None:
                dimension = output.jacobian.shape[1]
                params = np.ones(dimension) if parameter_scales is None else np.asarray(parameter_scales,dtype=np.float64)
                if params.shape != (dimension,) or not np.isfinite(params).all() or np.any(params<=0):
                    raise ValueError('one positive scale per coefficient required')
                difference = (output.jacobian-previous.jacobian).detach().cpu().numpy()*params[None,:]/scales[:,None]
                row['jacobian_discrepancy'] = float(np.max(np.abs(difference)))
            row['converged'] = row['moment_discrepancy'] <= tolerance and (
                vertex_velocity is None or row['jacobian_discrepancy'] is not None and row['jacobian_discrepancy'] <= tolerance)
        records.append(row); previous=output
    return output, dict(records=records, converged=records[-1].get('converged',False),
        tolerance=tolerance, discrepancy_is_error_certificate=False,
        remaining_errors=['material representation','partial-cell rule','smoothing','native hard-cut mismatch'])
