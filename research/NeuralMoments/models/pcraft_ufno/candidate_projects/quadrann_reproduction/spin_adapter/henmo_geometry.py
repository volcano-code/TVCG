"""One-time conservative inputs and reusable P-CRAFT features for HENMO.

The native material side is independent of both the context and feature grids.
PreparedMaterial uses analytic run aggregation, so a 1024 material does not
require a 1024-cubed dense tensor or a complete expanded point pack.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
import torch

from .henmo_hierarchy import SparseMaterialHierarchy
from .henmo_cubature import moment_basis
from .v21_material import PreparedMaterial


def geometry_versions(model):
    # Runtime query choices (coarse level/execution) do not change an encoding.
    # Its input grids and deterministic bypass do: reject previously learned
    # tables if those semantics change even when parameter tensors do not.
    result = [('encoding_inputs', model.config.feature_grid,
               tuple(model.config.context_grids), model.config.response_path)]
    for prefix, module in (('encoder',model.shared_encoder),('summary',model.geometry_attention),
                           ('local_geometry',getattr(model,'local_geometry',None))):
        if module is None:
            continue
        for kind, values in (('parameter',module.named_parameters()),('buffer',module.named_buffers())):
            for name,p in values:
                result.append((prefix+'.'+kind+'.'+name,id(p),p._version,p.data_ptr(),p.dtype,p.device,p.requires_grad))
    return tuple(result)


class GraphLifetime:
    def __init__(self):
        self.consumed = False

    def backward(self, value):
        self.consumed = True
        return value


@dataclass(frozen=True)
class MaterialHandle:
    identity: str
    side: int
    pitch: float
    origin: np.ndarray


@dataclass(frozen=True)
class HENMOGeometry:
    hierarchy: SparseMaterialHierarchy
    blocks: object
    fixed_moments: np.ndarray
    table: torch.Tensor
    feature_grid: int
    owner: int
    versions: tuple
    table_version: int
    fixed_bytes: bytes
    inference_only: bool
    graph_lifetime: GraphLifetime | None
    node_contexts: dict = field(default_factory=dict, compare=False)
    node_context_bytes: dict = field(default_factory=lambda: {'bytes':0}, compare=False)

    @property
    def nbytes(self):
        return (self.table.numel()*self.table.element_size()+self.fixed_moments.nbytes
                +self.node_context_bytes['bytes'])

    def context(self, points):
        tree = self.hierarchy
        origin = torch.tensor(tree.origin.copy(), dtype=points.dtype, device=points.device)
        index = ((points-origin) * (self.feature_grid/(tree.side*tree.pitch))).floor().long()
        index = index.clamp(0, self.feature_grid-1)
        keys = (index[:,0]*self.feature_grid + index[:,1])*self.feature_grid + index[:,2]
        return self.table[keys]


def make_hierarchy(source, config):
    if isinstance(source, SparseMaterialHierarchy):
        tree = source
    elif isinstance(source, PreparedMaterial):
        tree = SparseMaterialHierarchy.from_material(source, leaf_grid=config.leaf_grid,
                                                     cache_bytes=config.material_cache_bytes)
    elif hasattr(source, 'block_voxels'):
        if source.side // source.block_voxels == config.leaf_grid:
            tree = SparseMaterialHierarchy.from_blocks(source, cache_bytes=config.material_cache_bytes)
        elif source.material is not None:
            tree = SparseMaterialHierarchy.from_material(source.material, leaf_grid=config.leaf_grid,
                                                         cache_bytes=config.material_cache_bytes)
        else:
            raise ValueError('changing packed leaf resolution requires original material provenance')
    else:
        raise ValueError('expected original PreparedMaterial, BlockMeasure or material hierarchy')
    if tree.side != config.native_material_resolution or tree.leaf_grid != config.leaf_grid:
        raise ValueError('actual material/leaf resolution differs from HENMO scientific config')
    tree.assert_source_unchanged()
    return tree


def aggregate_grid(tree, grid):
    """Positive, physical ten-moment rows on a chosen *context* grid."""
    if type(grid) is not int or grid < 1 or tree.side % grid:
        raise ValueError('aggregation grid must divide native material side')
    table = np.zeros((grid**3, 10), dtype=np.float64)
    if tree.material is not None:
        indices, moments = tree.material.block_moments(tree.side//grid)
        table[np.ravel_multi_index(indices.T, (grid,)*3)] = moments
    elif grid <= tree.leaf_grid:
        centers = np.array([tree.nodes[i].center for i in tree.leaf_nodes])
        index = np.floor((centers-tree.origin) * grid/(tree.pitch*tree.side)).astype(np.int64)
        keys = np.ravel_multi_index(index.T, (grid,)*3)
        np.add.at(table, keys, np.array([tree.nodes[i].moments for i in tree.leaf_nodes]))
    else:
        # Legacy packs may lack runs. Scan their existing positive atoms only
        # during preparation, never at each query and never invent fine support.
        for points, weights in tree.chunks(tree.root, max_nodes=131072, full_cell_rule='tetra4'):
            index = np.floor((points-tree.origin)*grid/(tree.pitch*tree.side)).astype(np.int64).clip(0,grid-1)
            keys = np.ravel_multi_index(index.T, (grid,)*3)
            values = moment_basis(points)*weights[:,None]
            for k in range(10):
                table[:,k] += np.bincount(keys, weights=values[:,k], minlength=grid**3)
    return table


def encoder_grid(tree, grid, density):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    width = tree.side*tree.pitch/grid
    fraction = aggregate_grid(tree, grid)[:,0].reshape((grid,)*3)/(density*width**3)
    if not np.isfinite(fraction).all() or fraction.min() < -1e-12 or fraction.max() > 1+1e-8:
        raise ValueError('material density or material volume fraction is inconsistent')
    fraction = np.clip(fraction,0.,1.)
    occupied = fraction > 0
    signed = (-np.ones_like(fraction) if occupied.all() else
              distance_transform_edt(~occupied)-distance_transform_edt(occupied))
    boundary = occupied & ~binary_erosion(occupied, structure=np.ones((3,)*3), border_value=0)
    return np.stack((fraction, occupied, np.tanh(signed/2), boundary)).astype(np.float32)


@dataclass(frozen=True)
class StaticEncoderInputs:
    """Weight-independent tensors only; never a learned encoding or AD graph."""
    active: torch.Tensor
    features: torch.Tensor
    coords: torch.Tensor
    stats: torch.Tensor
    log_mass: torch.Tensor
    images: tuple[torch.Tensor, ...]

    @property
    def tensors(self):
        return (self.active, self.features, self.coords, self.stats,
                self.log_mass, *self.images)

    @property
    def nbytes(self):
        return sum(value.numel()*value.element_size() for value in self.tensors)

    @property
    def versions(self):
        return tuple((id(value), value.data_ptr(), value._version, value.shape,
                      value.dtype, value.device) for value in self.tensors)


class StaticGeometryInputCache:
    """Byte-bounded, per-operator LRU for immutable geometry preparation.

    A hit skips material aggregation, distance fields and device transfer, but
    every ``encode_table`` still executes the encoder and attention. Entries
    contain ordinary non-grad tensors, including when first built inside
    ``inference_mode``; this permits a later training forward to save them for
    backward. The budget counts owned tensor storage, not allocator reservation
    or tensors temporarily retained by a currently live training graph.
    """
    def __init__(self, max_bytes=128*1024**2):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('static input cache max_bytes must be a nonnegative integer')
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._evictions = self._oversized = 0

    def clear(self):
        self._entries.clear()
        self._bytes = 0

    def snapshot(self):
        return dict(entries=len(self._entries), bytes=self._bytes,
                    max_bytes=self.max_bytes, hits=self._hits, misses=self._misses,
                    evictions=self._evictions, oversized=self._oversized,
                    learned_features_cached=False,
                    storage='weight-independent device tensors; no autograd graphs')

    def prepare(self, model, tree, fixed, density):
        # Do this on hits as well. In particular, a changed read-only mmap or
        # replaced material array must not be concealed by a stale cache hit.
        tree.assert_source_unchanged()
        key = _static_input_key(model, tree, fixed, density)
        previous = self._entries.get(key)
        if previous is not None:
            inputs, versions = previous
            if inputs.versions != versions:
                raise ValueError('static geometry input cache tensor was modified; clear the cache')
            self._entries.move_to_end(key)
            self._hits += 1
            return inputs
        self._misses += 1
        inputs = _prepare_static_inputs(model, tree, fixed, density)
        if inputs.nbytes > self.max_bytes:
            self._oversized += 1
            return inputs
        while self._entries and self._bytes+inputs.nbytes > self.max_bytes:
            _, (old, _) = self._entries.popitem(last=False)
            self._bytes -= old.nbytes
            self._evictions += 1
        self._entries[key] = (inputs, inputs.versions)
        self._bytes += inputs.nbytes
        return inputs


def _static_input_key(model, tree, fixed, density):
    fixed = np.asarray(fixed, dtype=np.float64)
    if fixed.shape != (10,) or not np.isfinite(fixed).all():
        raise ValueError('static encoding requires ten finite fixed moments')
    if not np.isfinite(density) or density <= 0:
        raise ValueError('static encoding requires positive finite density')
    parameter = next(model.shared_encoder.parameters())
    config = model.config
    return ('henmo.static-encoder-inputs.v1', tree.identity, tree.material_identity,
            tree.side, tree.pitch, tuple(tree.origin), tree.leaf_grid,
            config.feature_grid, tuple(config.context_grids), float(density),
            fixed.tobytes(), str(parameter.device), parameter.dtype)


def _prepare_static_inputs(model, tree, fixed, density):
    """The original preparation arithmetic, separated from learned forwards."""
    config = model.config
    grid = config.feature_grid
    masses = aggregate_grid(tree, grid)
    active = np.flatnonzero(masses[:,0] > 0)
    m = masses[active]
    editable = m.sum(0)
    span = tree.side*tree.pitch
    coord = span**np.array([0,1,1,1,2,2,2,2,2,2])
    means = m/m[:,:1]
    width = span/grid
    centers = tree.origin+(np.array(np.unravel_index(active,(grid,)*3)).T+.5)*width
    low, high = centers.min(0)-width/2, centers.max(0)+width/2
    scale = (editable[0]+fixed[0])*coord
    if scale[0] <= 0:
        raise ValueError('positive total material mass required')
    features = np.column_stack((means/coord,np.full((len(m),3),width/span)))
    stats = np.r_[(editable+fixed)/scale, fixed/scale, editable/scale,
                  (low+high)/(2*span),(high-low)/span]
    coords = (2*(means[:,1:4]-tree.origin)/span-1)[:,[2,1,0]].copy()
    parameter = next(model.shared_encoder.parameters())
    # Inputs first prepared during evaluation must remain legal saved tensors
    # when a subsequent real update reuses this static cache.
    with torch.inference_mode(False), torch.no_grad():
        def tensor(value):
            return torch.tensor(value, dtype=parameter.dtype, device=parameter.device)
        return StaticEncoderInputs(
            active=torch.tensor(active, device=parameter.device),
            features=tensor(features), coords=tensor(coords), stats=tensor(stats),
            log_mass=tensor(np.log(m[:,0]/editable[0]))[None],
            images=tuple(tensor(encoder_grid(tree, g, density))[None]
                         for g in config.context_grids))


def encode_table(model, tree, fixed, density, *, static_cache=None):
    """Recompute learned features, optionally reusing only static inputs."""
    if static_cache is None:
        tree.assert_source_unchanged()
        _static_input_key(model, tree, fixed, density)
        inputs = _prepare_static_inputs(model, tree, fixed, density)
    elif isinstance(static_cache, StaticGeometryInputCache):
        inputs = static_cache.prepare(model, tree, fixed, density)
    else:
        raise TypeError('static_cache must be a StaticGeometryInputCache or None')
    config = model.config
    grid = config.feature_grid
    encoder = model.shared_encoder
    maps, globals_ = [], []
    for image in inputs.images:
        pooled, feature_map = encoder.pino_encoder.forward_with_map(image, preserve_resolution=True)
        maps.append(feature_map)
        globals_.append(encoder.pino_global(pooled))
    glob, s = torch.stack(globals_,dim=1), inputs.stats[None]
    score = encoder.pino_scale_score(torch.cat((glob,s[:,None].expand(-1,len(maps),-1)),-1))
    shared = encoder.statistics_context(s)+(glob*score.softmax(1)).sum(1)
    local = encoder._interval_features(inputs.features,inputs.coords,maps,shared)
    summary,_ = model.geometry_attention(shared[:,None],local[None],local[None],
                                         attn_mask=inputs.log_mass,need_weights=False)
    return torch.index_copy(local.new_zeros((grid**3,local.shape[1])),0,
                            inputs.active,local+summary[0])
