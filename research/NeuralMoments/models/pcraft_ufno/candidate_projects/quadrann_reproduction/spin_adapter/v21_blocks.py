"""Disk-backed spatial index of a query-independent positive voxel cubature.

The material is sampled once. Queries can add whole blocks exactly when a
bounding sphere misses the complete cut transition; only crossing blocks
need to read their points. No query or learned feature enters this index.

New packs also retain compact, read-only material provenance. A nonlegacy full
cell rule is expanded from those runs only for requested blocks; the original
packed nodes continue to supply encoder inputs and the default tetra4 response.
"""
from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .v21_material import (
    PARTIAL_CELL_RULES, PreparedMaterial, _hash_arrays, _revision,
    validate_full_cell_rule,
)

_SCHEMA = "tvcg.positive-material-blocks.v1"
_PROVENANCE_SCHEMA = "tvcg.positive-material-block-provenance.v1"
_PROVENANCE_ARRAYS = ("full_runs", "partial_cells", "boundary_nodes")


def _save_provenance(material, directory):
    """Store only native runs and existing partial atoms, never refined full cells."""
    if not isinstance(material, PreparedMaterial):
        return None
    desc = dict(schema=_PROVENANCE_SCHEMA, identity=material.identity,
                source=material.source, partial_rule=material.partial_rule,
                revision=_revision(), density=float(material.density), audit=material.audit)
    arrays = [getattr(material, name) for name in _PROVENANCE_ARRAYS]
    desc["digest"] = _hash_arrays(json.dumps(desc, sort_keys=True, allow_nan=False), *arrays)
    for name, values in zip(_PROVENANCE_ARRAYS, arrays):
        np.save(directory / f"material_{name}.npy", values, allow_pickle=False)
    return desc


def _load_provenance(path, desc, *, identity, origin, pitch, side):
    if desc is None:
        return None
    if (desc.get("schema") != _PROVENANCE_SCHEMA or desc.get("identity") != identity
            or not isinstance(desc.get("revision"), str)):
        raise ValueError("invalid static block material provenance")
    source, rule = desc.get("source"), desc.get("partial_rule")
    if ((source == "mesh" and rule not in PARTIAL_CELL_RULES)
            or (source == "rle" and rule != "none") or source not in ("mesh", "rle")):
        raise ValueError("invalid static block material source or partial rule")
    density = desc.get("density")
    if not isinstance(density, (int, float)) or not np.isfinite(density) or density <= 0:
        raise ValueError("invalid static block material density")
    try:
        arrays = [np.load(path / f"material_{name}.npy", mmap_mode="r", allow_pickle=False)
                  for name in _PROVENANCE_ARRAYS]
    except (OSError, ValueError) as exc:
        raise ValueError("static block material provenance is missing or unreadable") from exc
    runs, cells, nodes = arrays
    if (any(a.ndim != 2 or a.shape[1] != 4 for a in arrays)
            or runs.dtype != np.int32 or cells.dtype != np.int32 or nodes.dtype != np.float64
            or int(cells[:, 3].sum(dtype=np.int64)) != len(nodes)
            or (source == "rle" and (len(cells) or len(nodes)))):
        raise ValueError("invalid static block material provenance arrays")
    payload = {k: v for k, v in desc.items() if k != "digest"}
    if _hash_arrays(json.dumps(payload, sort_keys=True, allow_nan=False), *arrays) != desc.get("digest"):
        raise ValueError("static block material provenance checksum mismatch")
    material_origin = np.array(origin, dtype=np.float64, copy=True)
    material_origin.flags.writeable = False
    return PreparedMaterial(material_origin, pitch, side, float(density), identity,
                            source, runs, cells, nodes, desc.get("audit", {}), rule)


def basis_at(points):
    x, y, z = points.T
    return np.column_stack((np.ones(len(x)), x, y, z, x*x, y*y, z*z, y*z, x*z, x*y))


@dataclass
class BlockMeasure:
    nodes: np.ndarray
    ids: np.ndarray
    offsets: np.ndarray
    moments: np.ndarray
    centers: np.ndarray
    radius: float
    origin: np.ndarray
    pitch: float
    side: int
    block_voxels: int
    identity: str
    material: PreparedMaterial | None = None

    @classmethod
    def open(cls, directory, *, identity, block_voxels):
        path = Path(directory)
        desc = json.loads((path / 'metadata.json').read_text())
        if (desc['schema'], desc['material'], desc['width']) != (_SCHEMA, identity, block_voxels):
            raise ValueError("static block cache identity or voxel width mismatch")
        if desc.get('stored_full_cell_rule', 'tetra4') != 'tetra4':
            raise ValueError("unsupported stored full-cell cubature rule")
        data = np.load(path / 'nodes.npy', mmap_mode='r', allow_pickle=False)
        with np.load(path / 'blocks.npz', allow_pickle=False) as f:
            ids, offsets, moments = f['ids'], f['offsets'], f['moments']
        side, pitch, origin = desc['side'], desc['pitch'], np.array(desc['origin'])
        if (data.shape != (desc['nodes'], 4) or len(ids) + 1 != len(offsets) or
                not len(ids) or offsets[0] != 0 or offsets[-1] != len(data) or
                np.any(np.diff(offsets) <= 0) or moments.shape != (len(ids), 10)):
            raise ValueError("invalid static block cache arrays")
        n = side // block_voxels
        ijk = np.array(np.unravel_index(ids, (n, n, n))).T
        edge = pitch * block_voxels
        material = _load_provenance(path, desc.get("material_provenance"), identity=identity,
                                    origin=origin, pitch=pitch, side=side)
        return cls(data, ids, offsets, moments, origin + (ijk + .5) * edge,
                   np.sqrt(3.) * edge / 2, origin, pitch, side, block_voxels, identity,
                   material)

    @classmethod
    def build(cls, material, directory, *, block_voxels=8, chunk_size=131072):
        path = Path(directory)
        if block_voxels < 1 or material.side % block_voxels or chunk_size < 1:
            raise ValueError("block width must divide grid side and chunk size must be positive")
        if path.exists():
            return cls.open(path, identity=material.identity, block_voxels=block_voxels)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=path.name + '.', dir=path.parent))
        try:
            n = material.side // block_voxels
            origin = np.asarray(material.origin, dtype=float)
            edge = material.pitch * block_voxels
            def block_ids(points):
                coordinates = (points - origin) / edge
                if np.any(coordinates < -1e-10) or np.any(coordinates > n + 1e-10):
                    raise ValueError("cubature point outside registered voxel grid")
                ijk = np.clip(np.floor(coordinates).astype(np.int64), 0, n-1)
                return np.ravel_multi_index(ijk.T, (n, n, n))

            masses = np.zeros((n**3, 10))
            counts = np.zeros(n**3, dtype=np.int64)
            with tempfile.TemporaryFile(dir=tmp) as spool:
                for points, weights in material.iter_chunks(chunk_size=chunk_size):
                    points, weights = np.asarray(points, dtype=float), np.asarray(weights, dtype=float)
                    if (points.shape != (len(weights), 3) or weights.ndim != 1 or
                            not np.isfinite(points).all() or not np.isfinite(weights).all() or
                            np.any(weights <= 0)):
                        raise ValueError("invalid positive cubature")
                    keys = block_ids(points)
                    counts += np.bincount(keys, minlength=len(counts))
                    basis = basis_at(points) * weights[:, None]
                    for j in range(10):
                        masses[:,j] += np.bincount(keys, weights=basis[:,j], minlength=len(counts))
                    np.column_stack((points, weights)).astype('<f8', copy=False).tofile(spool)
                offsets = np.r_[0, counts.cumsum()]
                if not offsets[-1]:
                    raise ValueError("empty material")
                out = np.lib.format.open_memmap(tmp / 'nodes.npy', mode='w+', dtype='<f8',
                                                shape=(int(offsets[-1]), 4))
                cursor = offsets[:-1].copy()
                spool.seek(0)
                while True:
                    rows = np.fromfile(spool, dtype='<f8', count=4*chunk_size).reshape(-1, 4)
                    if not len(rows):
                        break
                    keys = block_ids(rows[:, :3])
                    order = np.argsort(keys, kind='stable')
                    groups, first, sizes = np.unique(keys[order], return_index=True, return_counts=True)
                    dst = np.repeat(cursor[groups] - first, sizes) + np.arange(len(rows))
                    out[dst] = rows[order]
                    cursor[groups] += sizes
                if not np.array_equal(cursor, offsets[1:]):
                    raise RuntimeError("spatial index node count changed while packing")
                out.flush()
                del out
            active = np.flatnonzero(counts)
            np.savez(tmp / 'blocks.npz', ids=active, offsets=np.r_[offsets[active], offsets[-1]],
                     moments=masses[active])
            desc = dict(schema=_SCHEMA, material=material.identity, width=int(block_voxels),
                        pitch=float(material.pitch), side=int(material.side), origin=origin.tolist(),
                        nodes=int(offsets[-1]), stored_full_cell_rule="tetra4")
            provenance = _save_provenance(material, tmp)
            if provenance is not None:
                desc["material_provenance"] = provenance
            (tmp / 'metadata.json').write_text(json.dumps(desc, allow_nan=False))
            try:
                os.rename(tmp, path)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not path.is_dir():
                    raise
            return cls.open(path, identity=material.identity, block_voxels=block_voxels)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def classify(self, surface, bandwidth, operation):
        if not np.isfinite(bandwidth) or bandwidth <= 0 or operation not in ('intersection', 'difference'):
            raise ValueError("invalid bandwidth or Boolean operation")
        d = surface.signed_distance(self.centers)
        if not np.isfinite(d).all():
            raise ValueError("nonfinite signed distance")
        safety = 64 * np.finfo(float).eps * (1 + np.max(np.abs(self.centers)))
        threshold = self.radius + bandwidth + safety + getattr(surface, 'tolerance', 0.)
        inside, outside = d > threshold, d < -threshold
        return (inside if operation == 'intersection' else outside), ~(inside | outside)

    def chunks(self, mask, max_nodes=65536, *, full_cell_rule="tetra4"):
        """Read legacy atoms, or lazily refine only full cells in selected blocks.

        Existing packs without material provenance remain usable with tetra4.
        They cannot safely infer full/partial support from their ten moments.
        """
        rule = validate_full_cell_rule(full_cell_rule)
        mask = np.asarray(mask)
        if mask.shape != self.ids.shape or mask.dtype != bool or max_nodes < 1:
            raise ValueError("expected one Boolean per spatial block and positive chunk capacity")
        if rule != "tetra4":
            if self.material is None:
                raise ValueError("full_cell_rule=" + rule + " requires material provenance; "
                                 "rebuild the block pack from its original PreparedMaterial")
            material = self.material
            if (material.identity != self.identity or material.side != self.side
                    or material.pitch != self.pitch or not np.array_equal(material.origin, self.origin)):
                raise ValueError("static block material provenance no longer matches the pack")
            n = self.side // self.block_voxels
            edge = self.pitch * self.block_voxels
            for i in np.flatnonzero(mask):
                ijk = np.asarray(np.unravel_index(int(self.ids[i]), (n, n, n)))
                low = self.origin + ijk * edge
                high = self.origin + (ijk + 1) * edge
                yield from material.iter_chunks(chunk_size=max_nodes, box=(low, high),
                                                full_cell_rule=rule)
            return
        pending, used = [], 0
        for i in np.flatnonzero(mask):
            start, end = self.offsets[i:i+2]
            while start < end:
                take = min(end-start, max_nodes-used)
                pending.append(self.nodes[start:start+take])
                used += take
                start += take
                if used == max_nodes:
                    a = np.concatenate(pending)
                    yield a[:,:3], a[:,3]
                    used, pending = 0, []
        if pending:
            a = np.concatenate(pending)
            yield a[:,:3], a[:,3]

    def encoder_grid(self, grid, density):
        """The four existing P-CRAFT input channels from conservative material."""
        from scipy.ndimage import binary_erosion, distance_transform_edt
        if grid < 1 or self.side % grid or not np.isfinite(density) or density <= 0:
            raise ValueError("invalid encoder grid or density")
        cell = self.side * self.pitch / grid
        masses = np.zeros(grid**3)
        for start in range(0, len(self.nodes), 131072):
            rows = self.nodes[start:start+131072]
            ijk = np.floor((rows[:, :3]-self.origin)/cell).astype(np.int64).clip(0, grid-1)
            idx = np.ravel_multi_index(ijk.T, (grid,)*3)
            masses += np.bincount(idx, weights=rows[:, 3], minlength=grid**3)
        fraction = masses.reshape((grid,)*3) / (density * cell**3)
        if fraction.max() > 1 + 1e-8:
            raise ValueError("cell volume fraction exceeds one")
        fraction = np.minimum(fraction, 1)
        occupied = fraction > 0
        signed = (-np.ones_like(fraction) if occupied.all() else
                  distance_transform_edt(~occupied) - distance_transform_edt(occupied))
        boundary = occupied & ~binary_erosion(occupied, structure=np.ones((3,)*3), border_value=0)
        return np.stack((fraction, occupied, np.tanh(signed/2), boundary)).astype(np.float32)
