"""Positive, degree-two native-cell cubature of an oriented triangle solid.

``prepare_material`` is offline and consumes only a triangle mesh and its grid.
For each native voxel it integrates the signed flux of the original clipped
faces. A positive four-point rule that reproduces the local mass, centroid and
covariance is accepted only if every point lies in material. Remaining cells
use a vertical decomposition of their clipped faces, GL(3,2,2), and positive
moment-preserving reduction. An explicit partial rule can instead retain the
raw sweep or compress that same sweep, without first trying the four-point fit.
Full voxels are stored as half-open z runs and expand on demand to tetra4
(legacy), gl2 (8 points), or gl3 (27 points). All ten moments are certified against a
separate signed tetrahedral mesh integral. Inexact arithmetic is float64.

Mesh coordinates and ``origin`` must share the physical frame. ``origin`` is
corner (0,0,0), and ``pitch`` is a scalar. External shells face outward and
cavity shells inward. A cache contains editable material only. A distinct
fixed moment vector belongs in the final response, once per query.

``iter_chunks`` returns physical XYZ and positive density-weighted volumes.
Grid-aligned world boxes select native cells using half-open bounds. ``moments``
and ``block_moments`` integrate interior runs in closed form. Static caches are
bound to frame, density, geometry arrays, algorithm revision and input type.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
import hashlib
import json
import operator
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .rle_quadrature import native_moment_basis


_SCHEMA = "compact-v21-cell-material-v2"
_NATIVE = Path(__file__).with_name("v21_material_native.cpp")
_SIGNS = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
) / np.sqrt(12.0)
FULL_CELL_RULES = ("tetra4", "gl2", "gl3")
PARTIAL_CELL_RULES = ("fit_compressed", "sweep_compressed", "sweep_raw")


def validate_full_cell_rule(rule):
    """Validate the query quadrature independently of material/encoder geometry."""
    if not isinstance(rule, str) or rule not in FULL_CELL_RULES:
        raise ValueError(f"full_cell_rule must be one of {FULL_CELL_RULES}")
    return rule


def _partial_rule(rule):
    if not isinstance(rule, str) or rule not in PARTIAL_CELL_RULES:
        raise ValueError(f"partial_rule must be one of {PARTIAL_CELL_RULES}")
    return rule


def full_cell_cubature(rule="tetra4"):
    """Positive unit-cube rule, with offsets about its center and unit total mass.

    This creates at most 27 reference points, never a dense material grid. The
    legacy ordering and coordinates are retained exactly for tetra4.
    """
    return _full_cell_cubature(validate_full_cell_rule(rule))


@lru_cache(maxsize=3)
def _full_cell_cubature(rule):
    if rule == "tetra4":
        points, weights = _SIGNS.copy(), np.full(4, 0.25)
    else:
        order = 2 if rule == "gl2" else 3
        axis, weight = np.polynomial.legendre.leggauss(order)
        index = np.indices((order,) * 3).reshape(3, -1).T
        points = 0.5 * axis[index]
        weights = np.prod(0.5 * weight[index], axis=1)
    points.flags.writeable = weights.flags.writeable = False
    return points, weights


def _index(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        n = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if n < 1:
        raise ValueError(f"{name} must be a positive integer")
    return n


def _grid(origin, pitch, side, density):
    origin = np.ascontiguousarray(origin, dtype=np.float64)
    side = _index(side, "side")
    pitch, density = float(pitch), float(density)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be a finite three-vector")
    if not np.isfinite([pitch, density]).all() or min(pitch, density) <= 0:
        raise ValueError("pitch and density must be finite and positive")
    if side**2 > np.iinfo(np.int32).max:
        raise ValueError("native column count exceeds the compiled integer index range")
    return origin, pitch, side, density


def _mesh(vertices, faces):
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError("vertices must be finite mesh coordinates [N,3]")
    if f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in "iu":
        raise ValueError("faces must be nonempty integer triangles [F,3]")
    if (
        f.min() < 0
        or f.max() >= len(v)
        or max(int(f.max()), len(f)) > np.iinfo(np.int32).max
    ):
        raise ValueError("invalid face index or triangle count")
    return v, np.ascontiguousarray(f, dtype=np.int32)


def _balanced_edges(faces):
    # An oriented closed triangular boundary has each geometric directed edge
    # equally often in both directions. Independent touching components may
    # legitimately give an edge more than two incident faces.
    forward = np.asarray(faces, dtype=np.int64).reshape(-1)
    backward = np.roll(faces, -1, axis=1).reshape(-1).astype(np.int64, copy=False)
    ascending = forward < backward
    descending = backward < forward
    up = (forward[ascending] << 32) | backward[ascending]
    down = (backward[descending] << 32) | forward[descending]
    return len(up) == len(down) and np.array_equal(np.sort(up), np.sort(down))


def _require_closed(vertices, faces):
    if _balanced_edges(faces):
        return
    # Accept indexed triangle soups that duplicate the same vertex coordinates
    # at facet boundaries, without moving any physical vertex.
    _, index = np.unique(vertices, axis=0, return_inverse=True)
    if not _balanced_edges(index[faces]):
        raise ValueError("mesh must form a closed oriented triangle boundary")


def _hash_arrays(tag, *arrays):
    h = hashlib.sha256(tag.encode("utf8"))
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode("ascii"))
        h.update(memoryview(a).cast("B") if a.size else b"")
    return h.hexdigest()


@lru_cache(maxsize=1)
def _revision():
    return hashlib.sha256(
        _NATIVE.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()


def _identity(kind, origin, pitch, side, density, *geometry, partial_rule="fit_compressed"):
    return _hash_arrays(
        _SCHEMA + _revision() + kind + ":partial=" + partial_rule,
        origin,
        np.asarray([pitch, side, density]),
        *geometry,
    )


def material_identity(vertices, faces, origin, pitch, side=256, density=1.0,
                      *, partial_rule="fit_compressed"):
    """Content key for locating a mesh cache ahead of one-time preparation."""
    return _identity(
        "mesh", *_grid(origin, pitch, side, density), *_mesh(vertices, faces),
        partial_rule=_partial_rule(partial_rule),
    )


@lru_cache(maxsize=1)
def _builder():
    # Cache only a locally compiled shared object, never dataset geometry. Unique
    # build directories and atomic installation permit concurrent first runs.
    root = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "compact-pino-v21"
        / "material"
    )
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_NATIVE.read_bytes()).hexdigest()
    so = root / f"{digest}.{os.uname().machine}.so"
    if not so.exists():
        with tempfile.TemporaryDirectory(dir=root) as td:
            output = Path(td) / "material.so"
            process = subprocess.run(
                [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    "-fopenmp",
                    str(_NATIVE),
                    "-o",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )
            if process.returncode:
                raise RuntimeError(
                    "Cannot compile native material builder: " + process.stderr
                )
            os.replace(output, so)
    build = ctypes.CDLL(str(so)).build_material
    build.argtypes = [
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    build.restype = ctypes.c_int
    return build


def signed_mesh_moments(vertices, faces, density=1.0):
    """Independent closed-surface fan of signed tetrahedra, in native moment order."""
    v, f = _mesh(vertices, faces)
    reference = 0.5 * (v.max(axis=0) + v.min(axis=0))
    total = np.zeros(10)
    for start in range(0, len(f), 32768):
        t = v[f[start : start + 32768]]
        a, b, c = (t - reference).transpose(1, 0, 2)
        w = (float(density) / 6) * np.einsum("ij,ij->i", a, np.cross(b, c))
        sums = t.sum(axis=1) + reference
        second = (
            sums[:, :, None] * sums[:, None, :]
            + np.einsum("nki,nkj->nij", t, t)
            + np.outer(reference, reference)
        ) / 20
        basis = np.column_stack(
            (
                np.ones(len(t)),
                sums / 4,
                second.diagonal(axis1=1, axis2=2),
                second[:, 1, 2],
                second[:, 0, 2],
                second[:, 0, 1],
            )
        )
        total += (w[:, None] * basis).sum(axis=0)
    return total


def _cuboid_moments(runs, origin, pitch, density):
    lo = origin + pitch * runs[:, :3]
    extent = np.full(lo.shape, pitch)
    extent[:, 2] = pitch * (runs[:, 3] - runs[:, 2])
    center = lo + 0.5 * extent
    b = native_moment_basis(center)
    b[:, 4:7] += extent * extent / 12
    return (density * pitch**2 * extent[:, 2])[:, None] * b


@dataclass
class PreparedMaterial:
    """Replayable static material. Boundary nodes use grid units in storage."""

    origin: np.ndarray
    pitch: float
    side: int
    density: float
    identity: str
    source: str
    full_runs: np.ndarray
    partial_cells: np.ndarray  # [i,j,k,node_count], matching consecutive boundary_nodes
    boundary_nodes: np.ndarray  # [grid_x,grid_y,grid_z,unit_cell_volume_weight]
    audit: dict = field(default_factory=dict)
    partial_rule: str = "fit_compressed"

    @property
    def full_cell_count(self):
        return int(np.sum(self.full_runs[:, 3] - self.full_runs[:, 2], dtype=np.int64))

    @property
    def partial_cell_count(self):
        return len(self.partial_cells)

    @property
    def node_count(self):
        return 4 * self.full_cell_count + len(self.boundary_nodes)

    def node_count_for_rule(self, full_cell_rule="tetra4"):
        """Potential expanded size; selection/chunking need not materialize it."""
        _, weights = full_cell_cubature(full_cell_rule)
        return len(weights) * self.full_cell_count + len(self.boundary_nodes)

    @cached_property
    def _partial_ptr(self):
        return np.r_[0, np.cumsum(self.partial_cells[:, 3], dtype=np.int64)]

    def _box(self, box):
        if box is None:
            return np.zeros(3, dtype=int), np.full(3, self.side, dtype=int)
        grid = (np.asarray(box, dtype=np.float64) - self.origin) / self.pitch
        if (
            grid.shape != (2, 3)
            or not np.isfinite(grid).all()
            or (grid[1] < grid[0]).any()
        ):
            raise ValueError(
                "box must contain ordered finite [lower_xyz,upper_xyz] bounds"
            )
        rounded = np.rint(grid)
        if np.max(np.abs(grid - rounded)) > 2e-8:
            raise ValueError("box bounds must coincide with native grid planes")
        return np.clip(rounded, 0, self.side).astype(np.int64)

    def _selected_runs(self, bounds):
        if bounds is None:
            return self.full_runs
        lo, hi = bounds
        xs = self.full_runs[:, 0].searchsorted([lo[0], hi[0]])
        runs = self.full_runs[xs[0] : xs[1]]
        runs = runs[
            (runs[:, 1] >= lo[1])
            & (runs[:, 1] < hi[1])
            & (runs[:, 3] > lo[2])
            & (runs[:, 2] < hi[2])
        ].copy()
        runs[:, 2] = np.maximum(runs[:, 2], lo[2])
        runs[:, 3] = np.minimum(runs[:, 3], hi[2])
        return runs

    def _boundary_chunks(self, bounds, size):
        if bounds is None:
            for a in range(0, len(self.boundary_nodes), size):
                yield self.boundary_nodes[a : a + size]
            return
        lo, hi = bounds
        a, b = self.partial_cells[:, 0].searchsorted([lo[0], hi[0]])
        cells = self.partial_cells[a:b]
        rows = a + np.flatnonzero(
            (cells[:, 1] >= lo[1])
            & (cells[:, 1] < hi[1])
            & (cells[:, 2] >= lo[2])
            & (cells[:, 2] < hi[2])
        )
        if not len(rows):
            return
        end = np.cumsum(self.partial_cells[rows, 3], dtype=np.int64)
        offset = self._partial_ptr[rows] - (end - self.partial_cells[rows, 3])
        for a in range(0, int(end[-1]), size):
            index = np.arange(a, min(a + size, end[-1]), dtype=np.int64)
            yield self.boundary_nodes[
                index + offset[np.searchsorted(end, index, side="right")]
            ]

    def iter_chunks(self, chunk_size=65536, box=None, *, full_cell_rule="tetra4"):
        """Stream positive atoms only for selected cells, retaining partial support.

        The full-cell rule changes only this expansion. Analytic run moments,
        partial nodes, density, and the material identity are unchanged.
        """
        size = _index(chunk_size, "chunk_size")
        offsets, rule_weights = full_cell_cubature(full_cell_rule)
        points_per_cell = len(rule_weights)
        bounds = None if box is None else self._box(box)
        runs = self._selected_runs(bounds)
        unit_mass = self.density * self.pitch**3
        if len(runs):
            ncell = runs[:, 3] - runs[:, 2]
            ends = np.cumsum(ncell, dtype=np.int64)
            for a in range(0, points_per_cell * int(ends[-1]), size):
                node = np.arange(a, min(a + size, points_per_cell * ends[-1]), dtype=np.int64)
                cell = node // points_per_cell
                row = np.searchsorted(ends, cell, side="right")
                xyz = runs[row, :3].astype(np.float64)
                xyz[:, 2] += cell - (ends[row] - ncell[row])
                xyz += 0.5 + offsets[node % points_per_cell]
                yield (
                    self.origin + self.pitch * xyz,
                    unit_mass * rule_weights[node % points_per_cell],
                )
        for p in self._boundary_chunks(bounds, size):
            yield self.origin + self.pitch * p[:, :3], unit_mass * p[:, 3]

    def moments(self, box=None):
        """Ten physical raw moments; full runs need no expanded nodes."""
        bounds = None if box is None else self._box(box)
        total = _cuboid_moments(
            self._selected_runs(bounds), self.origin, self.pitch, self.density
        ).sum(0)
        for nodes in self._boundary_chunks(bounds, 65536):
            xyz = self.origin + self.pitch * nodes[:, :3]
            total += np.sum(
                (nodes[:, 3] * self.density * self.pitch**3)[:, None]
                * native_moment_basis(xyz),
                axis=0,
            )
        return total

    def block_moments(self, block_width=8):
        """Return occupied block indices [B,3] and exact ten-moment rows [B,10]."""
        width = _index(block_width, "block_width")
        n = (self.side + width - 1) // width
        # An arbitrary one-voxel block request should scale with occupied cells.
        # Sorting sparse keys avoids allocating all side**3 rows of ten doubles.
        keys, vals = [], []

        def accumulate(idx, m):
            if len(idx):
                code = (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]
                unique, inverse = np.unique(code, return_inverse=True)
                summed = np.stack(
                    [
                        np.bincount(inverse, weights=m[:, j], minlength=len(unique))
                        for j in range(10)
                    ],
                    axis=1,
                )
                keys.append(unique)
                vals.append(summed)

        runs = self.full_runs
        for k in range(n):
            hit = (runs[:, 3] > k * width) & (runs[:, 2] < (k + 1) * width)
            part = runs[hit].copy()
            part[:, 2] = np.maximum(part[:, 2], k * width)
            part[:, 3] = np.minimum(part[:, 3], (k + 1) * width)
            accumulate(
                part[:, :3].astype(np.int64) // width,
                _cuboid_moments(part, self.origin, self.pitch, self.density),
            )
        for nodes in self._boundary_chunks(None, 65536):
            xyz = self.origin + self.pitch * nodes[:, :3]
            weighted = (nodes[:, 3] * self.density * self.pitch**3)[
                :, None
            ] * native_moment_basis(xyz)
            accumulate(np.floor(nodes[:, :3] / width).astype(np.int64), weighted)
        if not keys:
            return np.empty((0, 3), dtype=np.int64), np.empty((0, 10))
        unique, inverse = np.unique(np.concatenate(keys), return_inverse=True)
        vals = np.concatenate(vals)
        total = np.stack(
            [
                np.bincount(inverse, weights=vals[:, j], minlength=len(unique))
                for j in range(10)
            ],
            axis=1,
        )
        return np.stack(np.unravel_index(unique, (n, n, n)), axis=1), total

    def save(self, path):
        """Atomically save the geometry-only payload and its content digest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            dict(
                schema=_SCHEMA,
                revision=_revision(),
                origin=self.origin.tolist(),
                pitch=self.pitch,
                side=self.side,
                density=self.density,
                identity=self.identity,
                source=self.source,
                partial_rule=self.partial_rule,
                default_full_cell_rule="tetra4",
                audit=self.audit,
            ),
            sort_keys=True,
            allow_nan=False,
        )
        digest = _hash_arrays(
            metadata, self.full_runs, self.partial_cells, self.boundary_nodes
        )
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            try:
                np.savez(
                    handle,
                    metadata=metadata,
                    digest=digest,
                    full_runs=self.full_runs,
                    partial_cells=self.partial_cells,
                    boundary_nodes=self.boundary_nodes,
                )
                handle.close()
                os.replace(handle.name, path)
            finally:
                Path(handle.name).unlink(missing_ok=True)


def load_material(path, expected_identity=None):
    """Read a certified static cache; optionally require a particular mesh/frame key."""
    with np.load(path, allow_pickle=False) as a:
        metadata = str(a["metadata"])
        info = json.loads(metadata)
        if info["schema"] != _SCHEMA or info["revision"] != _revision():
            raise ValueError("unsupported material cache revision")
        if expected_identity is not None and info["identity"] != expected_identity:
            raise ValueError("material cache identity mismatch")
        partial_rule = info.get("partial_rule")
        if ((info["source"] == "mesh" and partial_rule not in PARTIAL_CELL_RULES)
                or (info["source"] == "rle" and partial_rule != "none")
                or info["source"] not in ("mesh", "rle")
                or info.get("default_full_cell_rule") != "tetra4"):
            raise ValueError("invalid material cache source or quadrature provenance")
        runs, cells, nodes = (
            a[k] for k in ("full_runs", "partial_cells", "boundary_nodes")
        )
        if _hash_arrays(metadata, runs, cells, nodes) != str(a["digest"]):
            raise ValueError("material cache payload checksum mismatch")
    return PreparedMaterial(
        *_grid(info["origin"], info["pitch"], info["side"], info["density"]),
        info["identity"],
        info["source"],
        runs,
        cells,
        nodes,
        info["audit"],
        partial_rule,
    )


def prepare_material(vertices, faces, origin, pitch, side=256, density=1.0,
                     *, partial_rule="fit_compressed"):
    """Build an editable material cache from a registered oriented physical mesh."""
    origin, pitch, side, density = _grid(origin, pitch, side, density)
    partial_rule = _partial_rule(partial_rule)
    v, f = _mesh(vertices, faces)
    _require_closed(v, f)
    key = _identity("mesh", origin, pitch, side, density, v, f, partial_rule=partial_rule)
    g = np.ascontiguousarray((v - origin) / pitch)
    if (g < -1e-11).any() or (g > side + 1e-11).any():
        raise ValueError("native lattice does not contain mesh vertices")
    np.clip(g, 0, side, out=g)
    exact = signed_mesh_moments(v, f, density)
    if exact[0] <= 0:
        raise ValueError(
            "mesh must have positive signed material volume; check shell orientation"
        )
    threads = _index(
        int(os.environ.get("V21_MATERIAL_THREADS", min(8, os.cpu_count() or 1))),
        "V21_MATERIAL_THREADS",
    )
    with tempfile.TemporaryDirectory(prefix="v21-material-") as folder:
        error = ctypes.create_string_buffer(2048)
        if _builder()(
            g, f, len(f), side, PARTIAL_CELL_RULES.index(partial_rule),
            os.fsencode(folder), threads, error, len(error)
        ):
            raise ValueError(error.value.decode("utf8", "replace"))
        work = Path(folder)
        runs = np.fromfile(work / "runs", dtype=np.int32).reshape(-1, 4)
        cells = np.fromfile(work / "partials", dtype=np.int32).reshape(-1, 4)
        nodes = np.fromfile(work / "nodes", dtype=np.float64).reshape(-1, 4)
        fallback, outside, residual = (work / "stats").read_text().split()
    if (
        len(nodes) and (not np.isfinite(nodes).all() or np.min(nodes[:, 3]) <= 0)
    ) or int(cells[:, 3].sum()) != len(nodes):
        raise ValueError("invalid positive boundary cubature payload")
    m = PreparedMaterial(origin, pitch, side, density, key, "mesh", runs, cells, nodes,
                         partial_rule=partial_rule)
    measured = m.moments()
    length = float(np.ptp(v, axis=0).max())
    scale = exact[0] * np.array([1.0, length, length, length, *([length**2] * 6)])
    error = np.abs(measured - exact) / scale
    if not np.isfinite(error).all() or np.max(error) > 1e-8:
        raise ValueError(
            f"positive material failed signed-mesh ten-moment audit: {error.tolist()}"
        )
    m.audit = dict(
        signed_mesh_moments=exact.tolist(),
        material_moments=measured.tolist(),
        moment_scales=scale.tolist(),
        scale_relative_residuals=error.tolist(),
        max_scale_relative_error=float(error.max()),
        max_local_unit_cell_flux_error=float(residual),
        swept_cells=int(fallback),
        partial_centers_outside=int(outside),
        full_cells=m.full_cell_count,
        partial_cells=m.partial_cell_count,
        nodes=m.node_count,
        material_source="mesh",
        partial_rule=partial_rule,
        default_full_cell_rule="tetra4",
    )
    return m


def prepare_rle_material(columns, origin, pitch, side=256, density=1.0):
    """Pure center-occupancy RLE control, using the same positive four-point cell rule."""
    origin, pitch, side, density = _grid(origin, pitch, side, density)
    c = np.asarray(columns)
    if c.ndim != 2 or c.shape[1] != 4 or c.dtype.kind not in "iu":
        raise ValueError("columns must be integer half-open runs [x,y,z0,z1]")
    if len(c) and (
        (c < 0).any()
        or (c[:, :3] >= side).any()
        or (c[:, 3] > side).any()
        or (c[:, 3] <= c[:, 2]).any()
    ):
        raise ValueError("invalid native RLE bounds")
    c = np.ascontiguousarray(c[np.lexsort((c[:, 2], c[:, 1], c[:, 0]))], dtype=np.int32)
    if (
        len(c) > 1
        and (((c[1:, :2] == c[:-1, :2]).all(axis=1)) & (c[1:, 2] < c[:-1, 3])).any()
    ):
        raise ValueError("overlapping native RLE runs")
    key = _identity("rle", origin, pitch, side, density, c, partial_rule="none")
    return PreparedMaterial(
        origin,
        pitch,
        side,
        density,
        key,
        "rle",
        c,
        np.empty((0, 4), np.int32),
        np.empty((0, 4)),
        partial_rule="none",
    )
