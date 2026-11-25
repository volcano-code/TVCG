"""Bounded caching of parameter-independent Compact PINO geometry inputs.

This module stores only NumPy inputs, never learned features or an autograd
graph.  The input construction deliberately follows ``encode_geometry``'s
original reduction order.  Every call checks the material signature.  Small
metadata arrays and ordinary arrays are hashed by content; read-only file
mappings use a checked file stamp and mapping/view identity so that repeated
queries do not scan a potentially enormous material point file.

File stamps are a within-process cache invalidation mechanism, not a substitute
for the teacher/material SHA256 admission checks.  Concurrent mutation during
preparation is rejected.  As with the material loader, callers must not mutate
the material concurrently with an individual lookup.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import numbers
from pathlib import Path
import threading
import weakref

import numpy as np


_INPUT_VERSION = "compact-pino.geometry-inputs.v1"
_MATERIAL_SIGNATURE_VERSION = "compact-pino.material-signature.v1"
_mapping_lock = threading.Lock()
_mapping_ids = {}
_mapping_serial = itertools.count(1)


def _mapping_token(array):
    """Identity that cannot be reused after an old mapping is garbage-collected."""
    ident = id(array)
    with _mapping_lock:
        current = _mapping_ids.get(ident)
        if current is not None and current[0]() is array:
            return current[1]
        token = next(_mapping_serial)

        def discard(reference):
            with _mapping_lock:
                item = _mapping_ids.get(ident)
                if item is not None and item[0] is reference:
                    del _mapping_ids[ident]

        _mapping_ids[ident] = (weakref.ref(array, discard), token)
        return token


def _array_digest(value):
    """Hash shape/dtype and C-order bytes without one huge contiguous copy."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("material identity cannot contain object arrays")
    digest = hashlib.sha256()
    digest.update(repr((array.dtype.str, array.shape)).encode("ascii"))
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
    else:
        iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                             op_flags=["readonly"], order="C", buffersize=131072)
        for chunk in iterator:
            digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def _readonly_mapping_signature(value):
    """Return a view-aware stamp only when the entire array base chain is read-only.

    A read-only view of writable memory is not immutable.  In particular,
    read-only views of ``mode='r+'`` or copy-on-write memmaps take the content
    hash path.  Mapping identity distinguishes an old mapping from a newly
    opened file after an atomic replacement at the same pathname.
    """
    array = np.asarray(value)
    current, root, seen = value, None, set()
    while isinstance(current, np.ndarray) and id(current) not in seen:
        seen.add(id(current))
        if current.flags.writeable:
            return None
        if isinstance(current, np.memmap):
            if current.mode != "r":
                return None
            root = current
        current = current.base
    if root is None or array.flags.writeable or not getattr(root, "filename", None):
        return None
    try:
        path = Path(root.filename).resolve(strict=True)
        stamp = path.stat()
    except (OSError, ValueError):
        # A still-live mapping can outlive the file name.  Hash its actual bytes.
        return None
    offset = int(root.offset) + int(array.__array_interface__["data"][0]) - int(root.__array_interface__["data"][0])
    return ("readonly_memmap", _mapping_token(root), str(path),
            stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns,
            offset, array.shape, array.dtype.str, array.strides)


def material_signature(blocks):
    """Hashable cache identity for all material state used by either response path.

    The large read-only node mapping is inspected in constant work, while the
    much smaller block moments/coordinates/indices are content-hashed.  Mutable
    or ordinary in-memory nodes are content-hashed each time.  No caller's
    mutable data is silently frozen and no material array is retained here.
    """
    nodes = _readonly_mapping_signature(blocks.nodes)
    if nodes is None:
        nodes = ("content", _array_digest(blocks.nodes))
    arrays = tuple((name, _array_digest(getattr(blocks, name)))
                   for name in ("ids", "offsets", "moments", "centers", "origin"))
    return (_MATERIAL_SIGNATURE_VERSION, type(blocks).__module__, type(blocks).__qualname__,
            str(blocks.identity), int(blocks.side), int(blocks.block_voxels),
            float(blocks.pitch), float(blocks.radius), nodes, arrays)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _arguments(blocks, fixed_moments, density, token_grid, grids):
    token_grid = _positive_int(token_grid, "token_grid")
    if token_grid not in (8, 16):
        raise ValueError("expected Token8 or Token16")
    if blocks.side != 256 or blocks.side % (blocks.block_voxels * token_grid):
        raise ValueError("native256 blocks must nest exactly within the token grid")
    fixed = np.array(fixed_moments, dtype=float, copy=True)
    if fixed.shape != (10,) or not np.isfinite(fixed).all():
        raise ValueError("expected ten fixed moments")
    density = float(density)
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be positive and finite")
    grids = tuple(_positive_int(grid, "encoder grid") for grid in grids)
    if not grids or any(blocks.side % grid for grid in grids):
        raise ValueError("encoder grids must be nonempty and divide the native side")
    return fixed, density, token_grid, grids


def _readonly(value):
    array = np.array(value, copy=True, order="C")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class PreparedGeometryInputs:
    token_grid: int
    token_count: int
    grids: tuple[int, ...]
    fixed_moments: np.ndarray
    active: np.ndarray
    features: np.ndarray
    stats: np.ndarray
    coords: np.ndarray
    log_mass: np.ndarray
    encoder_inputs: tuple[np.ndarray, ...]

    @property
    def nbytes(self):
        """Resident NumPy payload bytes; Python bookkeeping is separately bounded."""
        return int(sum(array.nbytes for array in
                       (self.fixed_moments, self.active, self.features, self.stats,
                        self.coords, self.log_mass, *self.encoder_inputs)))


def prepare_geometry_inputs(blocks, fixed_moments, density=1., *, token_grid=16, grids=(32, 64)):
    """The original static encoder-input calculation, with no neural parameters."""
    fixed, density, token_grid, grids = _arguments(blocks, fixed_moments, density, token_grid, grids)
    token_count = token_grid**3
    span = blocks.side * blocks.pitch
    ijk = np.floor((blocks.centers - blocks.origin)/span * token_grid).astype(int)
    ids = np.ravel_multi_index(ijk.T, (token_grid,)*3)
    mass_table = np.zeros((token_count, 10))
    np.add.at(mass_table, ids, blocks.moments)
    active = np.flatnonzero(mass_table[:, 0] > 0)
    if not len(active):
        raise ValueError("static encoder inputs require nonempty editable material")
    m = mass_table[active]
    editable = m.sum(0)
    means = m / m[:, :1]
    width = span / token_grid
    coord = np.power(span, [0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    scale = (editable[0] + fixed[0]) * coord
    centers = blocks.origin + (np.array(np.unravel_index(active, (token_grid,)*3)).T + .5)*width
    lo, hi = centers.min(0)-width/2, centers.max(0)+width/2
    features = np.column_stack((means/coord, np.full((len(m), 3), width/span)))
    stats = np.r_[(editable+fixed)/scale, fixed/scale, editable/scale,
                  (lo+hi)/(2*span), (hi-lo)/span]
    coords = (2*(means[:, 1:4]-blocks.origin)/span - 1)[:, [2, 1, 0]].copy()
    log_mass = np.log(m[:, 0]/editable[0])
    encoder_inputs = tuple(_readonly(blocks.encoder_grid(grid, density)) for grid in grids)
    return PreparedGeometryInputs(token_grid, token_count, grids, _readonly(fixed), _readonly(active),
                                  _readonly(features), _readonly(stats), _readonly(coords),
                                  _readonly(log_mass), encoder_inputs)


class GeometryInputCache:
    """Per-model/worker LRU of static inputs with explicit byte and entry limits.

    ``max_bytes=0`` disables caching but uses the exact same preparation path.
    ``bytes`` measures retained NumPy payload; at most ``max_entries`` keys and
    wrappers are retained.  This cache contains CPU inputs only and survives
    optimizer updates.  It is not a learned-feature cache and is not serialized
    in checkpoints.  A cache instance is intended for one orchestration thread.
    """
    def __init__(self, max_bytes=128 * 1024**2, *, max_entries=128):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, numbers.Integral) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self.max_bytes = int(max_bytes)
        self.max_entries = _positive_int(max_entries, "max_entries")
        self._items = OrderedDict()
        self._bytes = 0
        self._counts = dict(requests=0, hits=0, misses=0, builds=0, bypasses=0,
                            evictions=0, source_changes=0)

    def _key(self, blocks, fixed, density, token_grid, grids):
        return (_INPUT_VERSION, material_signature(blocks), _array_digest(fixed),
                density, token_grid, grids)

    def get(self, blocks, fixed_moments, density=1., *, token_grid=16, grids=(32, 64)):
        fixed, density, token_grid, grids = _arguments(blocks, fixed_moments, density, token_grid, grids)
        self._counts["requests"] += 1
        if not self.max_bytes:
            self._counts["bypasses"] += 1
            self._counts["builds"] += 1
            return prepare_geometry_inputs(blocks, fixed, density, token_grid=token_grid, grids=grids)
        key = self._key(blocks, fixed, density, token_grid, grids)
        if key in self._items:
            self._counts["hits"] += 1
            self._items.move_to_end(key)
            return self._items[key]
        self._counts["misses"] += 1
        self._counts["builds"] += 1
        result = prepare_geometry_inputs(blocks, fixed, density, token_grid=token_grid, grids=grids)
        if self._key(blocks, fixed, density, token_grid, grids) != key:
            self._counts["source_changes"] += 1
            raise RuntimeError("material changed while preparing static geometry inputs")
        if result.nbytes > self.max_bytes:
            self._counts["bypasses"] += 1
            return result
        while self._items and (self._bytes + result.nbytes > self.max_bytes
                               or len(self._items) >= self.max_entries):
            _, evicted = self._items.popitem(last=False)
            self._bytes -= evicted.nbytes
            self._counts["evictions"] += 1
        self._items[key] = result
        self._bytes += result.nbytes
        return result

    def clear(self, *, reset_stats=False):
        self._items.clear()
        self._bytes = 0
        if reset_stats:
            self._counts = dict.fromkeys(self._counts, 0)

    def snapshot(self):
        return dict(self._counts, entries=len(self._items), bytes=self._bytes,
                    max_bytes=self.max_bytes, max_entries=self.max_entries,
                    scope="parameter_independent_cpu_inputs")
