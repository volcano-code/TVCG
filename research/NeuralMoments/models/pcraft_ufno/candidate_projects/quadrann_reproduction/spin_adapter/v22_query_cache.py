"""Bounded preparation cache for exactly repeated, frozen finite-cut queries.

Only the material/cut geometry calculation is retained.  No model output,
parameter-dependent tensor, vertex velocity or autograd graph is cached.  A
caller evaluates each returned stencil with the *current* native velocity and
current learned response.  New Newton operands have different content keys.

The byte budget covers conservatively counted retained NumPy buffers and their
containers.  A miss builds a candidate before admitting/evicting entries, so
peak preparation can additionally hold one candidate up to that budget and one
ordinary streaming chunk.  An oversized query is discarded and subsequently
bypassed using a bounded negative cache.  The original streaming evaluator
remains the fallback.  The budget is not a process-RSS or CUDA-memory limit.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import hashlib
import math
import sys

import numpy as np
import torch

from .v21_response import ResponseStencil, prepare_measure, prepare_stencil
from .v21_response_surface import FiniteSurface, prepare_surface
from .v22_geometry_inputs import material_signature


_VERSION = 'compact-pino.frozen-query-preparation.v1'


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_digest(value):
    array = np.ascontiguousarray(_numpy(value))
    if array.dtype.hasobject:
        raise ValueError('query preparation requires numeric arrays')
    digest = hashlib.sha256()
    digest.update(repr((array.dtype.str, array.shape)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def query_preparation_key(blocks, operand, bandwidth, chunk_size):
    """Bind material and all cut semantics, without binding independent motion.

    Motion is intentionally excluded: the prepared witnesses are geometry-only
    and evaluate_stencil contracts the supplied velocity on every evaluation.
    Caps are included even though their triangles already occur in faces.
    """
    parts = (_VERSION, material_signature(blocks),
             _array_digest(operand.vertices), _array_digest(operand.faces),
             _array_digest(operand.face_is_cap), str(operand.operation),
             float(bandwidth).hex(), int(chunk_size))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _array_bytes(array):
    # ndarray.__sizeof__ includes data only for an owning array.  Charging a
    # view for its apparent data too is conservative when arrays share storage.
    own_data = array.nbytes if array.flags.owndata else 0
    return int(array.nbytes + max(0, sys.getsizeof(array) - own_data))


def _surface_bytes(surface):
    seen = set()
    size = sys.getsizeof(surface) + sys.getsizeof(surface.__dict__) + sys.getsizeof(surface._data)
    for array in (surface.vertices, surface.faces, *surface._data):
        if isinstance(array, np.ndarray) and id(array) not in seen:
            seen.add(id(array))
            size += _array_bytes(array)
    return size


def _freeze_stencil(stencil):
    size = sys.getsizeof(stencil) + sys.getsizeof(stencil.__dict__)
    for item in fields(stencil):
        value = getattr(stencil, item.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
            size += _array_bytes(value)
    return size


@dataclass(frozen=True)
class PreparedPackedQuery:
    """One original-order set of streamed stencils for a frozen query.

    ``saturated_moments`` contains only whole retained editable blocks.  The
    caller adds fixed moments once, then each stencil result in tuple order.
    ``band_nodes`` counts all streamed atoms, including saturated atoms within
    a crossing block; active_nodes remains a per-evaluation stencil diagnostic.
    """
    surface: FiniteSurface
    operation: str
    bandwidth: float
    chunk_size: int
    full: np.ndarray
    band: np.ndarray
    saturated_moments: np.ndarray
    stencils: tuple[ResponseStencil, ...]
    band_nodes: int
    byte_size: int
    key: str


class QueryPreparationCache:
    """LRU static query cache; zero bytes disables it without preparing geometry."""

    def __init__(self, max_bytes=128 * 1024**2, *, negative_capacity=256):
        self._check_budget(max_bytes)
        if type(negative_capacity) is not int or negative_capacity < 0:
            raise ValueError('negative cache capacity must be a nonnegative integer')
        self._max_bytes = max_bytes
        self.negative_capacity = negative_capacity
        self._entries = OrderedDict()
        self._oversized = OrderedDict()
        self._bytes = 0
        self._counts = dict(hits=0, misses=0, builds=0, evictions=0,
                            disabled_bypasses=0, live_vertex_bypasses=0,
                            oversize_bypasses=0, negative_hits=0)

    @staticmethod
    def _check_budget(value):
        if type(value) is not int or value < 0:
            raise ValueError('query cache max_bytes must be a nonnegative integer')

    @property
    def max_bytes(self):
        return self._max_bytes

    def info(self):
        return dict(self._counts, entries=len(self._entries), bytes=self._bytes,
                    max_bytes=self._max_bytes, negative_entries=len(self._oversized),
                    negative_capacity=self.negative_capacity,
                    scope='static_query_preparation_no_motion_or_learning_graph')

    def clear(self, *, reset_stats=False):
        self._entries.clear()
        self._oversized.clear()
        self._bytes = 0
        if reset_stats:
            self._counts = dict.fromkeys(self._counts, 0)

    def resize(self, max_bytes):
        self._check_budget(max_bytes)
        self._max_bytes = max_bytes
        self._oversized.clear()
        self._evict_until(0)

    def _evict_until(self, incoming_bytes):
        while self._entries and self._bytes + incoming_bytes > self._max_bytes:
            _, removed = self._entries.popitem(last=False)
            self._bytes -= removed.byte_size
            self._counts['evictions'] += 1

    def _too_large(self, key):
        self._counts['oversize_bypasses'] += 1
        if self.negative_capacity:
            self._oversized[key] = None
            self._oversized.move_to_end(key)
            while len(self._oversized) > self.negative_capacity:
                self._oversized.popitem(last=False)
        return None

    def get_or_prepare(self, blocks, operand, bandwidth, chunk_size=65536):
        """Return immutable geometry preparation, or None for streaming fallback.

        A differentiable vertex tensor always bypasses this cache.  Its live
        vertex graph must be constructed afresh by the original evaluator.
        Static native vertex velocities may change freely between calls: they
        are neither read nor retained here.
        """
        width = float(bandwidth)
        if not math.isfinite(width) or width <= 0:
            raise ValueError('query bandwidth must be positive and finite')
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('query chunk_size must be a positive integer')
        if operand.operation not in ('intersection', 'difference'):
            raise ValueError('query operation must be intersection or difference')
        if not self._max_bytes:
            self._counts['disabled_bypasses'] += 1
            return None
        if isinstance(operand.vertices, torch.Tensor) and operand.vertices.requires_grad:
            self._counts['live_vertex_bypasses'] += 1
            return None
        key = query_preparation_key(blocks, operand, width, chunk_size)
        if key in self._entries:
            self._counts['hits'] += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        if key in self._oversized:
            self._counts['negative_hits'] += 1
            self._oversized.move_to_end(key)
            return None
        self._counts['misses'] += 1
        self._counts['builds'] += 1
        surface = prepare_surface(operand.vertices, operand.faces)
        # Container/key accounting is deliberately conservative and does not
        # depend on how CPython happens to resize an OrderedDict allocation.
        size = _surface_bytes(surface) + 1024 + len(key)
        if size > self._max_bytes:
            return self._too_large(key)
        full, band = blocks.classify(surface, width, operand.operation)
        full = np.array(full, dtype=bool, copy=True)
        band = np.array(band, dtype=bool, copy=True)
        saturated = np.array(blocks.moments[full].sum(0), dtype=np.float64, copy=True)
        for array in (full, band, saturated):
            array.flags.writeable = False
            size += _array_bytes(array)
        if size > self._max_bytes:
            return self._too_large(key)
        stencils = []
        band_nodes = 0
        for nodes, weights in blocks.chunks(band, max_nodes=chunk_size):
            measure = prepare_measure(nodes, weights)
            stencil = prepare_stencil(measure, surface, bandwidth=width, chunk_size=chunk_size)
            size += _freeze_stencil(stencil) + 16  # tuple slot + list preparation slot
            if size > self._max_bytes:
                return self._too_large(key)
            stencils.append(stencil)
            band_nodes += len(weights)
        # No old input may be admitted if the caller changed arrays/files while
        # preparation ran.  Read-only memmap identity uses the same shared
        # material-signature rule as static encoder input preparation.
        if key != query_preparation_key(blocks, operand, width, chunk_size):
            raise RuntimeError('material or operand changed during query preparation')
        packed = PreparedPackedQuery(surface, operand.operation, width, chunk_size,
                                     full, band, saturated, tuple(stencils),
                                     band_nodes, int(size), key)
        self._evict_until(packed.byte_size)
        self._entries[key] = packed
        self._bytes += packed.byte_size
        return packed
