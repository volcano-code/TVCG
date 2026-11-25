"""Bounded, evaluation-session-only storage for learned geometry features.

This cache never survives an evaluation session or an optimizer update. Its
budget counts retained feature tensors, not model parameters, material memmaps
or the temporary activations needed to encode a new geometry.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from .v22_geometry_inputs import material_signature


def geometry_feature_key(blocks, fixed_moments, density):
    fixed = np.asarray(fixed_moments, dtype=np.float64)
    return material_signature(blocks), fixed.shape, fixed.tobytes(), float(density)


class InferenceGeometryCache:
    def __init__(self, max_bytes=64*1024**2, *, max_entries=128):
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries < 1:
            raise ValueError('nonnegative byte budget and positive entry limit required')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = 0

    def get(self, key):
        item = self._entries.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return item[0]

    def __setitem__(self, key, features):
        if not features.inference_only:
            raise ValueError('training graph cannot enter an inference feature cache')
        tensors = [features.table]
        projection = getattr(features, 'projected_context', None)
        if projection is not None:
            tensors.append(projection.values)
        storages, size = set(), 0
        for value in tensors:
            if not isinstance(value, torch.Tensor) or value.requires_grad or value.grad_fn is not None:
                raise ValueError('only graph-free feature tensors can be retained')
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            if identity not in storages:
                size += storage.nbytes()
                storages.add(identity)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        if size > self.max_bytes or self.max_bytes == 0:
            self.bypasses += 1
            return
        while self._entries and (self.bytes + size > self.max_bytes or len(self._entries) >= self.max_entries):
            _, (_, old_size) = self._entries.popitem(last=False)
            self.bytes -= old_size
            self.evictions += 1
        self._entries[key] = features, size
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def snapshot(self):
        return dict(entries=len(self._entries), bytes=self.bytes, peak_bytes=self.peak_bytes,
                    max_bytes=self.max_bytes, hits=self.hits, misses=self.misses,
                    evictions=self.evictions, bypasses=self.bypasses,
                    scope='one_evaluation_session_feature_tensor_storage')

    def clear(self):
        self._entries.clear()
        self.bytes = 0
