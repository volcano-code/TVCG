"""Bounded graph-free geometry features for an explicitly frozen encoder.

Only the unprojected token table is retained. Response parameters and response
autograd graphs never enter this cache. Full frozen parameter/buffer contents
are digested once per PyTorch mutation-version generation, so normal updates,
state loads, replacement and device/dtype conversion invalidate the cache
without copying a large frozen CUDA state back to CPU on every query. As with
PyTorch autograd itself, callers must not mutate tensors through ``.data`` or
untracked external storage aliases.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib

import torch


def tensor_version(value):
    """Detect supported in-place mutation, replacement and dtype/device changes."""
    try:
        version = value._version
    except RuntimeError:  # Inference tensors have no mutation version counter.
        version = None
    return (id(value), version, value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, value.requires_grad)


class FrozenStateIdentity:
    """Digest all frozen parameters and buffers, including nonpersistent buffers."""
    def __init__(self):
        self._versions = self._signature = None
        self.digest_builds = 0

    def clear(self):
        self._versions = self._signature = None

    def signature(self, named_modules):
        tensors, modes = [], []
        for label, module in named_modules:
            for name, child in module.named_modules():
                modes.append((label, name, id(child), type(child).__module__,
                              type(child).__qualname__, child.training))
                if child.training:
                    raise ValueError('frozen geometry modules must remain in eval mode')
            for name, value in module.named_parameters():
                if value.requires_grad:
                    raise ValueError('frozen geometry parameters must not require gradients')
                tensors.append((label + '.parameter.' + name, value))
            for name, value in module.named_buffers():
                if value.requires_grad:
                    raise ValueError('frozen geometry buffers must not require gradients')
                tensors.append((label + '.buffer.' + name, value))
        versions = (tuple(modes), tuple((name, tensor_version(value)) for name, value in tensors))
        unversioned = any(item[1][1] is None for item in versions[1])
        if self._signature is None or versions != self._versions or unversioned:
            digest = hashlib.sha256()
            for name, value in tensors:
                digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode('utf8'))
                # Byte view also supports bfloat16 and complex Fourier tensors.
                raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                digest.update(raw.numpy().tobytes())
            self._versions = versions
            self._signature = (digest.hexdigest(), versions)
            self.digest_builds += 1
        return self._signature


def validate_feature_payload(features):
    if features.table_version is not None and tensor_version(features.table) != features.table_version:
        raise ValueError('learned geometry feature table was mutated or replaced')
    if features.fixed_key is not None and features.fixed_moments.tobytes() != features.fixed_key:
        raise ValueError('learned geometry fixed moments were mutated')


class FrozenFeatureCache:
    """An LRU of unprojected frozen tables, bounded by storage bytes and entries."""
    def __init__(self, max_bytes=64*1024**2, *, max_entries=128):
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries < 1:
            raise ValueError('nonnegative cache byte budget and positive entry count required')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._items = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = self.invalidations = 0

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        validate_feature_payload(item[0])
        self.hits += 1
        self._items.move_to_end(key)
        return item[0]

    def put(self, key, features):
        if (not features.frozen_encoder or features.projected_context is not None
                or features.table.requires_grad or features.table.grad_fn is not None
                or features.table.is_inference()):
            raise ValueError('frozen cache accepts only normal, graph-free, unprojected token tables')
        validate_feature_payload(features)
        size = features.table.untyped_storage().nbytes() + features.fixed_moments.nbytes
        previous = self._items.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        if not self.max_bytes or size > self.max_bytes:
            self.bypasses += 1
            return
        while self._items and (self.bytes + size > self.max_bytes or len(self._items) >= self.max_entries):
            _, (_, old_size) = self._items.popitem(last=False)
            self.bytes -= old_size
            self.evictions += 1
        self._items[key] = (features, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def clear(self, *, invalidated=False):
        self._items.clear()
        self.bytes = 0
        self.invalidations += int(invalidated)

    def snapshot(self):
        return dict(entries=len(self._items), bytes=self.bytes, peak_bytes=self.peak_bytes,
                    max_bytes=self.max_bytes, max_entries=self.max_entries,
                    hits=self.hits, misses=self.misses, evictions=self.evictions,
                    bypasses=self.bypasses, invalidations=self.invalidations,
                    scope='frozen_encoder_unprojected_tables_across_response_updates')
