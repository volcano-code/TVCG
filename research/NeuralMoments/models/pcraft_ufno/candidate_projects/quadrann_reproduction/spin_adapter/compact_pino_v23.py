"""V2.3 execution of the unchanged conservative V2.2 neural response.

Weights/state_dict, material semantics and derivative definitions are unchanged.
This entry enables fused reductions, bounded versioned inference projections and
a solve-lifetime surface executor. It introduces no trained student or J head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .compact_pino_v21 import CompactPINOv21
from .v21_response import _projection_version
from .v22_frozen_features import tensor_version


@dataclass(frozen=True)
class RuntimeConfig:
    fused_contraction: bool = True
    cache_inference_projection: bool = True
    projection_cache_bytes: int = 8 * 1024**2
    surface_workers: int | None = None

    def __post_init__(self):
        for value in (self.fused_contraction, self.cache_inference_projection):
            if type(value) is not bool:
                raise ValueError("runtime switches must be explicit Booleans")
        if type(self.projection_cache_bytes) is not int or self.projection_cache_bytes < 0:
            raise ValueError("projection cache bytes must be a nonnegative integer")
        if self.surface_workers is not None and (
                type(self.surface_workers) is not int or self.surface_workers < 1):
            raise ValueError("surface workers must be a positive integer or None")


class CompactPINOv23(CompactPINOv21):
    """Same physical operator and checkpoint layout, with explicit runtime policy.

    Query preparation caching defaults to zero for new Newton queries. Inference
    projection caching holds at most one table; it never retains a training graph.
    Use close() (or a with block) after a solve to release the private executor.
    """
    def __init__(self, token_grid=16, *, runtime=None, query_cache_bytes=0,
                 response_type="monotone_warp", training_scope="response", **kwargs):
        if runtime is not None and not isinstance(runtime, RuntimeConfig):
            raise ValueError("runtime requires RuntimeConfig")
        self.runtime_config = runtime or RuntimeConfig()
        self._projection_entry = None
        self._projection_counts = dict(hits=0, misses=0, bypasses=0)
        self._runtime_closed = False
        super().__init__(token_grid, query_cache_bytes=query_cache_bytes,
                         response_type=response_type, training_scope=training_scope, **kwargs)
        # Imported lazily: ordinary V2.1 consumers do not acquire a new executor.
        from .v23_surface_runtime import SurfaceExecutionContext
        self.surface_execution = SurfaceExecutionContext(max_workers=self.runtime_config.surface_workers)

    def _ensure_runtime_open(self):
        if self._runtime_closed:
            raise RuntimeError("V2.3 runtime is closed; create a new operator for another solve")

    def _drop_projection(self):
        self._projection_entry = None

    def _invalidate_frozen_features(self):
        self._drop_projection()
        super()._invalidate_frozen_features()

    def clear_efficiency_caches(self):
        self._drop_projection()
        super().clear_efficiency_caches()

    def train(self, mode=True):
        # Discard inference-only projections on a mode transition, including
        # eval()->train(). The frozen feature cache has its own checked lifetime.
        self._drop_projection()
        return super().train(mode)

    def encode_geometry(self, *args, **kwargs):
        self._ensure_runtime_open()
        return super().encode_geometry(*args, **kwargs)

    def _project_query_context(self, cache):
        inference = (not self.training and
                     (cache.inference_only or not torch.is_grad_enabled()))
        if not self.efficiency or not inference or not self.runtime_config.cache_inference_projection:
            self._projection_counts["bypasses"] += 1
            return super()._project_query_context(cache)
        key = (id(cache.table), tensor_version(cache.table),
               _projection_version(self.response.local[0]))
        entry = self._projection_entry
        if entry is not None and entry[0] == key:
            self._projection_counts["hits"] += 1
            return entry[2]
        self._projection_counts["misses"] += 1
        self._projection_entry = None
        with torch.inference_mode(False), torch.no_grad():
            projected = self.response.project_context(cache.table)
        size = projected.values.numel() * projected.values.element_size()
        if size <= self.runtime_config.projection_cache_bytes:
            # Keep the source table alive: object/allocator identity cannot be
            # accidentally recycled into a false cache hit.
            self._projection_entry = (key, cache.table, projected)
        else:
            self._projection_counts["bypasses"] += 1
        return projected

    def evaluate_operand(self, cache, operand, *, with_jacobian=True, bandwidth=None,
                         chunk_size=65536, trace=None):
        self._ensure_runtime_open()
        if type(with_jacobian) is not bool:
            raise ValueError("with_jacobian must be an explicit Boolean")
        result = super().evaluate_operand(
            cache, operand, with_jacobian=with_jacobian, bandwidth=bandwidth,
            chunk_size=chunk_size, fused_contraction=self.runtime_config.fused_contraction,
            execution_context=self.surface_execution, trace=trace)
        result.diagnostics.update(runtime_version="compact-pino-v23-execution-v1",
                                  request_mode="joint" if with_jacobian else "M",
                                  learned_response_unchanged=True)
        return result

    def efficiency_statistics(self):
        result = super().efficiency_statistics()
        entry = self._projection_entry
        size = 0 if entry is None else entry[2].values.numel()*entry[2].values.element_size()
        result.update(runtime_version="compact-pino-v23-execution-v1",
                      runtime_config=asdict(self.runtime_config),
                      inference_projection={**self._projection_counts, "bytes": size,
                                            "entries": int(entry is not None)},
                      surface_execution=self.surface_execution.snapshot(),
                      runtime_closed=self._runtime_closed)
        return result

    def close(self):
        if self._runtime_closed:
            return
        self.clear_efficiency_caches()
        self.surface_execution.close()
        self._runtime_closed = True

    def __enter__(self):
        self._ensure_runtime_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
