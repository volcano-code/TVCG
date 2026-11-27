"""Scientific and execution contract for the HENMO V1+ research candidate.

Resolution describes the material, not the context grid or an accuracy claim.
Adaptive discrepancies are diagnostics, never certified mesh error bounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class HENMOConfig:
    native_material_resolution: int = 512
    context_grids: tuple[int, ...] = (64, 128)
    feature_grid: int = 32
    coarse_grid: int = 32
    leaf_grid: int = 64
    geometry_width: int = 96
    pino_width: int = 16
    pino_modes: int = 8
    pino_layers: int = 2
    interval_depth: int = 2
    response_hidden: int = 64
    cut_token_count: int = 16
    cut_attention: bool = True
    response_path: str = 'neural'
    response_profile: str = 'compact_warp'
    legacy_response_type: str = 'monotone_warp'
    response_zero_init: bool = True
    training_scope: str = 'all'
    training_execution: str = 'coarse'
    training_coarse_grids: tuple[int, ...] | None = (32,64)
    execution: str = 'adaptive'
    max_coarse_atoms: int = 32
    leaf_rule: str = 'gl3'
    bandwidth_voxels: float = 1.0
    coarse_support_fraction: float = 1.0
    moment_tolerance: float = 1.e-3
    jacobian_tolerance: float = 5.e-2
    blend_start: float = 1.0
    blend_end: float = 4.0
    chunk_size: int = 65536
    material_cache_bytes: int = 128 * 1024**2
    feature_cache_bytes: int = 128 * 1024**2
    surface_workers: int = 1
    require_native_acceptance: bool = True
    schema: str = 'henmo.v1plus.config.v1'

    def __post_init__(self):
        if isinstance(self.context_grids, list):
            object.__setattr__(self, 'context_grids', tuple(self.context_grids))
        if isinstance(self.training_coarse_grids,list):
            object.__setattr__(self,'training_coarse_grids',tuple(self.training_coarse_grids))
        for name in ('native_material_resolution', 'feature_grid', 'coarse_grid', 'leaf_grid',
                     'geometry_width', 'pino_width', 'pino_modes', 'pino_layers', 'interval_depth',
                     'response_hidden', 'cut_token_count', 'max_coarse_atoms', 'chunk_size', 'surface_workers'):
            n = getattr(self, name)
            if type(n) is not int or n < 1:
                raise ValueError(name + ' must be a positive integer')
        side = self.native_material_resolution
        if side < 8 or side & (side-1):
            raise ValueError('native material side must be a power of two >= 8')
        grids = self.context_grids
        if (not isinstance(grids, tuple) or len(grids) < 2 or tuple(sorted(set(grids))) != grids
                or any(type(g) is not int or g < 8 or g % 4 or side % g for g in grids)):
            raise ValueError('context grids must be sorted distinct >=8 divisors of native resolution')
        for name in ('feature_grid', 'coarse_grid', 'leaf_grid'):
            grid = getattr(self, name)
            if grid & (grid-1) or side % grid:
                raise ValueError(name + ' must be a power-of-two divisor of material resolution')
        if self.coarse_grid > self.leaf_grid:
            raise ValueError('coarse grid cannot exceed leaf grid')
        if self.geometry_width < 8 or self.geometry_width % 4 or self.pino_width < 2:
            raise ValueError('geometry width >=8 divisible by four and pino width >=2 required')
        if self.max_coarse_atoms < 10:
            raise ValueError('ten-moment positive compression needs a budget of at least ten atoms')
        if self.response_path not in ('neural', 'legacy', 'deterministic'):
            raise ValueError('response_path must be neural, legacy or deterministic')
        if self.response_profile not in ('compact_warp','bounded_residual'):
            raise ValueError('explicit supported response profile required')
        if self.training_coarse_grids is not None and (
                not self.training_coarse_grids or any(type(g) is not int or g < 1 or g & (g-1)
                    or g > self.leaf_grid or side % g for g in self.training_coarse_grids)
                or len(set(self.training_coarse_grids)) != len(self.training_coarse_grids)):
            raise ValueError('training coarse grids must be distinct valid hierarchy grids')
        if self.pino_layers < 2:
            raise ValueError('shared P-CRAFT encoder requires at least two layers')
        if self.execution not in ('adaptive', 'coarse', 'leaf') or self.training_execution not in ('adaptive','coarse','leaf'):
            raise ValueError('execution must be adaptive, coarse or leaf')
        if self.training_scope not in ('all', 'response'):
            raise ValueError('training scope must be all or response')
        if self.legacy_response_type not in ('bounded_cubic', 'monotone_warp'):
            raise ValueError('unknown legacy response semantics')
        if self.leaf_rule not in ('tetra4', 'gl2', 'gl3'):
            raise ValueError('leaf_rule must preserve the registered full-cell cubature semantics')
        for name in ('bandwidth_voxels', 'coarse_support_fraction','moment_tolerance', 'jacobian_tolerance', 'blend_start', 'blend_end'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        if self.blend_start >= self.blend_end:
            raise ValueError('blend_start must precede blend_end')
        for name in ('material_cache_bytes', 'feature_cache_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        for name in ('cut_attention', 'response_zero_init', 'require_native_acceptance'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + ' must be an explicit Boolean')
        if not self.require_native_acceptance or self.schema != 'henmo.v1plus.config.v1':
            raise ValueError('native acceptance and versioned semantics are mandatory')

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @classmethod
    def e16_compatible(cls, **overrides):
        # Values come from the actual Compact preset rather than a guessed layout.
        from .unified_factory import _make_native_config, get_config
        c = _make_native_config(get_config('compact_pino_v2_256'))
        values = dict(native_material_resolution=256, context_grids=c.multiscale_grids,
                      feature_grid=16, coarse_grid=16, leaf_grid=32,
                      geometry_width=c.interval_width, pino_width=c.pino_width,
                      pino_modes=c.pino_modes, pino_layers=c.pino_layers,
                      interval_depth=c.interval_depth, response_hidden=64,
                      response_path='legacy', response_zero_init=False,
                      training_scope='response', training_execution='leaf',training_coarse_grids=None,
                      execution='leaf', leaf_rule='tetra4', cut_attention=False)
        return cls(**(values | overrides))

    @classmethod
    def cpu_smoke(cls, **overrides):
        return cls(**(dict(native_material_resolution=16, context_grids=(8,16),
                           feature_grid=2, coarse_grid=2, leaf_grid=4,
                           geometry_width=8, pino_width=2, pino_modes=2,
                           pino_layers=2, interval_depth=1, response_hidden=8,
                           cut_token_count=4, max_coarse_atoms=12,
                           training_coarse_grids=(2,4),
                           material_cache_bytes=8*1024**2, feature_cache_bytes=8*1024**2) | overrides))
