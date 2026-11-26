"""Compact V2.1 single physical response to a deployed finite operand.

The phase-A entry point, evaluate_packed, uses no learned weights. The optional
encoder reuses P-CRAFT's two-scale U-FNO and local feature layers with conservative
voxel channels. Stage B training admission lives outside this module.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    PcraftNativeMomentOperator, parameter_versions,
)
from .unified_factory import _make_native_config, get_config
from .v21_response import LocalRetention, prepare_measure, evaluate_response, evaluate_stencil, prepare_surface
from .v22_geometry_inputs import GeometryInputCache, material_signature
from .v22_query_cache import QueryPreparationCache
from .v21_material import validate_full_cell_rule
from .v22_frozen_features import (FrozenFeatureCache, FrozenStateIdentity,
                                  tensor_version, validate_feature_payload)


@dataclass(frozen=True)
class PackedPrediction:
    moments: torch.Tensor
    jacobian: torch.Tensor | None
    diagnostics: dict


def native_proposal_status(prediction, fixed_moments):
    """The numerical response alone never certifies a Boolean result or minimizer."""
    m = prediction.moments
    if not bool(torch.isfinite(m).all()) or float(m[0].detach()) <= float(fixed_moments[0]):
        return "NATIVE_REQUIRED_EMPTY_EDITABLE"
    if prediction.jacobian is None or not bool(torch.isfinite(prediction.jacobian).all()):
        return "NATIVE_REQUIRED_DERIVATIVE"
    return "PROPOSAL_REQUIRES_NATIVE_ACCEPTANCE"


def evaluate_packed(blocks, surface, bandwidth, operation, fixed_moments, vertex_velocity=None,
                    *, model=None, context=None, chunk_size=65536, full_cell_rule='tetra4'):
    """Use the signed-distance bound on complete blocks and stream the remainder."""
    full_cell_rule = validate_full_cell_rule(full_cell_rule)
    full, band = blocks.classify(surface, bandwidth, operation)
    device = next(model.parameters()).device if model is not None else 'cpu'
    fixed = np.asarray(fixed_moments, dtype=float)
    if fixed.shape != (10,) or not np.all(np.isfinite(fixed)):
        raise ValueError("fixed material requires ten finite moments")
    value = torch.as_tensor(fixed + blocks.moments[full].sum(0), dtype=torch.float64, device=device)
    jac = None
    if vertex_velocity is not None:
        motion = np.asarray(vertex_velocity, dtype=float)
        if motion.ndim != 3 or motion.shape[1] != 3 or motion.shape[2] not in (3, 8, 10) or not np.isfinite(motion).all():
            raise ValueError("invalid raw or stratum motion")
        if hasattr(surface, 'vertices') and len(motion) != len(surface.vertices):
            raise ValueError("vertex velocity must follow the supplied operand")
        jac = value.new_zeros((10, motion.shape[2]))
    diag = dict(total_blocks=len(full), saturated_retained_blocks=int(full.sum()), band_blocks=int(band.sum()),
                band_nodes=0, active_nodes=0, ambiguous_nodes=0, chunks=0,
                full_cell_rule=full_cell_rule, bandwidth=float(bandwidth))
    options = {} if full_cell_rule == 'tetra4' else dict(full_cell_rule=full_cell_rule)
    for nodes, weights in blocks.chunks(band, max_nodes=chunk_size, **options):
        result = evaluate_response(prepare_measure(nodes, weights), surface, bandwidth=bandwidth,
                                   operation=operation, vertex_velocity=vertex_velocity, model=model,
                                   context=context, chunk_size=chunk_size)
        value = value + result.moments
        if jac is not None:
            jac = None if result.jacobian is None else jac + result.jacobian
        diag['chunks'] += 1
        diag['band_nodes'] += len(weights)
        for key in ('active_nodes', 'ambiguous_nodes'):
            diag[key] += result.diagnostics[key]
    if not band.any() and model is not None and torch.is_grad_enabled():
        zero = next(model.parameters()).reshape(-1)[0] * 0.
        value = value + zero
        jac = jac + zero if jac is not None else None
    prediction = PackedPrediction(value, jac, diag)
    diag["solver_status"] = native_proposal_status(prediction, fixed)
    return prediction


@dataclass(frozen=True)
class GeometryFeatures:
    blocks: object
    fixed_moments: np.ndarray
    table: torch.Tensor
    owner: int
    versions: tuple
    inference_only: bool
    projected_context: object | None = None
    material_key: object | None = None
    frozen_encoder: bool = False
    table_version: tuple | None = None
    fixed_key: bytes | None = None
    graph_lifetime: object | None = None


class _GeometryGraphLifetime:
    """All-parameter training features belong to one forward/backward graph."""
    def __init__(self):
        self.consumed = False

    def backward(self, gradient):
        self.consumed = True
        return gradient


def _evaluate_prepared_packed(prepared, operation, fixed_moments, vertex_velocity=None,
                              *, model, context, chunk_size):
    """Reduce the same streamed stencils after parameter-independent preparation.

    QueryPreparationCache has checked the actual material and operand bytes.
    Neither a learned value nor a vertex-motion contraction is cached here.
    """
    device = next(model.parameters()).device
    fixed = np.asarray(fixed_moments, dtype=float)
    if fixed.shape != (10,) or not np.isfinite(fixed).all():
        raise ValueError('fixed material requires ten finite moments')
    value = torch.as_tensor(fixed + prepared.saturated_moments, dtype=torch.float64, device=device)
    jac = None
    if vertex_velocity is not None:
        motion = np.asarray(vertex_velocity, dtype=float)
        if (motion.ndim != 3 or motion.shape[1] != 3 or motion.shape[2] not in (3, 8, 10)
                or not np.isfinite(motion).all() or len(motion) != len(prepared.surface.vertices)):
            raise ValueError('invalid raw or stratum motion for prepared operand')
        jac = value.new_zeros((10, motion.shape[2]))
    diag = dict(total_blocks=len(prepared.full), saturated_retained_blocks=int(prepared.full.sum()),
                band_blocks=int(prepared.band.sum()), band_nodes=0, active_nodes=0,
                ambiguous_nodes=0, chunks=0, full_cell_rule=prepared.full_cell_rule,
                bandwidth=prepared.bandwidth)
    for stencil in prepared.stencils:
        result = evaluate_stencil(stencil, operation=operation, vertex_velocity=vertex_velocity,
                                  model=model, context=context, chunk_size=chunk_size)
        value = value + result.moments
        if jac is not None:
            jac = None if result.jacobian is None else jac + result.jacobian
        diag['chunks'] += 1
        diag['band_nodes'] += stencil.total_nodes
        for key in ('active_nodes', 'ambiguous_nodes'):
            diag[key] += result.diagnostics[key]
    if not prepared.band.any() and torch.is_grad_enabled():
        zero = next(model.parameters()).reshape(-1)[0] * 0.
        value = value + zero
        jac = jac + zero if jac is not None else None
    result = PackedPrediction(value, jac, diag)
    diag['solver_status'] = native_proposal_status(result, fixed)
    return result


class CompactPINOv21(nn.Module):
    """A Token16/Token8 candidate with one bounded local retention field."""
    def __init__(self, token_grid=16, *, efficiency=True,
                 static_cache_bytes=128*1024**2, query_cache_bytes=128*1024**2,
                 full_cell_rule='tetra4', response_type='bounded_cubic',
                 response_zero_init=False, training_scope='all',
                 frozen_feature_cache_bytes=64*1024**2):
        super().__init__()
        if token_grid not in (8, 16):
            raise ValueError("expected Token8 or Token16")
        if type(efficiency) is not bool:
            raise ValueError('efficiency must be a Boolean')
        for budget in (static_cache_bytes, query_cache_bytes, frozen_feature_cache_bytes):
            if type(budget) is not int or budget < 0:
                raise ValueError('cache byte limits must be nonnegative integers')
        self.token_grid = token_grid
        self.full_cell_rule = validate_full_cell_rule(full_cell_rule)
        self.efficiency = efficiency
        self.geometry_input_cache = GeometryInputCache(max_bytes=static_cache_bytes if efficiency else 0)
        self.query_preparation_cache = QueryPreparationCache(max_bytes=query_cache_bytes if efficiency else 0)
        self.frozen_feature_cache = FrozenFeatureCache(frozen_feature_cache_bytes if efficiency else 0)
        self._frozen_state = FrozenStateIdentity()
        self._last_frozen_signature = None
        self.encoder_calls = 0
        self.encoder_grid_calls = 0
        cfg = _make_native_config(get_config('compact_pino_v2_256'))
        self.shared_encoder = PcraftNativeMomentOperator(cfg, encoder_only=True)
        width = cfg.interval_width
        self.geometry_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.response = LocalRetention(context_dim=width, hidden_dim=64,
                                       response_type=response_type, zero_init=response_zero_init)
        self.response.use_analytic_slope = efficiency
        self.response_type = response_type
        self.response_zero_init = response_zero_init
        self.set_training_scope(training_scope)

    def set_training_scope(self, scope):
        """Explicitly change which parameters may learn; never reuse old graphs."""
        if scope not in ('all', 'response'):
            raise ValueError("training_scope must be 'all' or 'response'")
        self.training_scope = scope
        for module in (self.shared_encoder, self.geometry_attention):
            for parameter in module.parameters():
                parameter.requires_grad_(scope == 'all')
                parameter.grad = None
        for parameter in self.response.parameters():
            parameter.requires_grad_(True)
        self._invalidate_frozen_features()
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'training_scope', 'all') == 'response':
            self.shared_encoder.eval()
            self.geometry_attention.eval()
        return self

    def _invalidate_frozen_features(self):
        self.frozen_feature_cache.clear(invalidated=True)
        self._frozen_state.clear()
        self._last_frozen_signature = None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, 'frozen_feature_cache'):
            self.clear_efficiency_caches()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.clear_efficiency_caches()
        self.train(self.training)
        return result

    def _frozen_signature(self):
        if self.training_scope != 'response':
            raise ValueError('frozen geometry features require response-only training scope')
        signature = (self.token_grid, repr(self.shared_encoder.config),
                     self._frozen_state.signature((('encoder', self.shared_encoder),
                                                   ('attention', self.geometry_attention))))
        if self._last_frozen_signature is not None and self._last_frozen_signature != signature:
            self.frozen_feature_cache.clear(invalidated=True)
        self._last_frozen_signature = signature
        return signature

    def clear_efficiency_caches(self):
        """Release inputs, queries and frozen tables; checkpoints contain no caches."""
        self.geometry_input_cache.clear()
        self.query_preparation_cache.clear()
        self._invalidate_frozen_features()

    def efficiency_statistics(self):
        return dict(enabled=self.efficiency, static_inputs=self.geometry_input_cache.snapshot(),
                    prepared_queries=self.query_preparation_cache.info(), full_cell_rule=self.full_cell_rule,
                    frozen_features=self.frozen_feature_cache.snapshot(),
                    encoder_calls=self.encoder_calls, encoder_grid_calls=self.encoder_grid_calls,
                    frozen_state_digest_builds=self._frozen_state.digest_builds,
                    training_scope=self.training_scope, response_type=self.response_type)

    def encode_geometry(self, blocks, fixed_moments, density=1., *, inference=False):
        if blocks.side != 256 or blocks.side % (blocks.block_voxels * self.token_grid):
            raise ValueError("native256 blocks must nest exactly within the token grid")
        if inference and self.training:
            raise ValueError("inference cache requires eval()")
        encoder = self.shared_encoder
        material_key = material_signature(blocks)
        frozen = self.training_scope == 'response'
        inference_only = inference or not torch.is_grad_enabled()
        versions = self._frozen_signature() if frozen else parameter_versions(self)
        frozen_key = None
        if frozen:
            fixed = np.asarray(fixed_moments, dtype=np.float64)
            if fixed.shape != (10,) or not np.isfinite(fixed).all():
                raise ValueError('expected ten finite fixed moments')
            if not np.isfinite(density) or density <= 0:
                raise ValueError('density must be positive and finite')
            frozen_key = (material_key, fixed.tobytes(), float(density), versions)
            entry = self.frozen_feature_cache.get(frozen_key)
            if entry is not None:
                # Inference use must not mark the reusable underlying table as
                # inference-only for all future response-training requests.
                return replace(entry, inference_only=inference_only)
        inputs = self.geometry_input_cache.get(blocks, fixed_moments, density,
            token_grid=self.token_grid, grids=tuple(encoder.config.multiscale_grids))
        p = next(encoder.parameters())
        def tensor(a):
            # Prepared arrays are immutable and can outlive any one forward.
            return torch.tensor(a, device=p.device, dtype=p.dtype)
        self.encoder_calls += 1
        with torch.no_grad() if inference or frozen else nullcontext():
            maps, globals_ = [], []
            for grid_input in inputs.encoder_inputs:
                self.encoder_grid_calls += 1
                pooled, fmap = encoder.pino_encoder.forward_with_map(tensor(grid_input)[None],
                                                                     preserve_resolution=True)
                maps.append(fmap)
                globals_.append(encoder.pino_global(pooled))
            glob = torch.stack(globals_, dim=1)
            s = tensor(inputs.stats)[None]
            score = encoder.pino_scale_score(torch.cat((glob, s[:, None].expand(-1, len(maps), -1)), -1))
            shared = encoder.statistics_context(s) + (glob * score.softmax(1)).sum(1)
            local = encoder._interval_features(tensor(inputs.features), tensor(inputs.coords), maps, shared)
            # This attention is query independent; cut dependence enters once through
            # the actual operand and its motion in LocalRetention.
            log_mass = tensor(inputs.log_mass)[None]
            summary, _ = self.geometry_attention(shared[:,None], local[None], local[None],
                                                 attn_mask=log_mass, need_weights=False)
            table = torch.index_copy(local.new_zeros((inputs.token_count, local.shape[1])), 0,
                                     torch.tensor(inputs.active, device=p.device), local + summary[0])
            # A cached table may later feed trainable response weights. Inference
            # tensors cannot be saved for backward, so retain a normal tensor.
            if table.is_inference():
                with torch.inference_mode(False), torch.no_grad():
                    table = table.clone()
            projected = self.response.project_context(table) if self.efficiency and not frozen else None
        if material_signature(blocks) != material_key:
            raise ValueError('material changed while encoding geometry')
        if frozen and self._frozen_signature() != versions:
            raise ValueError('frozen geometry state changed while encoding')
        lifetime = None
        if not frozen and not inference_only and table.requires_grad:
            lifetime = _GeometryGraphLifetime()
            table.register_hook(lifetime.backward)
        result = GeometryFeatures(blocks, inputs.fixed_moments, table, id(self), versions,
                                  inference_only, projected, material_key, frozen,
                                  tensor_version(table), inputs.fixed_moments.tobytes(), lifetime)
        if frozen:
            self.frozen_feature_cache.put(frozen_key, replace(result, inference_only=False))
        return result

    def evaluate_operand(self, cache, operand, *, with_jacobian=True, bandwidth=None, chunk_size=65536):
        if cache.owner != id(self):
            raise ValueError("foreign or stale learned geometry cache")
        if cache.frozen_encoder:
            versions = self._frozen_signature()
        else:
            if self.training_scope != 'all':
                raise ValueError('feature scope changed; encode geometry again')
            versions = parameter_versions(self)
        if cache.versions != versions:
            raise ValueError("foreign or stale learned geometry cache")
        if cache.inference_only and self.training:
            raise ValueError("inference-only geometry features cannot train the model")
        validate_feature_payload(cache)
        if cache.graph_lifetime is not None and cache.graph_lifetime.consumed:
            raise ValueError('learned geometry training graph was consumed; encode geometry again')
        blocks = cache.blocks
        if cache.material_key != material_signature(blocks):
            raise ValueError('stale learned geometry cache: material changed')
        if with_jacobian and operand.vertex_velocity is None:
            raise ValueError("this operand has no native derivative")
        width = blocks.pitch if bandwidth is None else bandwidth
        prepared = self.query_preparation_cache.get_or_prepare(blocks, operand, width,
            chunk_size=chunk_size, full_cell_rule=self.full_cell_rule)
        surface = prepared.surface if prepared is not None else prepare_surface(operand.vertices, operand.faces)
        with torch.no_grad() if cache.inference_only else nullcontext():
            # Response W_h is trainable even when the encoder is frozen. This
            # projection belongs to this query graph and is never cached across
            # updates or backwards. The unprojected table is safe to reuse.
            projected = (self.response.project_context(cache.table)
                         if cache.frozen_encoder and self.efficiency else cache.projected_context)
            def context(points):
                origin = torch.as_tensor(blocks.origin, device=points.device, dtype=points.dtype)
                idx = ((points-origin) * self.token_grid/(blocks.side*blocks.pitch)).floor().long().clamp(0, self.token_grid-1)
                keys = (idx[:,0]*self.token_grid+idx[:,1])*self.token_grid+idx[:,2]
                return projected.gather(keys) if projected is not None else cache.table[keys]
            if prepared is not None:
                return _evaluate_prepared_packed(prepared, operand.operation, cache.fixed_moments,
                    operand.vertex_velocity if with_jacobian else None,
                    model=self.response, context=context, chunk_size=chunk_size)
            return evaluate_packed(blocks, surface, width,
                                   operand.operation, cache.fixed_moments,
                                   operand.vertex_velocity if with_jacobian else None,
                                   model=self.response, context=context, chunk_size=chunk_size,
                                   full_cell_rule=self.full_cell_rule)
