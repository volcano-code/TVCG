"""Compact V2.1 single physical response to a deployed finite operand.

The phase-A entry point, evaluate_packed, uses no learned weights. The optional
encoder reuses P-CRAFT's two-scale U-FNO and local feature layers with conservative
voxel channels. Stage B training admission lives outside this module.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    PcraftNativeMomentOperator, parameter_versions,
)
from .unified_factory import _make_native_config, get_config
from .v21_response import LocalRetention, prepare_measure, evaluate_response, prepare_surface


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
                    *, model=None, context=None, chunk_size=65536):
    """Use the signed-distance bound on complete blocks and stream the remainder."""
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
                band_nodes=0, active_nodes=0, ambiguous_nodes=0, chunks=0)
    for nodes, weights in blocks.chunks(band, max_nodes=chunk_size):
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


class CompactPINOv21(nn.Module):
    """A Token16/Token8 candidate with one bounded local retention field."""
    def __init__(self, token_grid=16):
        super().__init__()
        if token_grid not in (8, 16):
            raise ValueError("expected Token8 or Token16")
        self.token_grid = token_grid
        cfg = _make_native_config(get_config('compact_pino_v2_256'))
        self.shared_encoder = PcraftNativeMomentOperator(cfg, encoder_only=True)
        width = cfg.interval_width
        self.geometry_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.response = LocalRetention(context_dim=width, hidden_dim=64)

    def encode_geometry(self, blocks, fixed_moments, density=1., *, inference=False):
        if blocks.side != 256 or blocks.side % (blocks.block_voxels * self.token_grid):
            raise ValueError("native256 blocks must nest exactly within the token grid")
        if inference and self.training:
            raise ValueError("inference cache requires eval()")
        fixed = np.array(fixed_moments, dtype=float, copy=True)
        if fixed.shape != (10,) or not np.isfinite(fixed).all():
            raise ValueError("expected ten fixed moments")
        token_count = self.token_grid**3
        span = blocks.side * blocks.pitch
        ijk = np.floor((blocks.centers - blocks.origin)/span * self.token_grid).astype(int)
        ids = np.ravel_multi_index(ijk.T, (self.token_grid,)*3)
        mass_table = np.zeros((token_count, 10))
        np.add.at(mass_table, ids, blocks.moments)
        active = np.flatnonzero(mass_table[:,0] > 0)
        m = mass_table[active]
        editable = m.sum(0)
        means = m / m[:, :1]
        width = span / self.token_grid
        coord = np.power(span, [0,1,1,1,2,2,2,2,2,2])
        scale = (editable[0] + fixed[0]) * coord
        centers = blocks.origin + (np.array(np.unravel_index(active, (self.token_grid,)*3)).T + .5)*width
        lo, hi = centers.min(0)-width/2, centers.max(0)+width/2
        features = np.column_stack((means/coord, np.full((len(m), 3), width/span)))
        stats = np.r_[(editable+fixed)/scale, fixed/scale, editable/scale, (lo+hi)/(2*span), (hi-lo)/span]
        encoder = self.shared_encoder
        p = next(encoder.parameters())
        def tensor(a):
            return torch.as_tensor(a, device=p.device, dtype=p.dtype)
        with torch.no_grad() if inference else nullcontext():
            maps, globals_ = [], []
            for grid in encoder.config.multiscale_grids:
                pooled, fmap = encoder.pino_encoder.forward_with_map(tensor(blocks.encoder_grid(grid, density))[None],
                                                                     preserve_resolution=True)
                maps.append(fmap)
                globals_.append(encoder.pino_global(pooled))
            glob = torch.stack(globals_, dim=1)
            s = tensor(stats)[None]
            score = encoder.pino_scale_score(torch.cat((glob, s[:, None].expand(-1, len(maps), -1)), -1))
            shared = encoder.statistics_context(s) + (glob * score.softmax(1)).sum(1)
            coords = tensor((2*(means[:,1:4]-blocks.origin)/span - 1)[:, [2,1,0]].copy())
            local = encoder._interval_features(tensor(features), coords, maps, shared)
            # This attention is query independent; cut dependence enters once through
            # the actual operand and its motion in LocalRetention.
            log_mass = tensor(np.log(m[:,0]/editable[0]))[None]
            summary, _ = self.geometry_attention(shared[:,None], local[None], local[None],
                                                 attn_mask=log_mass, need_weights=False)
            table = torch.index_copy(local.new_zeros((token_count, local.shape[1])), 0,
                                     torch.as_tensor(active, device=p.device), local + summary[0])
        fixed.flags.writeable = False
        return GeometryFeatures(blocks, fixed, table, id(self), parameter_versions(self),
                                inference or not torch.is_grad_enabled())

    def evaluate_operand(self, cache, operand, *, with_jacobian=True, bandwidth=None, chunk_size=65536):
        if cache.owner != id(self) or cache.versions != parameter_versions(self):
            raise ValueError("foreign or stale learned geometry cache")
        if cache.inference_only and self.training:
            raise ValueError("inference-only geometry features cannot train the model")
        blocks = cache.blocks
        if with_jacobian and operand.vertex_velocity is None:
            raise ValueError("this operand has no native derivative")
        surface = prepare_surface(operand.vertices, operand.faces)
        def context(points):
            origin = torch.as_tensor(blocks.origin, device=points.device, dtype=points.dtype)
            idx = ((points-origin) * self.token_grid/(blocks.side*blocks.pitch)).floor().long().clamp(0, self.token_grid-1)
            keys = (idx[:,0]*self.token_grid+idx[:,1])*self.token_grid+idx[:,2]
            return cache.table[keys]
        with torch.no_grad() if cache.inference_only else nullcontext():
            return evaluate_packed(blocks, surface, blocks.pitch if bandwidth is None else bandwidth,
                                   operand.operation, cache.fixed_moments,
                                   operand.vertex_velocity if with_jacobian else None,
                                   model=self.response, context=context, chunk_size=chunk_size)
