"""Mass-weighted geometry attention with an exact raw-query Jacobian."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .memory_profile import capture_profile, profile_count, profile_recompute, profile_stage, profile_tensors, use_profile


def _query_tiles(count, tile):
    return (slice(i, min(i + tile, count)) for i in range(0, count, tile))


def _scores(query, tangent, keys, log_mass, scale):
    """Score and directional derivatives, with raw coefficient as last axis."""
    score = torch.einsum("qhd,nhd->qhn", query, keys) * scale + log_mass
    direction = torch.einsum("qhda,nhd->qhna", tangent, keys) * scale
    return score, direction


def _streaming_statistics(q, dq, features, log_mass, key_weight, value_weight,
                          interval_chunk, query_chunk, validate, projected_kv=None):
    """One common reduction for training and the opt-in inference projection cache."""
    nq, heads, dim = q.shape
    axes = dq.shape[-1]
    scale = dim ** -0.5
    max_score = q.new_full((nq, heads, 1), -torch.inf)
    denom = torch.zeros_like(max_score)
    values_sum = torch.zeros_like(q)
    directions_sum = q.new_zeros(nq, heads, axes)
    cross_sum = torch.zeros_like(dq)
    scores_sum = torch.zeros_like(denom)

    with profile_stage("attention_moments", phase="forward"):
        for ns in _query_tiles(len(features), interval_chunk):
            feature = features[ns]
            if validate and not torch.isfinite(feature).all():
                raise ValueError(f"attention interval features must be finite (chunk {ns.start}:{ns.stop})")
            if projected_kv is None:
                key = F.linear(feature, key_weight).reshape(-1, heads, dim)
                value = F.linear(feature, value_weight).reshape(-1, heads, dim)
            else:
                key, value = projected_kv[0][ns], projected_kv[1][ns]
            for qs in _query_tiles(nq, query_chunk):
                score, direction = _scores(q[qs], dq[qs], key, log_mass[ns], scale)
                new_max = torch.maximum(max_score[qs], score.amax(-1, keepdim=True))
                old_scale = torch.exp(max_score[qs] - new_max)
                weight = torch.exp(score - new_max)
                weighted_direction = weight[..., None] * direction
                denom[qs] = denom[qs] * old_scale + weight.sum(-1, keepdim=True)
                scores_sum[qs] = scores_sum[qs] * old_scale + (weight * score).sum(-1, keepdim=True)
                values_sum[qs] = values_sum[qs] * old_scale + torch.einsum(
                    "qhn,nhd->qhd", weight, value)
                directions_sum[qs] = directions_sum[qs] * old_scale + weighted_direction.sum(-2)
                cross_sum[qs] = cross_sum[qs] * old_scale[..., None] + torch.einsum(
                    "qhna,nhd->qhda", weighted_direction, value)
                max_score[qs] = new_max
                if ns.start == 0 and qs.start == 0:
                    profile_tensors(score_tile=score, direction_tile=direction, key_tile=key)
            profile_count(interval_chunks=1, projected_intervals=len(feature) if projected_kv is None else 0,
                          query_tiles=math.ceil(nq / query_chunk))

    means = (values_sum / denom, directions_sum / denom, cross_sum / denom[..., None],
             scores_sum / denom, max_score + denom.log())
    return means


class _StreamingMoments(torch.autograd.Function):
    """Compute E[v], E[t], E[v t^T], E[z], log Z for t = dz/dc.

    The VJP reprojects each interval tile once and reconstructs its global
    probabilities from log Z. The inputs and small per-query reductions suffice
    for mixed parameter/query supervision. Returning all differentiable saved
    reductions also allows an explicit double backward through this VJP.
    """

    @staticmethod
    def forward(ctx, q, dq, features, log_mass, key_weight, value_weight,
                interval_chunk, query_chunk, validate):
        ctx.profile = capture_profile()
        means = _streaming_statistics(q, dq, features, log_mass, key_weight, value_weight,
                                      interval_chunk, query_chunk, validate)
        ctx.save_for_backward(q, dq, features, log_mass, key_weight, value_weight, *means)
        ctx.interval_chunk = interval_chunk
        ctx.query_chunk = query_chunk
        return means

    @staticmethod
    def backward(ctx, gv, gt, gvt, gz, glogz):
        q, dq, features, log_mass, wk, wv, ev, et, evt, ez, logz = ctx.saved_tensors
        nq, heads, dim = q.shape
        scale = dim ** -0.5
        grad_q = torch.zeros_like(q)
        grad_dq = torch.zeros_like(dq)
        grad_wk = torch.zeros_like(wk)
        grad_wv = torch.zeros_like(wv)
        grad_features = torch.empty_like(features) if ctx.needs_input_grad[2] else None
        grad_mass = torch.empty_like(log_mass) if ctx.needs_input_grad[3] else None
        # Subtract the global expectation from the softmax score adjoint.
        offset = (gv * ev).sum(-1, keepdim=True) + (gt * et).sum(-1, keepdim=True)
        offset = offset + (gvt * evt).sum((-1, -2)).unsqueeze(-1)
        with use_profile(ctx.profile), profile_stage("attention_moments", phase="backward"):
            for ns in _query_tiles(len(features), ctx.interval_chunk):
                feature = features[ns]
                with profile_recompute("attention_key_value_projection"):
                    key = F.linear(feature, wk).reshape(-1, heads, dim)
                    value = F.linear(feature, wv).reshape(-1, heads, dim)
                grad_key = torch.zeros_like(key)
                grad_value = torch.zeros_like(value)
                grad_prior = torch.zeros_like(log_mass[ns]) if grad_mass is not None else None
                for qs in _query_tiles(nq, ctx.query_chunk):
                    with profile_recompute("attention_scores_and_directions"):
                        score, direction = _scores(q[qs], dq[qs], key, log_mass[ns], scale)
                        weight = torch.exp(score - logz[qs])
                    # d <Gvt, v t^T> / dt = Gvt^T v
                    value_pullback = torch.einsum("qhda,nhd->qhna", gvt[qs], value)
                    direction_pullback = value_pullback + gt[qs].unsqueeze(-2)
                    score_pullback = torch.einsum("qhd,nhd->qhn", gv[qs], value)
                    score_pullback = score_pullback + (direction_pullback * direction).sum(-1) - offset[qs]
                    score_pullback = score_pullback + glogz[qs] + gz[qs] * (1 + score - ez[qs])
                    grad_score = weight * score_pullback
                    grad_direction = weight[..., None] * direction_pullback
                    grad_q[qs] += scale * torch.einsum("qhn,nhd->qhd", grad_score, key)
                    grad_dq[qs] += scale * torch.einsum("qhna,nhd->qhda", grad_direction, key)
                    grad_key = grad_key + scale * (
                        torch.einsum("qhn,qhd->nhd", grad_score, q[qs])
                        + torch.einsum("qhna,qhda->nhd", grad_direction, dq[qs]))
                    grad_value = grad_value + torch.einsum("qhn,qhd->nhd", weight, gv[qs])
                    grad_value = grad_value + torch.einsum(
                        "qhna,qhda->nhd", weight[..., None] * direction, gvt[qs])
                    if grad_prior is not None:
                        grad_prior = grad_prior + grad_score.sum((0, 1))
                grad_key = grad_key.flatten(1)
                grad_value = grad_value.flatten(1)
                grad_wk += grad_key.T @ feature
                grad_wv += grad_value.T @ feature
                if grad_features is not None:
                    grad_features[ns] = grad_key @ wk + grad_value @ wv
                if grad_mass is not None:
                    grad_mass[ns] = grad_prior
                profile_count(interval_chunks=1, projected_intervals=len(feature),
                              query_tiles=math.ceil(nq / ctx.query_chunk))
        return grad_q, grad_dq, grad_features, grad_mass, grad_wk, grad_wv, None, None, None


def _gelu_prime(x):
    return 0.5 * torch.erfc(-x / math.sqrt(2)) + x * torch.exp(-0.5 * x.square()) / math.sqrt(2 * math.pi)


def _mlp_with_tangent(mlp, x, dx):
    linear1, activation, linear2 = mlp
    h = linear1(x)
    dh = (linear1.weight @ dx) * _gelu_prime(h).unsqueeze(-1)
    return linear2(activation(h)), linear2.weight @ dh


class GeometryPhysicsCrossAttention(nn.Module):
    """Quadric queries attend to all intervals with a normalized physical-mass prior."""

    def __init__(self, width: int, *, heads: int = 2, chunk_intervals: int = 256,
                 hidden: int | None = None, output_scale: float = .05,
                 validate_inputs: bool = False, chunk_queries: int = 16):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        if chunk_intervals < 1 or chunk_queries < 1 or output_scale < 0:
            raise ValueError("invalid attention chunk or output scale")
        self.width = int(width)
        self.heads = int(heads)
        self.dim = width // heads
        self.chunk_intervals = int(chunk_intervals)
        self.chunk_queries = int(chunk_queries)
        self.output_scale = float(output_scale)
        self.validate_inputs = bool(validate_inputs)
        hidden = int(hidden or width)
        self.query_encoder = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, hidden),
                                    nn.GELU(), nn.Linear(hidden, 10))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @torch.no_grad()
    def prepare_projected_kv(self, features):
        if self.training:
            raise ValueError("KV projection cache requires eval()")
        if features.ndim != 2 or features.shape[1] != self.width or len(features) == 0:
            raise ValueError("invalid interval features for KV cache")
        shape = (len(features), self.heads, self.dim)
        keys = features.new_empty(shape)
        values = features.new_empty(shape)
        for ns in _query_tiles(len(features), self.chunk_intervals):
            keys[ns] = self.key(features[ns]).reshape(-1,self.heads,self.dim)
            values[ns] = self.value(features[ns]).reshape(-1,self.heads,self.dim)
        return keys, values

    def _evaluate(self, coefficients, features, mass, with_jacobian, projected_kv=None):
        if coefficients.ndim != 2 or coefficients.shape[1] != 10 or len(coefficients) == 0:
            raise ValueError("coefficients must have non-empty shape [Q,10]")
        if features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError("interval_features must have shape [N,width]")
        if mass.ndim != 1 or mass.shape[0] != len(features):
            raise ValueError("interval_mass must have shape [N]")
        if len(features) == 0:
            raise ValueError("geometry has no intervals")
        if self.validate_inputs:
            if not torch.isfinite(coefficients).all():
                raise ValueError("attention coefficients must be finite")
            if not torch.isfinite(mass).all() or (mass <= 0).any():
                raise ValueError("physical interval masses must be finite and positive")

        norm = torch.linalg.vector_norm(coefficients, dim=-1, keepdim=True)
        denominator = norm.clamp_min(1e-30)
        unit = coefficients / denominator
        if with_jacobian:
            eye = torch.eye(10, device=unit.device, dtype=unit.dtype)
            outer = unit.unsqueeze(-1) * unit.unsqueeze(-2)
            dunit = (eye - (norm >= 1e-30).unsqueeze(-1) * outer) / denominator.unsqueeze(-1)
            token, dtoken = _mlp_with_tangent(self.query_encoder, unit.to(features), dunit.to(features))
            dq = (self.query.weight @ dtoken).reshape(-1, self.heads, self.dim, 10)
        else:
            token = self.query_encoder(unit.to(features))
            dq = token.new_empty(len(token), self.heads, self.dim, 0)
        q = self.query(token).reshape(-1, self.heads, self.dim)
        mass_work = mass.to(features)
        log_prior = mass_work.clamp_min(1e-30).log() - mass_work.sum().clamp_min(1e-30).log()
        if projected_kv is None:
            context, et, evt, ez, logz = _StreamingMoments.apply(
                q, dq, features, log_prior, self.key.weight, self.value.weight,
                self.chunk_intervals, self.chunk_queries, self.validate_inputs)
        else:
            if self.training or torch.is_grad_enabled():
                raise ValueError("projected KV is inference-only and requires no_grad")
            expected = (len(features), self.heads, self.dim)
            if len(projected_kv)!=2 or any(k.shape!=expected or k.dtype!=features.dtype or
                    k.device!=features.device for k in projected_kv):
                raise ValueError("invalid projected KV cache")
            context, et, evt, ez, logz = _streaming_statistics(
                q,dq,features,log_prior,self.key.weight,self.value.weight,
                self.chunk_intervals,self.chunk_queries,self.validate_inputs,projected_kv)
        joined = torch.cat((token, context.flatten(1)), dim=1)
        jacobian = None
        if with_jacobian:
            # d E[v] / dc = E[v t^T] - E[v] E[t]^T.
            dcontext = evt - context.unsqueeze(-1) * et.unsqueeze(-2)
            tangent = torch.cat((dtoken, dcontext.flatten(1, 2)), dim=1)
            layernorm = self.output[0]
            variance, mean = torch.var_mean(joined, dim=-1, unbiased=False, keepdim=True)
            inv_std = torch.rsqrt(variance + layernorm.eps)
            normalized = (joined - mean) * inv_std
            projected = tangent - tangent.mean(dim=1, keepdim=True)
            projected = projected - normalized.unsqueeze(-1) * (
                normalized.unsqueeze(-1) * tangent).mean(dim=1, keepdim=True)
            dnormalized = (layernorm.weight * inv_std).unsqueeze(-1) * projected
            correction, jacobian = _mlp_with_tangent(
                (self.output[1], self.output[2], self.output[3]), layernorm(joined), dnormalized)
            jacobian = (jacobian * self.output_scale).to(coefficients)
        else:
            correction = self.output(joined)
        entropy = (logz - ez).squeeze(-1).clamp_min(0)
        diagnostics = {
            "attention_entropy": entropy.mean(dim=-1),
            "attention_effective_support": entropy.exp().mean(dim=-1),
            "attention_mass_sum": mass.sum().expand(len(coefficients)),
            "attention_query_norm": norm.squeeze(-1),
        }
        return correction * self.output_scale, jacobian, diagnostics

    def forward(self, coefficients: Tensor, interval_features: Tensor, interval_mass: Tensor, *, projected_kv=None):
        with profile_stage("geometry_physics_attention", phase="forward"):
            correction, _, diagnostics = self._evaluate(coefficients, interval_features, interval_mass, False, projected_kv)
        return correction, diagnostics

    def forward_with_query_jacobian(self, coefficients: Tensor, interval_features: Tensor, interval_mass: Tensor, *, projected_kv=None):
        """Return correction, d(correction)/d(raw coefficients) [Q,10,10], and diagnostics."""
        with profile_stage("geometry_physics_attention", phase="forward"):
            return self._evaluate(coefficients, interval_features, interval_mass, True, projected_kv)
