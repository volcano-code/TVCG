"""Stable public prediction contract for the unified Spin Neural Moments operator.

This adapter composes existing model methods; it does not duplicate the
P-CRAFT, M80, SQNO, UQ or Event networks. Exact solver acceptance remains
outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .unified_m80_sqno import UnifiedGeometryCache, UnifiedM80SQNO


@dataclass
class SpinMomentPrediction:
    """Typed outputs for one geometry and one or more raw quadric queries.

    The query dimension is always preserved: Q=1 returns [1,10] and
    [1,10,10]. Missing auxiliary heads are None, never fabricated zeros.
    """

    moments: Tensor
    jacobian_coefficients: Tensor
    moment_log_std: Tensor | None
    jacobian_log_std: Tensor | None
    event_logits: Tensor | None
    event_probability: Tensor | None
    teacher_derivative_validity_available: bool
    event_prediction_available: bool
    uq_prediction_available: bool
    prediction_finite: Tensor
    diagnostics: dict[str, Any]


def _evaluate_chunk(
    model: UnifiedM80SQNO,
    cache: UnifiedGeometryCache,
    q: Tensor,
    *,
    create_graph: bool,
) -> SpinMomentPrediction:
    result = model.evaluate_queries(
        cache, q, create_graph=create_graph, with_jacobian=True
    )
    moments = result["moments"]
    jacobian = result["jacobian_coefficients"]

    uq = None
    if model.shared_encoder.config.uq_enabled:
        uq = model.shared_encoder.predict_uq(cache.shared_cache, q)

    event_logits = None
    if model.shared_encoder.config.event_validity_head_enabled:
        event_logits = model.shared_encoder.predict_event_validity(
            cache.shared_cache, q
        )
    event_probability = None if event_logits is None else torch.sigmoid(event_logits)

    finite = torch.isfinite(moments).all(dim=-1) & torch.isfinite(
        jacobian
    ).all(dim=(-2, -1))
    if uq is not None:
        finite = finite & torch.isfinite(uq["moment_log_std"]).all(dim=-1)
        finite = finite & torch.isfinite(uq["jacobian_log_std"]).all(dim=(-2, -1))
    if event_logits is not None:
        finite = finite & torch.isfinite(event_logits)

    excluded = {
        "moments", "jacobian_coefficients", "jacobian_q",
        "m80_moments", "m80_jacobian_coefficients", "m80_jacobian_q",
        "sqno_moments", "sqno_jacobian_coefficients", "sqno_jacobian_q",
    }
    diagnostics = {key: value for key, value in result.items() if key not in excluded}
    return SpinMomentPrediction(
        moments=moments,
        jacobian_coefficients=jacobian,
        moment_log_std=None if uq is None else uq["moment_log_std"],
        jacobian_log_std=None if uq is None else uq["jacobian_log_std"],
        event_logits=event_logits,
        event_probability=event_probability,
        # Teacher derivative validity is a label, not a model prediction.
        # It is deliberately unavailable from inference alone.
        teacher_derivative_validity_available=False,
        event_prediction_available=event_logits is not None,
        uq_prediction_available=uq is not None,
        prediction_finite=finite,
        diagnostics=diagnostics,
    )


def predict_full(
    model: UnifiedM80SQNO,
    cache: UnifiedGeometryCache,
    raw_coefficients: Tensor,
    *,
    create_graph: bool = False,
    query_chunk_size: int | None = None,
) -> SpinMomentPrediction:
    """Evaluate fused moments/Jacobians and configured auxiliary heads.

    `query_chunk_size` bounds per-call query memory. Chunks share the same
    geometry cache. During training, callers must create that cache with
    autograd enabled and use create_graph=True if losses differentiate through
    the returned Jacobian.
    """
    q = torch.as_tensor(
        raw_coefficients,
        device=cache.shared_cache.latent.device,
        dtype=torch.float64,
    )
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if q.ndim != 2 or q.shape[-1] != 10:
        raise ValueError("raw_coefficients must have shape [Q,10]")
    if q.shape[0] < 1 or not torch.isfinite(q).all():
        raise ValueError("raw_coefficients must be nonempty and finite")
    if (q.abs().amax(dim=-1) <= 0).any():
        raise ValueError("raw_coefficients must be nonzero")

    chunk = q.shape[0] if query_chunk_size is None else int(query_chunk_size)
    if chunk < 1:
        raise ValueError("query_chunk_size must be positive")
    parts = [
        _evaluate_chunk(model, cache, q[start:start + chunk], create_graph=create_graph)
        for start in range(0, q.shape[0], chunk)
    ]

    def cat_optional(name: str) -> Tensor | None:
        values = [getattr(part, name) for part in parts]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RuntimeError(f"inconsistent auxiliary output availability: {name}")
        return torch.cat(values, dim=0)

    diagnostic_keys = set.intersection(
        *(set(part.diagnostics) for part in parts)
    ) if parts else set()
    diagnostics: dict[str, Any] = {
        "chunk_count": len(parts),
        "query_chunk_size": chunk,
        "geometry_id": cache.geometry_id,
    }
    for key in diagnostic_keys:
        values = [part.diagnostics[key] for part in parts]
        if all(isinstance(value, Tensor) and value.ndim > 0 for value in values):
            try:
                diagnostics[key] = torch.cat(values, dim=0)
                continue
            except RuntimeError:
                pass
        diagnostics[key] = values[0] if len(values) == 1 else values

    return SpinMomentPrediction(
        moments=torch.cat([part.moments for part in parts], dim=0),
        jacobian_coefficients=torch.cat(
            [part.jacobian_coefficients for part in parts], dim=0
        ),
        moment_log_std=cat_optional("moment_log_std"),
        jacobian_log_std=cat_optional("jacobian_log_std"),
        event_logits=cat_optional("event_logits"),
        event_probability=cat_optional("event_probability"),
        teacher_derivative_validity_available=False,
        event_prediction_available=parts[0].event_prediction_available,
        uq_prediction_available=parts[0].uq_prediction_available,
        prediction_finite=torch.cat(
            [part.prediction_finite for part in parts], dim=0
        ),
        diagnostics=diagnostics,
    )


__all__ = ["SpinMomentPrediction", "predict_full"]
