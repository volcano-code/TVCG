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

    Query dimension is preserved: Q=1 returns [1,10] and [1,10,10].
    Unavailable auxiliary heads are represented by None plus explicit flags,
    never by fabricated zero tensors.
    """

    moments: Tensor
    jacobian_coefficients: Tensor
    moment_log_std: Tensor | None
    jacobian_log_std: Tensor | None
    event_logits: Tensor | None
    event_probability: Tensor | None
    derivative_validity_available: bool
    event_prediction_available: bool
    uq_prediction_available: bool
    prediction_finite: Tensor
    diagnostics: dict[str, Any]


def predict_full(
    model: UnifiedM80SQNO,
    cache: UnifiedGeometryCache,
    raw_coefficients: Tensor,
    *,
    create_graph: bool = False,
) -> SpinMomentPrediction:
    """Evaluate the fused moment function and all configured auxiliary heads.

    The Jacobian is obtained from UnifiedM80SQNO.evaluate_queries, whose
    implementation includes the raw-coefficient chain rule for query
    conditioned fusion and attention corrections.
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

    result = model.evaluate_queries(
        cache, q, create_graph=create_graph, with_jacobian=True
    )
    moments = result["moments"]
    jacobian = result["jacobian_coefficients"]

    uq = None
    if model.shared_encoder.config.uq_enabled:
        uq = model.shared_encoder.predict_uq(cache.shared_cache, q)

    event_logits = None
    event_probability = None
    if model.shared_encoder.config.event_validity_head_enabled:
        event_logits = model.shared_encoder.predict_event_validity(
            cache.shared_cache, q
        )
        event_probability = torch.sigmoid(event_logits)

    finite = torch.isfinite(moments).all(dim=-1) & torch.isfinite(
        jacobian
    ).all(dim=(-2, -1))
    if uq is not None:
        finite = finite & torch.isfinite(uq["moment_log_std"]).all(dim=-1)
        finite = finite & torch.isfinite(uq["jacobian_log_std"]).all(dim=(-2, -1))
    if event_logits is not None:
        finite = finite & torch.isfinite(event_logits)

    diagnostics = {
        key: value
        for key, value in result.items()
        if key not in {
            "moments", "jacobian_coefficients", "jacobian_q",
            "m80_moments", "m80_jacobian_coefficients", "m80_jacobian_q",
            "sqno_moments", "sqno_jacobian_coefficients", "sqno_jacobian_q",
        }
    }
    return SpinMomentPrediction(
        moments=moments,
        jacobian_coefficients=jacobian,
        moment_log_std=None if uq is None else uq["moment_log_std"],
        jacobian_log_std=None if uq is None else uq["jacobian_log_std"],
        event_logits=event_logits,
        event_probability=event_probability,
        derivative_validity_available=True,
        event_prediction_available=event_logits is not None,
        uq_prediction_available=uq is not None,
        prediction_finite=finite,
        diagnostics=diagnostics,
    )


__all__ = ["SpinMomentPrediction", "predict_full"]
