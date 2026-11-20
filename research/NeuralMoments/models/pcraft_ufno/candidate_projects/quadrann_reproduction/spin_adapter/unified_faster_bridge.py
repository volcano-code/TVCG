"""Bridge Unified M80+SQNO predictions into the existing M69/S9 KKT solver.

The bridge reuses Phase 2.3/2.4/2.5 solver mathematics. It does not invent a
new Faster objective or replace the exact M69 merit/oracle. The caller must
supply the authoritative TaskSpec, original moments, active set, and exact
merit/fallback callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

from .unified_m80_sqno import UnifiedGeometryCache, UnifiedM80SQNO
from .unified_prediction import predict_full
from research.NeuralMoments.phase2_4_neural_newton import (
    NewtonDirection,
    S9Chart,
    StepAcceptance,
 )
from research.NeuralMoments.phase2_5_end_to_end_solver import (
    accept_m69_neural_newton_step,
    propose_m69_neural_newton_step,
)


@dataclass
class UnifiedM69Proposal:
    chart: S9Chart
    kkt_system: Any
    direction: NewtonDirection
    current_beta: Tensor
    prediction_diagnostics: dict[str, Any]


def make_unified_moment_callback(
    model: UnifiedM80SQNO,
    cache: UnifiedGeometryCache,
    *,
    create_graph: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Return beta->[10] fused moments from the same unified M80+SQNO function.

    create_graph=True is required when the M69 KKT builder differentiates the
    Moment Jacobian to form its stationarity Jacobian/Hessian.
    """
    if create_graph is False:
        raise ValueError("M69 KKT bridge requires create_graph=True")

    def beta_to_moments(beta: Tensor) -> Tensor:
        value = torch.as_tensor(
            beta,
            device=cache.shared_cache.latent.device,
            dtype=torch.float64,
        )
        if value.shape != (10,):
            raise ValueError("beta must be a raw ten-coefficient vector")
        result = model.evaluate_queries(
            cache,
            value.unsqueeze(0),
            create_graph=True,
            with_jacobian=True,
        )
        moments = result["moments"]
        if moments.shape != (1, 10) or not torch.isfinite(moments).all():
            raise ValueError("unified model returned invalid Moment output")
        return moments[0]

    return beta_to_moments


def propose_unified_m69_step(
    *,
    model: UnifiedM80SQNO,
    cache: UnifiedGeometryCache,
    local: Tensor,
    beta_anchor: Tensor,
    original_moments: Tensor,
    task: dict,
    length_scale: float,
    active_mask,
    trust_radius: float,
    max_event_probability: float = 0.25,
    max_moment_log_std: float = 0.0,
    max_jacobian_log_std: float = 0.0,
    event_tolerance: float = 1e-8,
) -> UnifiedM69Proposal:
    """Build a Faster-compatible S9/M69 KKT proposal from unified predictions.

    Event/UQ heads are conservative proposal gates only. Exact M69 event,
    feasibility, merit and acceptance remain authoritative.
    """
    beta_anchor = torch.as_tensor(
        beta_anchor,
        device=cache.shared_cache.latent.device,
        dtype=torch.float64,
    )
    preview = predict_full(
        model, cache, beta_anchor.unsqueeze(0), create_graph=False
    )
    if not bool(preview.prediction_finite.all()):
        raise ValueError("unified prediction is nonfinite; exact fallback required")
    if preview.event_probability is not None:
        if float(preview.event_probability.max()) > max_event_probability:
            raise ValueError("predicted event risk exceeds proposal threshold")
    if preview.moment_log_std is not None:
        if float(preview.moment_log_std.max()) > max_moment_log_std:
            raise ValueError("Moment uncertainty exceeds proposal threshold")
    if preview.jacobian_log_std is not None:
        if float(preview.jacobian_log_std.max()) > max_jacobian_log_std:
            raise ValueError("Jacobian uncertainty exceeds proposal threshold")

    beta_fn = make_unified_moment_callback(model, cache, create_graph=True)
    chart, system, direction = propose_m69_neural_newton_step(
        local,
        beta_anchor,
        beta_fn,
        original_moments,
        task,
        length_scale=length_scale,
        active_mask=active_mask,
        trust_radius=trust_radius,
        event_tolerance=event_tolerance,
    )
    if not system.derivative_valid:
        raise ValueError(
            "M69 KKT derivative invalid: " + ",".join(system.invalid_reasons)
        )
    return UnifiedM69Proposal(
        chart=chart,
        kkt_system=system,
        direction=direction,
        current_beta=beta_anchor.detach().clone(),
        prediction_diagnostics={
            "prediction_finite": preview.prediction_finite.detach(),
            "event_probability": None if preview.event_probability is None else preview.event_probability.detach(),
            "moment_log_std": None if preview.moment_log_std is None else preview.moment_log_std.detach(),
            "jacobian_log_std": None if preview.jacobian_log_std is None else preview.jacobian_log_std.detach(),
        },
    )


def accept_unified_m69_step(
    state: Any,
    proposal: UnifiedM69Proposal,
    *,
    decode: Callable[[Any, Tensor], Any],
    exact_m69_merit: Callable[[Any], float],
    predicted_reduction: float,
    exact_solver_fallback: Callable[[Any], Any] | None = None,
) -> StepAcceptance:
    """Delegate acceptance/rejection/fallback to the existing exact-merit loop."""
    return accept_m69_neural_newton_step(
        state,
        proposal.direction,
        decode=decode,
        exact_m69_merit=exact_m69_merit,
        predicted_reduction=predicted_reduction,
        exact_solver_fallback=exact_solver_fallback,
    )


__all__ = [
    "UnifiedM69Proposal",
    "make_unified_moment_callback",
    "propose_unified_m69_step",
    "accept_unified_m69_step",
]
