"""Neural Newton proposal and exact-evaluation gate.

This module deliberately does not implement or replace the Faster optimizer.
It computes a bounded candidate from a supplied optimizer residual/Jacobian,
then requires an authoritative exact evaluator to decide acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProposalPolicy:
    damping: float = 1e-6
    trust_radius: float = 0.1
    max_step_norm: float = 0.1
    max_moment_log_std: float = 0.0
    max_jacobian_log_std: float = 0.0
    max_event_probability: float = 0.25
    armijo: float = 1e-4
    minimum_objective_decrease: float = 0.0
    feasibility_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.damping <= 0 or self.trust_radius <= 0 or self.max_step_norm <= 0:
            raise ValueError("damping and step limits must be positive")
        if not 0.0 <= self.max_event_probability <= 1.0:
            raise ValueError("max_event_probability must be in [0,1]")
        if not 0.0 <= self.armijo < 1.0:
            raise ValueError("armijo must be in [0,1)")
        if self.feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be nonnegative")


@dataclass
class NeuralNewtonProposal:
    current_coefficients: Tensor
    candidate_coefficients: Tensor
    step: Tensor
    predicted_residual_norm: Tensor
    predicted_step_norm: Tensor
    eligible_for_exact_evaluation: bool
    fallback_reasons: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass
class ExactCorrectionResult:
    accepted: bool
    coefficients: Tensor
    objective_before: float
    objective_after: float
    constraint_violation: float
    reason: str
    exact_payload: Any = None


def _vector_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value.reshape(-1))


def make_neural_newton_proposal(
    *,
    current_coefficients: Tensor,
    predicted_residual: Tensor,
    predicted_jacobian: Tensor,
    moment_log_std: Tensor | None = None,
    jacobian_log_std: Tensor | None = None,
    event_probability: Tensor | None = None,
    policy: ProposalPolicy = ProposalPolicy(),
) -> NeuralNewtonProposal:
    """Solve a damped least-squares local model and clip it to a trust region.

    Residual and Jacobian are supplied by the Faster optimizer's objective
    adapter. This routine does not assume that raw ten-moment residuals are
    themselves the complete Faster stationarity system.
    """
    beta = torch.as_tensor(current_coefficients)
    residual = torch.as_tensor(predicted_residual, device=beta.device, dtype=torch.float64).reshape(-1)
    jac = torch.as_tensor(predicted_jacobian, device=beta.device, dtype=torch.float64)
    beta = beta.to(dtype=torch.float64).reshape(-1)
    if beta.numel() != 10:
        raise ValueError("current_coefficients must contain 10 raw coefficients")
    if jac.ndim != 2 or jac.shape != (residual.numel(), beta.numel()):
        raise ValueError("predicted_jacobian must have shape [residual_dim,10]")
    if residual.numel() == 0 or not torch.isfinite(beta).all() or not torch.isfinite(residual).all() or not torch.isfinite(jac).all():
        raise ValueError("proposal inputs must be nonempty and finite")

    reasons: list[str] = []
    if moment_log_std is not None:
        if not torch.isfinite(moment_log_std).all():
            reasons.append("nonfinite_moment_uncertainty")
        elif float(torch.as_tensor(moment_log_std).max()) > policy.max_moment_log_std:
            reasons.append("moment_uncertainty_above_policy")
    if jacobian_log_std is not None:
        if not torch.isfinite(jacobian_log_std).all():
            reasons.append("nonfinite_jacobian_uncertainty")
        elif float(torch.as_tensor(jacobian_log_std).max()) > policy.max_jacobian_log_std:
            reasons.append("jacobian_uncertainty_above_policy")
    if event_probability is not None:
        event = torch.as_tensor(event_probability, dtype=torch.float64, device=beta.device)
        if not torch.isfinite(event).all():
            reasons.append("nonfinite_event_probability")
        elif float(event.max()) > policy.max_event_probability:
            reasons.append("event_risk_above_policy")

    lhs = jac.T @ jac + policy.damping * torch.eye(10, dtype=jac.dtype, device=jac.device)
    rhs = -(jac.T @ residual)
    try:
        step = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        step = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
        reasons.append("linear_solve_fallback")
    raw_norm = _vector_norm(step)
    radius = min(policy.trust_radius, policy.max_step_norm)
    scale = torch.clamp(torch.as_tensor(radius, dtype=step.dtype, device=step.device) / raw_norm.clamp_min(1e-30), max=1.0)
    step = step * scale
    candidate = beta + step
    finite = torch.isfinite(candidate).all() and torch.isfinite(step).all()
    if not bool(finite):
        reasons.append("nonfinite_candidate")
    eligible = len(reasons) == 0 and bool(finite)
    return NeuralNewtonProposal(
        current_coefficients=beta,
        candidate_coefficients=candidate,
        step=step,
        predicted_residual_norm=_vector_norm(residual),
        predicted_step_norm=_vector_norm(step),
        eligible_for_exact_evaluation=eligible,
        fallback_reasons=tuple(reasons),
        diagnostics={
            "damping": policy.damping,
            "trust_radius": policy.trust_radius,
            "max_step_norm": policy.max_step_norm,
            "unclipped_step_norm": raw_norm.detach(),
            "clipped_scale": scale.detach(),
        },
    )


def exact_correction_and_accept(
    proposal: NeuralNewtonProposal,
    *,
    exact_evaluator: Callable[[Tensor], dict[str, Any]],
    policy: ProposalPolicy = ProposalPolicy(),
) -> ExactCorrectionResult:
    """Ask the exact solver to evaluate a proposal and apply a conservative gate.

    exact_evaluator must return objective, constraint_violation and may return
    any exact solver payload. This adapter never fabricates exact values.
    """
    current = exact_evaluator(proposal.current_coefficients)
    before = float(current["objective"])
    if not proposal.eligible_for_exact_evaluation:
        return ExactCorrectionResult(
            False, proposal.current_coefficients, before, before,
            float(current["constraint_violation"]),
            "neural proposal rejected by uncertainty/event/finite policy",
            current,
        )
    trial = exact_evaluator(proposal.candidate_coefficients)
    after = float(trial["objective"])
    violation = float(trial["constraint_violation"])
    if not all(torch.isfinite(torch.tensor(v)) for v in (before, after, violation)):
        return ExactCorrectionResult(
            False, proposal.current_coefficients, before, after, violation,
            "exact evaluator returned nonfinite acceptance metrics", trial,
        )
    required_decrease = max(
        policy.minimum_objective_decrease,
        policy.armijo * float(proposal.predicted_residual_norm) * float(proposal.predicted_step_norm),
    )
    accepted = violation <= policy.feasibility_tolerance and after <= before - required_decrease
    return ExactCorrectionResult(
        accepted,
        proposal.candidate_coefficients if accepted else proposal.current_coefficients,
        before,
        after,
        violation,
        "accepted_by_exact_gate" if accepted else "rejected_by_exact_gate",
        trial,
    )


__all__ = [
    "ProposalPolicy",
    "NeuralNewtonProposal",
    "ExactCorrectionResult",
    "make_neural_newton_proposal",
    "exact_correction_and_accept",
]
