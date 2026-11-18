"""Phase 2.6 R2: Unified M80 + Quadric + Local-Correction optimization.

The learned M80 operator remains the global geometry-conditioned Moment
potential. Local Correction is introduced as a residual material-response
field. The hybrid surrogate is deliberately anchored so psi=0 is exactly the
M80/Pure-Quadric path:
    M_h(beta,0) = M80(beta).
For psi != 0:
    M_h = M80(beta) + [M_local(beta,psi)-M_local(beta,0)].

This keeps the learned global operator intact while exposing beta and psi to a
single differentiable M69 KKT/Newton optimization state. Exact binary
acceptance remains an independent oracle over (beta, psi).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
import math
import torch
from torch import Tensor

from research.NeuralMoments.phase2_4_neural_newton import (
    S9Chart, NewtonDirection, StepAcceptance, make_s9_chart,
    safeguarded_kkt_newton_direction, exact_merit_acceptance,
)
from research.NeuralMoments.phase2_3_differentiable_m69_kkt import (
    KKTSystem, M69TorchConstraintResult, m69_kkt_from_moment_operator,
    m69_constraint_slacks_torch,
)
from research.NeuralMoments.phase2_6_local_correction import (
    LocalCorrectionField, MaterialMasks, integrate_hybrid_moments,
)


@dataclass(frozen=True)
class HybridMaterialChart:
    """Nine projective Quadric coordinates plus K local-correction coordinates."""
    quadric_chart: S9Chart
    local_dimension: int

    @property
    def dimension(self) -> int:
        return 9 + self.local_dimension

    def decode(self, state: Tensor) -> tuple[Tensor, Tensor]:
        if state.shape != (self.dimension,) or not torch.isfinite(state).all():
            raise ValueError(f"hybrid state must be finite [{self.dimension}]")
        beta = self.quadric_chart.decode(state[:9])
        psi = state[9:]
        return beta, psi

    def encode_zero_correction(self, beta: Tensor) -> Tensor:
        if beta.shape != (10,):
            raise ValueError("beta must be [10]")
        local = self.quadric_chart.encode_tangent(beta)
        return torch.cat((local, torch.zeros(
            self.local_dimension, device=beta.device, dtype=beta.dtype)))


@dataclass(frozen=True)
class HybridM80LocalProblem:
    base_moment_fn: Callable[[Tensor], Tensor]
    local_field: LocalCorrectionField
    points: Tensor
    quadrature_mass: Tensor
    masks: MaterialMasks
    original_moments: Tensor
    task: Mapping[str, Any]
    length_scale: float
    beta_anchor: Tensor


@dataclass(frozen=True)
class HybridNeuralProposal:
    chart: HybridMaterialChart
    state: Tensor
    beta: Tensor
    psi: Tensor
    moments: Tensor
    constraints: M69TorchConstraintResult
    kkt: KKTSystem
    direction: NewtonDirection
    active_mask: Tensor


@dataclass(frozen=True)
class ExactHybridEvaluation:
    moments: Tensor
    constraints: M69TorchConstraintResult
    connected: bool
    topology_valid: bool
    event_valid: bool
    source: str


@dataclass(frozen=True)
class HybridExactAcceptance:
    decision: StepAcceptance
    exact_evaluations: int
    exact_fallback_used: bool
    exact_fallback_result: Any


def make_hybrid_m80_local_problem(
    *,
    base_moment_fn: Callable[[Tensor], Tensor],
    local_field: LocalCorrectionField,
    points: Tensor,
    quadrature_mass: Tensor,
    masks: MaterialMasks,
    original_moments: Tensor,
    task: Mapping[str, Any],
    length_scale: float,
    beta_anchor: Tensor,
) -> HybridM80LocalProblem:
    if not callable(base_moment_fn):
        raise TypeError("base_moment_fn must be callable")
    if beta_anchor.shape != (10,) or not torch.isfinite(beta_anchor).all():
        raise ValueError("beta_anchor must be finite [10]")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if quadrature_mass.shape != (len(points),):
        raise ValueError("quadrature_mass must be [N]")
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    if original_moments.shape != (10,) or not torch.isfinite(original_moments).all():
        raise ValueError("original_moments must be finite [10]")
    masks.validate(len(points), points.device)
    if local_field.centers.device != points.device or local_field.centers.dtype != points.dtype:
        raise ValueError("local field and integration samples must share dtype/device")
    if beta_anchor.device != points.device or beta_anchor.dtype != points.dtype:
        raise ValueError("beta_anchor and integration samples must share dtype/device")
    return HybridM80LocalProblem(
        base_moment_fn, local_field, points, quadrature_mass, masks,
        original_moments, dict(task), float(length_scale),
        beta_anchor.detach().clone()
    )


def make_hybrid_material_chart(problem: HybridM80LocalProblem) -> HybridMaterialChart:
    return HybridMaterialChart(
        make_s9_chart(problem.beta_anchor),
        problem.local_field.centers.shape[0],
    )


def hybrid_moment_function(problem: HybridM80LocalProblem) -> Callable[[Tensor], Tensor]:
    """Build the single beta/psi -> Moment operator used by KKT/Newton."""
    chart = make_hybrid_material_chart(problem)

    def moment_from_state(state: Tensor) -> Tensor:
        beta, psi = chart.decode(state)
        global_moments = problem.base_moment_fn(beta)
        if global_moments.shape != (10,) or not torch.isfinite(global_moments).all():
            raise ValueError("base M80 moment function must return finite [10]")
        local = integrate_hybrid_moments(
            problem.local_field, beta, psi, problem.points,
            problem.quadrature_mass, problem.masks
        )
        pure = integrate_hybrid_moments(
            problem.local_field, beta,
            torch.zeros_like(psi), problem.points,
            problem.quadrature_mass, problem.masks
        )
        # Residual correction: no local-correction double counting at psi=0.
        return global_moments.to(local) + local - pure

    return moment_from_state


def hybrid_zero_correction_identity(
    problem: HybridM80LocalProblem, state: Tensor
) -> tuple[Tensor, Tensor]:
    chart = make_hybrid_material_chart(problem)
    beta, psi = chart.decode(state)
    if not torch.allclose(psi, torch.zeros_like(psi), atol=1e-12, rtol=0):
        raise ValueError("identity check requires psi=0")
    fn = hybrid_moment_function(problem)
    return fn(state), problem.base_moment_fn(beta)


def build_hybrid_m69_kkt_system(
    problem: HybridM80LocalProblem,
    state: Tensor,
    active_mask: Tensor,
    *,
    event_tolerance: float = 1e-8,
) -> tuple[HybridMaterialChart, KKTSystem]:
    chart = make_hybrid_material_chart(problem)
    if state.shape != (chart.dimension,):
        raise ValueError("hybrid state dimension mismatch")
    if active_mask.shape != (10,):
        raise ValueError("M69 active_mask must be [10]")
    moment_fn = hybrid_moment_function(problem)
    system = m69_kkt_from_moment_operator(
        state, moment_fn, problem.original_moments, dict(problem.task),
        length_scale=problem.length_scale,
        active_mask=active_mask, event_tolerance=event_tolerance
    )
    return chart, system


def propose_hybrid_m69_newton_step(
    problem: HybridM80LocalProblem,
    state: Tensor,
    active_mask: Tensor,
    *,
    trust_radius: float,
    event_tolerance: float = 1e-8,
) -> HybridNeuralProposal:
    chart, system = build_hybrid_m69_kkt_system(
        problem, state, active_mask, event_tolerance=event_tolerance
    )
    if not system.derivative_valid:
        raise ValueError(
            "Hybrid M69 KKT derivative invalid: "
            + ",".join(system.invalid_reasons)
        )
    if system.residual.numel() <= chart.dimension:
        raise ValueError("Hybrid KKT needs at least one active constraint")
    direction = safeguarded_kkt_newton_direction(
        system.residual, system.jacobian,
        primal_dimension=chart.dimension, trust_radius=trust_radius
    )
    moment_fn = hybrid_moment_function(problem)
    moments = moment_fn(state)
    constraints = m69_constraint_slacks_torch(
        moments, problem.original_moments, dict(problem.task),
        length_scale=problem.length_scale, event_tolerance=event_tolerance
    )
    beta, psi = chart.decode(state)
    return HybridNeuralProposal(
        chart, state.detach().clone(), beta, psi, moments, constraints,
        system, direction, active_mask.detach().clone()
    )


def evaluate_exact_hybrid_candidate(
    beta: Tensor,
    psi: Tensor,
    *,
    exact_material_oracle: Callable[[Tensor, Tensor], Mapping[str, Any]],
    original_moments: Tensor,
    task: Mapping[str, Any],
    length_scale: float,
    event_tolerance: float = 1e-9,
) -> ExactHybridEvaluation:
    raw = exact_material_oracle(beta.detach(), psi.detach())
    required = ("moments", "connected", "topology_valid", "event_valid", "source")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError("exact hybrid oracle result missing fields: " + ",".join(missing))
    if not str(raw["source"]).strip():
        raise ValueError("exact hybrid oracle source must be named")
    moments = torch.as_tensor(
        raw["moments"], device=original_moments.device, dtype=torch.float64
    )
    if moments.shape != (10,):
        raise ValueError("exact hybrid oracle moments must be [10]")
    constraints = m69_constraint_slacks_torch(
        moments, original_moments, dict(task),
        length_scale=length_scale, event_tolerance=event_tolerance
    )
    return ExactHybridEvaluation(
        moments, constraints, bool(raw["connected"]),
        bool(raw["topology_valid"]), bool(raw["event_valid"]),
        str(raw["source"])
    )


def exact_hybrid_merit(
    evaluation: ExactHybridEvaluation,
    *,
    connectivity_penalty: float,
    feasibility_penalty: float,
) -> float:
    if connectivity_penalty <= 0 or feasibility_penalty <= 0:
        raise ValueError("exact merit penalties must be positive")
    violation = torch.clamp(evaluation.constraints.g_le_zero, min=0).sum()
    merit = float(evaluation.moments[0].detach()) + feasibility_penalty * float(violation.detach())
    if not evaluation.connected or not evaluation.topology_valid or not evaluation.event_valid:
        merit += connectivity_penalty
    return merit


def make_exact_hybrid_merit_callback(
    problem: HybridM80LocalProblem,
    *,
    exact_material_oracle: Callable[[Tensor, Tensor], Mapping[str, Any]],
    state_to_material: Callable[[Any], tuple[Tensor, Tensor]],
    connectivity_penalty: float = 1e6,
    feasibility_penalty: float = 1e4,
    event_tolerance: float = 1e-9,
) -> Callable[[Any], float]:
    if connectivity_penalty <= 0 or feasibility_penalty <= 0:
        raise ValueError("exact merit penalties must be positive")

    def evaluate(state: Any) -> float:
        beta, psi = state_to_material(state)
        result = evaluate_exact_hybrid_candidate(
            beta, psi, exact_material_oracle=exact_material_oracle,
            original_moments=problem.original_moments,
            task=problem.task, length_scale=problem.length_scale,
            event_tolerance=event_tolerance
        )
        evaluate.evaluation_count += 1
        return exact_hybrid_merit(
            result, connectivity_penalty=connectivity_penalty,
            feasibility_penalty=feasibility_penalty
        )

    evaluate.evaluation_count = 0
    return evaluate


def accept_hybrid_proposal_with_exact_oracle(
    state: Any,
    proposal: HybridNeuralProposal,
    *,
    decode: Callable[[Any, Tensor], Any],
    exact_merit: Callable[[Any], float],
    predicted_reduction: float,
    exact_solver_fallback: Callable[[Any], Any] | None = None,
) -> HybridExactAcceptance:
    before = getattr(exact_merit, "evaluation_count", 0)
    decision = exact_merit_acceptance(
        state, proposal.direction.step, decode=decode,
        exact_merit=exact_merit, predicted_reduction=predicted_reduction
    )
    fallback_result = None
    if not decision.accepted and exact_solver_fallback is not None:
        fallback_result = exact_solver_fallback(state)
    after = getattr(exact_merit, "evaluation_count", before)
    return HybridExactAcceptance(
        decision, max(0, after - before),
        fallback_result is not None, fallback_result
    )
