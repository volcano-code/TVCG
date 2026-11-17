"""Phase 2.6: real M80 operator wiring with an independent exact M69 oracle boundary.

The learned path uses the repository's actual task-conditioned M80 wrapper and
native geometry cache. The exact path is a separate caller-supplied material
integration/oracle callback. No learned Moment output is relabelled as exact.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Mapping
import math
import torch
from torch import Tensor

from research.NeuralMoments.m69_task_conditioned_m80_v1 import (
    M69TaskConditionedM80, MaterialMapContract, encode_m69_task_features
)
from research.NeuralMoments.phase2_3_differentiable_m69_kkt import (
    M69TorchConstraintResult, m69_constraint_slacks_torch
)
from research.NeuralMoments.phase2_4_neural_newton import (
    S9Chart, NewtonDirection, StepAcceptance,
    build_s9_m69_kkt_system, safeguarded_kkt_newton_direction,
    exact_merit_acceptance
)


@dataclass(frozen=True)
class M80M69Problem:
    model: M69TaskConditionedM80
    geometry_cache: Any
    task: Mapping[str, Any]
    task_features: Tensor
    original_moments: Tensor
    length_scale: float
    beta_anchor: Tensor
    material_map_contract: MaterialMapContract


@dataclass(frozen=True)
class NeuralProposal:
    chart: S9Chart
    moments: Tensor
    constraint_result: M69TorchConstraintResult
    residual: Tensor
    jacobian: Tensor
    direction: NewtonDirection
    active_mask: Tensor


@dataclass(frozen=True)
class ExactM69Evaluation:
    moments: Tensor
    constraints: M69TorchConstraintResult
    connected: bool
    topology_valid: bool
    event_valid: bool
    source: str


@dataclass(frozen=True)
class M80ExactAcceptance:
    decision: StepAcceptance
    exact_evaluations: int
    exact_fallback_used: bool
    exact_fallback_result: Any


def make_m80_m69_problem(
    model:M69TaskConditionedM80,geometry:Mapping[str,Any],task:Mapping[str,Any],
    *,length_scale:float,beta_anchor:Tensor,material_map_contract:MaterialMapContract,
    chunk:int|None=None
)->M80M69Problem:
    """Encode native geometry once and bind the exact M69 task to the M80 wrapper."""
    if not isinstance(model,M69TaskConditionedM80):
        raise TypeError("model must be the repository M69TaskConditionedM80 wrapper")
    material_map_contract.validate()
    if not math.isfinite(length_scale) or length_scale<=0:
        raise ValueError("length_scale must be finite and positive")
    if beta_anchor.shape!=(10,) or not torch.isfinite(beta_anchor).all():
        raise ValueError("beta_anchor must be finite [10]")
    cache=model.encode_geometry(geometry,chunk)
    features=torch.as_tensor(
        encode_m69_task_features(task,length_scale=length_scale),
        device=cache.latent.device,dtype=cache.latent.dtype
    )
    original=torch.as_tensor(geometry["original_moments"],device=cache.latent.device,
                             dtype=torch.float64)
    beta=beta_anchor.to(device=cache.latent.device,dtype=torch.float64)
    return M80M69Problem(model,cache,dict(task),features,original,float(length_scale),
                         beta,material_map_contract)


def m80_moment_function(problem:M80M69Problem)->Callable[[Tensor],Tensor]:
    """Return the actual task-conditioned M80 beta-query -> ten-Moment function."""
    problem.material_map_contract.validate()
    def predict(beta:Tensor)->Tensor:
        return problem.model.predict(problem.geometry_cache,beta,problem.task_features)
    return predict


def propose_m80_m69_newton_step(
    problem:M80M69Problem,local:Tensor,active_mask:Tensor,*,trust_radius:float,
    event_tolerance:float=1e-8
)->NeuralProposal:
    """Construct M80->S9->M69 KKT and return a safeguarded Newton proposal."""
    moment_fn=m80_moment_function(problem)
    chart,system=build_s9_m69_kkt_system(
        local,problem.beta_anchor,moment_fn,problem.original_moments,
        dict(problem.task),length_scale=problem.length_scale,
        active_mask=active_mask,event_tolerance=event_tolerance
    )
    if not system.derivative_valid:
        raise ValueError("M80/M69 KKT derivative invalid: "+",".join(system.invalid_reasons))
    if system.residual.numel()<=9:
        raise ValueError("KKT system needs at least one active constraint for a saddle-point step")
    direction=safeguarded_kkt_newton_direction(
        system.residual,system.jacobian,primal_dimension=9,trust_radius=trust_radius
    )
    beta=chart.decode(local)
    moments=moment_fn(beta)
    constraint_result=m69_constraint_slacks_torch(
        moments,problem.original_moments,dict(problem.task),
        length_scale=problem.length_scale,event_tolerance=event_tolerance
    )
    return NeuralProposal(chart,moments,constraint_result,system.residual,
                          system.jacobian,direction,active_mask.detach().clone())


def evaluate_exact_m69_candidate(
    beta:Tensor,*,exact_material_oracle:Callable[[Tensor],Mapping[str,Any]],
    original_moments:Tensor,task:Mapping[str,Any],length_scale:float,
    event_tolerance:float=1e-9
)->ExactM69Evaluation:
    """Evaluate a candidate through an independent exact material integration.

    Callback must return {'moments': Tensor[10], 'connected': bool,
    'topology_valid': bool, 'event_valid': bool, 'source': nonempty str}.
    It must not call the M80 model or reuse its predicted moments.
    """
    raw=exact_material_oracle(beta.detach())
    required=("moments","connected","topology_valid","event_valid","source")
    missing=[k for k in required if k not in raw]
    if missing: raise ValueError("exact oracle result missing fields: "+",".join(missing))
    if not str(raw["source"]).strip(): raise ValueError("exact oracle source must be named")
    moments=torch.as_tensor(raw["moments"],device=original_moments.device,dtype=torch.float64)
    constraints=m69_constraint_slacks_torch(
        moments,original_moments,dict(task),length_scale=length_scale,
        event_tolerance=event_tolerance
    )
    return ExactM69Evaluation(moments,constraints,bool(raw["connected"]),
        bool(raw["topology_valid"]),bool(raw["event_valid"]),str(raw["source"]))


def exact_m69_merit(
    evaluation:ExactM69Evaluation,*,connectivity_penalty:float,feasibility_penalty:float
)->float:
    """Exact objective + L1 constraint penalty; invalid topology is never feasible."""
    if connectivity_penalty<=0 or feasibility_penalty<=0:
        raise ValueError("exact merit penalties must be positive")
    violation=torch.clamp(evaluation.constraints.g_le_zero,min=0).sum()
    merit=float(evaluation.moments[0].detach())+feasibility_penalty*float(violation.detach())
    if not evaluation.connected or not evaluation.topology_valid or not evaluation.event_valid:
        merit+=connectivity_penalty
    return merit


def accept_m80_proposal_with_exact_oracle(
    state:Any,proposal:NeuralProposal,*,decode:Callable[[Any,Tensor],Any],
    exact_merit:Callable[[Any],float],predicted_reduction:float,
    exact_solver_fallback:Callable[[Any],Any]|None=None
)->M80ExactAcceptance:
    """Use independent exact M69 merit for acceptance; call exact solver on rejection."""
    before=getattr(exact_merit,"evaluation_count",0)
    decision=exact_merit_acceptance(
        state,proposal.direction.step,decode=decode,exact_merit=exact_merit,
        predicted_reduction=predicted_reduction
    )
    fallback_result=None
    if not decision.accepted and exact_solver_fallback is not None:
        fallback_result=exact_solver_fallback(state)
    after=getattr(exact_merit,"evaluation_count",before)
    return M80ExactAcceptance(decision,max(0,after-before),fallback_result is not None,
                              fallback_result)
