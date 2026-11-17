"""Phase 2.4: S9 material chart and safeguarded differentiable KKT Newton.

This module implements a local projective S9 chart, a symmetric-KKT Newton
direction with primal-block regularization (never a blanket diagonal shift on
the saddle-point matrix), trust-radius control, and exact-merit acceptance /
oracle fallback hooks. It does not differentiate through discrete active-set
changes or claim an end-to-end physical speedup.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import math
import torch
from torch import Tensor
from research.NeuralMoments.phase2_3_differentiable_m69_kkt import KKTSystem, m69_kkt_from_moment_operator


@dataclass(frozen=True)
class S9Chart:
    anchor: Tensor
    basis: Tensor

    @property
    def dimension(self)->int:
        return 9

    def decode(self,local:Tensor)->Tensor:
        if local.shape!=(9,) or not torch.isfinite(local).all():
            raise ValueError("S9 local coordinate must be finite [9]")
        raw=self.anchor+self.basis@local
        norm=torch.linalg.vector_norm(raw)
        if float(norm.detach())<=1e-14:
            raise ValueError("S9 chart decode reached zero vector")
        beta=raw/norm
        if float((beta@self.anchor).detach())<=0:
            raise ValueError("S9 chart crossed projective sign boundary")
        return beta

    def encode_tangent(self,beta:Tensor)->Tensor:
        if beta.shape!=(10,) or not torch.isfinite(beta).all():
            raise ValueError("beta must be finite [10]")
        u=beta/torch.linalg.vector_norm(beta)
        if float((u@self.anchor).detach())<=0:
            raise ValueError("beta lies outside anchor's local projective chart")
        return self.basis.T@(u-self.anchor)


def make_s9_chart(beta:Tensor,*,orthogonality_tolerance:float=1e-10)->S9Chart:
    """Construct a deterministic orthonormal 10x9 tangent basis at beta."""
    if beta.shape!=(10,) or not torch.isfinite(beta).all():
        raise ValueError("beta must be finite [10]")
    anchor=beta/torch.linalg.vector_norm(beta)
    # QR of projected coordinate axes. Choose the least-aligned coordinate
    # as the omitted seed so the projected matrix remains well-conditioned.
    omitted=int(torch.argmin(torch.abs(anchor)).detach())
    seed=torch.eye(10,device=beta.device,dtype=beta.dtype)
    seed=seed[:,[i for i in range(10) if i!=omitted]]
    projected=seed-anchor[:,None]*(anchor@seed)[None,:]
    basis,_=torch.linalg.qr(projected,mode="reduced")
    if basis.shape!=(10,9): raise RuntimeError("failed to construct S9 tangent basis")
    err=torch.linalg.norm(basis.T@basis-torch.eye(9,device=beta.device,dtype=beta.dtype))
    radial=torch.linalg.norm(basis.T@anchor)
    if float(err.detach())>orthogonality_tolerance or float(radial.detach())>orthogonality_tolerance:
        raise RuntimeError("S9 tangent basis failed orthonormality audit")
    return S9Chart(anchor.detach(),basis.detach())


@dataclass(frozen=True)
class NewtonDirection:
    step:Tensor
    primal_step_norm:float
    multiplier_step_norm:float
    primal_regularization:float
    linear_residual_norm:float
    kkt_symmetry_error:float
    trust_radius_limited:bool


def safeguarded_kkt_newton_direction(
    residual:Tensor,jacobian:Tensor,*,primal_dimension:int,
    trust_radius:float,multiplier_radius:float=1.0,
    initial_regularization:float=1e-8,max_regularization:float=1e6,
    symmetry_tolerance:float=1e-7
)->NewtonDirection:
    """Solve J dz=-F; regularize only primal Hessian block of symmetric KKT.

    Assumes ordering z=(primal variables, equality multipliers, active
    inequality multipliers). KKT Jacobian must be symmetric within tolerance.
    """
    if residual.ndim!=1 or jacobian.shape!=(residual.numel(),residual.numel()):
        raise ValueError("KKT residual/Jacobian dimensions mismatch")
    n=residual.numel();p=int(primal_dimension)
    if p<=0 or p>=n: raise ValueError("primal_dimension must be in (0,n)")
    if not torch.isfinite(residual).all() or not torch.isfinite(jacobian).all():
        raise ValueError("KKT system must be finite")
    for name,value in (("trust_radius",trust_radius),("multiplier_radius",multiplier_radius),
                       ("initial_regularization",initial_regularization),("max_regularization",max_regularization)):
        if not math.isfinite(float(value)) or float(value)<=0:
            raise ValueError(f"{name} must be finite and positive")
    symerr=float(torch.linalg.norm(jacobian-jacobian.T).detach())
    scale=max(1.0,float(torch.linalg.norm(jacobian).detach()))
    if symerr>symmetry_tolerance*scale:
        raise ValueError(f"KKT Jacobian is not symmetric: relative error={symerr/scale:.3e}")
    H=0.5*(jacobian[:p,:p]+jacobian[:p,:p].T)
    A=jacobian[p:,:p]
    # Preserve the saddle-point zero block and off-diagonal constraint coupling.
    zero=jacobian[p:,p:]
    rhs=-residual
    lam=max(initial_regularization,1e-14)
    solution=None
    while lam<=max_regularization:
        top=torch.cat((H+lam*torch.eye(p,device=H.device,dtype=H.dtype),
                       jacobian[:p,p:]),dim=1)
        bottom=torch.cat((A,zero),dim=1)
        regularized=torch.cat((top,bottom),dim=0)
        try:
            candidate=torch.linalg.solve(regularized,rhs)
            if torch.isfinite(candidate).all():
                solution=candidate
                break
        except RuntimeError:
            pass
        lam*=10.0
    if solution is None:
        raise RuntimeError("regularized KKT system could not be solved")
    primal=solution[:p]
    multipliers=solution[p:]
    pn=float(torch.linalg.vector_norm(primal).detach())
    mn=float(torch.linalg.vector_norm(multipliers).detach())
    limited=False
    if pn>trust_radius:
        solution=solution*(trust_radius/pn)
        limited=True
    if mn>multiplier_radius:
        solution[p:]=solution[p:]*(multiplier_radius/mn)
        limited=True
    lin=float(torch.linalg.norm(jacobian@solution+residual).detach())
    return NewtonDirection(solution,float(torch.linalg.vector_norm(solution[:p]).detach()),
        float(torch.linalg.vector_norm(solution[p:]).detach()),lam,lin,symerr,limited)


@dataclass(frozen=True)
class StepAcceptance:
    accepted:bool
    trial_scale:float
    predicted_reduction:float
    actual_reduction:float
    ratio:float
    reason:str
    accepted_state:Any


def exact_merit_acceptance(
    state:Any,step:Tensor,*,decode:Callable[[Any,Tensor],Any],
    exact_merit:Callable[[Any],float],predicted_reduction:float,
    scales=(1.0,.5,.25,.125,.0625),minimum_ratio:float=0.05
)->StepAcceptance:
    """Backtrack using exact-oracle merit; reject if no scale passes.

    The model may propose a direction, but acceptance is decided by exact_merit.
    The caller owns the exact residual/feasibility definition and can use the
    M69 oracle without differentiating through it.
    """
    base=float(exact_merit(state))
    if not math.isfinite(base): raise ValueError("base exact merit must be finite")
    if not math.isfinite(predicted_reduction) or predicted_reduction<=0:
        raise ValueError("predicted_reduction must be positive and finite")
    for scale in scales:
        scale=float(scale)
        if not 0<scale<=1: raise ValueError("backtracking scales must be in (0,1]")
        trial=decode(state,step*scale)
        value=float(exact_merit(trial))
        actual=base-value
        predicted=predicted_reduction*scale
        ratio=actual/predicted
        if math.isfinite(value) and actual>0 and ratio>=minimum_ratio:
            return StepAcceptance(True,scale,predicted,actual,ratio,"exact_merit_decreased",trial)
    return StepAcceptance(False,0.0,0.0,0.0,float("-inf"),
        "all_trials_rejected_exact_oracle_fallback_required",state)


@dataclass(frozen=True)
class NewtonIterationResult:
    state:Any
    accepted:bool
    used_fallback:bool
    acceptance:StepAcceptance
    fallback_result:Any


def safeguarded_newton_iteration(
    state:Any,direction:NewtonDirection,*,decode:Callable[[Any,Tensor],Any],
    exact_merit:Callable[[Any],float],predicted_reduction:float,
    exact_fallback:Callable[[Any],Any]|None=None
)->NewtonIterationResult:
    decision=exact_merit_acceptance(state,direction.step,decode=decode,
        exact_merit=exact_merit,predicted_reduction=predicted_reduction)
    if decision.accepted:
        return NewtonIterationResult(decision.accepted_state,True,False,decision,None)
    if exact_fallback is None:
        return NewtonIterationResult(state,False,False,decision,None)
    fallback=exact_fallback(state)
    return NewtonIterationResult(fallback,False,True,decision,fallback)


def build_s9_m69_kkt_system(
    local:Tensor,beta_anchor:Tensor,beta_to_moments:Callable[[Tensor],Tensor],
    original_moments:Tensor,task:dict,*,length_scale:float,
    active_mask, event_tolerance:float=1e-8
)->tuple[S9Chart,KKTSystem]:
    """Compose the S9 material chart, Moment operator and M69 KKT system.

    beta_to_moments receives normalized ten-dimensional query coefficients.
    Its derivative is propagated through chart.decode and into the KKT
    residual/Jacobian. The caller must separately certify the Phase 2.1
    query-to-material contract before interpreting this as a physical design
    Jacobian.
    """
    chart=make_s9_chart(beta_anchor)
    if local.shape!=(9,):
        raise ValueError("S9 local material coordinate must have shape [9]")
    def moment_fn(xi):
        beta=chart.decode(xi)
        return beta_to_moments(beta)
    system=m69_kkt_from_moment_operator(
        local,moment_fn,original_moments,task,length_scale=length_scale,
        active_mask=active_mask,event_tolerance=event_tolerance
    )
    return chart,system
