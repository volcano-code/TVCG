"""Phase 2.5 synthetic end-to-end safeguarded optimization loop.

This is a solver integration benchmark on a convex quadratic program. The
"neural proposal" is an explicitly synthetic surrogate objective, NOT a
trained M80 model or physical oracle. Exact objective/constraints decide
acceptance and final feasibility. The implementation is intended to validate
active-set, KKT, trust-region and fallback control flow before physical use.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Callable
import math
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ActiveSetState:
    mask: Tensor
    multipliers: Tensor


@dataclass(frozen=True)
class QPStep:
    primal: Tensor
    multipliers: Tensor
    active_indices: tuple[int,...]
    predicted_reduction: float
    actual_reduction: float
    ratio: float
    accepted: bool
    used_exact_fallback: bool
    oracle_evaluations: int
    reason: str


@dataclass(frozen=True)
class QPSolveReport:
    method: str
    converged: bool
    x: tuple[float,...]
    objective: float
    max_constraint_violation: float
    stationarity_norm: float
    complementarity_norm: float
    iterations: int
    oracle_evaluations: int
    fallback_count: int
    history: tuple[dict,...]


def update_active_set(g:Tensor,mask:Tensor,multipliers:Tensor,*,
                      primal_tolerance:float=1e-7,dual_tolerance:float=1e-8):
    """Update g<=0 active set; release active constraints with negative lambda."""
    if g.ndim!=1 or mask.shape!=g.shape:
        raise ValueError("constraint and active mask dimensions differ")
    active_ids=torch.nonzero(mask,as_tuple=False).flatten()
    if multipliers.shape!=(len(active_ids),):
        raise ValueError("active multiplier shape mismatch")
    new=mask.clone()
    for local,global_id in enumerate(active_ids.tolist()):
        if float(multipliers[local].detach()) < -dual_tolerance:
            new[global_id]=False
    new |= g >= -primal_tolerance
    ids=torch.nonzero(new,as_tuple=False).flatten()
    # Preserve multipliers by global constraint index; newly active starts at 0.
    old={int(i):multipliers[k] for k,i in enumerate(active_ids.tolist())}
    lam=torch.stack([old.get(int(i),torch.zeros((),device=g.device,dtype=g.dtype)) for i in ids.tolist()]) if len(ids) else torch.empty(0,device=g.device,dtype=g.dtype)
    return ActiveSetState(new,lam)


def kkt_direction(objective_fn:Callable[[Tensor],Tensor],constraints_fn:Callable[[Tensor],Tensor],
                  x:Tensor,active:Tensor,multipliers:Tensor,*,trust_radius:float,
                  regularization:float=1e-8):
    """Autodiff fixed-active-set Newton direction for min f, g<=0, lambda>=0."""
    x0=x.detach().clone().requires_grad_(True)
    f=objective_fn(x0)
    g=constraints_fn(x0)
    ids=torch.nonzero(active,as_tuple=False).flatten()
    if multipliers.shape!=(len(ids),): raise ValueError("active multipliers mismatch")
    grad=torch.autograd.grad(f,x0,create_graph=True)[0]
    if len(ids):
        ga=g[ids]
        jacg=torch.autograd.functional.jacobian(lambda v:constraints_fn(v)[ids],x0)
        lag=f+(multipliers.detach()*ga).sum()
        H=torch.autograd.functional.hessian(lambda v:objective_fn(v)+(multipliers.detach()*constraints_fn(v)[ids]).sum(),x0)
        H=.5*(H+H.T)
        n=x.numel();m=len(ids)
        K=torch.cat((torch.cat((H+regularization*torch.eye(n,dtype=x.dtype,device=x.device),jacg.T),1),
                     torch.cat((jacg,torch.zeros((m,m),dtype=x.dtype,device=x.device)),1)),0)
        F=torch.cat((grad+jacg.T@multipliers.detach(),ga))
        dz=torch.linalg.solve(K,-F)
        dx=dz[:n];dl=dz[n:]
    else:
        H=torch.autograd.functional.hessian(objective_fn,x0)
        H=.5*(H+H.T)+regularization*torch.eye(x.numel(),dtype=x.dtype,device=x.device)
        dx=torch.linalg.solve(H,-grad)
        dl=torch.empty(0,dtype=x.dtype,device=x.device)
    norm=torch.linalg.vector_norm(dx)
    if float(norm.detach())>trust_radius:
        scale=trust_radius/float(norm.detach())
        dx=dx*scale
        dl=dl*scale
    return dx,dl,ids


def qp_merit(objective:float,constraints:Tensor,penalty:float)->float:
    violation=torch.clamp(constraints,min=0)
    return float(objective+penalty*torch.dot(violation,violation))


def _one_step(x,model_objective,exact_objective,constraints_fn,active_state,*,
              trust_radius,penalty,minimum_ratio,scales,oracle_counter):
    g=constraints_fn(x).detach()
    state=update_active_set(g,active_state.mask,active_state.multipliers)
    model_fn=model_objective
    dx,dl,ids=kkt_direction(model_fn,constraints_fn,x,state.mask,state.multipliers,
                            trust_radius=trust_radius)
    model_base=qp_merit(float(model_fn(x).detach()),g,penalty)
    exact_base=qp_merit(float(exact_objective(x).detach()),g,penalty)
    oracle_counter[0]+=1
    predicted_best=0.0
    for scale in scales:
        trial=x+float(scale)*dx
        tg=constraints_fn(trial).detach()
        model_trial=qp_merit(float(model_fn(trial).detach()),tg,penalty)
        exact_trial=qp_merit(float(exact_objective(trial).detach()),tg,penalty)
        oracle_counter[0]+=1
        predicted=model_base-model_trial
        actual=exact_base-exact_trial
        ratio=actual/predicted if predicted>0 else float("-inf")
        if predicted>0 and actual>0 and ratio>=minimum_ratio:
            new_lam=torch.clamp(state.multipliers+float(scale)*dl,min=0)
            new_mask=state.mask.clone()
            next_state=update_active_set(tg,new_mask,new_lam)
            return trial,next_state,QPStep(tuple(float(v) for v in dx.detach()),
                tuple(float(v) for v in new_lam.detach()),tuple(int(i) for i in ids.tolist()),
                predicted,actual,ratio,True,False,oracle_counter[0],"exact_ratio_accepted")
        predicted_best=max(predicted_best,predicted)
    # Explicit exact Newton fallback uses the exact objective and same active set.
    dx_e,dl_e,ids_e=kkt_direction(exact_objective,constraints_fn,x,state.mask,
                                  state.multipliers,trust_radius=trust_radius)
    for scale in scales:
        trial=x+float(scale)*dx_e
        tg=constraints_fn(trial).detach()
        exact_trial=qp_merit(float(exact_objective(trial).detach()),tg,penalty)
        oracle_counter[0]+=1
        actual=exact_base-exact_trial
        if actual>0:
            new_lam=torch.clamp(state.multipliers+float(scale)*dl_e,min=0)
            next_state=update_active_set(tg,state.mask,new_lam)
            return trial,next_state,QPStep(tuple(float(v) for v in dx_e.detach()),
                tuple(float(v) for v in new_lam.detach()),tuple(int(i) for i in ids_e.tolist()),
                predicted_best,actual,float("nan"),True,True,oracle_counter[0],"exact_newton_fallback")
    return x,state,QPStep(tuple(float(v) for v in dx.detach()),tuple(float(v) for v in state.multipliers.detach()),
        tuple(int(i) for i in ids.tolist()),predicted_best,0.,float("-inf"),False,True,
        oracle_counter[0],"model_and_exact_trials_rejected")


def solve_synthetic_qp(*,method:str,seed:int=0,max_iterations:int=30,
                       tolerance:float=1e-8,trust_radius:float=.5,penalty:float=100.,
                       exact_fallback:bool=True)->QPSolveReport:
    """Solve a fixed 2D convex QP with exact or synthetic-surrogate proposals."""
    if method not in ("exact_newton","surrogate_newton"):
        raise ValueError("method must be exact_newton or surrogate_newton")
    dtype=torch.float64
    H=torch.tensor([[2.,.2],[.2,1.]],dtype=dtype)
    target=torch.tensor([1.,1.],dtype=dtype)
    # A deterministic small surrogate bias, varied by seed, represents model error.
    rng=np.random.default_rng(seed)
    bias=torch.tensor(rng.normal(0.,.035,size=2),dtype=dtype)
    H_model=H+torch.tensor([[.015,0.],[0.,-.01]],dtype=dtype)
    target_model=target+bias
    def exact_obj(x):
        d=x-target
        return .5*d@H@d
    def model_obj(x):
        if method=="exact_newton": return exact_obj(x)
        d=x-target_model
        return .5*d@H_model@d
    def constraints(x):
        return torch.stack((x[0]+x[1]-.2,))
    x=torch.zeros(2,dtype=dtype)
    active=ActiveSetState(torch.tensor([False]),torch.empty(0,dtype=dtype))
    oracle=[0];fallbacks=0;history=[]
    scales=(1.,.5,.25,.125,.0625,.03125)
    for iteration in range(max_iterations):
        x_new,state,step=_one_step(x,model_obj,exact_obj,constraints,active,
            trust_radius=trust_radius,penalty=penalty,minimum_ratio=.05,
            scales=scales,oracle_counter=oracle)
        if step.used_exact_fallback: fallbacks+=1
        x=x_new;active=state
        g=constraints(x).detach()
        grad=torch.autograd.grad(exact_obj(x.requires_grad_(True)),x,create_graph=False)[0].detach()
        ids=torch.nonzero(active.mask,as_tuple=False).flatten()
        if len(ids):
            # KKT stationarity and complementarity use nonnegative multipliers.
            lam=active.multipliers
            jac=torch.autograd.functional.jacobian(constraints,x)
            stationarity=grad+jac[ids].T@lam
            comp=torch.max(torch.abs(lam*constraints(x)[ids]))
        else:
            stationarity=grad
            comp=torch.zeros((),dtype=dtype)
        violation=torch.clamp(g,min=0).max()
        history.append({"iteration":iteration+1,"objective":float(exact_obj(x).detach()),
            "max_violation":float(violation),"stationarity":float(torch.linalg.vector_norm(stationarity)),
            "complementarity":float(comp),"accepted":step.accepted,
            "fallback":step.used_exact_fallback,"ratio":step.ratio})
        if (float(violation)<=tolerance and float(torch.linalg.vector_norm(stationarity))<=1e-6
            and float(comp)<=1e-6):
            break
        if not step.accepted:
            break
    g=constraints(x).detach()
    grad=torch.autograd.grad(exact_obj(x.requires_grad_(True)),x,create_graph=False)[0].detach()
    ids=torch.nonzero(active.mask,as_tuple=False).flatten()
    if len(ids):
        jac=torch.autograd.functional.jacobian(constraints,x)
        stationarity=grad+jac[ids].T@active.multipliers
        comp=torch.max(torch.abs(active.multipliers*constraints(x)[ids]))
    else:
        stationarity=grad;comp=torch.zeros((),dtype=dtype)
    return QPSolveReport(method,
        bool(float(torch.clamp(g,min=0).max())<=tolerance and
             float(torch.linalg.vector_norm(stationarity))<=1e-6 and float(comp)<=1e-6),
        tuple(float(v) for v in x.detach()),float(exact_obj(x).detach()),
        float(torch.clamp(g,min=0).max()),float(torch.linalg.vector_norm(stationarity)),
        float(comp),len(history),oracle[0],fallbacks,tuple(history))


def compare_exact_and_surrogate(*,seeds=(2501,2502,2503),max_iterations=30):
    rows=[]
    for seed in seeds:
        exact=solve_synthetic_qp(method="exact_newton",seed=seed,max_iterations=max_iterations)
        surrogate=solve_synthetic_qp(method="surrogate_newton",seed=seed,max_iterations=max_iterations)
        rows.extend((asdict(exact),asdict(surrogate)))
    return {
      "protocol":{"kind":"synthetic_convex_qp_solver_integration_not_physical_oracle",
        "seeds":list(seeds),"same_initial_state":[0.,0.],"same_exact_objective":True,
        "same_constraints":True,"same_tolerance":1e-8,"max_iterations":max_iterations,
        "methods":["exact_newton","surrogate_newton"],"exact_merit_penalty":100.},
      "results":rows
    }
