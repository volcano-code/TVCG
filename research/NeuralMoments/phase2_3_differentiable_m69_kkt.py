"""Phase 2.3 differentiable M69 moment constraints and fixed-active-set KKT.

The ten M69 physical inequalities are represented as g(x)<=0. This module
uses exact unsmoothed formulas. Derivatives are refused at eigenvalue
multiplicity, zero-norm alignment metrics, zero mass, or active-set events.
It does not smooth discrete connectivity/topology into a scalar constraint.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import math
import torch
from torch import Tensor

M69_CONSTRAINT_NAMES=(
 "positive_mass","no_added_mass","minimum_removal","polar_retention",
 "no_added_polar_inertia","axis_alignment_com","axis_alignment_inertia",
 "principal_dominance","centroid_height","physical_second_moments"
)

@dataclass(frozen=True)
class M69TorchConstraintResult:
    slack: Tensor
    g_le_zero: Tensor
    derivative_valid: bool
    invalid_reasons: tuple[str,...]
    active_set_event: bool
    constraint_names: tuple[str,...]

@dataclass(frozen=True)
class KKTSystem:
    unknowns: Tensor
    residual: Tensor
    jacobian: Tensor
    stationarity: Tensor
    equality: Tensor
    active_inequality: Tensor
    inactive_primal_violation: Tensor
    active_dual_violation: Tensor
    derivative_valid: bool
    invalid_reasons: tuple[str,...]


def _as_moment(x:Tensor,name:str)->Tensor:
    if not isinstance(x,Tensor) or x.shape!=(10,) or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be a finite torch tensor [10]")
    if float(x[0].detach())<=0:
        raise ValueError(f"{name} mass must be positive")
    return x

def _task_tensor(task,key,device,dtype):
    value=torch.as_tensor(task[key],device=device,dtype=dtype)
    if key in ("axis","axis_origin","contact"):
        if value.shape!=(3,) or not torch.isfinite(value).all():
            raise ValueError(f"task {key} must be finite [3]")
    return value

def m69_constraint_slacks_torch(
    moments:Tensor,original_moments:Tensor,task:dict,*,length_scale:float,
    event_tolerance:float=1e-9,active_tolerance:float=1e-8
)->M69TorchConstraintResult:
    """Return exact M69 slacks (feasible iff every slack >= 0).

    Derivative validity is conservative. The numerical values are still
    returned at events, but callers must not use their Jacobian/Hessian there.
    """
    if task.get("schema")!="m69.fixed-axis-task.v1":
        raise ValueError("unsupported M69 TaskSpec")
    if task.get("objective")!="minimum_retained_mass" or task.get("material_policy")!="single_density_removal_only":
        raise ValueError("unsupported M69 physical profile")
    m=_as_moment(moments,"moments"); ref=_as_moment(original_moments,"original_moments")
    if m.device!=ref.device or m.dtype!=ref.dtype:
        raise ValueError("candidate and original moments must share device/dtype")
    L=float(length_scale)
    if not math.isfinite(L) or L<=0: raise ValueError("length_scale must be positive")
    axis=_task_tensor(task,"axis",m.device,m.dtype)
    axis=axis/torch.linalg.vector_norm(axis)
    origin=_task_tensor(task,"axis_origin",m.device,m.dtype)
    contact=_task_tensor(task,"contact",m.device,m.dtype)
    density=float(task["density"])
    if not math.isfinite(density) or density<=0: raise ValueError("invalid task density")
    axis_offset=contact-origin
    off_axis=axis_offset-(axis_offset@axis)*axis
    if float(torch.linalg.vector_norm(off_axis).detach())>1e-10*max(1.,float(torch.linalg.vector_norm(axis_offset).detach())):
        raise ValueError("contact is not on the fixed axis")

    def second(v):
        return torch.stack((
            torch.stack((v[4],v[9],v[8])),
            torch.stack((v[9],v[5],v[7])),
            torch.stack((v[8],v[7],v[6]))
        ))
    center=m[1:4]/m[0]; center_ref=ref[1:4]/ref[0]
    S=second(m); Sref=second(ref)
    cov=S-m[0]*torch.outer(center,center)
    cov_ref=Sref-ref[0]*torch.outer(center_ref,center_ref)
    Ic=torch.trace(cov)*torch.eye(3,device=m.device,dtype=m.dtype)-cov
    shifted=S-torch.outer(origin,m[1:4])-torch.outer(m[1:4],origin)+m[0]*torch.outer(origin,origin)
    shifted_ref=Sref-torch.outer(origin,ref[1:4])-torch.outer(ref[1:4],origin)+ref[0]*torch.outer(origin,origin)
    polar=torch.trace(shifted)-axis@shifted@axis
    polar_ref=torch.trace(shifted_ref)-axis@shifted_ref@axis
    if float(polar_ref.detach())<=0: raise ValueError("original fixed-axis inertia must be positive")
    mass_fraction=m[0]/ref[0]; polar_ratio=polar/polar_ref
    displacement=center-origin
    transverse=displacement-(displacement@axis)*axis
    com_norm=torch.linalg.vector_norm(transverse)
    axial=axis@Ic@axis
    align_vec=Ic@axis-axial*axis
    align_norm=torch.linalg.vector_norm(align_vec)/polar_ref
    seed=torch.eye(3,device=m.device,dtype=m.dtype)[int(torch.argmin(torch.abs(axis)).detach())]
    u=torch.linalg.cross(axis,seed);u=u/torch.linalg.vector_norm(u)
    v=torch.linalg.cross(axis,u)
    plane=torch.stack((u,v),dim=1)
    transverse_I=plane.T@Ic@plane
    eig2=torch.linalg.eigvalsh(transverse_I)
    eig_cov=torch.linalg.eigvalsh(cov)
    principal_gap=(axial-eig2[-1])/polar_ref
    centroid_height=(center-contact)@axis
    centroid_height_ref=(center_ref-contact)@axis
    psd=eig_cov[0]/(ref[0]*L*L)
    tol_mass=float(task["mass_tolerance"]);tol_I=float(task["inertia_tolerance"])
    slack=torch.stack((
        m[0]/ref[0],
        1.+tol_mass-mass_fraction,
        1.-mass_fraction-float(task["min_removal_fraction"])+tol_mass,
        polar_ratio-float(task["polar_retention"])+tol_I,
        1.+tol_I-polar_ratio,
        float(task["com_tolerance"])-com_norm/L,
        float(task["alignment_tolerance"])-align_norm,
        principal_gap-float(task["principal_gap_min"]),
        float(task["max_centroid_rise"])-(centroid_height-centroid_height_ref)/L,
        psd+float(task["moment_psd_tolerance"])
    ))
    reasons=[]
    if float(m[0].detach())<=event_tolerance: reasons.append("zero_mass")
    if float(com_norm.detach())<=event_tolerance: reasons.append("zero_norm_com_alignment")
    if float(torch.linalg.vector_norm(align_vec).detach())<=event_tolerance*max(1.,float(polar_ref.detach())):
        reasons.append("zero_norm_inertia_alignment")
    if float((eig2[-1]-eig2[-2]).detach())<=event_tolerance:
        reasons.append("transverse_eigenvalue_multiplicity")
    if float((eig_cov[1]-eig_cov[0]).detach())<=event_tolerance:
        reasons.append("covariance_min_eigenvalue_multiplicity")
    if not torch.isfinite(slack).all(): reasons.append("nonfinite_constraint")
    active_event=bool(torch.any(torch.abs(slack.detach())<=active_tolerance))
    if active_event: reasons.append("active_set_switch_or_boundary")
    return M69TorchConstraintResult(slack=slack,g_le_zero=-slack,
        derivative_valid=len(reasons)==0,invalid_reasons=tuple(reasons),
        active_set_event=active_event,constraint_names=M69_CONSTRAINT_NAMES)


def fixed_active_set_kkt(
    x:Tensor,objective_fn:Callable[[Tensor],Tensor],
    equality_fn:Callable[[Tensor],Tensor],
    inequality_fn:Callable[[Tensor],Tensor],
    active_mask:Sequence[bool]|Tensor,*,
    equality_multipliers:Tensor|None=None,
    active_inequality_multipliers:Tensor|None=None,
    event_tolerance:float=1e-8
)->KKTSystem:
    """Build F_KKT and dF/dz for min f, h=0, g<=0, L=f+nu*h+lambda*g_A.

    z=(x,nu,lambda_A); F=[grad_x L,h,g_A]. The active set is held fixed.
    x should be local feasible coordinates (e.g. 9D S9 chart), not redundant
    unconstrained ten-dimensional normalized quadric coefficients.
    """
    if x.ndim!=1 or not torch.isfinite(x).all(): raise ValueError("x must be finite vector")
    x0=x.detach().clone().requires_grad_(True)
    f=objective_fn(x0)
    if f.ndim!=0 or not torch.isfinite(f): raise ValueError("objective_fn must return finite scalar")
    h=equality_fn(x0)
    g=inequality_fn(x0)
    if h.ndim!=1 or g.ndim!=1 or not torch.isfinite(h).all() or not torch.isfinite(g).all():
        raise ValueError("constraint functions must return finite vectors")
    mask=torch.as_tensor(active_mask,dtype=torch.bool,device=x.device)
    if mask.shape!=g.shape: raise ValueError("active_mask shape mismatch")
    na=int(mask.sum()); nh=h.numel()
    nu=(torch.zeros(nh,dtype=x.dtype,device=x.device) if equality_multipliers is None
        else equality_multipliers.to(device=x.device,dtype=x.dtype))
    lam=(torch.zeros(na,dtype=x.dtype,device=x.device) if active_inequality_multipliers is None
         else active_inequality_multipliers.to(device=x.device,dtype=x.dtype))
    if nu.shape!=(nh,) or lam.shape!=(na,): raise ValueError("KKT multiplier shape mismatch")
    if not torch.isfinite(nu).all() or not torch.isfinite(lam).all(): raise ValueError("nonfinite multipliers")
    if (~mask).any() and torch.any(torch.abs(g.detach()[~mask])<=event_tolerance):
        raise ValueError("inactive constraint is at active-set switching boundary")
    if not f.requires_grad: raise ValueError("objective must depend differentiably on x")
    grad_f=torch.autograd.grad(f,x0,create_graph=True,retain_graph=True)[0]
    grad_l=grad_f
    if nh:
        for i in range(nh):
            grad_h=torch.autograd.grad(h[i],x0,create_graph=True,retain_graph=True,allow_unused=False)[0]
            grad_l=grad_l+nu[i]*grad_h
    ga=g[mask]
    for i in range(na):
        grad_g=torch.autograd.grad(ga[i],x0,create_graph=True,retain_graph=True,allow_unused=False)[0]
        grad_l=grad_l+lam[i]*grad_g
    residual=torch.cat((grad_l,h,ga))
    z=torch.cat((x0,nu,lam))
    def residual_from_z(zz):
        xx=zz[:x.numel()]
        nnu=zz[x.numel():x.numel()+nh]
        nlam=zz[x.numel()+nh:]
        ff=objective_fn(xx)
        hh=equality_fn(xx)
        gg=inequality_fn(xx)
        grad=torch.autograd.grad(ff,xx,create_graph=True,retain_graph=True)[0]
        for j in range(nh):
            gj=torch.autograd.grad(hh[j],xx,create_graph=True,retain_graph=True)[0]
            grad=grad+nnu[j]*gj
        active_g=gg[mask]
        for j in range(na):
            gj=torch.autograd.grad(active_g[j],xx,create_graph=True,retain_graph=True)[0]
            grad=grad+nlam[j]*gj
        return torch.cat((grad,hh,active_g))
    jac=torch.autograd.functional.jacobian(residual_from_z,z,create_graph=True,vectorize=False)
    inactive_violation=(torch.clamp(g[~mask],min=0).max() if (~mask).any()
                        else torch.zeros((),device=x.device,dtype=x.dtype))
    dual_violation=torch.clamp(-lam,min=0).max() if na else torch.zeros((),device=x.device,dtype=x.dtype)
    return KKTSystem(z,residual,jac,grad_l,h,ga,inactive_violation,dual_violation,
        bool(torch.isfinite(jac).all()),tuple() if torch.isfinite(jac).all() else ("nonfinite_kkt_jacobian",))


def m69_kkt_from_moment_operator(
    x:Tensor,moment_fn:Callable[[Tensor],Tensor],original_moments:Tensor,task:dict,
    *,length_scale:float,active_mask:Sequence[bool]|Tensor,
    event_tolerance:float=1e-8
)->KKTSystem:
    """Compose a differentiable moment operator with M69's exact constraints.

    Objective is retained mass. All ten M69 inequalities are represented as
    g<=0. This wrapper refuses derivative-invalid M69 branches before KKT.
    """
    def evaluate(xx):
        m=moment_fn(xx)
        result=m69_constraint_slacks_torch(m,original_moments,task,
            length_scale=length_scale,event_tolerance=event_tolerance,
            active_tolerance=event_tolerance)
        return m,result
    def obj(xx): return evaluate(xx)[0][0]
    def ineq(xx):
        _,result=evaluate(xx)
        if not result.derivative_valid:
            raise ValueError("M69 derivative invalid: "+",".join(result.invalid_reasons))
        return result.g_le_zero
    return fixed_active_set_kkt(x,obj,lambda xx:torch.empty(0,dtype=xx.dtype,device=xx.device),
        ineq,active_mask,event_tolerance=event_tolerance)
