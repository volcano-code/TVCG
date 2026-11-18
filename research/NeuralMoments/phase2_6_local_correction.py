"""Phase 2.6 Local Correction: bounded, mask-safe Quadric residual field.

Pure Quadric is the zero-correction baseline. The hybrid field adds compactly
supported local logit corrections only on explicitly editable samples.
Coordinates are assumed to be expressed in the registered normalized frame.
The returned field is a continuous material-retention relaxation; binary
thresholding/connectivity remain separate exact acceptance operations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import math
import torch
from torch import Tensor, nn

MOMENT_ORDER=("mass","x","y","z","xx","yy","zz","yz","xz","xy")
QUERY_ORDER=("1","x","y","z","xx","yy","zz","2yz","2xz","2xy")


def quadric_query(beta:Tensor,points:Tensor)->Tensor:
    if beta.shape!=(10,) or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError("beta must be [10] and points must be [N,3]")
    if not torch.isfinite(beta).all() or not torch.isfinite(points).all():
        raise ValueError("beta and points must be finite")
    x,y,z=points.unbind(dim=-1)
    basis=torch.stack((torch.ones_like(x),x,y,z,x*x,y*y,z*z,
                       2*y*z,2*x*z,2*x*y),dim=-1)
    return basis@beta


def wendland_c2_basis(points:Tensor,centers:Tensor,radius:float)->Tensor:
    if points.ndim!=2 or points.shape[1]!=3 or centers.ndim!=2 or centers.shape[1]!=3:
        raise ValueError("points and centers must have shapes [N,3] and [K,3]")
    if points.device!=centers.device or points.dtype!=centers.dtype:
        raise ValueError("points and centers must share dtype/device")
    if not math.isfinite(radius) or radius<=0:
        raise ValueError("support radius must be finite and positive")
    r=torch.cdist(points,centers)/radius
    t=torch.clamp(1.-r,min=0.)
    return torch.where(r<1.,t.pow(4)*(4.*r+1.),torch.zeros_like(r))


@dataclass(frozen=True)
class MaterialMasks:
    fixed:Tensor
    protected:Tensor
    editable:Tensor

    def validate(self,n:int,device)->None:
        masks=(self.fixed,self.protected,self.editable)
        if any(m.shape!=(n,) or m.dtype!=torch.bool or m.device!=device for m in masks):
            raise ValueError("fixed/protected/editable masks must be bool [N] on points device")
        if torch.any(self.fixed & self.protected) or torch.any(self.fixed & self.editable) or torch.any(self.protected & self.editable):
            raise ValueError("fixed/protected/editable masks must be pairwise disjoint")
        if not torch.all(self.fixed | self.protected | self.editable):
            raise ValueError("material masks must partition all integration samples")


class LocalCorrectionField(nn.Module):
    """Pure Quadric baseline plus bounded local correction in editable region."""
    def __init__(self,centers:Tensor,*,support_radius:float,
                 temperature:float=0.05,max_logit_correction:float=1.0):
        super().__init__()
        if centers.ndim!=2 or centers.shape[1]!=3 or centers.shape[0]<1:
            raise ValueError("centers must have shape [K,3], K>=1")
        if not torch.isfinite(centers).all():
            raise ValueError("centers must be finite")
        if not math.isfinite(temperature) or temperature<=0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(max_logit_correction) or max_logit_correction<=0:
            raise ValueError("max_logit_correction must be finite and positive")
        self.register_buffer("centers",centers.detach().clone())
        self.support_radius=float(support_radius)
        self.temperature=float(temperature)
        self.max_logit_correction=float(max_logit_correction)
        self.coefficients=nn.Parameter(torch.zeros(centers.shape[0],dtype=centers.dtype,device=centers.device))

    def local_logit_correction(self,points:Tensor,editable:Tensor)->Tensor:
        basis=wendland_c2_basis(points,self.centers,self.support_radius)
        raw=basis@self.coefficients
        bounded=self.max_logit_correction*torch.tanh(raw)
        return torch.where(editable,bounded,torch.zeros_like(bounded))

    def forward(self,beta:Tensor,points:Tensor,masks:MaterialMasks)->Tensor:
        masks.validate(len(points),points.device)
        if points.device!=self.centers.device or points.dtype!=self.centers.dtype:
            raise ValueError("points must match local basis dtype/device")
        if beta.device!=points.device or beta.dtype!=points.dtype:
            raise ValueError("beta must match points dtype/device")
        q=quadric_query(beta,points)
        baseline=torch.sigmoid(q/self.temperature)
        correction=self.local_logit_correction(points,masks.editable)
        corrected=torch.sigmoid(q/self.temperature+correction)
        retention=torch.where(masks.editable,corrected,baseline)
        retention=torch.where(masks.fixed|masks.protected,torch.ones_like(retention),retention)
        return retention

    def pure_quadric(self,beta:Tensor,points:Tensor,masks:MaterialMasks)->Tensor:
        masks.validate(len(points),points.device)
        q=quadric_query(beta,points)
        baseline=torch.sigmoid(q/self.temperature)
        return torch.where(masks.fixed|masks.protected,torch.ones_like(baseline),baseline)

    @torch.no_grad()
    def threshold(self,retention:Tensor,*,threshold:float=0.5)->Tensor:
        if not 0.<threshold<1.:
            raise ValueError("threshold must be in (0,1)")
        return retention>=threshold


def integrate_retained_moments(points:Tensor,quadrature_mass:Tensor,retention:Tensor)->Tensor:
    """Integrate [1,x,y,z,x²,y²,z²,yz,xz,xy] from weighted samples."""
    n=len(points)
    if points.shape!=(n,3) or quadrature_mass.shape!=(n,) or retention.shape!=(n,):
        raise ValueError("points, quadrature_mass and retention shapes mismatch")
    if not torch.isfinite(points).all() or not torch.isfinite(quadrature_mass).all() or not torch.isfinite(retention).all():
        raise ValueError("moment integration inputs must be finite")
    if torch.any(quadrature_mass<0) or torch.any(retention<0) or torch.any(retention>1):
        raise ValueError("quadrature mass must be nonnegative and retention in [0,1]")
    x,y,z=points.unbind(dim=-1)
    basis=torch.stack((torch.ones_like(x),x,y,z,x*x,y*y,z*z,y*z,x*z,x*y),dim=-1)
    return basis.T@(quadrature_mass*retention)


@dataclass(frozen=True)
class HybridAblationResult:
    pure_moments:Tensor
    hybrid_moments:Tensor
    pure_mass:Tensor
    hybrid_mass:Tensor
    correction_l2:Tensor
    correction_max_abs:Tensor


def evaluate_hybrid_ablation(field:LocalCorrectionField,beta:Tensor,points:Tensor,
                             quadrature_mass:Tensor,masks:MaterialMasks)->HybridAblationResult:
    pure=field.pure_quadric(beta,points,masks)
    hybrid=field(beta,points,masks)
    pm=integrate_retained_moments(points,quadrature_mass,pure)
    hm=integrate_retained_moments(points,quadrature_mass,hybrid)
    delta=hybrid-pure
    editable_delta=delta[masks.editable]
    return HybridAblationResult(pm,hm,pm[0],hm[0],
        torch.linalg.vector_norm(editable_delta),
        editable_delta.abs().max() if editable_delta.numel() else torch.zeros((),device=points.device,dtype=points.dtype))


def validate_local_correction_contract(contract:Mapping[str,bool])->None:
    required=("same_frame","same_moment_order","fixed_mask_immutable",
              "protected_mask_immutable","editable_mask_explicit",
              "bounded_correction","exact_binary_topology_check")
    missing=[k for k in required if k not in contract]
    if missing: raise ValueError("local correction contract missing: "+",".join(missing))
    failed=[k for k in required if not bool(contract[k])]
    if failed: raise ValueError("local correction contract not admitted: "+",".join(failed))
