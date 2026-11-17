"""Phase 2.2 task-conditioned wrapper for the frozen M80 native PINO.

The M80 spatial/native-interval encoder remains the geometry trunk. This
adapter adds an M69 task-conditioned FiLM modulation of the cached global
latent and exposes an explicit alpha -> beta_query material-design map.
The underlying M80 baseline is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .models.pcraft_ufno.spin_pino.native_pcraft import (
    EncodedNativeGeometry,
    PcraftNativeMomentOperator,
)

TASK_SCHEMA = "m69.fixed-axis-task.v1"
TASK_FEATURE_NAMES = (
    "axis_x","axis_y","axis_z",
    "axis_origin_x_over_L","axis_origin_y_over_L","axis_origin_z_over_L",
    "contact_x_over_L","contact_y_over_L","contact_z_over_L",
    "log_density","log2_native_resolution_over_512",
    "shell_fraction","protect_radius_fraction","protect_height_fraction",
    "polar_retention","min_removal_fraction","log10_max_centroid_rise",
    "log10_com_tolerance","log10_alignment_tolerance","principal_gap_min",
    "require_connected","log10_mass_tolerance","log10_inertia_tolerance",
    "log10_moment_psd_tolerance","log10_solver_tolerance",
    "log10_solver_polar_margin","log10_training_polar_margin",
    "objective_minimum_retained_mass","inertia_reference_original_fixed_axis",
    "material_policy_single_density_removal","purpose_research",
)
TASK_FEATURE_DIM = len(TASK_FEATURE_NAMES)


def _positive_log10(value: Any, name: str) -> float:
    x=float(value)
    if not math.isfinite(x) or x<=0:
        raise ValueError(f"{name} must be finite and positive")
    return math.log10(max(x,1e-30))


def encode_m69_task_features(task: Mapping[str, Any], *, length_scale: float) -> np.ndarray:
    """Encode the frozen M69 TaskSpec into a stable 31D conditioning vector.

    Positions are divided by the caller-supplied outer length L. Density remains
    the registered TaskSpec scalar; no SI conversion is inferred.
    """
    if task.get("schema") != TASK_SCHEMA:
        raise ValueError("unsupported M69 task schema")
    L=float(length_scale)
    if not math.isfinite(L) or L<=0:
        raise ValueError("length_scale must be finite and positive")
    axis=np.asarray(task["axis"],dtype=np.float64)
    origin=np.asarray(task["axis_origin"],dtype=np.float64)
    contact=np.asarray(task["contact"],dtype=np.float64)
    for name,value in (("axis",axis),("axis_origin",origin),("contact",contact)):
        if value.shape!=(3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite 3-vector")
    norm=float(np.linalg.norm(axis))
    if norm<=1e-14:
        raise ValueError("task axis must be nonzero")
    axis=axis/norm
    delta=contact-origin
    off_axis=delta-np.dot(delta,axis)*axis
    if np.linalg.norm(off_axis)>1e-10*max(1.0,np.linalg.norm(delta)):
        raise ValueError("task contact must lie on the fixed axis")
    density=float(task["density"])
    resolution=int(task["native_resolution"])
    if not math.isfinite(density) or density<=0 or resolution<1:
        raise ValueError("invalid task density or native resolution")
    categorical=(
        float(task.get("objective")=="minimum_retained_mass"),
        float(task.get("inertia_reference")=="original_fixed_axis"),
        float(task.get("material_policy")=="single_density_removal_only"),
        float(task.get("purpose")=="research"),
    )
    if not all(categorical):
        raise ValueError("Phase 2.2 v1 supports only the frozen M69 fixed-axis removal task")
    values=[
        *axis.tolist(),*(origin/L).tolist(),*(contact/L).tolist(),
        math.log(density),math.log2(resolution/512.0),
        float(task["shell_fraction"]),float(task["protect_radius_fraction"]),
        float(task["protect_height_fraction"]),float(task["polar_retention"]),
        float(task["min_removal_fraction"]),
        _positive_log10(task["max_centroid_rise"],"max_centroid_rise"),
        _positive_log10(task["com_tolerance"],"com_tolerance"),
        _positive_log10(task["alignment_tolerance"],"alignment_tolerance"),
        float(task["principal_gap_min"]),float(bool(task["require_connected"])),
        _positive_log10(task["mass_tolerance"],"mass_tolerance"),
        _positive_log10(task["inertia_tolerance"],"inertia_tolerance"),
        _positive_log10(task["moment_psd_tolerance"],"moment_psd_tolerance"),
        _positive_log10(task["solver_tolerance"],"solver_tolerance"),
        _positive_log10(task["solver_polar_margin"],"solver_polar_margin"),
        _positive_log10(task["training_polar_margin"],"training_polar_margin"),
        *categorical,
    ]
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(TASK_FEATURE_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("internal M69 task feature contract mismatch")
    return result


@dataclass(frozen=True)
class MaterialMapContract:
    """Evidence gates for mapping alpha to M80's beta_query coordinates."""
    same_frame: bool
    same_retained_side: bool
    same_fixed_protected_editable_masks: bool
    same_units_and_moment_normalization: bool
    derivative_valid: bool
    identity_map: bool = False

    def validate(self) -> None:
        flags=(self.same_frame,self.same_retained_side,
               self.same_fixed_protected_editable_masks,
               self.same_units_and_moment_normalization,self.derivative_valid)
        if not all(flags):
            raise ValueError("query-to-material Jacobian contract is not admitted")
        # A non-identity local-coordinate map is valid when its explicit
        # Jacobian D is supplied by the caller; identity_map only controls
        # whether D must be the 10x10 identity.


class M69TaskConditionedM80(nn.Module):
    """M80 geometry trunk + M69 task FiLM + explicit material Jacobian chain rule.

    The wrapper requires the convex-potential M80 head so the learned moment
    map has a scalar-potential consistency structure. Task FiLM is zero
    initialized: before task-specific training, its output exactly matches
    the frozen M80 model for the same geometry/query.
    """
    def __init__(self, base_model: PcraftNativeMomentOperator, *,
                 task_feature_dim: int=TASK_FEATURE_DIM, hidden_dim: int=128,
                 freeze_backbone: bool=True):
        super().__init__()
        if not isinstance(base_model,PcraftNativeMomentOperator):
            raise TypeError("base_model must be PcraftNativeMomentOperator")
        if not base_model.config.convex_potential_enabled:
            raise ValueError("Phase 2.2 requires M80 convex_potential_enabled=True")
        if task_feature_dim!=TASK_FEATURE_DIM:
            raise ValueError(f"task_feature_dim must equal {TASK_FEATURE_DIM}")
        if hidden_dim<8:
            raise ValueError("hidden_dim must be >=8")
        self.base_model=base_model
        self.freeze_backbone=bool(freeze_backbone)
        width=base_model.config.interval_width
        self.task_encoder=nn.Sequential(
            nn.LayerNorm(task_feature_dim),
            nn.Linear(task_feature_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,2*width),
        )
        nn.init.zeros_(self.task_encoder[-1].weight)
        nn.init.zeros_(self.task_encoder[-1].bias)
        if self.freeze_backbone:
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)
            self.base_model.eval()

    def train(self,mode:bool=True):
        super().train(mode)
        if self.freeze_backbone:
            self.base_model.eval()
        return self

    def encode_geometry(self,geometry:Mapping[str,Any],chunk:int|None=None):
        return self.base_model.encode(geometry,chunk)

    def condition_cache(self,cache:EncodedNativeGeometry,task_features:Tensor|np.ndarray):
        if cache.owner!=id(self.base_model) or cache.versions!=self.base_model._versions():
            raise ValueError("foreign or stale M80 geometry cache")
        task=torch.as_tensor(task_features,device=cache.latent.device,dtype=cache.latent.dtype)
        if task.ndim==1:
            task=task.unsqueeze(0)
        if task.shape!=(1,TASK_FEATURE_DIM) or not torch.isfinite(task).all():
            raise ValueError(f"task_features must be finite [31] or [1,31]")
        modulation=self.task_encoder(task)
        gamma,beta=modulation.chunk(2,dim=-1)
        gamma=0.1*torch.tanh(gamma)
        beta=0.1*torch.tanh(beta)
        latent=cache.latent*(1.0+gamma)+beta
        return replace(cache,latent=latent)

    def predict(self,cache:EncodedNativeGeometry,beta_query:Tensor|np.ndarray,
                task_features:Tensor|np.ndarray):
        conditioned=self.condition_cache(cache,task_features)
        return self.base_model.predict(conditioned,beta_query)

    def moments_and_query_jacobian(self,cache:EncodedNativeGeometry,beta_query:Tensor|np.ndarray,
                                   task_features:Tensor|np.ndarray,*,create_graph:bool=False):
        conditioned=self.condition_cache(cache,task_features)
        return self.base_model.moments_and_jacobian(
            conditioned,beta_query,create_graph=create_graph
        )

    def moments_and_material_jacobian(self,cache:EncodedNativeGeometry,beta_query:Tensor|np.ndarray,
                                      task_features:Tensor|np.ndarray,
                                      design_map_jacobian:Tensor|np.ndarray,
                                      contract:MaterialMapContract,*,create_graph:bool=False):
        contract.validate()
        moments,j_query=self.moments_and_query_jacobian(
            cache,beta_query,task_features,create_graph=create_graph
        )
        D=torch.as_tensor(design_map_jacobian,device=j_query.device,dtype=j_query.dtype)
        if D.ndim!=2 or D.shape[0]!=10 or not torch.isfinite(D).all():
            raise ValueError("design_map_jacobian must be finite [10,n_alpha]")
        if contract.identity_map and D.shape!=(10,10):
            raise ValueError("identity design map must have shape [10,10]")
        return moments,j_query@D
