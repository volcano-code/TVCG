"""Unified M80 + SQNO with one shared P-CRAFT encoder (Phase 2.8.4)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from contextlib import nullcontext
import hashlib, math, os, random, tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import Tensor, nn
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    EncodedNativeGeometry, PcraftNativeMomentOperator,
)
from .spin_quadrann_v12 import SpinQuadrANNV12, SQNOGeometryCache
from .geometry_physics_attention import GeometryPhysicsCrossAttention
from .cross_scale_attention import QueryConditionedCrossScaleAttention
from .masked_derivative_supervision import derivative_supervision_loss
from .unified_training_state import (
    parameter_partition, make_training_contract, optimizer_contract,
    checkpoint_training_metadata, validate_resume,
)

SCHEMA="tvcg.unified-m80-sqno.v1"
SEMANTICS="one-shared-pcraft-encoder-m80-and-sqno-query-many"

@dataclass
class UnifiedGeometryCache:
    geometry_id: str
    shared_cache: EncodedNativeGeometry
    sqno_cache: SQNOGeometryCache
    owner: int
    versions: tuple[int,...]
    native_intervals: int
    inference_only: bool = False

def _source_fingerprints():
    here=Path(__file__).resolve().parent
    pcraft_root=here.parents[2]/"spin_pino"
    files={
      "native_pcraft.py":pcraft_root/"native_pcraft.py",
      "spin_quadrann_v11.py":here/"spin_quadrann_v11.py",
      "spin_quadrann_v12.py":here/"spin_quadrann_v12.py",
      "batched_moment_operator.py":here/"batched_moment_operator.py",
      "unified_m80_sqno.py":here/"unified_m80_sqno.py",
      "geometry_physics_attention.py":here/"geometry_physics_attention.py",
      "cross_scale_attention.py":here/"cross_scale_attention.py",
    }
    out={}
    for name,path in files.items():
        if not path.is_file(): raise FileNotFoundError(path)
        out[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out

class UnifiedM80SQNO(nn.Module):
    """Single geometry trunk, M80 Moment head, SQNO query-many branch.

    Global scalar and query-conditioned convex fusion share the same raw
    coefficient Jacobian and training/checkpoint contracts.
    """
    def __init__(self,shared_encoder:PcraftNativeMomentOperator,sqno:SpinQuadrANNV12,
                 *,fusion_initial_weight=.5,architecture_metadata:Mapping[str,Any]|None=None,
                 cross_attention_enabled:bool=False,cross_attention_heads:int=2,
                 cross_attention_chunk:int=1024,cross_attention_output_scale:float=.05,
                 cross_scale_attention_enabled:bool=False,cross_scale_heads:int=2,
                 cross_scale_max_scales:int=16,cross_scale_disabled_ids:tuple[int,...]=(),
                 interaction_variant:str="independent",
                 fusion_mode:str="global_scalar",query_fusion_hidden:int=32):
        super().__init__()
        if not isinstance(shared_encoder,PcraftNativeMomentOperator):
            raise TypeError("shared_encoder must be PcraftNativeMomentOperator")
        if not isinstance(sqno,SpinQuadrANNV12):
            raise TypeError("sqno must be SpinQuadrANNV12")
        if not getattr(sqno,"external_pcraft_required",False) or sqno.pcraft is not None:
            raise ValueError("SQNO must use external pcraft_width mode and own no P-CRAFT adapter")
        width=shared_encoder.config.interval_width
        if sqno.pcraft_width!=width: raise ValueError("shared P-CRAFT feature width mismatch")
        if not 0<float(fusion_initial_weight)<1: raise ValueError("fusion weight must be between zero and one")
        valid_variants={"independent","gno_only","geometry_attention_only","cross_scale_only","gno_then_attention","parallel_gno_attention"}
        if interaction_variant not in valid_variants:
            raise ValueError(f"unsupported interaction_variant: {interaction_variant}")
        if interaction_variant in {"gno_only","gno_then_attention","parallel_gno_attention"} and not shared_encoder.config.gno_enabled:
            raise ValueError(f"{interaction_variant} requires shared_encoder.config.gno_enabled=True")
        if interaction_variant=="gno_only" and (cross_attention_enabled or cross_scale_attention_enabled):
            raise ValueError("gno_only cannot enable attention branches")
        if interaction_variant=="geometry_attention_only" and (not cross_attention_enabled or cross_scale_attention_enabled):
            raise ValueError("geometry_attention_only requires only geometry cross-attention")
        if interaction_variant=="cross_scale_only" and (cross_attention_enabled or not cross_scale_attention_enabled):
            raise ValueError("cross_scale_only requires only cross-scale attention")
        if interaction_variant=="gno_then_attention" and not (cross_attention_enabled or cross_scale_attention_enabled):
            raise ValueError("gno_then_attention requires at least one attention branch")
        if interaction_variant=="parallel_gno_attention" and not (cross_attention_enabled and cross_scale_attention_enabled):
            raise ValueError("parallel_gno_attention requires both attention branches")
        if fusion_mode not in {"global_scalar","query_conditioned"}:
            raise ValueError("fusion_mode must be global_scalar or query_conditioned")
        if query_fusion_hidden<1: raise ValueError("query_fusion_hidden must be positive")
        self.interaction_variant=interaction_variant
        self.fusion_mode=fusion_mode
        self.query_fusion_hidden=int(query_fusion_hidden)
        self.shared_encoder=shared_encoder
        self.sqno=sqno
        self.fusion_initial_weight=float(fusion_initial_weight)
        self.cross_attention_enabled=bool(cross_attention_enabled)
        self.cross_attention_heads=int(cross_attention_heads)
        self.cross_attention_chunk=int(cross_attention_chunk)
        self.cross_attention_output_scale=float(cross_attention_output_scale)
        self.cross_scale_attention_enabled=bool(cross_scale_attention_enabled)
        self.cross_scale_heads=int(cross_scale_heads)
        self.cross_scale_max_scales=int(cross_scale_max_scales)
        self.cross_scale_disabled_ids=tuple(int(v) for v in cross_scale_disabled_ids)
        if interaction_variant=="parallel_gno_attention":
            self.parallel_interaction_logit=nn.Parameter(torch.tensor(0.0,dtype=torch.float64))
        else:
            self.register_parameter("parallel_interaction_logit",None)
        self.cross_scale_attention=(QueryConditionedCrossScaleAttention(width,heads=self.cross_scale_heads,
            max_scales=self.cross_scale_max_scales) if self.cross_scale_attention_enabled else None)
        if self.cross_attention_enabled:
            self.geometry_physics_attention=GeometryPhysicsCrossAttention(
                width,heads=self.cross_attention_heads,
                chunk_intervals=self.cross_attention_chunk,
                output_scale=self.cross_attention_output_scale)
        else:
            self.geometry_physics_attention=None
        logit=np.log(fusion_initial_weight/(1-fusion_initial_weight))
        if self.fusion_mode=="global_scalar":
            self.fusion_logit=nn.Parameter(torch.tensor(float(logit),dtype=torch.float64))
            self.query_fusion_gate=None
        else:
            self.register_parameter("fusion_logit",None)
            self.query_fusion_gate=nn.Sequential(nn.Linear(10+width,self.query_fusion_hidden),
                nn.GELU(),nn.Linear(self.query_fusion_hidden,1))
            nn.init.zeros_(self.query_fusion_gate[-1].weight)
            nn.init.constant_(self.query_fusion_gate[-1].bias,float(logit))
        self.architecture_metadata=dict(architecture_metadata or {})
        self.geometry_encode_calls=0
        self.query_evaluation_calls=0

    @property
    def fusion_weight(self):
        if self.fusion_logit is None:
            raise RuntimeError("query-conditioned fusion has no global scalar fusion_weight")
        return torch.sigmoid(self.fusion_logit)

    def _versions(self):
        return tuple(p._version for p in self.parameters())

    def encode_geometry(self,geometry:Mapping[str,Any],*,chunk:int|None=None,inference:bool=False):
        """Encode one geometry.

        Training caches retain the full autograd graph. Inference caches are
        encoded under no_grad, consume less memory, and are rejected by train()
        or create_graph=True evaluation.
        """
        if inference and self.training:
            raise ValueError("inference=True requires model.eval()")
        if "xyz" not in geometry:
            from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import geometry_from_columns
            geometry=geometry_from_columns(geometry)
        context=torch.no_grad() if inference else nullcontext()
        with context:
            shared=self.shared_encoder.encode(geometry,chunk,retain_interval_features=True)
            payload=self.shared_encoder.export_shared_geometry_features(shared)
            sqno_cache=self.sqno.encode_geometry(geometry,pcraft_features=payload)
        self.geometry_encode_calls+=1
        return UnifiedGeometryCache(shared.geometry_sha256,shared,sqno_cache,id(self),
                                     self._versions(),shared.native_intervals,bool(inference))

    def evaluate_queries(self,cache:UnifiedGeometryCache,coefficients:Tensor,*,
                         create_graph=False,with_jacobian=True):
        if cache.owner!=id(self) or cache.versions!=self._versions():
            raise ValueError("foreign or stale unified geometry cache")
        if cache.inference_only and (self.training or create_graph):
            raise ValueError("inference-only geometry cache cannot be used for training or higher-order gradients")
        if not with_jacobian: raise ValueError("unified contract currently requires Moment Jacobians")
        q=torch.as_tensor(coefficients,device=cache.shared_cache.latent.device,dtype=torch.float64)
        single=q.ndim==1
        qb=q.unsqueeze(0) if single else q
        if qb.ndim!=2 or qb.shape[1]!=10: raise ValueError("coefficients must be [10] or [Q,10]")
        if not torch.isfinite(qb).all() or (qb.abs().amax(-1)<=0).any():
            raise ValueError("coefficients must be finite and nonzero")
        m80,j80=self.shared_encoder.moments_and_jacobian(cache.shared_cache,qb,create_graph=create_graph)
        sq=self.sqno.evaluate_queries(cache.sqno_cache,qb,with_jacobian=True)
        ms=sq["moments"]
        # M80.moments_and_jacobian differentiates with respect to the raw
        # coefficient tensor supplied to this wrapper. SQNO's jacobian_q is
        # with respect to its internal normalized quadrature coordinates;
        # jacobian_input is the chain-rule derivative with respect to the raw
        # input coefficients and is therefore the compatible quantity.
        js=sq["jacobian_input"]
        if m80.shape!=ms.shape or j80.shape!=js.shape:
            raise RuntimeError("M80/SQNO output contract mismatch")
        if cache.inference_only:
            # M80 internally enables autograd to obtain dM/dq. Do not retain
            # that temporary graph or parameter gradients in inference mode.
            m80,j80,ms,js=(value.detach() for value in (m80,j80,ms,js))
        alpha_jacobian=None
        if self.fusion_mode=="global_scalar":
            alpha=self.fusion_weight.to(device=m80.device,dtype=m80.dtype).expand(len(qb),1)
        else:
            # Gate conditions on scale-invariant quadric coefficients and
            # cached global geometry context. Its complete derivative is
            # included in the raw-coefficient Moment Jacobian below.
            with torch.enable_grad():
                q_gate=qb if qb.requires_grad else qb.detach().requires_grad_(True)
                qnorm=q_gate/torch.linalg.vector_norm(q_gate,dim=-1,keepdim=True).clamp_min(1e-30)
                geometry_context=cache.shared_cache.latent.reshape(1,-1).expand(len(qb),-1).to(
                    device=q_gate.device,dtype=q_gate.dtype)
                gate_input=torch.cat((qnorm.to(geometry_context.dtype),geometry_context),dim=-1)
                # Neural gate parameters may remain FP32 while the physical
                # coefficient/Jacobian path uses FP64. Cast only at the MLP
                # boundary; autograd preserves the chain rule through this cast.
                gate_dtype=self.query_fusion_gate[0].weight.dtype
                gate_input=gate_input.to(dtype=gate_dtype,
                    device=self.query_fusion_gate[0].weight.device)
                alpha=torch.sigmoid(self.query_fusion_gate(gate_input))
                alpha_jacobian=torch.autograd.grad(alpha.sum(),q_gate,create_graph=create_graph,
                    retain_graph=True)[0].unsqueeze(1)
        alpha_value=alpha.to(device=m80.device,dtype=m80.dtype)
        fused_moments=alpha_value*m80+(1-alpha_value)*ms
        fused_jacobian=alpha_value.unsqueeze(-1)*j80+(1-alpha_value).unsqueeze(-1)*js
        if alpha_jacobian is not None:
            fused_jacobian=fused_jacobian+(m80-ms).unsqueeze(-1)*alpha_jacobian.to(
                device=m80.device,dtype=m80.dtype)
        if cache.inference_only:
            fused_moments=fused_moments.detach()
            fused_jacobian=fused_jacobian.detach()
            alpha_value=alpha_value.detach()
            if alpha_jacobian is not None: alpha_jacobian=alpha_jacobian.detach()
        attention_diagnostics={}
        corrections=[]; correction_jacobians=[]; branch_names=[]
        if self.geometry_physics_attention is not None:
            with torch.enable_grad():
                q_attention=qb if qb.requires_grad else qb.detach().requires_grad_(True)
                correction,diag=self.geometry_physics_attention(q_attention,
                    cache.shared_cache.interval_features,cache.shared_cache.interval_mass)
                jac=torch.stack([torch.autograd.grad(correction[:,k].sum(),q_attention,
                    create_graph=create_graph,retain_graph=True)[0] for k in range(10)],dim=1)
            corrections.append(correction); correction_jacobians.append(jac); branch_names.append("geometry")
            attention_diagnostics.update(diag)
        if self.cross_scale_attention is not None:
            with torch.enable_grad():
                q_scale=qb if qb.requires_grad else qb.detach().requires_grad_(True)
                correction,diag=self.cross_scale_attention(q_scale,cache.shared_cache.scale_features,
                    cache.shared_cache.scale_ids,disabled_scale_ids=self.cross_scale_disabled_ids)
                jac=torch.stack([torch.autograd.grad(correction[:,k].sum(),q_scale,
                    create_graph=create_graph,retain_graph=True)[0] for k in range(10)],dim=1)
            corrections.append(correction); correction_jacobians.append(jac); branch_names.append("scale")
            attention_diagnostics.update(diag)
        if corrections:
            if self.interaction_variant=="parallel_gno_attention" and len(corrections)==2:
                gate=torch.sigmoid(self.parallel_interaction_logit).to(device=qb.device,dtype=qb.dtype)
                attention_correction=gate*corrections[0]+(1-gate)*corrections[1]
                attention_jacobian=gate*correction_jacobians[0]+(1-gate)*correction_jacobians[1]
                attention_diagnostics["parallel_attention_gate"]=gate.detach()
            else:
                attention_correction=sum(corrections)
                attention_jacobian=sum(correction_jacobians)
            fused_moments=fused_moments+attention_correction.to(dtype=fused_moments.dtype)
            fused_jacobian=fused_jacobian+attention_jacobian.to(dtype=fused_jacobian.dtype)
            if cache.inference_only:
                fused_moments=fused_moments.detach(); fused_jacobian=fused_jacobian.detach()
                attention_diagnostics={k:v.detach() if isinstance(v,Tensor) else v for k,v in attention_diagnostics.items()}
        result={
          "moments":fused_moments,
          "jacobian_coefficients":fused_jacobian,
          "jacobian_q":fused_jacobian,
          "m80_moments":m80,"m80_jacobian_coefficients":j80,
          "m80_jacobian_q":j80,
          "sqno_moments":ms,"sqno_jacobian_coefficients":js,
          "sqno_jacobian_q":js,
          "fusion_weight":(self.fusion_weight.to(device=m80.device,dtype=m80.dtype)
            if self.fusion_mode=="global_scalar" else alpha_value.squeeze(-1)),
          "fusion_mode":self.fusion_mode,
          "fusion_gate_jacobian_norm":(torch.zeros(len(qb),device=qb.device,dtype=qb.dtype)
            if alpha_jacobian is None else torch.linalg.vector_norm(alpha_jacobian.squeeze(1),dim=-1)),
          "geometry_id":cache.geometry_id,
          "native_intervals":torch.as_tensor(cache.native_intervals,device=m80.device),
          "mass_error":sq["total_mass_error"],
          **attention_diagnostics,
        }
        self.query_evaluation_calls+=1
        if single:
            for key in ("moments","jacobian_coefficients","jacobian_q","m80_moments",
                        "m80_jacobian_coefficients","m80_jacobian_q","sqno_moments",
                        "sqno_jacobian_coefficients","sqno_jacobian_q"):
                result[key]=result[key][0]
        return result

    @staticmethod
    def _masked_mean_square(error:Tensor,mask:Tensor|None,*,name:str)->Tensor:
        if mask is not None:
            mask=torch.as_tensor(mask,device=error.device)
            if mask.dtype!=torch.bool or mask.ndim!=1 or mask.shape[0]!=error.shape[0]:
                raise ValueError(f"{name} mask must be boolean with shape [Q]")
            if not mask.any():
                raise ValueError(f"{name} mask contains no valid queries")
            error=error[mask]
        if not torch.isfinite(error).all():
            raise ValueError(f"{name} selected targets/predictions must be finite")
        return error.square().mean()

    def _joint_task_losses_and_metadata(
        self,cache:UnifiedGeometryCache,coefficients:Tensor,target_moments:Tensor,target_jacobian:Tensor|None=None,*,
        moment_scale:Tensor|float=1.0,jacobian_scale:Tensor|float=1.0,
        valid_mask:Tensor|None=None,jacobian_valid_mask:Tensor|None=None,
        jacobian_weight:float=0.1,branch_aux_weight:float=0.25,
        restricted_basis:Tensor|None=None,restricted_jacobian:Tensor|None=None,
        restricted_valid:Tensor|None=None,restricted_dimension:Tensor|None=None,
        restricted_scale:Tensor|float=1.0,_create_graph:bool=True,
    )->tuple[dict[str,Tensor],dict[str,Tensor]]:
        if any(not math.isfinite(float(value)) or value<0
               for value in (jacobian_weight,branch_aux_weight)):
            raise ValueError("loss weights must be finite and nonnegative")
        def real_tensor(value,name):
            tensor=torch.as_tensor(value,device=cache.shared_cache.latent.device)
            if tensor.is_complex() or tensor.dtype==torch.bool:
                raise ValueError(f"{name} must be real numeric data, not complex or boolean")
            # Preserve Python scalar precision instead of first materializing float32.
            return torch.as_tensor(value,device=cache.shared_cache.latent.device,dtype=torch.float64)
        q=real_tensor(coefficients,"coefficients")
        if q.ndim not in {1,2} or q.shape[-1]!=10 or not q.numel():
            raise ValueError("coefficients must be [10] or nonempty [Q,10]")
        qn=1 if q.ndim==1 else q.shape[0]
        target_m=real_tensor(target_moments,"target_moments").detach()
        target_j=(None if target_jacobian is None else
                  real_tensor(target_jacobian,"target_jacobian").detach())
        if target_m.shape!=(qn,10) and not (q.ndim==1 and target_m.shape==(10,)):
            raise ValueError("target_moments must be [Q,10] or [10] for a single query")
        if target_j is not None and target_j.shape!=(qn,10,10) and not (q.ndim==1 and target_j.shape==(10,10)):
            raise ValueError("target_jacobian must be [Q,10,10] or [10,10] for a single query")
        if target_m.ndim==1: target_m=target_m.unsqueeze(0)
        if target_j is not None and target_j.ndim==2: target_j=target_j.unsqueeze(0)
        ms=real_tensor(moment_scale,"moment_scale").detach()
        if not torch.isfinite(ms).all() or (ms<=0).any():
            raise ValueError("moment scales must be finite and positive")
        try: target_m_norm=target_m/ms
        except RuntimeError as exc: raise ValueError("normalization scales are not broadcastable") from exc
        if target_m_norm.shape!=target_m.shape:
            raise ValueError("normalization scales must preserve target shapes")
        result=self.evaluate_queries(cache,q,create_graph=_create_graph,with_jacobian=True)
        def norm_m(key):
            value=result[key]
            if value.ndim==1:value=value.unsqueeze(0)
            return value/ms
        def raw_j(key):
            value=result[key]
            if value.ndim==2:value=value.unsqueeze(0)
            return value
        mask=valid_mask
        jmask=jacobian_valid_mask if jacobian_valid_mask is not None else (valid_mask if target_j is not None else None)
        derivative_options=dict(derivative_valid=jmask,restricted_basis=restricted_basis,
            restricted_target=restricted_jacobian,restricted_valid=restricted_valid,
            restricted_dimension=restricted_dimension,jacobian_scale=jacobian_scale,
            restricted_scale=restricted_scale)
        fm=self._masked_mean_square(norm_m("moments")-target_m_norm,mask,name="moment")
        m80m=self._masked_mean_square(norm_m("m80_moments")-target_m_norm,mask,name="M80 moment")
        sqnom=self._masked_mean_square(norm_m("sqno_moments")-target_m_norm,mask,name="SQNO moment")
        fused=derivative_supervision_loss(raw_j("jacobian_q"),target_j,**derivative_options)
        m80=derivative_supervision_loss(raw_j("m80_jacobian_q"),target_j,**derivative_options)
        sqno=derivative_supervision_loss(raw_j("sqno_jacobian_q"),target_j,**derivative_options)
        fj,m80j,sqnoj=fused["loss"],m80["loss"],sqno["loss"]
        branch=(m80m+sqnom+jacobian_weight*(m80j+sqnoj))*0.5
        tasks={"moment":fm,"jacobian":fj,"branch_aux":branch,
            "m80_moment":m80m,"sqno_moment":sqnom,"m80_jacobian":m80j,"sqno_jacobian":sqnoj}
        for prefix,report in (("",fused),("m80_",m80),("sqno_",sqno)):
            tasks[prefix+"full_jacobian"]=report["full_loss"]
            tasks[prefix+"restricted_jacobian"]=report["restricted_loss"]
        def valid_count(value):
            return (torch.tensor(qn,device=q.device,dtype=torch.int64) if value is None
                    else torch.as_tensor(value,device=q.device,dtype=torch.bool).sum())
        metadata={"fusion_weight":result["fusion_weight"].detach(),
            "valid_queries":valid_count(mask),"valid_jacobian_queries":fused["full_count"],
            "valid_restricted_queries":fused["restricted_count"],
            "valid_derivative_queries":fused["full_count"]+fused["restricted_count"],
            "restricted_dimensions":fused["restricted_dimensions"]}
        return tasks,metadata

    def joint_task_losses(
        self,cache:UnifiedGeometryCache,coefficients:Tensor,target_moments:Tensor,target_jacobian:Tensor|None=None,*,
        moment_scale:Tensor|float=1.0,jacobian_scale:Tensor|float=1.0,
        valid_mask:Tensor|None=None,jacobian_valid_mask:Tensor|None=None,
        jacobian_weight:float=0.1,branch_aux_weight:float=0.25,
        restricted_basis:Tensor|None=None,restricted_jacobian:Tensor|None=None,
        restricted_valid:Tensor|None=None,restricted_dimension:Tensor|None=None,
        restricted_scale:Tensor|float=1.0,
    )->dict[str,Tensor]:
        """Return three task objectives plus branch and derivative diagnostics.

        Full and restricted derivatives are disjoint supervision modes per query.
        Restricted labels compare raw ``J @ basis`` before normalization, with
        only the first ``restricted_dimension`` columns active for each row.
        An absent derivative task is a detached zero and is skipped by PCGrad.
        """
        tasks,_=self._joint_task_losses_and_metadata(cache,coefficients,target_moments,target_jacobian,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,valid_mask=valid_mask,
            jacobian_valid_mask=jacobian_valid_mask,jacobian_weight=jacobian_weight,
            branch_aux_weight=branch_aux_weight,restricted_basis=restricted_basis,
            restricted_jacobian=restricted_jacobian,restricted_valid=restricted_valid,
            restricted_dimension=restricted_dimension,restricted_scale=restricted_scale)
        return tasks

    def joint_training_loss(self,cache:UnifiedGeometryCache,coefficients:Tensor,
        target_moments:Tensor,target_jacobian:Tensor|None=None,*,moment_scale:Tensor|float=1.0,
        jacobian_scale:Tensor|float=1.0,valid_mask:Tensor|None=None,
        jacobian_valid_mask:Tensor|None=None,jacobian_weight:float=0.1,
        branch_aux_weight:float=0.25,restricted_basis:Tensor|None=None,
        restricted_jacobian:Tensor|None=None,restricted_valid:Tensor|None=None,
        restricted_dimension:Tensor|None=None,restricted_scale:Tensor|float=1.0)->dict[str,Tensor]:
        tasks,metadata=self._joint_task_losses_and_metadata(cache,coefficients,target_moments,target_jacobian,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,valid_mask=valid_mask,
            jacobian_valid_mask=jacobian_valid_mask,jacobian_weight=jacobian_weight,
            branch_aux_weight=branch_aux_weight,restricted_basis=restricted_basis,
            restricted_jacobian=restricted_jacobian,restricted_valid=restricted_valid,
            restricted_dimension=restricted_dimension,restricted_scale=restricted_scale)
        total=tasks["moment"]+jacobian_weight*tasks["jacobian"]+branch_aux_weight*tasks["branch_aux"]
        return {"loss":total,**{name+"_loss":value.detach() for name,value in tasks.items()},**metadata}

    def train_step(
        self,optimizer,cache:UnifiedGeometryCache,coefficients:Tensor,
        target_moments:Tensor,target_jacobian:Tensor|None=None,*,
        moment_scale:Tensor|float=1.0,jacobian_scale:Tensor|float=1.0,
        valid_mask:Tensor|None=None,jacobian_valid_mask:Tensor|None=None,
        jacobian_weight:float=0.1,branch_aux_weight:float=0.25,
        gradient_clip:float=1.0,
        gradient_mode:str="fixed",
        pcgrad_reduction:str="sum",
        restricted_basis:Tensor|None=None,restricted_jacobian:Tensor|None=None,
        restricted_valid:Tensor|None=None,restricted_dimension:Tensor|None=None,
        restricted_scale:Tensor|float=1.0,
    )->dict[str,Any]:
        """One explicit joint update; caller must re-encode after optimizer.step."""
        if not self.training:
            raise RuntimeError("train_step requires model.train()")
        if cache.inference_only:
            raise ValueError("train_step cannot consume an inference-only cache")
        contract=self.training_contract(gradient_mode=gradient_mode,pcgrad_reduction=pcgrad_reduction,
            jacobian_weight=jacobian_weight,branch_aux_weight=branch_aux_weight,gradient_clip=gradient_clip,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,restricted_scale=restricted_scale)
        resumed=getattr(self,"_resume_training_contract",None)
        if resumed is not None and resumed!=contract:
            raise ValueError("train_step differs from the resumed training contract")
        optimizer_contract(self,optimizer)
        optimizer.zero_grad(set_to_none=True)
        task_losses,metadata=self._joint_task_losses_and_metadata(cache,coefficients,target_moments,target_jacobian,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,valid_mask=valid_mask,
            jacobian_valid_mask=jacobian_valid_mask,jacobian_weight=jacobian_weight,
            branch_aux_weight=branch_aux_weight,restricted_basis=restricted_basis,
            restricted_jacobian=restricted_jacobian,restricted_valid=restricted_valid,
            restricted_dimension=restricted_dimension,restricted_scale=restricted_scale)
        weights={"moment":1.0,"jacobian":float(jacobian_weight),"branch_aux":float(branch_aux_weight)}
        active=[name for name in weights if weights[name]>0 and task_losses[name].requires_grad]
        total=sum((task_losses[k]*weights[k] for k in active),
                  torch.zeros((),device=task_losses["moment"].device,dtype=task_losses["moment"].dtype))
        parameters=[p for p in self.parameters() if p.requires_grad]
        if gradient_mode=="pcgrad":
            from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import pcgrad_assign_gradients
            named=dict(self.named_parameters())
            partition=contract["parameter_partition"]
            shared=[named[name] for name in partition["shared"]]
            private=[named[name] for name in partition["private"]]
            multitask_report=pcgrad_assign_gradients(task_losses,weights,shared,private,reduction=pcgrad_reduction)
            multitask_report["diagnostics_available"]=True
        else:
            total.backward()
            multitask_report={"task_names":active,"task_count":len(active),
                "projection_counts":[0]*len(active),"cosine_matrix":[],
                "shared_gradient_norms":{},"private_gradient_norms":{},
                "reduction":"sum","diagnostics_available":False}
        grad_norm=torch.nn.utils.clip_grad_norm_(parameters,gradient_clip,error_if_nonfinite=True)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite unified gradient norm")
        optimizer.step()
        self._last_training_contract=contract
        out={key:float(value.detach()) for key,value in task_losses.items()}
        out.update({key+"_loss":value for key,value in list(out.items())})
        out.update({"fusion_weight":float(metadata["fusion_weight"].mean()),
            **{key:int(value) for key,value in metadata.items() if key!="fusion_weight"}})
        out["loss"]=float(total.detach())
        out["gradient_norm"]=float(grad_norm.detach())
        out["gradient_mode"]=gradient_mode
        out["gradient_reduction"]=contract["pcgrad_reduction"]
        out["multitask_report"]=multitask_report
        out["cache_invalidated"]=1.0
        return out

    def training_parameter_partition(self):
        """Return ordered, exhaustive shared/private/inactive parameter names."""
        return parameter_partition(self)

    def training_contract(self,*,gradient_mode="fixed",pcgrad_reduction="sum",jacobian_weight=.1,
                          branch_aux_weight=.25,gradient_clip=1.,moment_scale=1.,jacobian_scale=1.,
                          restricted_scale=1.):
        return make_training_contract(self,gradient_mode=gradient_mode,pcgrad_reduction=pcgrad_reduction,
            jacobian_weight=jacobian_weight,branch_aux_weight=branch_aux_weight,gradient_clip=gradient_clip,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,restricted_scale=restricted_scale)

    def unique_trainable_parameters(self):
        seen=set()
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); yield p

    def build_optimizer(self,*,learning_rate=3e-4,weight_decay=1e-4):
        if (not math.isfinite(float(learning_rate)) or not math.isfinite(float(weight_decay))
            or learning_rate<=0 or weight_decay<0): raise ValueError("invalid optimizer parameters")
        params=list(self.unique_trainable_parameters())
        if not params: raise ValueError("no trainable parameters")
        if len({id(p) for p in params})!=len(params): raise RuntimeError("duplicate optimizer parameters")
        return torch.optim.AdamW(params,lr=learning_rate,weight_decay=weight_decay)

    def architecture_contract(self):
        s=self.sqno
        return {
          "schema":SCHEMA,"semantics":SEMANTICS,
          "shared_encoder_config":asdict(self.shared_encoder.config),
          "sqno":{"external_pcraft_required":s.external_pcraft_required,"pcraft_width":s.pcraft_width,
            "hidden_dim":s.hidden_dim,"final_hidden_dim":s.final_hidden_dim,"num_layers":s.num_layers,
            "neighbors":s.neighbors,"quadrature_order":s.quadrature_order,
            "chunk_intervals":s.chunk_intervals,"chunk_nodes":s.chunk_nodes,
            "query_chunk":s.query_chunk,"temperature":s.moment_operator.temperature},
          "fusion":{"kind":"global_scalar_convex" if self.fusion_mode=="global_scalar" else "query_conditioned",
            "initial_weight":self.fusion_initial_weight,"query_hidden":self.query_fusion_hidden},
          "fusion_mode":self.fusion_mode,
          "query_fusion_hidden":self.query_fusion_hidden,
          "geometry_physics_attention":{"enabled":self.cross_attention_enabled,
            "heads":self.cross_attention_heads,"chunk_intervals":self.cross_attention_chunk,
            "output_scale":self.cross_attention_output_scale},
          "cross_scale_attention":{"enabled":self.cross_scale_attention_enabled,
            "heads":self.cross_scale_heads,"max_scales":self.cross_scale_max_scales,
            "disabled_ids":self.cross_scale_disabled_ids},
          "interaction_variant":self.interaction_variant,
          "metadata":self.architecture_metadata,
        }

    def save_checkpoint(self,path,*,optimizer=None,scheduler=None,data_fingerprint=None,extra=None,
                        training_contract=None):
        training_metadata=checkpoint_training_metadata(self,optimizer,scheduler,data_fingerprint,training_contract)
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        payload={"schema":SCHEMA,"semantics":SEMANTICS,"architecture":self.architecture_contract(),
          "source_fingerprints":_source_fingerprints(),"data_fingerprint":data_fingerprint,
          "model":self.state_dict(),"optimizer":None if optimizer is None else optimizer.state_dict(),
          "scheduler":None if scheduler is None else scheduler.state_dict(),
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
          "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"extra":dict(extra or {}),
          **training_metadata}
        tmp=None
        try:
            with tempfile.NamedTemporaryFile("wb",dir=target.parent,prefix=target.name+".",suffix=".tmp",delete=False) as f:
                tmp=Path(f.name); torch.save(payload,f); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,target)
        finally:
            if tmp is not None: tmp.unlink(missing_ok=True)

    def load_checkpoint(self,path,*,optimizer=None,scheduler=None,expected_data_fingerprint=None,
                        restore_rng=False,map_location="cpu",expected_training_contract=None):
        payload=torch.load(path,map_location=map_location,weights_only=False)
        if payload.get("schema")!=SCHEMA or payload.get("semantics")!=SEMANTICS:
            raise ValueError("incompatible unified checkpoint schema")
        saved_architecture=payload.get("architecture",{})
        current_architecture=self.architecture_contract()
        legacy_global=(not self.cross_attention_enabled
            and "geometry_physics_attention" not in saved_architecture)
        if saved_architecture!=current_architecture:
            saved_compare=dict(saved_architecture)
            current_compare=dict(current_architecture)
            # Phase 2.8.5.3 adds opt-in adaptive slicing config defaults.
            # Normalize absent fields from older Phase 2.8.4 checkpoints.
            saved_encoder=dict(saved_compare.get("shared_encoder_config",{}))
            current_encoder=dict(current_compare.get("shared_encoder_config",{}))
            for key,value in (("transolver_adaptive",False),("transolver_min_slices",4),("transolver_adaptive_temperature",False),("transolver_temperature_min",0.05),("transolver_temperature_max",2.0)):
                saved_encoder.setdefault(key,value)
            saved_compare["shared_encoder_config"]=saved_encoder
            saved_compare.setdefault("fusion_mode","global_scalar")
            saved_compare.setdefault("query_fusion_hidden",32)
            saved_fusion=dict(saved_compare.get("fusion",{}))
            # Preserve the historical fusion-kind token. The top-level
            # fusion_mode is normalized separately; rewriting this token
            # would make a current global-scalar checkpoint fail migration
            # against its own architecture contract.
            saved_fusion.setdefault("query_hidden",32)
            saved_compare["fusion"]=saved_fusion
            current_compare.pop("geometry_physics_attention",None)
            if not legacy_global or saved_compare!=current_compare:
                raise ValueError("unified checkpoint architecture contract mismatch")
        saved_sources=payload.get("source_fingerprints",{})
        current_sources=_source_fingerprints()
        if saved_sources!=current_sources:
            legacy_keys=set(current_sources)-{"geometry_physics_attention.py"}
            source_ok=(legacy_global and set(saved_sources)==legacy_keys
                and all(saved_sources.get(key)==current_sources.get(key)
                        for key in legacy_keys if key!="unified_m80_sqno.py"))
            if not source_ok:
                raise ValueError("unified checkpoint source fingerprint mismatch")
        if expected_data_fingerprint is not None and payload.get("data_fingerprint")!=expected_data_fingerprint:
            raise ValueError("unified checkpoint data fingerprint mismatch")
        resumed_contract=validate_resume(self,payload,optimizer,scheduler,expected_training_contract,
            expected_data_fingerprint,restore_rng)
        self.load_state_dict(payload["model"],strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if restore_rng:
            torch.set_rng_state(payload["torch_rng"].cpu())
            if torch.cuda.is_available() and payload.get("cuda_rng"):
                torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
            np.random.set_state(payload["numpy_rng"]); random.setstate(payload["python_rng"])
        self._resume_training_contract=resumed_contract
        self._last_training_contract=resumed_contract
        return payload
