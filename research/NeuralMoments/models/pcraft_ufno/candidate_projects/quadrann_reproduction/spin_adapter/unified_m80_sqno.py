"""Unified M80 + SQNO with one shared P-CRAFT encoder (Phase 2.8.4)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from contextlib import nullcontext
import hashlib, os, random, tempfile
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

    A trainable global scalar convex fusion is used as a simple auditable
    baseline. Query-conditioned fusion is reserved for Phase 2.8.5.
    """
    def __init__(self,shared_encoder:PcraftNativeMomentOperator,sqno:SpinQuadrANNV12,
                 *,fusion_initial_weight=.5,architecture_metadata:Mapping[str,Any]|None=None,
                 cross_attention_enabled:bool=False,cross_attention_heads:int=2,
                 cross_attention_chunk:int=1024,cross_attention_output_scale:float=.05,
                 cross_scale_attention_enabled:bool=False,cross_scale_heads:int=2,
                 cross_scale_max_scales:int=16,cross_scale_disabled_ids:tuple[int,...]=()):
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
        self.fusion_logit=nn.Parameter(torch.tensor(float(logit),dtype=torch.float64))
        self.architecture_metadata=dict(architecture_metadata or {})
        self.geometry_encode_calls=0
        self.query_evaluation_calls=0

    @property
    def fusion_weight(self): return torch.sigmoid(self.fusion_logit)

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
        alpha=self.fusion_weight.to(device=m80.device,dtype=m80.dtype)
        fused_moments=alpha*m80+(1-alpha)*ms
        fused_jacobian=alpha*j80+(1-alpha)*js
        attention_diagnostics={}
        if self.geometry_physics_attention is not None:
            # The query-conditioned correction is differentiated with respect
            # to the raw input coefficients and added to the branch Jacobian.
            with torch.enable_grad():
                q_attention=qb if qb.requires_grad else qb.detach().requires_grad_(True)
                correction,attention_diagnostics=self.geometry_physics_attention(
                    q_attention,cache.shared_cache.interval_features,
                    cache.shared_cache.interval_mass)
                correction_jacobian=torch.stack([
                    torch.autograd.grad(correction[:,out_index].sum(),q_attention,
                        create_graph=create_graph,retain_graph=True)[0]
                    for out_index in range(10)
                ],dim=1)
            fused_moments=fused_moments+correction.to(dtype=fused_moments.dtype)
            fused_jacobian=fused_jacobian+correction_jacobian.to(dtype=fused_jacobian.dtype)
            if cache.inference_only:
                fused_moments=fused_moments.detach()
                fused_jacobian=fused_jacobian.detach()
                attention_diagnostics={k:v.detach() for k,v in attention_diagnostics.items()}
        if self.cross_scale_attention is not None:
            correction,scale_diag=self.cross_scale_attention(qb,cache.shared_cache.scale_features,
                cache.shared_cache.scale_ids,disabled_scale_ids=self.cross_scale_disabled_ids)
            with torch.enable_grad():
                q_scale=qb if qb.requires_grad else qb.detach().requires_grad_(True)
                correction,scale_diag=self.cross_scale_attention(q_scale,cache.shared_cache.scale_features,
                    cache.shared_cache.scale_ids,disabled_scale_ids=self.cross_scale_disabled_ids)
                scale_jacobian=torch.stack([torch.autograd.grad(correction[:,k].sum(),q_scale,
                    create_graph=create_graph,retain_graph=True)[0] for k in range(10)],dim=1)
            fused_moments=fused_moments+correction.to(dtype=fused_moments.dtype)
            fused_jacobian=fused_jacobian+scale_jacobian.to(dtype=fused_jacobian.dtype)
            if cache.inference_only:
                fused_moments=fused_moments.detach(); fused_jacobian=fused_jacobian.detach()
                scale_diag={k:v.detach() for k,v in scale_diag.items()}
            attention_diagnostics.update(scale_diag)
        result={
          "moments":fused_moments,
          "jacobian_coefficients":fused_jacobian,
          "jacobian_q":fused_jacobian,
          "m80_moments":m80,"m80_jacobian_coefficients":j80,
          "m80_jacobian_q":j80,
          "sqno_moments":ms,"sqno_jacobian_coefficients":js,
          "sqno_jacobian_q":js,
          "fusion_weight":alpha,"geometry_id":cache.geometry_id,
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
        values=error.square()
        if mask is None:
            return values.mean()
        mask=torch.as_tensor(mask,device=values.device,dtype=torch.bool)
        if mask.ndim!=1 or mask.shape[0]!=values.shape[0]:
            raise ValueError(f"{name} mask must have shape [Q]")
        if not mask.any():
            raise ValueError(f"{name} mask contains no valid queries")
        return values[mask].mean()

    def joint_training_loss(
        self,
        cache:UnifiedGeometryCache,
        coefficients:Tensor,
        target_moments:Tensor,
        target_jacobian:Tensor,
        *,
        moment_scale:Tensor|float=1.0,
        jacobian_scale:Tensor|float=1.0,
        valid_mask:Tensor|None=None,
        jacobian_valid_mask:Tensor|None=None,
        jacobian_weight:float=0.1,
        branch_aux_weight:float=0.25,
    )->dict[str,Tensor]:
        """Compute a shared, mask-aware M80/SQNO/fused supervision objective.

        Scales are explicit physical normalization factors; no dataset-specific
        normalization is inferred. All three outputs use the same valid-query
        masks and targets. Jacobian supervision uses create_graph=True so the
        loss differentiates through the M80 autograd Jacobian as well as SQNO's
        analytic quadrature Jacobian.
        """
        if jacobian_weight<0 or branch_aux_weight<0:
            raise ValueError("loss weights must be nonnegative")
        q=torch.as_tensor(coefficients,device=cache.shared_cache.latent.device,dtype=torch.float64)
        qn=1 if q.ndim==1 else q.shape[0]
        target_m=torch.as_tensor(target_moments,device=q.device,dtype=torch.float64)
        target_j=torch.as_tensor(target_jacobian,device=q.device,dtype=torch.float64)
        if target_m.shape!=(qn,10) and not (q.ndim==1 and target_m.shape==(10,)):
            raise ValueError("target_moments must be [Q,10] or [10] for a single query")
        if target_j.shape!=(qn,10,10) and not (q.ndim==1 and target_j.shape==(10,10)):
            raise ValueError("target_jacobian must be [Q,10,10] or [10,10] for a single query")
        if q.ndim==1:
            target_m=target_m.unsqueeze(0)
            target_j=target_j.unsqueeze(0)
        if not torch.isfinite(target_m).all() or not torch.isfinite(target_j).all():
            raise ValueError("training targets must be finite")
        ms=torch.as_tensor(moment_scale,device=q.device,dtype=torch.float64)
        js=torch.as_tensor(jacobian_scale,device=q.device,dtype=torch.float64)
        if (not torch.isfinite(ms).all() or (ms<=0).any()
                or not torch.isfinite(js).all() or (js<=0).any()):
            raise ValueError("moment/jacobian scales must be finite and positive")
        try:
            target_m_norm=target_m/ms
            target_j_norm=target_j/js
        except RuntimeError as exc:
            raise ValueError("normalization scales are not broadcastable to target shapes") from exc
        result=self.evaluate_queries(cache,q,create_graph=True,with_jacobian=True)
        def norm_m(key):
            value=result[key]
            if value.ndim==1: value=value.unsqueeze(0)
            return value/ms
        def norm_j(key):
            value=result[key]
            if value.ndim==2: value=value.unsqueeze(0)
            return value/js
        mask=valid_mask
        jmask=jacobian_valid_mask if jacobian_valid_mask is not None else valid_mask
        fm=self._masked_mean_square(norm_m("moments")-target_m_norm,mask,name="moment")
        fj=self._masked_mean_square(norm_j("jacobian_q")-target_j_norm,jmask,name="jacobian")
        m80m=self._masked_mean_square(norm_m("m80_moments")-target_m_norm,mask,name="M80 moment")
        sqnom=self._masked_mean_square(norm_m("sqno_moments")-target_m_norm,mask,name="SQNO moment")
        m80j=self._masked_mean_square(norm_j("m80_jacobian_q")-target_j_norm,jmask,name="M80 Jacobian")
        sqnoj=self._masked_mean_square(norm_j("sqno_jacobian_q")-target_j_norm,jmask,name="SQNO Jacobian")
        branch=(m80m+sqnom+jacobian_weight*(m80j+sqnoj))*0.5
        total=fm+jacobian_weight*fj+branch_aux_weight*branch
        return {
            "loss":total,
            "moment_loss":fm.detach(),
            "jacobian_loss":fj.detach(),
            "m80_moment_loss":m80m.detach(),
            "sqno_moment_loss":sqnom.detach(),
            "m80_jacobian_loss":m80j.detach(),
            "sqno_jacobian_loss":sqnoj.detach(),
            "branch_aux_loss":branch.detach(),
            "fusion_weight":result["fusion_weight"].detach(),
            "valid_queries":torch.as_tensor(qn if mask is None else torch.as_tensor(mask).sum().item(),device=q.device),
        }

    def train_step(
        self,optimizer,cache:UnifiedGeometryCache,coefficients:Tensor,
        target_moments:Tensor,target_jacobian:Tensor,*,
        moment_scale:Tensor|float=1.0,jacobian_scale:Tensor|float=1.0,
        valid_mask:Tensor|None=None,jacobian_valid_mask:Tensor|None=None,
        jacobian_weight:float=0.1,branch_aux_weight:float=0.25,
        gradient_clip:float=1.0,
    )->dict[str,float]:
        """One explicit joint update; caller must re-encode after optimizer.step."""
        if not self.training:
            raise RuntimeError("train_step requires model.train()")
        if cache.inference_only:
            raise ValueError("train_step cannot consume an inference-only cache")
        if gradient_clip<=0:
            raise ValueError("gradient_clip must be positive")
        optimizer.zero_grad(set_to_none=True)
        losses=self.joint_training_loss(
            cache,coefficients,target_moments,target_jacobian,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,
            valid_mask=valid_mask,jacobian_valid_mask=jacobian_valid_mask,
            jacobian_weight=jacobian_weight,branch_aux_weight=branch_aux_weight,
        )
        losses["loss"].backward()
        parameters=[p for p in self.parameters() if p.requires_grad]
        grad_norm=torch.nn.utils.clip_grad_norm_(parameters,gradient_clip,error_if_nonfinite=True)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite unified gradient norm")
        optimizer.step()
        out={key:float(value.detach()) for key,value in losses.items() if key!="loss"}
        out["loss"]=float(losses["loss"].detach())
        out["gradient_norm"]=float(grad_norm.detach())
        out["cache_invalidated"]=1.0
        return out

    def unique_trainable_parameters(self):
        seen=set()
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); yield p

    def build_optimizer(self,*,learning_rate=3e-4,weight_decay=1e-4):
        if learning_rate<=0 or weight_decay<0: raise ValueError("invalid optimizer parameters")
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
          "fusion":{"kind":"global_scalar_convex","initial_weight":self.fusion_initial_weight},
          "geometry_physics_attention":{"enabled":self.cross_attention_enabled,
            "heads":self.cross_attention_heads,"chunk_intervals":self.cross_attention_chunk,
            "output_scale":self.cross_attention_output_scale},
          "cross_scale_attention":{"enabled":self.cross_scale_attention_enabled,
            "heads":self.cross_scale_heads,"max_scales":self.cross_scale_max_scales,
            "disabled_ids":self.cross_scale_disabled_ids},
          "metadata":self.architecture_metadata,
        }

    def save_checkpoint(self,path,*,optimizer=None,scheduler=None,data_fingerprint=None,extra=None):
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        payload={"schema":SCHEMA,"semantics":SEMANTICS,"architecture":self.architecture_contract(),
          "source_fingerprints":_source_fingerprints(),"data_fingerprint":data_fingerprint,
          "model":self.state_dict(),"optimizer":None if optimizer is None else optimizer.state_dict(),
          "scheduler":None if scheduler is None else scheduler.state_dict(),
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
          "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"extra":dict(extra or {})}
        tmp=None
        try:
            with tempfile.NamedTemporaryFile("wb",dir=target.parent,prefix=target.name+".",suffix=".tmp",delete=False) as f:
                tmp=Path(f.name); torch.save(payload,f); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,target)
        finally:
            if tmp is not None: tmp.unlink(missing_ok=True)

    def load_checkpoint(self,path,*,optimizer=None,scheduler=None,expected_data_fingerprint=None,
                        restore_rng=False,map_location="cpu"):
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
        self.load_state_dict(payload["model"],strict=True)
        if optimizer is not None:
            if payload.get("optimizer") is None: raise ValueError("checkpoint has no optimizer state")
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            if payload.get("scheduler") is None: raise ValueError("checkpoint has no scheduler state")
            scheduler.load_state_dict(payload["scheduler"])
        if restore_rng:
            torch.set_rng_state(payload["torch_rng"])
            if torch.cuda.is_available() and payload.get("cuda_rng"): torch.cuda.set_rng_state_all(payload["cuda_rng"])
            np.random.set_state(payload["numpy_rng"]); random.setstate(payload["python_rng"])
        return payload
