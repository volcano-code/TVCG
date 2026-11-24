"""Single-response native256 Compact PINO V2, independent of UnifiedM80SQNO."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import Tensor,nn
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    PcraftNativeMomentOperator,PreparedNativeGeometry,parameter_versions,
)
from .compact_v2_response import ResponseMemory,CutAwareMomentResponse,physical_tokens,pool_features
from .masked_derivative_supervision import derivative_supervision_loss
from .unified_training_state import make_training_contract,optimizer_contract,checkpoint_training_metadata,validate_resume

SCHEMA="tvcg.compact-pino.v2"
SEMANTICS="native-rle-token-cut-aware-single-moment-response"


@dataclass(frozen=True)
class CompactV2Cache:
    geometry_id: str
    shared_cache: object
    token_cache: ResponseMemory
    owner: int
    versions: tuple
    native_intervals: int
    split_segments: int
    inference_only: bool


def _sources():
    here=Path(__file__).resolve().parent
    paths=[here/"compact_pino_v2.py",here/"compact_v2_response.py",here/"masked_derivative_supervision.py",
           here/"unified_training_state.py",here/"unified_factory.py",
           here.parents[2]/"spin_pino"/"native_pcraft.py",
           here.parents[2]/"spin_pino"/"geometry_tiles.py"]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class CompactPINOv2(nn.Module):
    operator_kind="compact_pino_v2"
    default_branch_aux_weight=0.

    def __init__(self,encoder,*,native_resolution=256,token_grid=16,width=64,heads=4,
                 query_chunk=16,architecture_metadata=None):
        super().__init__()
        if not isinstance(encoder,PcraftNativeMomentOperator) or not encoder.encoder_only:
            raise ValueError("Compact V2 requires encoder-only P-CRAFT")
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1
               for v in (native_resolution,token_grid,width,heads,query_chunk)) or token_grid>native_resolution or native_resolution%token_grid:
            raise ValueError("invalid Compact V2 dimensions")
        self.shared_encoder=encoder
        self.response=CutAwareMomentResponse(encoder.config.interval_width,width,heads)
        self.native_resolution=native_resolution
        self.token_grid=token_grid
        self.query_chunk=query_chunk
        self.architecture_metadata=dict(architecture_metadata or {})
        self.geometry_encode_calls=0
        self.query_evaluation_calls=0

    def _versions(self):return parameter_versions(self)

    @staticmethod
    def prepare_geometry(geometry):return PreparedNativeGeometry.prepare(geometry)

    def encode_geometry(self,geometry,*,chunk=None,inference=False):
        if inference and self.training:
            raise ValueError("inference=True requires eval()")
        geometry=PreparedNativeGeometry.prepare(geometry,borrow=True)
        if int(np.asarray(geometry["native_resolution"]))!=self.native_resolution:
            raise ValueError(f"exact native resolution {self.native_resolution} required")
        from contextlib import nullcontext
        with torch.no_grad() if inference else nullcontext():
            encoded=self.shared_encoder.encode(geometry,chunk,retain_interval_features=True)
            tokens=physical_tokens(geometry,self.token_grid,device=encoded.latent.device)
            torch.testing.assert_close(tokens.moments.sum(0),encoded.design_moments,
                                       rtol=1e-8,atol=1e-10)
            pool=pool_features(encoded.interval_features,encoded.interval_mass,tokens)
            memory=self.response.prepare(pool,tokens)
            # Keep the shared global metadata, but not the full native learned
            # feature table in the inference cache. Autograd owns dependencies
            # during training until the one joint update completes.
            from dataclasses import replace
            encoded=replace(encoded,interval_features=None,descriptor=None,normalized_xyz=None,interval_mass=None)
        self.geometry_encode_calls+=1
        return CompactV2Cache(encoded.geometry_sha256,encoded,memory,id(self),self._versions(),
                              encoded.native_intervals,len(tokens.inverse),bool(inference or not torch.is_grad_enabled()))

    def _check_cache(self,cache,create_graph=False):
        if cache.owner!=id(self) or cache.versions!=self._versions():
            raise ValueError("foreign or stale Compact V2 geometry cache")
        if cache.inference_only and (self.training or create_graph):
            raise ValueError("inference-only cache cannot be used for training")

    def evaluate_queries(self,cache,coefficients,*,create_graph=False,with_jacobian=True):
        self._check_cache(cache,create_graph)
        q=torch.as_tensor(coefficients,device=cache.token_cache.moments.device,dtype=torch.float64)
        single=q.ndim==1
        q=q[None] if single else q
        if q.ndim!=2 or q.shape[1]!=10 or not len(q):
            raise ValueError("coefficients must be nonempty [10] or [Q,10]")
        ms=[];js=[]
        for block in q.split(self.query_chunk):
            if with_jacobian:
                with torch.enable_grad():
                    value=block if block.requires_grad else block.detach().requires_grad_(True)
                    m=self.response(cache.token_cache,cache.shared_cache.fixed_moments,value)
                    j=torch.stack([torch.autograd.grad(m[:,i].sum(),value,create_graph=create_graph,
                                                     retain_graph=True)[0] for i in range(10)],dim=1)
                js.append(j.detach() if cache.inference_only else j)
            else:
                m=self.response(cache.token_cache,cache.shared_cache.fixed_moments,block)
            ms.append(m.detach() if cache.inference_only else m)
        result={"moments":torch.cat(ms),"geometry_id":cache.geometry_id,
                "physical_token_count":len(cache.token_cache.moments),"native_intervals":cache.native_intervals,
                "split_segments":cache.split_segments}
        if with_jacobian:
            result["jacobian_coefficients"]=torch.cat(js)
            result["jacobian_q"]=result["jacobian_coefficients"]
        if single:
            for key in ("moments","jacobian_coefficients","jacobian_q"):
                if key in result:result[key]=result[key][0]
        self.query_evaluation_calls+=1
        return result

    @torch.no_grad()
    def predict_moments(self,cache,coefficients):
        if self.training or not cache.inference_only:
            raise ValueError("predict_moments requires eval and inference-only cache")
        return self.evaluate_queries(cache,coefficients,with_jacobian=False)["moments"]

    def architecture_contract(self):
        return {"schema":SCHEMA,"semantics":SEMANTICS,"shared_encoder_config":asdict(self.shared_encoder.config),
                "native_resolution":self.native_resolution,"token_grid":self.token_grid,
                "response_width":self.response.width,"attention_heads":self.response.heads,
                "query_chunk":self.query_chunk,"metadata":self.architecture_metadata,
                "response_domain":"positive_quadric_on_editable_domain","single_response":True}

    def training_contract(self,*,gradient_mode="fixed",pcgrad_reduction="sum",jacobian_weight=.1,
                          branch_aux_weight=0.,gradient_clip=1.,moment_scale=1.,jacobian_scale=1.,restricted_scale=1.):
        if branch_aux_weight!=0:
            raise ValueError("Compact V2 has no auxiliary Moment branches; branch_aux_weight must be zero")
        return make_training_contract(self,gradient_mode=gradient_mode,pcgrad_reduction=pcgrad_reduction,
            jacobian_weight=jacobian_weight,branch_aux_weight=branch_aux_weight,gradient_clip=gradient_clip,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,restricted_scale=restricted_scale)

    def build_optimizer(self,*,learning_rate=3e-4,weight_decay=1e-4):
        if not math.isfinite(learning_rate) or learning_rate<=0 or not math.isfinite(weight_decay) or weight_decay<0:
            raise ValueError("invalid optimizer parameters")
        return torch.optim.AdamW(self.parameters(),lr=learning_rate,weight_decay=weight_decay)

    def train_step(self,optimizer,cache,coefficients,target_moments,target_jacobian=None,*,
                   moment_scale=1.,jacobian_scale=1.,restricted_scale=1.,valid_mask=None,
                   jacobian_valid_mask=None,restricted_basis=None,restricted_jacobian=None,
                   restricted_valid=None,restricted_dimension=None,jacobian_weight=.1,branch_aux_weight=0.,
                   gradient_clip=1.,gradient_mode="fixed",pcgrad_reduction="sum"):
        if not self.training or cache.inference_only:
            raise ValueError("training requires an autograd-enabled geometry cache")
        contract=self.training_contract(gradient_mode=gradient_mode,pcgrad_reduction=pcgrad_reduction,
            jacobian_weight=jacobian_weight,branch_aux_weight=branch_aux_weight,gradient_clip=gradient_clip,
            moment_scale=moment_scale,jacobian_scale=jacobian_scale,restricted_scale=restricted_scale)
        resumed=getattr(self,"_resume_training_contract",None)
        if resumed is not None and resumed!=contract:
            raise ValueError("training contract differs from resumed checkpoint")
        optimizer_contract(self,optimizer)
        optimizer.zero_grad(set_to_none=True)
        result=self.evaluate_queries(cache,coefficients,create_graph=True)
        m=result["moments"].reshape(-1,10)
        j=result["jacobian_coefficients"].reshape(-1,10,10)
        target=torch.as_tensor(target_moments,device=m.device,dtype=m.dtype).detach()
        if target.ndim==1 and len(m)==1:target=target[None]
        if target.shape!=m.shape:raise ValueError("target Moment shape mismatch")
        valid=torch.ones(len(m),device=m.device,dtype=torch.bool) if valid_mask is None else torch.as_tensor(valid_mask,device=m.device)
        if valid.shape!=(len(m),) or valid.dtype!=torch.bool or not valid.any():
            raise ValueError("invalid Moment mask")
        scale=torch.as_tensor(moment_scale,device=m.device,dtype=m.dtype).detach()
        if scale.shape not in ((),(10,)):
            raise ValueError("Moment scale must be scalar or [10]")
        ml=((m[valid]-target[valid])/scale).square().mean()
        if target_jacobian is not None and jacobian_valid_mask is None:jacobian_valid_mask=valid
        for mask in (jacobian_valid_mask,restricted_valid):
            if mask is not None and (torch.as_tensor(mask,device=m.device,dtype=torch.bool)&~valid).any():
                raise ValueError("derivative mask outside Moment validity")
        if target_jacobian is not None:
            target_jacobian=torch.as_tensor(target_jacobian,device=j.device)
            if target_jacobian.ndim==2 and len(j)==1:target_jacobian=target_jacobian[None]
        deriv=derivative_supervision_loss(j,target_jacobian,derivative_valid=jacobian_valid_mask,
            restricted_basis=restricted_basis,restricted_target=restricted_jacobian,restricted_valid=restricted_valid,
            restricted_dimension=restricted_dimension,jacobian_scale=jacobian_scale,restricted_scale=restricted_scale)
        jl=deriv["loss"]
        total=ml+jacobian_weight*jl
        if not torch.isfinite(total):raise FloatingPointError("nonfinite Compact V2 loss")
        if gradient_mode=="pcgrad":
            from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import pcgrad_assign_gradients
            multitask=pcgrad_assign_gradients({"moment":ml,"jacobian":jl},
                {"moment":1.,"jacobian":float(jacobian_weight)},list(self.parameters()),[],reduction=pcgrad_reduction)
        else:
            total.backward()
            multitask={"diagnostics_available":False}
        norm=torch.nn.utils.clip_grad_norm_(self.parameters(),gradient_clip,error_if_nonfinite=True)
        optimizer.step()
        self._last_training_contract=contract
        return {"loss":float(total.detach()),"moment":float(ml.detach()),"jacobian":float(jl.detach()),
                "full_jacobian":float(deriv["full_loss"].detach()),"restricted_jacobian":float(deriv["restricted_loss"].detach()),
                "gradient_norm":float(norm.detach()),"valid_queries":int(valid.sum()),
                "valid_jacobian_queries":int(deriv["full_count"]),"valid_restricted_queries":int(deriv["restricted_count"]),
                "restricted_dimensions":int(deriv["restricted_dimensions"]),"cache_invalidated":1.0,
                "multitask_report":multitask,"physical_token_count":len(cache.token_cache.moments)}

    def save_checkpoint(self,path,*,optimizer=None,scheduler=None,data_fingerprint=None,extra=None,training_contract=None):
        training=checkpoint_training_metadata(self,optimizer,scheduler,data_fingerprint,training_contract)
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        payload={"schema":SCHEMA,"semantics":SEMANTICS,"architecture":self.architecture_contract(),
                 "source_fingerprints":_sources(),"data_fingerprint":data_fingerprint,"model":self.state_dict(),
                 "optimizer":None if optimizer is None else optimizer.state_dict(),
                 "scheduler":None if scheduler is None else scheduler.state_dict(),
                 "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"extra":dict(extra or {}),**training}
        tmp=None
        try:
            with tempfile.NamedTemporaryFile("wb",dir=target.parent,delete=False) as stream:
                tmp=Path(stream.name);torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,target)
        finally:
            if tmp is not None:tmp.unlink(missing_ok=True)

    def load_checkpoint(self,path,*,optimizer=None,scheduler=None,expected_data_fingerprint=None,
                        restore_rng=False,map_location="cpu",expected_training_contract=None):
        payload=torch.load(path,map_location=map_location,weights_only=False)
        if payload.get("schema")!=SCHEMA or payload.get("semantics")!=SEMANTICS:
            raise ValueError("incompatible Compact V2 checkpoint")
        if payload.get("architecture")!=self.architecture_contract() or payload.get("source_fingerprints")!=_sources():
            raise ValueError("Compact V2 architecture/source fingerprint mismatch")
        if expected_data_fingerprint is not None and payload.get("data_fingerprint")!=expected_data_fingerprint:
            raise ValueError("Compact V2 data fingerprint mismatch")
        contract=validate_resume(self,payload,optimizer,scheduler,expected_training_contract,expected_data_fingerprint,restore_rng)
        self.load_state_dict(payload["model"],strict=True)
        if optimizer is not None:optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:scheduler.load_state_dict(payload["scheduler"])
        if restore_rng:
            torch.set_rng_state(payload["torch_rng"].cpu())
            if torch.cuda.is_available() and payload.get("cuda_rng"):
                torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])
            np.random.set_state(payload["numpy_rng"]);random.setstate(payload["python_rng"])
        self._resume_training_contract=contract;self._last_training_contract=contract
        return payload
