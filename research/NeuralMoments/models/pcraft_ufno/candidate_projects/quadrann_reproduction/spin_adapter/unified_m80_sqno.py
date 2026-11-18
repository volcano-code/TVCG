"""Unified M80 + SQNO with one shared P-CRAFT encoder (Phase 2.8.4)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
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

def _source_fingerprints():
    here=Path(__file__).resolve().parent
    pcraft_root=here.parents[2]/"spin_pino"
    files={
      "native_pcraft.py":pcraft_root/"native_pcraft.py",
      "spin_quadrann_v11.py":here/"spin_quadrann_v11.py",
      "spin_quadrann_v12.py":here/"spin_quadrann_v12.py",
      "batched_moment_operator.py":here/"batched_moment_operator.py",
      "unified_m80_sqno.py":here/"unified_m80_sqno.py",
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
                 *,fusion_initial_weight=.5,architecture_metadata:Mapping[str,Any]|None=None):
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
        logit=np.log(fusion_initial_weight/(1-fusion_initial_weight))
        self.fusion_logit=nn.Parameter(torch.tensor(float(logit),dtype=torch.float64))
        self.architecture_metadata=dict(architecture_metadata or {})
        self.geometry_encode_calls=0
        self.query_evaluation_calls=0

    @property
    def fusion_weight(self): return torch.sigmoid(self.fusion_logit)

    def _versions(self):
        return tuple(p._version for p in self.parameters())

    def encode_geometry(self,geometry:Mapping[str,Any],*,chunk:int|None=None):
        if "xyz" not in geometry:
            from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import geometry_from_columns
            geometry=geometry_from_columns(geometry)
        shared=self.shared_encoder.encode(geometry,chunk,retain_interval_features=True)
        payload=self.shared_encoder.export_shared_geometry_features(shared)
        sqno_cache=self.sqno.encode_geometry(geometry,pcraft_features=payload)
        self.geometry_encode_calls+=1
        return UnifiedGeometryCache(shared.geometry_sha256,shared,sqno_cache,id(self),
                                     self._versions(),shared.native_intervals)

    def evaluate_queries(self,cache:UnifiedGeometryCache,coefficients:Tensor,*,
                         create_graph=False,with_jacobian=True):
        if cache.owner!=id(self) or cache.versions!=self._versions():
            raise ValueError("foreign or stale unified geometry cache")
        if not with_jacobian: raise ValueError("unified contract currently requires Moment Jacobians")
        q=torch.as_tensor(coefficients,device=cache.shared_cache.latent.device,dtype=torch.float64)
        single=q.ndim==1
        qb=q.unsqueeze(0) if single else q
        if qb.ndim!=2 or qb.shape[1]!=10: raise ValueError("coefficients must be [10] or [Q,10]")
        if not torch.isfinite(qb).all() or (qb.abs().amax(-1)<=0).any():
            raise ValueError("coefficients must be finite and nonzero")
        m80,j80=self.shared_encoder.moments_and_jacobian(cache.shared_cache,qb,create_graph=create_graph)
        sq=self.sqno.evaluate_queries(cache.sqno_cache,qb,with_jacobian=True)
        ms,js=sq["moments"],sq["jacobian_q"]
        if m80.shape!=ms.shape or j80.shape!=js.shape: raise RuntimeError("M80/SQNO output shape mismatch")
        alpha=self.fusion_weight.to(device=m80.device,dtype=m80.dtype)
        result={
          "moments":alpha*m80+(1-alpha)*ms,
          "jacobian_q":alpha*j80+(1-alpha)*js,
          "m80_moments":m80,"m80_jacobian_q":j80,
          "sqno_moments":ms,"sqno_jacobian_q":js,
          "fusion_weight":alpha,"geometry_id":cache.geometry_id,
          "native_intervals":torch.as_tensor(cache.native_intervals,device=m80.device),
          "mass_error":sq["total_mass_error"],
        }
        self.query_evaluation_calls+=1
        if single:
            for key in ("moments","jacobian_q","m80_moments","m80_jacobian_q","sqno_moments","sqno_jacobian_q"):
                result[key]=result[key][0]
        return result

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
        if payload.get("architecture")!=self.architecture_contract():
            raise ValueError("unified checkpoint architecture contract mismatch")
        if payload.get("source_fingerprints")!=_source_fingerprints():
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
