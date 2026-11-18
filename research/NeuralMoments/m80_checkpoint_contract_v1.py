"""Phase 2.6 Round 2: provenance-aware M80 checkpoint loading.

The loader is deliberately strict: it never invents a checkpoint, silently accepts
a stale schema, or treats an arbitrary state_dict as a trained M80 artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json, os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

CHECKPOINT_FORMAT = "tvcg.m80.checkpoint.v1"
EXPECTED_MODEL_CLASS = "research.NeuralMoments.m69_task_conditioned_m80_v1.M69TaskConditionedM80"
EXPECTED_BACKBONE_CLASS = "research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft.PcraftNativeMomentOperator"


@dataclass(frozen=True)
class M80CheckpointProvenance:
    path: str
    sha256: str
    checkpoint_format: str
    model_class: str
    backbone_class: str
    training_run_id: str
    training_commit: str
    config_sha256: str
    state_dict_sha256: str
    frozen_backbone: bool


@dataclass(frozen=True)
class LoadedM80Checkpoint:
    provenance: M80CheckpointProvenance
    payload: Mapping[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json_sha256(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return _sha256_bytes(raw)


def _state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    h=hashlib.sha256()
    for key in sorted(state_dict):
        value=state_dict[key]
        if not isinstance(value,torch.Tensor):
            raise ValueError(f"state_dict[{key!r}] is not a Tensor")
        h.update(key.encode()); h.update(b"\0")
        h.update(str(value.dtype).encode()); h.update(b"\0")
        h.update(json.dumps(list(value.shape),separators=(",",":")).encode()); h.update(b"\0")
        h.update(value.detach().cpu().contiguous().numpy().tobytes()); h.update(b"\0")
    return h.hexdigest()


def validate_checkpoint_payload(payload: Mapping[str, Any], *,
                                expected_model_class: str=EXPECTED_MODEL_CLASS,
                                expected_backbone_class: str=EXPECTED_BACKBONE_CLASS,
                                require_frozen_backbone: bool=True) -> None:
    required={"format","model_class","backbone_class","config","state_dict","provenance"}
    missing=sorted(required-set(payload))
    if missing: raise ValueError("checkpoint missing fields: "+",".join(missing))
    if payload["format"]!=CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload['format']!r}")
    if payload["model_class"]!=expected_model_class:
        raise ValueError("checkpoint model_class mismatch")
    if payload["backbone_class"]!=expected_backbone_class:
        raise ValueError("checkpoint backbone_class mismatch")
    config=payload["config"]
    if not isinstance(config,Mapping): raise ValueError("checkpoint config must be a mapping")
    for key in ("conditioning_mode","task_feature_dim","hidden_dim","freeze_backbone"):
        if key not in config: raise ValueError("checkpoint config missing "+key)
    if int(config["task_feature_dim"])!=31:
        raise ValueError("checkpoint task_feature_dim must be 31")
    if bool(config["freeze_backbone"]) is not require_frozen_backbone:
        raise ValueError("checkpoint freeze_backbone policy mismatch")
    prov=payload["provenance"]
    if not isinstance(prov,Mapping): raise ValueError("checkpoint provenance must be a mapping")
    for key in ("training_run_id","training_commit","config_sha256","state_dict_sha256"):
        if not str(prov.get(key,"")).strip():
            raise ValueError("checkpoint provenance missing "+key)
    actual_cfg=_stable_json_sha256(config)
    if prov["config_sha256"]!=actual_cfg:
        raise ValueError("checkpoint config SHA256 mismatch")
    sd=payload["state_dict"]
    if not isinstance(sd,Mapping): raise ValueError("checkpoint state_dict must be a mapping")
    if prov["state_dict_sha256"]!=_state_dict_digest(sd):
        raise ValueError("checkpoint state_dict SHA256 mismatch")


def load_m80_checkpoint(path: str|os.PathLike[str], model: nn.Module, *,
                        map_location: str|torch.device="cpu",
                        expected_model_class: str=EXPECTED_MODEL_CLASS,
                        expected_backbone_class: str=EXPECTED_BACKBONE_CLASS,
                        require_frozen_backbone: bool=True,
                        strict: bool=True) -> LoadedM80Checkpoint:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(str(p))
    raw=p.read_bytes()
    payload=torch.load(p,map_location=map_location,weights_only=False)
    if not isinstance(payload,Mapping): raise ValueError("checkpoint root must be a mapping")
    validate_checkpoint_payload(payload,expected_model_class=expected_model_class,
                                expected_backbone_class=expected_backbone_class,
                                require_frozen_backbone=require_frozen_backbone)
    expected_digest=str(payload["provenance"].get("file_sha256",""))
    if expected_digest and expected_digest!=_sha256_bytes(raw):
        raise ValueError("checkpoint file SHA256 mismatch")
    actual_class=f"{model.__class__.__module__}.{model.__class__.__qualname__}"
    if actual_class!=expected_model_class:
        raise TypeError(f"model class mismatch: {actual_class!r}")
    incompatible=model.load_state_dict(payload["state_dict"],strict=strict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("checkpoint state_dict/model key mismatch")
    frozen=True
    if hasattr(model,"base_model"):
        frozen=all(not p.requires_grad for p in model.base_model.parameters())
    if require_frozen_backbone and not frozen:
        raise ValueError("loaded model backbone is not frozen")
    prov=payload["provenance"]
    provenance=M80CheckpointProvenance(
        str(p),_sha256_bytes(raw),payload["format"],payload["model_class"],
        payload["backbone_class"],str(prov["training_run_id"]),
        str(prov["training_commit"]),str(prov["config_sha256"]),
        str(prov["state_dict_sha256"]),frozen
    )
    return LoadedM80Checkpoint(provenance,payload)


def build_checkpoint_payload(model: nn.Module, *, config: Mapping[str,Any],
                             training_run_id: str, training_commit: str,
                             backbone_class: str=EXPECTED_BACKBONE_CLASS,
                             model_class: str=EXPECTED_MODEL_CLASS) -> dict[str,Any]:
    if not training_run_id.strip() or not training_commit.strip():
        raise ValueError("training_run_id and training_commit are required")
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    config=dict(config)
    prov={"training_run_id":training_run_id,"training_commit":training_commit,
          "config_sha256":_stable_json_sha256(config),
          "state_dict_sha256":_state_dict_digest(state)}
    return {"format":CHECKPOINT_FORMAT,"model_class":model_class,
            "backbone_class":backbone_class,"config":config,
            "state_dict":state,"provenance":prov}
