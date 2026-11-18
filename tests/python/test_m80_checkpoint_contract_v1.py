import hashlib, json
import torch
from torch import nn
import pytest

from research.NeuralMoments.m80_checkpoint_contract_v1 import (
    CHECKPOINT_FORMAT, EXPECTED_BACKBONE_CLASS, EXPECTED_MODEL_CLASS,
    build_checkpoint_payload, validate_checkpoint_payload, load_m80_checkpoint,
)


class TinyM80(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model=nn.Linear(3,2)
        self.base_model.weight.requires_grad_(False)
        self.base_model.bias.requires_grad_(False)
        self.head=nn.Linear(2,1)


def config():
    return {"conditioning_mode":"conditional_norm","task_feature_dim":31,
            "hidden_dim":16,"freeze_backbone":True}


def test_checkpoint_payload_roundtrip(tmp_path):
    model=TinyM80()
    payload=build_checkpoint_payload(model,config=config(),
        training_run_id="m80-test-run",training_commit="abc123")
    validate_checkpoint_payload(payload)
    path=tmp_path/"m80.pt"
    torch.save(payload,path)
    target=TinyM80()
    loaded=load_m80_checkpoint(path,target,
        expected_model_class=f"{TinyM80.__module__}.{TinyM80.__qualname__}")
    assert loaded.provenance.training_run_id=="m80-test-run"
    assert loaded.provenance.training_commit=="abc123"
    assert loaded.provenance.frozen_backbone
    for a,b in zip(model.parameters(),target.parameters()):
        assert torch.equal(a,b)


def test_rejects_schema_mismatch():
    model=TinyM80()
    payload=build_checkpoint_payload(model,config=config(),
        training_run_id="run",training_commit="commit")
    payload["format"]="tvcg.m80.checkpoint.v0"
    with pytest.raises(ValueError,match="format"):
        validate_checkpoint_payload(payload)


def test_rejects_stale_config_digest():
    model=TinyM80()
    payload=build_checkpoint_payload(model,config=config(),
        training_run_id="run",training_commit="commit")
    payload["config"]["hidden_dim"]=99
    with pytest.raises(ValueError,match="config SHA256"):
        validate_checkpoint_payload(payload)


def test_rejects_stale_state_digest():
    model=TinyM80()
    payload=build_checkpoint_payload(model,config=config(),
        training_run_id="run",training_commit="commit")
    key=next(iter(payload["state_dict"]))
    payload["state_dict"][key]=payload["state_dict"][key].clone()
    payload["state_dict"][key].view(-1)[0]+=1
    with pytest.raises(ValueError,match="state_dict SHA256"):
        validate_checkpoint_payload(payload)


def test_rejects_nonfrozen_backbone():
    model=TinyM80()
    model.base_model.weight.requires_grad_(True)
    payload=build_checkpoint_payload(model,config={**config(),"freeze_backbone":False},
        training_run_id="run",training_commit="commit")
    with pytest.raises(ValueError,match="freeze_backbone"):
        validate_checkpoint_payload(payload)


def test_rejects_model_class_mismatch(tmp_path):
    model=TinyM80()
    payload=build_checkpoint_payload(model,config=config(),
        training_run_id="run",training_commit="commit")
    path=tmp_path/"m80.pt"; torch.save(payload,path)
    with pytest.raises(TypeError,match="model class mismatch"):
        load_m80_checkpoint(path,nn.Linear(3,2))


def test_missing_checkpoint_is_explicit():
    with pytest.raises(FileNotFoundError):
        load_m80_checkpoint("/definitely/missing/m80.pt",TinyM80())
