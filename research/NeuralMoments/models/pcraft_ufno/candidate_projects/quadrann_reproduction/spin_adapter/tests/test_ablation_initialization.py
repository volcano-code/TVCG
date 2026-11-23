import json

import numpy as np
import pytest
import torch
from torch import nn

from spin_adapter import ablation_initialization as initialization
from spin_adapter.unified_factory import build_model
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13 import train_unified


class TinyModel(nn.Module):
    def __init__(self, optional=True):
        super().__init__()
        if optional:
            self.optional = nn.Linear(3, 7)
        self.shared = nn.Linear(3, 3)
        self.zero_head = nn.Linear(3, 2)
        nn.init.zeros_(self.zero_head.weight)
        nn.init.zeros_(self.zero_head.bias)
        self.register_buffer("complex_state", torch.tensor([1 + 2j, 3 - 4j]))

    def architecture_contract(self):
        return {"schema": "synthetic-test", "optional": hasattr(self, "optional")}

    def build_optimizer(self, **kwargs):
        return torch.optim.AdamW(self.parameters(), lr=kwargs["learning_rate"],
                                 weight_decay=kwargs["weight_decay"])

    def training_contract(self, **kwargs):
        return {"schema": "synthetic-test-training"}

    def save_checkpoint(self, path, **kwargs):
        # The training integration test inspects the run record; checkpoint
        # serialization already has separate real-model regression coverage.
        self.last_checkpoint_extra = kwargs["extra"]


def test_reference_aligns_shared_state_and_preserves_constructor_zero_heads_and_rng(monkeypatch):
    monkeypatch.setattr(initialization, "build_model", lambda *args, **kwargs: TinyModel())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        reference = TinyModel()
        torch.manual_seed(71)
        target = TinyModel(optional=False)
        assert not torch.equal(target.shared.weight, reference.shared.weight)
        before_rng = torch.get_rng_state().clone()
        report = initialization.initialize_from_reference(target, seed=71)
        assert torch.equal(before_rng, torch.get_rng_state())
    for name, value in target.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
    assert torch.count_nonzero(target.zero_head.weight) == 0
    assert report["reference_only_keys"] == ["optional.bias", "optional.weight"]
    assert report["complete_target_coverage"]
    assert report["reference_state_sha256"] == initialization.state_fingerprint(reference.state_dict())
    assert report["target_architecture_manifest"]["optional"] is False
    assert not report["checkpoint_loaded"] and not report["optimizer_copied"]


def test_incompatible_reference_is_rejected_before_any_target_copy(monkeypatch):
    monkeypatch.setattr(initialization, "build_model", lambda *args, **kwargs: TinyModel())
    target = TinyModel(optional=False)
    target.shared = nn.Linear(3, 4)
    target.register_buffer("target_only", torch.ones(1))
    before = initialization.state_fingerprint(target.state_dict())
    with pytest.raises(ValueError, match="complete target state") as error:
        initialization.initialize_from_reference(target)
    assert "target_only" in str(error.value) and "shared.weight" in str(error.value)
    assert initialization.state_fingerprint(target.state_dict()) == before
    report = initialization.initialize_from_reference(target, require_complete=False)
    assert report["missing_keys"] == ["target_only"]
    assert {row["key"] for row in report["mismatched_keys"]} == {"shared.weight", "shared.bias"}
    assert not report["complete_target_coverage"]


@pytest.fixture(scope="module")
def full_reference_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(830251)
        model = build_model("full_v1_candidate", device="cpu")
    return model.state_dict()


@pytest.mark.parametrize("name", [
    "ablation_g0_t0", "ablation_g1_t0", "ablation_g0_t1", "ablation_g1_t1",
    "ablation_no_cross_scale", "ablation_two_scales", "ablation_one_scale", "compact_v1_candidate",
])
def test_all_real_candidate_states_are_exact_subsets_of_common_reference(name, full_reference_state):
    model = build_model(name, device="cpu")
    report = initialization.initialize_from_reference(model, seed=830251)
    assert report["complete_target_coverage"]
    assert not report["missing_keys"] and not report["mismatched_keys"]
    assert report["reference_state_sha256"] == initialization.state_fingerprint(full_reference_state)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, full_reference_state[key]), key
    assert torch.count_nonzero(model.geometry_physics_attention.output[-1].weight) == 0
    assert torch.count_nonzero(model.query_fusion_gate[-1].weight) == 0
    assert report["target_architecture_manifest"] == model.architecture_contract()


def test_resume_and_reference_are_rejected_before_creating_output(tmp_path):
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="cannot be combined with resume"):
        train_unified.train(tmp_path / "missing_manifest.json", output, resume="checkpoint.pt",
                            initialization_reference="full_v1_candidate")
    assert not output.exists()


@pytest.mark.parametrize("resumed", [False, True])
def test_training_records_fresh_initialization_and_preserves_resume_provenance(tmp_path, monkeypatch, resumed):
    class Corpus:
        manifest_sha256 = "a" * 64
        splits = {"train": [{"id": "synthetic"}], "val": [], "test": []}

        def __init__(self, *args, **kwargs):
            pass

        def load_labels(self, row):
            return {"valid_mask": np.ones(2, dtype=bool), "moments": np.ones((2, 10))}

        def split_summary(self):
            return {"train": {"records": 1}}

    # train_unified imports the full namespace while this test also exercises
    # the supported legacy spin_adapter namespace; patch its actual helper.
    helper_module = __import__(train_unified.initialize_from_reference.__module__, fromlist=["build_model"])
    monkeypatch.setattr(helper_module, "build_model", lambda *args, **kwargs: TinyModel())
    prior_initialization = {"reference_config_name": "full_v1_candidate", "reference_seed": 71,
                            "reference_state_sha256": "b" * 64}

    def target_model(*args, **kwargs):
        model = TinyModel(optional=False)
        if resumed:
            model.load_checkpoint = lambda *args, **kwargs: {"extra": {
                "epoch": 0, "global_step": 7, "config": {"initialization": prior_initialization}}}
        return model

    monkeypatch.setattr(train_unified, "build_model", target_model)
    monkeypatch.setattr(train_unified, "UnifiedOperatorCorpus", Corpus)
    monkeypatch.setattr(train_unified, "_load_training_batch", lambda *args: ({"geometry": {}}, {}))
    monkeypatch.setattr(train_unified, "_evaluate_split", lambda *args: {"moment_normalized_rmse": .25})
    captured = {}

    def update(model, optimizer, *args, **kwargs):
        captured["state"] = initialization.state_fingerprint(model.state_dict())
        captured["rng"] = torch.get_rng_state().clone()
        captured["model"] = model
        loss = sum(p.square().sum() for p in model.parameters())
        loss.backward()
        optimizer.step()
        return object(), {"loss": 1., "moment": .5, "jacobian": .1}

    monkeypatch.setattr(train_unified, "_train_geometry", update)
    output = tmp_path / "run"
    result = train_unified.train("synthetic_manifest", output, device="cpu", epochs=1, seed=81,
                                 resume="synthetic_checkpoint" if resumed else None,
                                 initialization_reference=None if resumed else "full_v1_candidate")
    record = json.loads((output / "training_config.json").read_text())
    assert result["global_step"] == (8 if resumed else 1)
    if resumed:
        assert record["initialization"] == prior_initialization
    else:
        assert record["initialization"]["target_initial_state_sha256"] == captured["state"]
        assert record["initialization"]["training_rng_reset_to_seed"] == 81
        assert torch.equal(captured["rng"], torch.Generator().manual_seed(81).get_state())
    assert record["architecture_manifest"] == {"schema": "synthetic-test", "optional": False}
    assert captured["model"].last_checkpoint_extra["config"] == record
