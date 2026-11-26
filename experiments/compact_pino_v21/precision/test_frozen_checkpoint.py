"""Small model tests of identity, paired routing and frozen evaluation only.

The mock is not a physical predictor. Actual full-operator inference belongs to
the separate synthetic runner smoke; these checks isolate orchestration.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import (
    load_baseline_case, sha256, write_synthetic_finite_case,
)
from experiments.compact_pino_v21.precision import frozen_checkpoint as checkpoint_module
from experiments.compact_pino_v21.precision.compare_rules import compare_records, main
from experiments.compact_pino_v21.training.admission import canonical_digest
from experiments.compact_pino_v21.training.runner import CHECKPOINT_SCHEMA, TrainConfig


class TinyOperator(torch.nn.Module):
    def __init__(self, *, full_cell_rule, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.125]))
        self.full_cell_rule = full_cell_rule
        self.encoding_calls, self.calls, self.clear_calls = 0, [], 0

    def encode_geometry(self, blocks, fixed_moments, *, density, inference):
        assert not self.training and inference and not torch.is_grad_enabled()
        self.encoding_calls += 1
        return SimpleNamespace(inference_only=True, table=self.weight.detach().clone(),
                               projected_context=None, blocks=blocks, fixed=np.array(fixed_moments))

    def evaluate_operand(self, encoded, operand, *, with_jacobian, bandwidth, chunk_size):
        assert not self.training and not torch.is_grad_enabled() and not self.weight.requires_grad
        self.calls.append((self.full_cell_rule, id(encoded), id(operand), bandwidth, with_jacobian))
        code = {"tetra4": 1., "gl2": 2., "gl3": 3.}[self.full_cell_rule]
        full = encoded.fixed+encoded.blocks.moments.sum(0)
        moments = torch.tensor(full, dtype=torch.float64)+self.weight.detach().double()*code*.001
        dimension = operand.vertex_velocity.shape[-1] if with_jacobian else None
        jacobian = None if dimension is None else torch.full((10, dimension), code*.01, dtype=torch.float64)
        return SimpleNamespace(moments=moments, jacobian=jacobian,
                               diagnostics=dict(band_nodes=int(code*4), active_nodes=int(code*4),
                                                scope="mock_routing_only"))

    def efficiency_statistics(self):
        return dict(scope="mock_routing_only", encoding_calls=self.encoding_calls, forwards=len(self.calls))

    def clear_efficiency_caches(self):
        self.clear_calls += 1


@pytest.fixture(scope="module")
def source_case(tmp_path_factory):
    path = write_synthetic_finite_case(tmp_path_factory.mktemp("frozen-routing")/"case")
    return path, load_baseline_case(path)


def _write_checkpoint(path, *, mutation=None):
    config = asdict(TrainConfig(token_grid=8, evaluation_cache_bytes=1024))
    state = dict(schema=CHECKPOINT_SCHEMA, config=config, config_sha256=canonical_digest(config),
                 model=TinyOperator(full_cell_rule="tetra4").state_dict(),
                 run_id="mock-software-only", global_step=104, attempts=104,
                 training_manifest_sha256="a"*64)
    if mutation is not None:
        mutation(state)
    torch.save(state, path)
    return path, sha256(path)


def test_checkpoint_identity_and_strict_format_are_required(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "CompactPINOv21", TinyOperator)
    path, digest = _write_checkpoint(tmp_path/"good.pt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checkpoint_module.load_frozen_checkpoint(path, "f"*64)
    with pytest.raises(ValueError, match="explicit lowercase SHA256"):
        checkpoint_module.load_frozen_checkpoint(path, None)
    with pytest.raises(FileNotFoundError):
        checkpoint_module.load_frozen_checkpoint(tmp_path/"missing.pt", digest)
    mutations = [
        (lambda s: s.update(schema="private_weights"), "CHECKPOINT_SCHEMA"),
        (lambda s: s["model"].update(unexpected=torch.ones(1)), "strictly match"),
        (lambda s: s["config"].update(token_grid=16), "config SHA256 mismatch"),
        (lambda s: s["model"]["weight"].fill_(float("nan")), "finite tensors"),
    ]
    for index, (mutation, reason) in enumerate(mutations):
        bad, bad_digest = _write_checkpoint(tmp_path/f"bad-{index}.pt", mutation=mutation)
        with pytest.raises(ValueError, match=reason):
            checkpoint_module.load_frozen_checkpoint(bad, bad_digest)


def test_frozen_rules_share_encoding_and_metrics_without_parameter_updates(tmp_path, monkeypatch, source_case):
    monkeypatch.setattr(checkpoint_module, "CompactPINOv21", TinyOperator)
    path, digest = _write_checkpoint(tmp_path/"model.pt")
    frozen = checkpoint_module.load_frozen_checkpoint(path, digest)
    case = source_case[1]
    saved = {name: value.clone() for name, value in frozen.model.state_dict().items()}
    # Repeated evaluation of the same already-qualified case isolates cache
    # reuse. A real manifest still forbids duplicate registered query identity.
    records = [dict(record_id=f"routing-repeat-{i}", role="train", case=case) for i in range(2)]
    report = compare_records(records, frozen_model=frozen, repeats=2, warmup=1)
    assert report["frozen_model"]["checkpoint"]["sha256"] == digest
    assert report["frozen_model"]["encoder_calls"] == 1
    assert report["frozen_model"]["geometry_features"]["hits"] == 1
    assert report["frozen_model"]["parameter_and_buffer_versions_unchanged"] is True
    assert report["parameter_updates"] == 0 and report["model_weights_used"] is True
    assert frozen.model.full_cell_rule == "tetra4"
    assert {call[1] for call in frozen.model.calls}.__len__() == 1
    assert [call[0] for call in frozen.model.calls] == [rule for _ in range(2) for rule in ("tetra4", "gl2", "gl3") for _ in range(3)]
    for item in report["cases"]:
        assert [r["full_cell_rule"] for r in item["records"]] == [r["full_cell_rule"] for r in item["learned_rules"]]
        for baseline, learned in zip(item["records"], item["learned_rules"]):
            assert baseline["model_weights_used"] is False and learned["model_weights_used"] is True
            assert learned["checkpoint_sha256"] == digest
            assert baseline["h"] == learned["h"] == case["blocks"].pitch
            assert baseline["moment_error"].keys() == learned["moment_error"].keys()
            assert baseline["derivative_error"].keys() == learned["derivative_error"].keys()
            assert baseline["physical_residual_error"]["availability"] == learned["physical_residual_error"]["availability"] == "UNAVAILABLE_PROFILE"
            assert learned["derivative_error"]["dimension"] == 3
            assert len(learned["query_seconds"]) == 2
    assert report["cases"][0]["learned_geometry_encoding"]["cache_hit"] is False
    assert report["cases"][1]["learned_geometry_encoding"]["cache_hit"] is True
    for name, value in frozen.model.state_dict().items():
        assert torch.equal(value, saved[name])
    assert all(not p.requires_grad and p.grad is None for p in frozen.model.parameters())
    frozen.close()


def test_checkpoint_cli_requires_hash_and_retains_both_result_sets(tmp_path, monkeypatch, source_case):
    monkeypatch.setattr(checkpoint_module, "CompactPINOv21", TinyOperator)
    path, digest = _write_checkpoint(tmp_path/"model.pt")
    case_path = source_case[0]
    output = tmp_path/"paired.json"
    with pytest.raises(SystemExit):
        main(["--case", str(case_path), "--checkpoint", str(path), "--output", str(output)])
    assert not output.exists()
    assert main(["--case", str(case_path), "--checkpoint", str(path), "--checkpoint-sha256", digest,
                 "--device", "cpu", "--rules", "gl2", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["schema"] == "tvcg.v22.common-query-cubature.v1"
    assert report["frozen_model"]["checkpoint"]["strict_model_state"] is True
    assert len(report["cases"][0]["records"]) == len(report["cases"][0]["learned_rules"]) == 1
    assert report["native_solver_acceptance"] == "NOT_EVALUATED"
    assert report["original_five_percent_gate"] == "UNCHANGED_NOT_EVALUATED"
