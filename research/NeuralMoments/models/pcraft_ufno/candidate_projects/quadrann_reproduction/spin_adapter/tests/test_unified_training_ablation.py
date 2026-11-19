from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_training_ablation import (
    main, run_training_ablation, synthetic_training_case,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_m80_sqno import UnifiedM80SQNO


def _rng_snapshot():
    return torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()), random.getstate()


def _assert_rng_equal(saved):
    assert torch.equal(torch.get_rng_state(), saved[0])
    current = np.random.get_state()
    assert current[0] == saved[1][0]
    np.testing.assert_array_equal(current[1], saved[1][1])
    assert current[2:] == saved[1][2:]
    assert random.getstate() == saved[2]


def _without_timing(report):
    result = copy.deepcopy(report)
    for arm in result["arms"].values():
        arm.pop("training_seconds")
        for step in arm["steps"]:
            step.pop("seconds")
    return result


def test_training_ablation_pairs_initial_state_batches_and_budget_with_real_updates():
    model, batches = synthetic_training_case()
    second = copy.deepcopy(batches[0])
    second["target_moments"].fill_(.01)
    second["valid_mask"] = torch.tensor([True, False, True, False])
    batches.append(second)
    report = run_training_ablation(model, batches, steps=3, scope="synthetic_smoke")
    fairness = report["fairness_contract"]
    fixed, pcgrad = report["arms"]["fixed"], report["arms"]["pcgrad"]
    assert fairness["batch_schedule"] == [0, 1, 0]
    assert len(set(fairness["batch_sha256"])) == 2
    assert fixed["initial_state_sha256"] == pcgrad["initial_state_sha256"]
    assert fixed["input_sha256"] == pcgrad["input_sha256"]
    assert fixed["initial_evaluation"] == pcgrad["initial_evaluation"]
    assert fairness["gradient_reduction"] == "sum"
    assert fairness["compute_budget_matched"] is False
    for arm in (fixed, pcgrad):
        assert arm["optimizer_steps"] == 3
        assert arm["training_geometry_encode_calls"] == 3
        assert arm["final_state_sha256"] != arm["initial_state_sha256"]
        delta = arm["parameter_delta_norms"]
        assert delta["real_l2"] > 0
        assert delta["imaginary_l2"] > 0
        assert delta["by_top_level_module"]["shared_encoder"] > 0
        assert delta["by_top_level_module"]["sqno"] > 0
        assert delta["by_top_level_module"]["fusion_logit"] > 0
        assert [row["batch_sha256"] for row in arm["steps"]] == [
            fairness["batch_sha256"][index] for index in (0, 1, 0)]
        assert [row["seed"] for row in arm["steps"]] == fairness["step_seeds"]
        initial = arm["initial_evaluation"]["per_batch"][0]
        metrics = arm["steps"][0]["metrics"]
        # This checks that evaluation uses the actual masked/normalized training objective.
        for key in ("loss", "moment", "jacobian", "branch_aux"):
            assert metrics[key] == pytest.approx(initial[key], rel=1e-8, abs=1e-10)
        assert metrics["valid_queries"] == 3
        assert metrics["valid_jacobian_queries"] == 2
    task_report = pcgrad["steps"][0]["metrics"]["multitask_report"]
    assert task_report["task_names"] == ["moment", "jacobian", "branch_aux"]
    assert len(task_report["cosine_matrix"]) == 3
    assert len(task_report["projection_counts"]) == 3
    assert set(task_report["shared_gradient_norms"]) == set(task_report["task_names"])
    assert report["claims"]["pcgrad_superiority_established"] is False
    assert report["claims"]["formal_accuracy_evidence"] is False
    json.dumps(report, allow_nan=False)


def test_training_ablation_preserves_caller_parameters_grads_modes_inputs_and_rng():
    model, batches = synthetic_training_case()
    model.train()
    model.sqno.eval()
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    batches[0]["coefficients"].requires_grad_(True)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_grad = first.grad.clone()
    modes = {name: module.training for name, module in model.named_modules()}
    counters = copy.deepcopy(model.shared_encoder.counters)
    inputs = copy.deepcopy(batches)
    resumed_contract = model.training_contract(gradient_mode="fixed")
    model._resume_training_contract = copy.deepcopy(resumed_contract)
    model._last_training_contract = copy.deepcopy(resumed_contract)
    scale = torch.ones(10, requires_grad=True)
    rng = _rng_snapshot()
    run_training_ablation(model, batches, steps=1, moment_scale=scale)
    _assert_rng_equal(rng)
    assert model.geometry_encode_calls == 0
    assert model.query_evaluation_calls == 0
    assert model.shared_encoder.counters == counters
    assert model._resume_training_contract == resumed_contract
    assert model._last_training_contract == resumed_contract
    assert {name: module.training for name, module in model.named_modules()} == modes
    assert all(torch.equal(value, initial_state[key]) for key, value in model.state_dict().items())
    assert torch.equal(first.grad, initial_grad)
    for key in ("coefficients", "target_moments", "target_jacobian", "valid_mask", "jacobian_valid_mask"):
        assert torch.equal(batches[0][key], inputs[0][key])
    assert batches[0]["coefficients"].requires_grad
    assert batches[0]["coefficients"].grad is None
    assert scale.requires_grad and scale.grad is None
    assert torch.equal(scale, torch.ones(10))
    for key in inputs[0]["geometry"]:
        np.testing.assert_equal(batches[0]["geometry"][key], inputs[0]["geometry"][key])


def test_training_ablation_replays_deterministically_and_records_normalization():
    model, batches = synthetic_training_case()
    kwargs = dict(steps=2, seed=123, moment_scale=torch.linspace(.7, 1.3, 10),
                  jacobian_scale=torch.full((10, 10), 1.5), jacobian_weight=.2,
                  branch_aux_weight=.3)
    one = run_training_ablation(model, batches, **kwargs)
    two = run_training_ablation(model, batches, **kwargs)
    assert _without_timing(one) == _without_timing(two)
    changed = run_training_ablation(model, batches, steps=1, seed=123, moment_scale=2.)
    assert one["fairness_contract"]["input_sha256"] != changed["fairness_contract"]["input_sha256"]
    for arm in one["arms"].values():
        expected = arm["initial_evaluation"]["per_batch"][0]
        assert arm["steps"][0]["metrics"]["loss"] == pytest.approx(expected["loss"], rel=1e-8)


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"steps": -1}, {"steps": 1.5}, {"steps": True}, {"steps": 101},
    {"seed": -1}, {"seed": True}, {"learning_rate": 0}, {"weight_decay": -1},
    {"jacobian_weight": float("nan")}, {"branch_aux_weight": float("inf")},
    {"gradient_clip": 0}, {"scope": "formal_performance"},
    {"moment_scale": -1}, {"jacobian_scale": torch.zeros(10, 10)},
])
def test_training_ablation_rejects_invalid_protocol_before_training(kwargs):
    model, batches = synthetic_training_case()
    rng = _rng_snapshot()
    with pytest.raises(ValueError):
        run_training_ablation(model, batches, **kwargs)
    _assert_rng_equal(rng)
    assert model.geometry_encode_calls == 0


def test_training_ablation_rejects_invalid_batches_and_restores_rng_after_failure(monkeypatch):
    model, batches = synthetic_training_case()
    with pytest.raises(ValueError, match="nonempty"):
        run_training_ablation(model, [], steps=1)
    bad = copy.deepcopy(batches)
    bad[0]["valid_mask"].zero_()
    with pytest.raises(ValueError, match="at least one valid"):
        run_training_ablation(model, bad, steps=1)
    bad = copy.deepcopy(batches)
    bad[0]["target_jacobian"] = torch.zeros(4, 10)
    with pytest.raises(ValueError, match="target_jacobian"):
        run_training_ablation(model, bad, steps=1)

    def fail(*args, **kwargs):
        torch.rand(3)
        np.random.rand(3)
        random.random()
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(UnifiedM80SQNO, "train_step", fail)
    rng = _rng_snapshot()
    with pytest.raises(RuntimeError, match="injected training failure"):
        run_training_ablation(model, batches, steps=1)
    _assert_rng_equal(rng)
    assert model.geometry_encode_calls == 0


def test_training_ablation_cli_writes_finite_explicitly_synthetic_report(tmp_path):
    output = tmp_path / "training-smoke.json"
    assert main(["--steps", "1", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["scope"] == "synthetic_smoke"
    assert report["synthetic_fixture"]["targets"] == "artificial_zero_targets"
    assert set(report["arms"]) == {"fixed", "pcgrad"}
    assert all(arm["optimizer_steps"] == 1 for arm in report["arms"].values())
    assert list(tmp_path.glob("*.tmp")) == []
