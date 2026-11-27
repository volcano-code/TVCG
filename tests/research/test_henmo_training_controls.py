"""No-update probes must not perturb the next actual optimizer update."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.compact_pino_v21.training.state import capture_rng
from experiments.henmo_v1plus.dataset import dataset_binding, model_contract
from experiments.henmo_v1plus.fixtures import synthetic_dataset
from experiments.henmo_v1plus.training import HENMOTrainer, HENMOTrainingConfig


@dataclass(frozen=True)
class ProbeConfig:
    execution: str = "adaptive"
    training_execution: str = "coarse"
    training_coarse_grids: tuple[int, ...] = (2, 4)
    coarse_grid: int = 2
    leaf_grid: int = 4


class ProbeFixture(nn.Module):
    """Stateful prediction adapter exercising the actual trainer contracts."""
    def __init__(self):
        super().__init__()
        self.config = ProbeConfig()
        self.theta = nn.Parameter(torch.tensor(.27, dtype=torch.float64))
        self.frozen_child = nn.Dropout(.1)
        self.options = []
        self.fail_diagnostic = False

    def invalidate_caches(self):
        pass

    def encode_geometry(self, *args, **kwargs):
        return object()

    def evaluate_operand(self, features, operand, *, with_jacobian, chunk_size,
                         parameter_scales=None, coarse_grid=None):
        self.options.append((self.training, self.config.execution, coarse_grid))
        # Deliberately consumes all three RNG streams even in eval mode to test
        # isolation from the next real update, including custom adapter behavior.
        noise = .01 * (random.random() + np.random.random() + torch.rand((), dtype=torch.float64))
        phi = torch.linspace(.1, 1., 10, dtype=torch.float64)
        moments = (self.theta + noise) * phi
        jacobian = self.theta * phi[:, None].expand(10, 3) if with_jacobian else None
        diagnostics = dict(execution=self.config.training_execution if self.training else self.config.execution,
                           coarse_grid=self.config.coarse_grid if coarse_grid is None else coarse_grid,
                           solver_status="NATIVE_REQUIRED_REFINEMENT",
                           nested={"blend_alpha": np.float64(.4), "values": torch.tensor([.2, .8])})
        if self.fail_diagnostic:
            diagnostics["invalid"] = float("nan")
        return SimpleNamespace(moments=moments, jacobian=jacobian, diagnostics=diagnostics)


def data_with_development():
    data = synthetic_dataset()
    train = data.train
    original = train[0]
    case = dict(original["case"])
    case["meta"] = dict(case["meta"], source_group="held-out-fixture", geometry_id="held-out-cube",
                        geometry_sha256="f" * 64)
    development = dict(original, case=case, role="development", record_id="dev-only", case_sha256="a" * 64)
    data.development = [development]
    data.records = [*train, development]
    return data


def make_trainer():
    data = data_with_development()
    model = ProbeFixture()
    trainer = HENMOTrainer(model, HENMOTrainingConfig(max_steps=3, max_attempts=6),
                          dataset_binding(data, model_contract(model)), records=data.train)
    return trainer, data


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_evaluation_retains_actual_outputs_diagnostics_and_preserves_rng_modes():
    trainer, data = make_trainer()
    model = trainer.model
    model.train()
    model.frozen_child.eval()
    model.theta.grad = torch.tensor(.73, dtype=torch.float64)
    before_rng = capture_rng()
    before_parameter = model.theta.detach().clone()
    first = trainer.evaluate(data.development)
    second = trainer.evaluate(data.development)
    assert_state_equal(before_rng, capture_rng())
    assert first == second
    assert model.training and not model.frozen_child.training
    assert torch.equal(before_parameter, model.theta)
    assert model.theta.grad == .73
    assert trainer.step_count == trainer.attempts == trainer.committed_exposures == 0
    row = first["rows"][0]
    assert len(row["predicted_moments"]) == 10
    assert np.asarray(row["predicted_jacobian"]).shape == (10, 3)
    assert row["diagnostics"]["solver_status"] == "NATIVE_REQUIRED_REFINEMENT"
    assert row["diagnostics"]["nested"]["blend_alpha"] == .4
    assert first["evaluation_scope"] == "held_out_development"
    json.dumps(first, allow_nan=False)


def test_fixed_training_probe_has_no_random_grid_and_no_optimizer_or_sampler_change(monkeypatch):
    trainer, data = make_trainer()
    original_config = trainer.model.config
    before_sampler = copy.deepcopy(trainer.sampler.state_dict())
    before_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    before_rng = capture_rng()
    monkeypatch.setattr(random, "choice", lambda *args: (_ for _ in ()).throw(AssertionError("random coarse probe")))
    report = trainer.evaluate_training_probe(data.train, execution="coarse", coarse_grid=4)
    assert report["complete"] and report["records"] == 2
    assert report["evaluation_scope"] == "bound_training_probe_not_held_out"
    assert report["optimizer_updates"] == 0
    assert report["execution_override"] == "coarse"
    assert report["coarse_grid_override"] == 4
    assert trainer.model.options == [(False, "coarse", 4), (False, "coarse", 4)]
    assert trainer.model.config is original_config
    assert_state_equal(before_sampler, trainer.sampler.state_dict())
    assert_state_equal(before_optimizer, trainer.optimizer.state_dict())
    assert_state_equal(before_rng, capture_rng())
    assert trainer.step_count == trainer.attempts == 0
    with pytest.raises(ValueError, match="registered training"):
        trainer.evaluate_training_probe(data.development, coarse_grid=4)
    with pytest.raises(ValueError, match="power-of-two"):
        trainer.evaluate_training_probe(data.train, coarse_grid=3)


def test_interposed_development_and_train_probes_leave_next_update_identical(tmp_path):
    trainer, data = make_trainer()
    first = trainer.step()
    assert first["committed"]
    checkpoint = tmp_path / "resume.pt"
    receipt = trainer.save_checkpoint(checkpoint)
    trainer.evaluate(data.development)
    trainer.evaluate_training_probe(data.train, coarse_grid=4)
    next_update = trainer.step()
    expected_parameters = copy.deepcopy(trainer.model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    restored, _ = make_trainer()
    restored.load_checkpoint(checkpoint, expected_sha256=receipt["sha256"])
    replay = restored.step()
    assert replay["committed"]
    assert next_update["record_ids"] == replay["record_ids"]
    assert next_update["coarse_grids"] == replay["coarse_grids"]
    assert next_update["loss"] == replay["loss"]
    assert_state_equal(expected_parameters, restored.model.state_dict())
    assert_state_equal(expected_optimizer, restored.optimizer.state_dict())
    assert all(value >= 0 for value in next_update["stages_host_s"].values())
    assert {"forward", "backward", "optimizer_backup", "optimizer_step", "update_audit"} <= set(next_update["stages_host_s"])
    assert next_update["parameter_backup_host_s"] >= 0
    assert next_update["parameter_backup_in_wall_s"] is False
    assert "not device kernel timing" in next_update["stage_timing_scope"]


def test_invalid_diagnostic_is_explicit_failed_row_and_rng_is_restored():
    trainer, data = make_trainer()
    trainer.model.fail_diagnostic = True
    before_rng = capture_rng()
    report = trainer.evaluate(data.development)
    assert not report["complete"] and report["successful_predictions"] == 0
    assert report["failed_predictions"][0]["error"]["type"] == "ValueError"
    assert_state_equal(before_rng, capture_rng())
    assert trainer.model.training


def test_real_henmo_fixed_probe_preserves_contract_and_reports_finite_same_response():
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        data = synthetic_dataset()
        config = HENMOConfig.cpu_smoke(response_zero_init=False)
        with HENMOOperator(config).double() as model:
            original_config = model.config
            trainer = HENMOTrainer(model, HENMOTrainingConfig(max_steps=2, max_attempts=4),
                                  dataset_binding(data, model_contract(model)), records=data.train)
            before = copy.deepcopy(model.state_dict())
            report = trainer.evaluate_training_probe(data.train[:1], execution="coarse", coarse_grid=2)
            assert report["complete"] and report["available_derivative_predictions"] == 1
            row = report["rows"][0]
            assert row["diagnostics"]["execution"] == "coarse"
            assert row["diagnostics"]["coarse_grid"] == 2
            assert np.isfinite(row["predicted_moments"]).all()
            assert np.isfinite(row["predicted_jacobian"]).all()
            assert model.config is original_config and model.training
            assert_state_equal(before, model.state_dict())
            model.config = replace(config, execution="coarse")
            model.eval()
            case = data.train[0]["case"]
            cache = model.encode_geometry(case["blocks"], case["fixed_moments"], inference=True)
            prediction = model.evaluate_operand(cache, case["operand"], coarse_grid=2,
                                                parameter_scales=case["parameter_scales"])
            np.testing.assert_array_equal(prediction.moments.detach().numpy(), row["predicted_moments"])
            np.testing.assert_array_equal(prediction.jacobian.detach().numpy(), row["predicted_jacobian"])
    finally:
        torch.set_num_threads(old_threads)
