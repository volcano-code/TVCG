"""End-to-end mixed full/restricted derivative supervision on the real model."""
from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_m80_sqno import (
    make_geometry, make_unified, queries,
)


def _partial_batch():
    generator = torch.Generator().manual_seed(251119)
    q = queries()
    moment = (torch.arange(40, dtype=torch.float64).reshape(4, 10) * .003).requires_grad_()
    full = torch.full((4, 10, 10), float("nan"), dtype=torch.float64)
    full[0] = .01 * torch.randn(10, 10, dtype=torch.float64, generator=generator)
    basis = torch.full((4, 10, 8), float("nan"), dtype=torch.float64)
    restricted = torch.full_like(basis, float("nan"))
    for row, dimension in ((1, 3), (2, 8)):
        basis[row, :, :dimension] = torch.randn(10, dimension, dtype=torch.float64, generator=generator)
        restricted[row, :, :dimension] = .02 * torch.randn(10, dimension, dtype=torch.float64, generator=generator)
    full.requires_grad_()
    options = {
        "jacobian_valid_mask": torch.tensor([True, False, False, False]),
        "restricted_basis": basis.requires_grad_(),
        "restricted_jacobian": restricted.requires_grad_(),
        "restricted_valid": torch.tensor([False, True, True, False]),
        "restricted_dimension": torch.tensor([float("nan"), 3., 8., float("nan")]),
        "moment_scale": torch.linspace(.6, 1.5, 10, dtype=torch.float64),
        "jacobian_scale": torch.linspace(.7, 1.4, 100, dtype=torch.float64).reshape(10, 10),
        "restricted_scale": torch.linspace(.8, 1.5, 10, dtype=torch.float64),
        "jacobian_weight": .2,
        "branch_aux_weight": .3,
    }
    return q, moment, full, options


def _assert_updated(before, model, prefix):
    names = [name for name, _ in model.named_parameters() if name.startswith(prefix)]
    assert names, prefix
    current = dict(model.named_parameters())
    assert any(not torch.equal(before[name], current[name]) for name in names), prefix
    gradients = [current[name].grad for name in names if current[name].grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients), prefix


def _manual_derivatives(prediction, full, options):
    # Deliberately compute each query separately, without using the production
    # supervision primitive or its grouped implementation.
    full_error = (prediction[0] - full[0].detach()) / options["jacobian_scale"]
    full_loss = full_error.square().sum() / 100
    rows = []
    for row, dimension in ((1, 3), (2, 8)):
        basis = options["restricted_basis"][row, :, :dimension].detach()
        target = options["restricted_jacobian"][row, :, :dimension].detach()
        error = (prediction[row] @ basis - target) / options["restricted_scale"][:, None]
        rows.append(error.square().sum() / (10 * dimension))
    restricted_loss = torch.stack(rows).mean()
    return (full_loss + sum(rows)) / 3, full_loss, restricted_loss


def test_unified_partial_derivative_losses_match_per_query_formulas_for_all_three_branches():
    model = make_unified().train()
    q, moment, full, options = _partial_batch()
    cache = model.encode_geometry(make_geometry())
    output = model.evaluate_queries(cache, q, create_graph=True)
    tasks = model.joint_task_losses(cache, q, moment, full, **options)
    for prefix, output_key in (("", "jacobian_q"), ("m80_", "m80_jacobian_q"),
                               ("sqno_", "sqno_jacobian_q")):
        expected, full_expected, restricted_expected = _manual_derivatives(output[output_key].detach(), full, options)
        torch.testing.assert_close(tasks[prefix + "jacobian"], expected, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(tasks[prefix + "full_jacobian"], full_expected, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(tasks[prefix + "restricted_jacobian"], restricted_expected, rtol=1e-11, atol=1e-12)
    expected_aux = .5 * (tasks["m80_moment"] + tasks["sqno_moment"]
                        + .2 * (tasks["m80_jacobian"] + tasks["sqno_jacobian"]))
    torch.testing.assert_close(tasks["branch_aux"], expected_aux)
    result = model.joint_training_loss(cache, q, moment, full, **options)
    torch.testing.assert_close(result["loss"], tasks["moment"] + .2 * tasks["jacobian"] + .3 * expected_aux)
    assert result["valid_queries"].item() == 4
    assert result["valid_jacobian_queries"].item() == 1
    assert result["valid_restricted_queries"].item() == 2
    assert result["valid_derivative_queries"].item() == 3
    assert result["restricted_dimensions"].item() == 11
    for name in tasks:
        torch.testing.assert_close(result[name + "_loss"], tasks[name].detach())
    assert all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in result.values())


@pytest.mark.parametrize("gradient_mode", ["fixed", "pcgrad"])
def test_unified_partial_supervision_updates_encoder_sqno_and_fusion_with_frozen_teachers(gradient_mode):
    model = make_unified().train()
    optimizer = model.build_optimizer(learning_rate=1e-4)
    q, moment, full, options = _partial_batch()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    teachers = (moment, full, options["restricted_basis"], options["restricted_jacobian"])
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert not optimized.intersection(id(teacher) for teacher in teachers)
    metrics = model.train_step(optimizer, model.encode_geometry(make_geometry()), q, moment, full,
                               gradient_mode=gradient_mode, **options)
    for prefix in ("shared_encoder.pino_encoder.", "sqno.", "fusion_logit"):
        _assert_updated(before, model, prefix)
    assert metrics["valid_jacobian_queries"] == 1
    assert metrics["valid_restricted_queries"] == 2
    assert metrics["valid_derivative_queries"] == 3
    assert metrics["restricted_dimensions"] == 11
    assert metrics["multitask_report"]["task_names"] == ["moment", "jacobian", "branch_aux"]
    assert all(math.isfinite(value) for value in metrics.values() if isinstance(value, float))
    for prefix in ("", "m80_", "sqno_"):
        for kind in ("full_jacobian", "restricted_jacobian"):
            key = prefix + kind
            assert metrics[key] == metrics[key + "_loss"]
    assert all(teacher.requires_grad and teacher.grad is None for teacher in teachers)


@pytest.mark.parametrize("gradient_mode", ["fixed", "pcgrad"])
@pytest.mark.parametrize("full_target_kind", ["none", "nan"])
def test_unified_all_derivatives_off_still_trains_moments_without_jacobian_task(gradient_mode, full_target_kind):
    model = make_unified().train()
    optimizer = model.build_optimizer(learning_rate=1e-4)
    q, moment, full, options = _partial_batch()
    options["jacobian_valid_mask"] = torch.zeros(4, dtype=torch.bool)
    options["restricted_valid"] = torch.zeros(4, dtype=torch.bool)
    with torch.no_grad():
        full.fill_(float("nan"))
        options["restricted_basis"].fill_(float("nan"))
        options["restricted_jacobian"].fill_(float("nan"))
        options["restricted_dimension"].fill_(float("nan"))
    target_j = None if full_target_kind == "none" else full
    tasks = model.joint_task_losses(model.encode_geometry(make_geometry()), q, moment, target_j, **options)
    assert tasks["jacobian"].item() == 0 and not tasks["jacobian"].requires_grad
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metrics = model.train_step(optimizer, model.encode_geometry(make_geometry()), q, moment, target_j,
                               gradient_mode=gradient_mode, **options)
    assert metrics["multitask_report"]["task_names"] == ["moment", "branch_aux"]
    assert metrics["jacobian_loss"] == metrics["full_jacobian_loss"] == metrics["restricted_jacobian_loss"] == 0.
    assert metrics["valid_queries"] == 4
    assert all(metrics[key] == 0 for key in ("valid_jacobian_queries", "valid_restricted_queries",
                                            "valid_derivative_queries", "restricted_dimensions"))
    for prefix in ("shared_encoder.pino_encoder.", "sqno.", "fusion_logit"):
        _assert_updated(before, model, prefix)
    assert moment.grad is None and full.grad is None
    assert options["restricted_basis"].grad is None and options["restricted_jacobian"].grad is None


@pytest.mark.parametrize("gradient_mode", ["fixed", "pcgrad"])
def test_unified_restricted_only_accepts_no_full_target_and_keeps_derivative_task(gradient_mode):
    model = make_unified().train()
    optimizer = model.build_optimizer(learning_rate=1e-4)
    q, moment, _, options = _partial_batch()
    options.pop("jacobian_valid_mask")
    metrics = model.train_step(optimizer, model.encode_geometry(make_geometry()), q, moment, None,
                               gradient_mode=gradient_mode, **options)
    assert metrics["valid_jacobian_queries"] == 0
    assert metrics["valid_restricted_queries"] == metrics["valid_derivative_queries"] == 2
    assert metrics["restricted_dimensions"] == 11
    assert metrics["full_jacobian_loss"] == 0.
    assert metrics["jacobian_loss"] == metrics["restricted_jacobian_loss"]
    assert "jacobian" in metrics["multitask_report"]["task_names"]
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_unified_moment_only_omits_all_derivative_arguments():
    model = make_unified().train()
    optimizer = model.build_optimizer(learning_rate=1e-4)
    metrics = model.train_step(optimizer, model.encode_geometry(make_geometry()), queries(),
                               torch.zeros(4, 10, dtype=torch.float64), gradient_mode="pcgrad")
    assert metrics["valid_derivative_queries"] == 0
    assert metrics["multitask_report"]["task_names"] == ["moment", "branch_aux"]


@pytest.mark.parametrize("invalid", ["full_target", "basis", "restricted_target"])
def test_unified_partial_supervision_rejects_active_nan_teacher_values(invalid):
    model = make_unified().train()
    q, moment, full, options = _partial_batch()
    with torch.no_grad():
        if invalid == "full_target": full[0, 0, 0] = float("nan")
        elif invalid == "basis": options["restricted_basis"][1, 0, 0] = float("nan")
        else: options["restricted_jacobian"][2, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.joint_training_loss(model.encode_geometry(make_geometry()), q, moment, full, **options)


@pytest.mark.parametrize("invalid,match", [
    ("overlap", "disjoint"), ("basis_shape", "shape"),
    ("full_shape", "target_jacobian"), ("full_mask_type", "boolean"),
])
def test_unified_partial_supervision_rejects_invalid_masks_and_shapes(invalid, match):
    model = make_unified().train()
    q, moment, full, options = _partial_batch()
    if invalid == "overlap": options["restricted_valid"][0] = True
    elif invalid == "basis_shape": options["restricted_basis"] = options["restricted_basis"][:3]
    elif invalid == "full_shape": full = full[:, :, 0]
    else: options["jacobian_valid_mask"] = options["jacobian_valid_mask"].to(torch.int64)
    with pytest.raises(ValueError, match=match):
        model.joint_training_loss(model.encode_geometry(make_geometry()), q, moment, full, **options)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.bool])
def test_unified_partial_supervision_rejects_nonreal_full_teacher_before_dtype_conversion(dtype):
    model = make_unified().train()
    q, moment, full, options = _partial_batch()
    full = full.detach().to(dtype)
    if dtype == torch.complex128:
        full[0, 0, 0] += 1j
    with pytest.raises(ValueError, match="real"):
        model.joint_training_loss(model.encode_geometry(make_geometry()), q, moment, full, **options)


@pytest.mark.parametrize("settings", [
    {"restricted_scale": torch.ones(10, 10)},
    {"jacobian_scale": torch.ones(10)},
    {"restricted_scale": True},
])
def test_unified_training_contract_rejects_unusable_derivative_normalizers(settings):
    with pytest.raises(ValueError, match="scale"):
        make_unified().training_contract(**settings)


def test_unified_partial_supervision_contract_records_semantics_and_rejects_restricted_scale_drift(tmp_path):
    model = make_unified().train()
    optimizer = model.build_optimizer(learning_rate=1e-4)
    q, moment, full, options = _partial_batch()
    model.train_step(optimizer, model.encode_geometry(make_geometry()), q, moment, full,
                     gradient_mode="pcgrad", **options)
    path = tmp_path / "partial.pt"
    fingerprint = "mixed-full-plane-paraboloid-moment-only-synthetic-v1"
    model.save_checkpoint(path, optimizer=optimizer, data_fingerprint=fingerprint)
    payload = torch.load(path, weights_only=False)
    saved_contract = payload["training_contract"]
    assert saved_contract["schema"] == "tvcg.unified-m80-sqno.training.v2"
    assert saved_contract["restricted_scale"] == options["restricted_scale"].tolist()
    assert saved_contract["derivative_supervision"] == "disjoint-full-or-restricted-per-query-mean-v1"
    restored = make_unified().train()
    restored_optimizer = restored.build_optimizer(learning_rate=1e-4)
    before = {name: value.detach().clone() for name, value in restored.state_dict().items()}
    expected = deepcopy(saved_contract)
    expected["restricted_scale"][0] *= 2.
    with pytest.raises(ValueError, match="contract"):
        restored.load_checkpoint(path, optimizer=restored_optimizer,
            expected_data_fingerprint=fingerprint, expected_training_contract=expected)
    assert all(torch.equal(value, before[name]) for name, value in restored.state_dict().items())
    assert not restored_optimizer.state


def test_paired_partial_ablation_replays_mixed_and_derivative_off_batches_with_finite_reports():
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_training_ablation import run_training_ablation

    model = make_unified().train()
    q, moment, full, options = _partial_batch()
    batch_fields = ("jacobian_valid_mask", "restricted_basis", "restricted_jacobian",
                    "restricted_valid", "restricted_dimension")
    mixed = {"geometry": make_geometry(), "coefficients": q,
             "target_moments": moment, "target_jacobian": full,
             **{name: options[name] for name in batch_fields}}
    moment_only = deepcopy(mixed)
    moment_only["target_jacobian"] = None
    moment_only["jacobian_valid_mask"].zero_()
    moment_only["restricted_valid"].zero_()
    with torch.no_grad():
        moment_only["restricted_basis"].fill_(float("nan"))
        moment_only["restricted_jacobian"].fill_(float("nan"))
        moment_only["restricted_dimension"].fill_(float("nan"))
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    report = run_training_ablation(model, [mixed, moment_only], steps=2, seed=251120,
        learning_rate=1e-4, scope="synthetic_smoke",
        **{key: options[key] for key in ("moment_scale", "jacobian_scale", "restricted_scale",
                                         "jacobian_weight", "branch_aux_weight")})
    fixed, pcgrad = report["arms"]["fixed"], report["arms"]["pcgrad"]
    fairness = report["fairness_contract"]
    assert fairness["batch_schedule"] == [0, 1]
    assert len(set(fairness["batch_sha256"])) == 2
    assert fixed["initial_state_sha256"] == pcgrad["initial_state_sha256"]
    assert fixed["input_sha256"] == pcgrad["input_sha256"] == fairness["input_sha256"]
    assert fixed["initial_evaluation"] == pcgrad["initial_evaluation"]
    for arm in (fixed, pcgrad):
        assert arm["optimizer_steps"] == arm["training_geometry_encode_calls"] == 2
        assert arm["final_state_sha256"] != arm["initial_state_sha256"]
        assert [record["batch_sha256"] for record in arm["steps"]] == fairness["batch_sha256"]
        assert [record["seed"] for record in arm["steps"]] == fairness["step_seeds"]
        first, second = [record["metrics"] for record in arm["steps"]]
        initial_batch = arm["initial_evaluation"]["per_batch"][0]
        for key in ("loss", "moment", "jacobian", "branch_aux", "full_jacobian",
                    "restricted_jacobian", "m80_jacobian", "sqno_jacobian"):
            assert first[key] == pytest.approx(initial_batch[key], rel=1e-8, abs=1e-10)
        assert first["valid_jacobian_queries"] == 1
        assert first["valid_restricted_queries"] == 2
        assert first["valid_derivative_queries"] == 3
        assert first["restricted_dimensions"] == 11
        assert second["valid_queries"] == 4
        assert second["valid_derivative_queries"] == second["restricted_dimensions"] == 0
        assert second["jacobian_loss"] == 0.
        assert second["multitask_report"]["task_names"] == ["moment", "branch_aux"]
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())
    assert moment.grad is None and full.grad is None
    assert options["restricted_basis"].grad is None and options["restricted_jacobian"].grad is None
    assert report["claims"]["formal_accuracy_evidence"] is False
    assert report["claims"]["pcgrad_superiority_established"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("gradient_mode", ["fixed", "pcgrad"])
def test_masked_moment_nan_does_not_change_updates_or_disable_valid_directional_labels(gradient_mode):
    """An unavailable Moment row must not erase its independently valid derivative."""
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_training_state import _assert_identical

    reference = make_unified().train()
    padded = deepcopy(reference)
    q, moment, full, options = _partial_batch()
    mask = torch.tensor([True, False, True, False])
    options["valid_mask"] = mask
    padded_moment = moment.detach().clone().requires_grad_()
    with torch.no_grad():
        padded_moment[~mask] = float("nan")
    # These independent teacher scales must stay detached even during ordinary
    # backward, and restricted row 1 remains active despite its Moment mask.
    for key in ("moment_scale", "jacobian_scale", "restricted_scale"):
        options[key].requires_grad_()
    reference_optimizer = reference.build_optimizer(learning_rate=1e-4)
    padded_optimizer = padded.build_optimizer(learning_rate=1e-4)
    expected = reference.train_step(reference_optimizer, reference.encode_geometry(make_geometry()),
        q, moment, full, gradient_mode=gradient_mode, **options)
    actual = padded.train_step(padded_optimizer, padded.encode_geometry(make_geometry()),
        q, padded_moment, full, gradient_mode=gradient_mode, **options)
    _assert_identical(padded.state_dict(), reference.state_dict())
    _assert_identical(padded_optimizer.state_dict(), reference_optimizer.state_dict())
    _assert_identical(actual, expected)
    assert actual["valid_queries"] == 2
    assert actual["valid_derivative_queries"] == 3
    assert actual["valid_restricted_queries"] == 2
    assert padded_moment.grad is None and moment.grad is None
    assert all(options[key].grad is None for key in ("moment_scale", "jacobian_scale", "restricted_scale"))


@pytest.mark.parametrize("gradient_mode", ["fixed", "pcgrad"])
def test_mixed_supervision_resume_preserves_updates_when_derivative_task_disappears_and_returns(tmp_path, gradient_mode):
    """Resume a populated optimizer through changing supervision availability."""
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_training_state import (
        _assert_identical, _optimizer_and_scheduler, _state,
    )

    model = make_unified().train()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    q, moment, full, options = _partial_batch()
    model.train_step(optimizer, model.encode_geometry(make_geometry()), q, moment, full,
        gradient_mode=gradient_mode, **options)
    scheduler.step()
    assert optimizer.state
    assert any(parameter.is_complex() and parameter in optimizer.state for parameter in model.parameters())
    path = tmp_path / "mixed_resume.pt"
    fingerprint = "mixed-and-moment-only-query-schedule-synthetic-v1"
    model.save_checkpoint(path, optimizer=optimizer, scheduler=scheduler, data_fingerprint=fingerprint)
    contract_options = {key: options[key] for key in (
        "moment_scale", "jacobian_scale", "restricted_scale", "jacobian_weight", "branch_aux_weight",
    )}
    contract = model.training_contract(gradient_mode=gradient_mode, **contract_options)

    moment_only_options = deepcopy(options)
    moment_only_options["jacobian_valid_mask"].zero_()
    moment_only_options["restricted_valid"].zero_()
    with torch.no_grad():
        for key in ("restricted_basis", "restricted_jacobian", "restricted_dimension"):
            moment_only_options[key].fill_(float("nan"))

    def continue_updates(current, current_optimizer, current_scheduler):
        metrics = []
        for target_j, step_options in ((None, moment_only_options), (full, options)):
            metrics.append(current.train_step(current_optimizer,
                current.encode_geometry(make_geometry()), q, moment, target_j,
                gradient_mode=gradient_mode, **step_options))
            current_scheduler.step()
        return metrics, _state(current, current_optimizer, current_scheduler)

    expected_metrics, expected_state = continue_updates(model, optimizer, scheduler)
    restored = make_unified().train()
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler(restored)
    restored.load_checkpoint(path, optimizer=restored_optimizer, scheduler=restored_scheduler,
        expected_data_fingerprint=fingerprint, expected_training_contract=contract, restore_rng=True)
    actual_metrics, actual_state = continue_updates(restored, restored_optimizer, restored_scheduler)
    _assert_identical(actual_metrics, expected_metrics)
    _assert_identical(actual_state, expected_state)
    assert actual_metrics[0]["multitask_report"]["task_names"] == ["moment", "branch_aux"]
    assert actual_metrics[1]["multitask_report"]["task_names"] == ["moment", "jacobian", "branch_aux"]
