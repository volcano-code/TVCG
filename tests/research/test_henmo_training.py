from __future__ import annotations

import copy
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.henmo_v1plus.dataset import (
    check_source_isolation, coverage_report, model_contract,
)
from experiments.henmo_v1plus.migration import (
    ModuleSemantics, TransferRule, transfer_compatible_state, transfer_from_checkpoint,
)
from experiments.henmo_v1plus.training import HENMOTrainer, HENMOTrainingConfig, default_predict


class TinyAnalyticResponse(nn.Module):
    """A two-parameter, ten-output fixture; not a native geometry experiment."""
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([.3, -.2], dtype=torch.float64))
        self.config = dict(response="analytic_training_fixture", native_material_resolution=512)
        self.invalidations = 0

    def invalidate_caches(self):
        self.invalidations += 1


def fixture_record(rid, *, qualified=True, role="train", source="train-source", mass_weight=1.):
    q = .2 if rid.endswith("0") else .6
    phi = np.linspace(.3, 1.2, 10)
    derivative = np.zeros((10, 3)); derivative[:, 0] = .8 * phi
    meta = dict(editable_mass=1., length_scale=2., derivative_valid=qualified,
                derivative_dimension=3 if qualified else None, derivative_reason="event" if not qualified else None,
                density=1., split="synthetic", source_group=source, geometry_id=source + "-geometry",
                geometry_sha256=("a" if role == "train" else "b") * 64,
                material_identity="c" * 64, query_id=rid)
    case = dict(meta=meta, q=q, phi=phi,
                native_moments=(.7 + .8 * q) * phi,
                native_jacobian=derivative if qualified else np.full((10, 10), np.nan),
                parameter_scales=np.array([2., 1., 1.]),
                fixed_moments=np.zeros(10), native_full_editable_moments=np.r_[1., np.zeros(9)],
                blocks=SimpleNamespace(side=512))
    return dict(record_id=rid, role=role, loader_eligible=True, case_sha256="d" * 64,
                moment_weight=mass_weight, jacobian_weight=1., case=case,
                surface_family="PLANE", trajectory_stage="near_convergence")


def fixture_predict(model, row, *, evaluation, chunk_size, jacobian_loss_weight):
    case = row["case"]
    noise = .01 * (torch.rand((), dtype=torch.float64) + random.random() + np.random.random())
    phi = model.theta.new_tensor(case["phi"])
    moments = (model.theta[0] + model.theta[1] * case["q"] + noise) * phi
    jacobian = torch.stack((model.theta[1] * phi, torch.zeros_like(phi), torch.zeros_like(phi)), dim=1)
    return SimpleNamespace(moments=moments, jacobian=jacobian, diagnostics={})


def make_trainer(*, config=None, records=None, model=None, predict=fixture_predict):
    model = model or TinyAnalyticResponse()
    records = records or [fixture_record("q0"), fixture_record("q1")]
    binding = dict(schema="henmo.training-binding.v1", manifest_sha256="e" * 64,
                   snapshot_id="synthetic-fixture", scope="synthetic", architecture=model_contract(model),
                   record_identities=[dict(record_id=row["record_id"], role="train",
                                           case_sha256=row["case_sha256"]) for row in records])
    trainer = HENMOTrainer(model, config or HENMOTrainingConfig(max_steps=4, max_attempts=8),
                          binding, records=records, predict=predict)
    return trainer


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_tree_equal(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_tree_equal(x, y)
    else:
        assert left == right


def test_actual_joint_update_and_exact_next_step_resume(tmp_path):
    original = make_trainer()
    initial = original.model.theta.detach().clone()
    first = original.step()
    assert first["committed"] and first["moment_denominator"] == 1
    assert first["jacobian_denominator"] == 1
    assert not torch.equal(initial, original.model.theta)
    path = tmp_path / "resume.pt"
    receipt = original.save_checkpoint(path)
    second = original.step()
    expected = copy.deepcopy(original.optimizer.state_dict())
    final_weights = original.model.theta.detach().clone()
    resumed = make_trainer()
    resumed.load_checkpoint(path, expected_sha256=receipt["sha256"])
    repeated = resumed.step()
    assert second["record_ids"] == repeated["record_ids"]
    assert first["coarse_grids"] == [None]  # Unrelated predictors retain their behavior.
    assert second["loss"] == repeated["loss"]
    assert torch.equal(final_weights, resumed.model.theta)
    assert_tree_equal(expected, resumed.optimizer.state_dict())
    assert original.step_count == resumed.step_count == 2
    assert resumed.model.invalidations >= 2


def test_jacobian_loss_really_updates_response_without_moment_loss():
    records = [fixture_record("q0", mass_weight=0.)]
    trainer = make_trainer(records=records)
    before = trainer.model.theta.detach().clone()
    receipt = trainer.step()
    assert receipt["moment_denominator"] == 0
    assert receipt["jacobian_denominator"] == 1
    assert receipt["committed"]
    assert trainer.model.theta[0] == before[0]
    assert trainer.model.theta[1] != before[1]


def test_unqualified_jacobian_is_not_read_and_has_no_fake_denominator():
    trainer = make_trainer(records=[fixture_record("q0", qualified=False)])
    receipt = trainer.step()
    assert receipt["committed"]
    assert receipt["jacobian_denominator"] == 0
    assert receipt["jacobian_mask"] == [False]


def test_resume_rejects_changed_material_semantics_and_config_before_update(tmp_path):
    trainer = make_trainer()
    trainer.step()
    path = tmp_path / "resume.pt"; trainer.save_checkpoint(path)
    other = make_trainer()
    initial = other.model.theta.detach().clone()
    other.binding["record_identities"][0]["material_identity"] = "f" * 64
    with pytest.raises(ValueError, match="semantics mismatch"):
        other.load_checkpoint(path)
    assert torch.equal(initial, other.model.theta)
    other = make_trainer(config=HENMOTrainingConfig(max_steps=4, max_attempts=8, learning_rate=1e-4))
    with pytest.raises(ValueError, match="semantics mismatch"):
        other.load_checkpoint(path)


def test_nonfinite_prediction_does_not_change_weights_or_optimizer():
    def broken(model, row, **kwargs):
        prediction = fixture_predict(model, row, **kwargs)
        prediction.moments = prediction.moments * float("nan")
        return prediction
    trainer = make_trainer(predict=broken)
    initial = trainer.model.theta.detach().clone()
    with pytest.raises(ValueError, match="nonfinite"):
        trainer.step()
    assert torch.equal(initial, trainer.model.theta)
    assert trainer.step_count == 0 and trainer.attempts == 1
    assert not trainer.optimizer.state
    assert trainer.history[-1]["error"] and not trainer.history[-1]["committed"]


def test_no_step_for_zero_gradient_and_no_automatic_budget_extension():
    def zero(model, row, **kwargs):
        moment = model.theta.sum() * 0 + model.theta.new_tensor(row["case"]["native_moments"])
        return SimpleNamespace(moments=moment, jacobian=None, diagnostics={})
    trainer = make_trainer(config=HENMOTrainingConfig(max_steps=1, max_attempts=1), predict=zero)
    receipt = trainer.step()
    assert not receipt["committed"] and trainer.step_count == 0
    with pytest.raises(RuntimeError, match="budget"):
        trainer.step()


def test_held_out_data_and_unqualified_physics_objective_rejected():
    row = fixture_record("q0", role="development", source="held-out")
    with pytest.raises(ValueError, match="training rows"):
        make_trainer(records=[row])
    with pytest.raises(ValueError, match="qualified targets"):
        make_trainer(config=HENMOTrainingConfig(residual_loss_weight=1.))


def test_source_isolation_and_missing_derivative_coverage_are_explicit():
    train = fixture_record("q0", qualified=False)
    development = fixture_record("q1", role="development", source="dev-source")
    report = coverage_report([train, development])
    assert report["rows"][0]["native_resolution"] == 512
    assert any(row.get("stratum") == "Full-J" for row in report["missing"])
    development["case"]["meta"]["source_group"] = "train-source"
    with pytest.raises(ValueError, match="leakage"):
        check_source_isolation([train, development])


def test_semantic_transfer_preserves_new_head_and_rejects_shape_only_reuse():
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.response = nn.Linear(2, 2)
    old, new = Network(), Network()
    response_before = new.response.weight.detach().clone()
    meaning = ModuleSemantics("physical_xyz", "native_material", "geometry_context", "v1")
    rule = TransferRule("encoder", "encoder", meaning, meaning)
    receipt = transfer_compatible_state(new, old.state_dict(), [rule], source_checkpoint_sha256="a" * 64)
    assert torch.equal(new.encoder.weight, old.encoder.weight)
    assert torch.equal(new.response.weight, response_before)
    assert not receipt["optimizer_transferred"]
    assert "response.weight" in receipt["newly_initialized"]
    changed_meaning = replace(meaning, output_meaning="coarse_cut_moments")
    bad = TransferRule("response", "response", meaning, changed_meaning)
    with pytest.raises(ValueError, match="semantics differ"):
        transfer_compatible_state(new, old.state_dict(), [bad], source_checkpoint_sha256="a" * 64)


def test_response_transfer_is_allowed_when_function_semantics_are_unchanged():
    old, new = nn.Linear(3, 2), nn.Linear(3, 2)
    meaning = ModuleSemantics("physical_xyz", "same_static_context", "monotone_warp", "same_bandwidth")
    rule = TransferRule("", "", meaning, meaning)
    transfer_compatible_state(new, old.state_dict(), [rule], source_checkpoint_sha256="b" * 64)
    assert torch.equal(new.weight, old.weight)


def test_migration_verifies_actual_checkpoint_bytes_before_loading(tmp_path):
    import hashlib
    old, new = nn.Linear(3, 2), nn.Linear(3, 2)
    meaning = ModuleSemantics("physical_xyz", "same_static_context", "monotone_warp", "same_bandwidth")
    rules = [TransferRule("", "", meaning, meaning)]
    path = tmp_path / "legacy.pt"; torch.save({"model": old.state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = transfer_from_checkpoint(new, path, rules, expected_sha256=digest)
    assert torch.equal(new.weight, old.weight)
    assert receipt["source_state_key"] == "model"
    with pytest.raises(ValueError, match="file SHA256"):
        transfer_from_checkpoint(new, path, rules, expected_sha256="f" * 64)


def test_training_contract_cannot_change_after_optimizer_construction():
    trainer = make_trainer()
    trainer.model.config["native_material_resolution"] = 1024
    with pytest.raises(ValueError, match="architecture changed"):
        trainer.step()


def test_training_selects_a_coarse_scale_but_evaluation_and_legacy_do_not():
    class Adapter:
        def __init__(self, config):
            self.config = config
            self.options = []
        def encode_geometry(self, *args, **kwargs):
            return object()
        def evaluate_operand(self, features, operand, **kwargs):
            self.options.append(kwargs)
            return object()
    row = fixture_record("q0"); row["case"]["operand"] = object()
    model = Adapter(SimpleNamespace(training_coarse_grids=(2, 4)))
    random.seed(830251)
    for _ in range(12):
        default_predict(model, row, evaluation=False, chunk_size=16, jacobian_loss_weight=1.)
    assert {options["coarse_grid"] for options in model.options} == {2, 4}
    random_state = random.getstate()
    default_predict(model, row, evaluation=True, chunk_size=16, jacobian_loss_weight=1.)
    assert "coarse_grid" not in model.options[-1] and random.getstate() == random_state
    legacy = Adapter(SimpleNamespace())
    default_predict(legacy, row, evaluation=False, chunk_size=16, jacobian_loss_weight=1.)
    assert "coarse_grid" not in legacy.options[-1]


def test_native512_requires_explicit_admission_and_teacher_resolution_binding():
    from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import (
        validate_case, ORDER, COEFFICIENT_MULTIPLIERS, LABEL_IDENTITY_FIELDS,
        operand_fingerprint, query_fingerprint,
    )
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    operand = FiniteOperand(vertices, faces, np.ones(4, dtype=bool), "intersection")
    coefficients = np.r_[1., np.zeros(9)]
    meta = dict(geometry_id="fixture", query_id="query", source_group="source", split="synthetic",
                teacher_version="analytic-fixture-v1", coordinate_frame="physical", units="m_kg",
                moment_order=list(ORDER), coefficient_basis="raw_coefficients",
                coefficient_multipliers=list(COEFFICIENT_MULTIPLIERS), geometry_sha256="a" * 64,
                material_identity="b" * 64, operand_content_sha256=operand_fingerprint(operand),
                query_content_sha256=query_fingerprint(coefficients), operation="intersection",
                reference_kind="analytic_fixture", moments_include_fixed=True, derivative_valid=False,
                derivative_dimension=None, derivative_reason="fixture value only", chart_id="value-only",
                length_scale=1., length_scale_definition="fixture", editable_mass=1., density=1.,
                native_material_resolution=512)
    reference = {key: meta[key] for key in LABEL_IDENTITY_FIELDS}
    reference["native_material_resolution"] = 512
    case = dict(meta=meta, reference_meta=reference, coefficients=coefficients, operand=operand,
                blocks=SimpleNamespace(side=512, pitch=1/512, identity="b" * 64),
                fixed_moments=np.zeros(10), native_moments=np.r_[.5, np.zeros(9)],
                native_full_editable_moments=np.r_[1., np.zeros(9)])
    with pytest.raises(ValueError, match="admission contract"):
        validate_case(case)
    assert validate_case(case, allowed_native_resolutions=(256, 512, 1024)).shape == (10,)
    case["reference_meta"]["native_material_resolution"] = 256
    with pytest.raises(ValueError, match="teacher metadata"):
        validate_case(case, allowed_native_resolutions=(512,))


def test_native512_export_preserves_teacher_values_and_requires_opt_in(tmp_path, monkeypatch):
    from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import (
        write_synthetic_finite_case, load_baseline_case,
    )
    from experiments.compact_pino_v21.p0_diagnostics.test_export_cases import request
    from experiments.compact_pino_v21.p0_diagnostics.export_cases import export_cases, verify_export_qualification
    from experiments.henmo_v1plus.dataset import load_henmo_dataset
    source = write_synthetic_finite_case(tmp_path / "raw")
    request_path, payload = request(tmp_path, source)
    payload["records"][0]["grid"].update(side=512, pitch=1/32)
    request_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exporter admission"):
        export_cases(request_path, tmp_path / "legacy-rejected", synthetic=True)
    manifest = export_cases(request_path, tmp_path / "highres", synthetic=True,
                            allowed_native_resolutions=(512,))
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure
    monkeypatch.setattr(BlockMeasure, "build", lambda *args, **kwargs: pytest.fail("HENMO must not expand a full-volume atom pack"))
    data, report = load_henmo_dataset(manifest, mode="smoke", allowed_native_resolutions=(512,))
    record = data.train[0]
    case = record["case"]
    assert case["blocks"].side == 512
    assert case["reference_meta"]["native_material_resolution"] == 512
    assert case["meta"]["native_material_resolution"] == 512
    assert report["rows"][0]["native_resolution"] == 512
    assert case["material_layout"] == "prepared"
    assert all(not receipt["full_volume_atoms_expanded"] for receipt in data.physics_receipts.values())
    with np.load(tmp_path / "teacher.npz", allow_pickle=False) as original:
        assert np.array_equal(case["native_moments"], original["native_moments"])
        assert np.array_equal(case["native_jacobian"], original["native_jacobian"])
    assert verify_export_qualification(case, record["record_id"])["status"] == "BOUND_EXTERNAL_DECLARATION_VERIFIED"
    with pytest.raises(ValueError, match="admission contract"):
        load_baseline_case(record["case_path"])


def test_sparse_material_admission_rejects_overlapping_runs_without_expansion():
    from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import audit_prepared_material
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_rle_material
    material = prepare_rle_material(np.array([[1, 1, 0, 16]], dtype=np.int32), [0.,0.,0.], 1/16, 16)
    scale = np.ones(10)
    material.iter_chunks = lambda *args, **kwargs: pytest.fail("audit must not expand full voxels")
    assert audit_prepared_material(material, scale)["full_volume_atoms_expanded"] is False
    material.full_runs = np.array([[1,1,0,10],[1,1,8,16]], dtype=np.int32)
    with pytest.raises(ValueError, match="overlapping"):
        audit_prepared_material(material, scale)


def test_prepared_manifest_reuses_material_without_query_local_decompression(tmp_path, monkeypatch):
    from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import write_synthetic_finite_case, sha256
    from experiments.compact_pino_v21.p0_diagnostics.test_export_cases import request
    from experiments.compact_pino_v21.p0_diagnostics.export_cases import export_cases
    from experiments.henmo_v1plus.dataset import load_henmo_dataset
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import v21_material
    source = write_synthetic_finite_case(tmp_path / "raw")
    request_path, payload = request(tmp_path, source)
    payload["records"][0]["grid"].update(side=512, pitch=1/32)
    request_path.write_text(json.dumps(payload))
    manifest_path = export_cases(request_path, tmp_path / "data", synthetic=True,
                                 allowed_native_resolutions=(512,))
    manifest = json.loads(manifest_path.read_text())
    first_path = manifest_path.parent / manifest["records"][0]["case_path"]
    spec = json.loads(first_path.read_text())
    # A repeated analytic query in a distinct directory exercises cache identity;
    # this is synthetic loader evidence, never a new native qualified sample.
    second_directory = tmp_path / "second-query"; second_directory.mkdir()
    assets = list(spec["assets"].values()) + [spec[key] for key in ("source_qualification","source_teacher_targets") if key in spec]
    assets += spec.get("source_evidence",[])
    for asset in assets:
        asset["path"] = str((first_path.parent / asset["path"]).resolve())
    with np.load(spec["assets"]["targets"]["path"], allow_pickle=False) as loaded:
        targets = {key: loaded[key].copy() for key in loaded.files}
    spec["meta"]["query_id"] = "synthetic-repeated-query"
    target_meta = json.loads(str(targets["metadata"].item()))
    target_meta["query_id"] = spec["meta"]["query_id"]
    targets["metadata"] = np.asarray(json.dumps(target_meta))
    target_path = second_directory / "targets.npz"; np.savez(target_path, **targets)
    spec["assets"]["targets"] = dict(path=str(target_path), sha256=sha256(target_path))
    second_path = second_directory / "case.json"; second_path.write_text(json.dumps(spec))
    second = copy.deepcopy(manifest["records"][0])
    second.update(record_id="synthetic-repeat", case_path=str(second_path), case_sha256=sha256(second_path))
    manifest["records"].append(second)
    manifest_path.write_text(json.dumps(manifest))
    actual_load, calls = v21_material.load_material, []
    def counted(*args, **kwargs):
        calls.append(args[0])
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(v21_material, "load_material", counted)
    data, _ = load_henmo_dataset(manifest_path, mode="smoke", allowed_native_resolutions=(512,))
    assert len(data.train) == 2 and len(calls) == 1
    assert data.train[0]["case"]["blocks"] is data.train[1]["case"]["blocks"]


def test_comparison_checkpoint_requires_hash_dtype_and_exact_snapshot(tmp_path):
    from experiments.henmo_v1plus.cli import _load_comparison_checkpoint
    from experiments.henmo_v1plus.dataset import dataset_binding
    records = [fixture_record("q0")]
    data = SimpleNamespace(records=records, train=records, development=[], snapshot_id="fixture",
                           manifest_sha256="e"*64, scope="synthetic", source_admission=None)
    model = TinyAnalyticResponse()
    trainer = HENMOTrainer(model,HENMOTrainingConfig(max_steps=2,max_attempts=4),
                          dataset_binding(data,model_contract(model)),records=records,predict=fixture_predict)
    trainer.step()
    path = tmp_path / "compare.pt"; receipt = trainer.save_checkpoint(path)
    restored = TinyAnalyticResponse()
    result = _load_comparison_checkpoint(restored,path,data,receipt["sha256"])
    assert result["step_count"] == 1 and torch.equal(model.theta,restored.theta)
    with pytest.raises(ValueError, match="dtype"):
        _load_comparison_checkpoint(TinyAnalyticResponse().float(),path,data)
    changed = copy.copy(data); changed.snapshot_id = "different"
    with pytest.raises(ValueError, match="snapshot binding"):
        _load_comparison_checkpoint(TinyAnalyticResponse(),path,changed)
    with pytest.raises(ValueError, match="explicitly requested"):
        _load_comparison_checkpoint(TinyAnalyticResponse(),path,data,"a"*64)
    receipt_path = path.with_suffix(".pt.json")
    receipt["sha256"] = "b"*64; receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt SHA256"):
        _load_comparison_checkpoint(TinyAnalyticResponse(),path,data)


@pytest.mark.parametrize("resume_step", [0, 1])
def test_cli_does_not_replace_better_starting_state_with_worse_updates(tmp_path, monkeypatch, resume_step):
    """Selection controller test with real updates and scripted evaluation scores."""
    from dataclasses import asdict
    from experiments.henmo_v1plus import cli
    from experiments.henmo_v1plus.dataset import dataset_binding
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
    records = [fixture_record("q0")]
    dev = [fixture_record("dev0", role="development", source="dev-source")]
    data = SimpleNamespace(records=records+dev, train=records, development=dev, snapshot_id="selection-fixture",
                           manifest_sha256="e"*64, scope="synthetic", source_admission=None)
    class ControlledEvaluation(HENMOTrainer):
        def __init__(self, model, config, binding, *, records):
            super().__init__(model,config,binding,records=records,predict=fixture_predict)
        def evaluate(self, records):
            return dict(complete=True,source_macro_raw_moment_rmse=float(1+self.step_count))
    def model_factory(*args):
        model = TinyAnalyticResponse(); model.close = lambda: None
        return model
    monkeypatch.setattr(cli,"_place_model",model_factory)
    monkeypatch.setattr(cli,"load_henmo_dataset",lambda *args,**kwargs: (data,{}))
    monkeypatch.setattr(cli,"HENMOTrainer",ControlledEvaluation)
    model_path = tmp_path/"model.json"
    model_path.write_text(json.dumps(asdict(HENMOConfig.cpu_smoke(native_material_resolution=512))))
    config = HENMOTrainingConfig(max_steps=2,max_attempts=4)
    config_path = tmp_path/"training.json"; config_path.write_text(json.dumps(asdict(config)))
    args = ["train","--config",str(model_path),"--manifest","synthetic-controller-fixture",
            "--mode","smoke","--output",str(tmp_path/"run"),"--device","cpu",
            "--dtype","float64","--training-config",str(config_path),"--evaluate-every","1"]
    if resume_step:
        model = model_factory()
        trainer = ControlledEvaluation(model,config,dataset_binding(data,model_contract(model)),records=records)
        trainer.step()
        receipt = trainer.save_checkpoint(tmp_path/"resume.pt")
        args += ["--resume",str(tmp_path/"resume.pt"),"--resume-sha256",receipt["sha256"]]
    cli.main(args)
    report = json.loads((tmp_path/"run"/"run.json").read_text())
    assert report["last"]["step_count"] == 2
    assert report["selected_step"] == resume_step and not report["selected_has_new_updates"]
    assert report["selection_scope"]["start_step"] == resume_step
    assert report["selection_scope"]["scope"] == ("current_continuation_only" if resume_step else "entire_current_run")
    assert "not_a_replay_of_V21" in report["selection_protocol"]
    with pytest.raises(ValueError,match="new or empty directory"):
        cli.main(args)


def test_cli_rejects_orphan_resume_and_comparison_hashes_before_loading():
    from experiments.henmo_v1plus.cli import main
    for command,flag in (("train","--resume-sha256"),("compare","--checkpoint-sha256")):
        with pytest.raises(ValueError,match="requires"):
            main([command,"--manifest","unused","--config","unused","--output","unused",flag,"a"*64])


def test_compare_driver_runs_actual_deterministic_and_neural_queries(tmp_path, monkeypatch):
    """Direct synthetic driver fixture; production admission is tested separately."""
    from dataclasses import asdict
    from experiments.henmo_v1plus import cli
    from experiments.henmo_v1plus.fixtures import synthetic_dataset
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
    data = synthetic_dataset()
    for row in data.records:
        row["role"] = "development"
    data.development = data.records; data.train = []
    monkeypatch.setattr(cli,"load_henmo_dataset",lambda *args,**kwargs: (data,coverage_report(data.records)))
    config_path = tmp_path/"model.json"
    config_path.write_text(json.dumps(asdict(HENMOConfig.cpu_smoke())))
    output = tmp_path/"compare.json"
    old_threads = torch.get_num_threads()
    try:
        cli.main(["compare","--manifest","explicit-synthetic-driver-fixture","--config",str(config_path),
                  "--mode","smoke","--output",str(output),"--device","cpu","--dtype","float64","--threads","1"])
        report = json.loads(output.read_text())
        assert not report["model_trained"] and not report["acceleration_proven"]
        assert len(report["shared_material_preparation"]) == 1
        assert report["candidate_geometry_encode_s"] > 0
        assert all(arm["records"] == 2 and arm["complete"] for arm in report["arms"].values())
        assert all(row["includes_finite_surface_preparation"] and row["includes_moment_jacobian_cpu_transfer"]
                   for row in report["timings"])
    finally:
        torch.set_num_threads(old_threads)


def test_cli_audits_exported_synthetic512_manifest_and_trains_one_actual_step(tmp_path):
    from dataclasses import asdict
    from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import write_synthetic_finite_case
    from experiments.compact_pino_v21.p0_diagnostics.test_export_cases import request
    from experiments.compact_pino_v21.p0_diagnostics.export_cases import export_cases
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
    from experiments.henmo_v1plus.cli import main
    source = write_synthetic_finite_case(tmp_path / "raw")
    request_path, payload = request(tmp_path, source)
    payload["records"][0]["grid"].update(side=512, pitch=1/32)
    request_path.write_text(json.dumps(payload))
    manifest = export_cases(request_path, tmp_path / "dataset", synthetic=True,
                            allowed_native_resolutions=(512,))
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(asdict(HENMOConfig.cpu_smoke(
        native_material_resolution=512, response_zero_init=False, training_scope="response"))))
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(asdict(HENMOTrainingConfig(max_steps=1, max_attempts=2))))
    audit_path = tmp_path / "audit.json"
    common = ["--manifest", str(manifest), "--config", str(config_path), "--mode", "smoke"]
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        main(["audit-data", *common, "--output", str(audit_path)])
        audit = json.loads(audit_path.read_text())
        assert audit["material_layout"] == "prepared"
        assert audit["rows"][0]["native_resolution"] == 512
        main(["train", *common, "--output", str(tmp_path / "run"), "--device", "cpu",
              "--dtype", "float64", "--training-config", str(training_path), "--evaluate-every", "1"])
        run = json.loads((tmp_path / "run" / "run.json").read_text())
        assert run["last"]["step_count"] == 1
        assert not run["physics_promotion"] and not run["long_training_promotion"]
        steps = [json.loads(line) for line in (tmp_path / "run" / "steps.jsonl").read_text().splitlines()]
        assert steps[0]["committed"] and steps[0]["jacobian_denominator"] == 1
        assert steps[0]["coarse_grids"][0] in (2, 4)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("mass_weight", [1., 0.])
def test_actual_henmo_joint_training_and_resume_through_default_adapter(tmp_path, mass_weight):
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_rle_material
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
    side, cut = 16, .37
    runs = np.array([[x, y, 0, side] for x in range(side) for y in range(side)], dtype=np.int32)
    material = prepare_rle_material(runs, [0., 0., 0.], 1/side, side)
    v = np.array([[-2,-2,-2],[cut,-2,-2],[cut,2,-2],[-2,2,-2],
                  [-2,-2,2],[cut,-2,2],[cut,2,2],[-2,2,2]], dtype=float)
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    motion = np.zeros((8,3,3)); active = v[:,0] == cut
    motion[active,0,:] = np.column_stack((np.ones(active.sum()), v[active,1], v[active,2]))
    operand = FiniteOperand(v, f, np.ones(12, dtype=bool), "intersection", motion)
    powers = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),
              (0,2,0),(0,0,2),(0,1,1),(1,0,1),(1,1,0)]
    moments = np.array([cut**(x+1)/(x+1)/(y+1)/(z+1) for x,y,z in powers])
    jacobian = np.array([[cut**x/(y+dy+1)/(z+dz+1) for dy,dz in ((0,0),(1,0),(0,1))]
                         for x,y,z in powers])
    row = fixture_record("analytic-0", mass_weight=mass_weight)
    row["case"].update(blocks=material, operand=operand, native_moments=moments,
                        native_jacobian=jacobian, native_full_editable_moments=material.moments(),
                        parameter_scales=np.ones(3))
    row["case"]["meta"].update(length_scale=1., material_identity=material.identity)
    config = HENMOConfig.cpu_smoke(response_zero_init=False)
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        first_model = HENMOOperator(config).double()
        trainer = make_trainer(model=first_model, records=[row], predict=None)
        first = trainer.step()
        assert first["committed"] and first["jacobian_denominator"] == 1
        assert first["moment_denominator"] == mass_weight
        assert any(name.startswith("response.") for name in first["changed_parameters"])
        assert any(name.startswith("shared_encoder.") for name in first["changed_parameters"])
        path = tmp_path / "actual-henmo.pt"; trainer.save_checkpoint(path)
        next_step = trainer.step()
        expected = {name: value.detach().clone() for name,value in trainer.model.state_dict().items()}
        with HENMOOperator(config).double() as second_model:
            resumed = make_trainer(model=second_model, records=[row], predict=None)
            resumed.load_checkpoint(path)
            replay = resumed.step()
            assert replay["coarse_grids"] == next_step["coarse_grids"]
            assert all(grid in config.training_coarse_grids for grid in first["coarse_grids"] + next_step["coarse_grids"])
            assert replay["loss"] == next_step["loss"]
            for name, value in resumed.model.state_dict().items():
                assert torch.equal(value, expected[name]), name
        first_model.close()
    finally:
        torch.set_num_threads(old_threads)
