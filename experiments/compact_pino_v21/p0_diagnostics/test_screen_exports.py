"""Integration-format tests with explicitly manufactured data and predictions."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.compact_pino_v21.p0_diagnostics import promotion_gate as gate
from experiments.compact_pino_v21.p0_diagnostics.screen_exports import write_screen_bundle


def setup(tmp_path, scope="synthetic", qualification=True):
    records, evaluations = [], {"epoch0":[], "candidate":[], "no_learning":[]}
    for i, (name, fraction, d) in enumerate((("low", .05, 3), ("middle", .5, 8), ("high", .9, 10))):
        phi = np.array([1., .3, .2, .1, .09, .04, .01, .02, .03, .06])
        native, fixed, full = phi*(fraction+.2), phi*.2, phi
        target_file = tmp_path / (name + "-targets.npz")
        native_j = np.ones((10,d))
        np.savez(target_file, native_moments=native, fixed_moments=fixed,
                 native_full_editable_moments=full, native_jacobian=native_j)
        target_receipt = dict(path=str(target_file), sha256=gate.sha256(target_file))
        meta = dict(geometry_id=name, source_group="source-"+name, query_id=name,
                    derivative_valid=True, derivative_dimension=d, chart_id="chart-"+name,
                    teacher_version="manufactured-only", editable_mass=1., length_scale=.5)
        case_path = tmp_path / (name + "-case.json")
        case_path.write_text(json.dumps(dict(meta=meta, targets=target_receipt)))
        case = dict(meta=meta, native_moments=native, fixed_moments=fixed,
                    native_full_editable_moments=full, native_jacobian=native_j,
                    parameter_scales=np.ones(d), asset_receipts={"targets":target_receipt})
        if qualification:
            qualifier = tmp_path / (name + "-qualification.json")
            qualifier.write_text(json.dumps(dict(scope="manufactured_contract_test", qualified=True)))
            case["asset_receipts"]["source_qualification"] = dict(path=str(qualifier), sha256=gate.sha256(qualifier))
        records.append(dict(record_id=name, role="development", loader_eligible=True,
                            case_path=str(case_path), case_sha256=gate.sha256(case_path), case=case))
        for role, error in (("epoch0",1.), ("candidate",.8), ("no_learning",.9)):
            evaluations[role].append(dict(record_id=name, status="available", moments=(native+error).tolist(),
                                           jacobian_status="available", chart_id=meta["chart_id"],
                                           jacobian=(native_j+error).tolist(), failure_reason=None))
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps(dict(scope="manufactured_contract_test", records=[{k:v for k,v in row.items() if k != "case"} for row in records])))
    data = SimpleNamespace(path=manifest, manifest_sha256=gate.sha256(manifest), scope=scope, records=records)
    registered = tmp_path / "registered.json"
    registered.write_text(json.dumps(dict(scope="manufactured_contract_test", improvement=.05)))
    protocol = dict(schema=gate.PROTOCOL_SCHEMA, protocol_id="manufactured-contract-test",
                    registered_protocol_sha256=gate.sha256(registered),
                    primary_aggregation="source_macro_group_rmse", jacobian_aggregation="source_macro_group_rmse",
                    primary_relative_improvement=.05, protection_max_ratio=1.05,
                    protected_retention_metric="scaled10_moment_rmse", protected_jacobian_metric="scaled_jacobian_rmse",
                    require_no_learning_baseline=True, expected_record_ids=["low","middle","high"])
    protocol_path = tmp_path / "screen_protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    checkpoints = []
    for name in ("epoch0", "candidate"):
        path = tmp_path / (name+".checkpoint")
        path.write_text("manufactured checkpoint identity: "+name)
        checkpoints.append(path)
    args = dict(protocol_path=protocol_path, registered_protocol_path=registered,
                epoch0_evaluation={"rows":evaluations["epoch0"]}, candidate_evaluation={"predictions":evaluations["candidate"]},
                no_learning_evaluation={"records":evaluations["no_learning"]}, run_id="manufactured-contract-test",
                config=dict(seed=830251, test_only=True), epoch0_checkpoint_path=checkpoints[0],
                candidate_checkpoint_path=checkpoints[1], output_directory=tmp_path / "screen")
    return data, args


def test_actual_runner_row_format_connects_to_screen_without_manual_remapping(tmp_path):
    data, args = setup(tmp_path)
    path = write_screen_bundle(data, **args)
    result = gate.load_and_screen(path)
    assert result["decision"] == "GO" and result["scope"] == "synthetic"
    assert not result["eligible_for_long_training_review"]
    labels = json.loads((path.parent / "labels.json").read_text())
    assert labels["metadata"]["teacher_identity_kind"].startswith("canonical_sorted")
    assert {row["record_id"] for row in labels["records"]} == {"low", "middle", "high"}
    assert result["identity"]["training_manifest_sha256"] == data.manifest_sha256
    assert result["identity"]["config_sha256"] == gate.canonical_sha256(args["config"])


def test_real_scope_cannot_turn_a_declared_j_into_a_receipted_j(tmp_path):
    data, args = setup(tmp_path, scope="development_training", qualification=False)
    result = gate.load_and_screen(write_screen_bundle(data, **args))
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["coverage"]["declared_but_unreceipted_jacobians"] == 3
    assert result["coverage"]["qualified_jacobian_queries"] == 0


def test_synthetic_analytic_receipt_does_not_require_a_fake_native_qualification(tmp_path):
    data, args = setup(tmp_path, qualification=False)
    path = write_screen_bundle(data, **args)
    labels = json.loads((path.parent/"labels.json").read_text())
    assert all(row["qualification_receipt_kind"] == "synthetic_analytic_target_asset" for row in labels["records"])
    result = gate.load_and_screen(path)
    assert result["decision"] == "GO" and not result["eligible_for_long_training_review"]


def test_source_mutation_or_dropped_development_rows_cannot_export(tmp_path):
    data, args = setup(tmp_path)
    target = Path(data.records[0]["case"]["asset_receipts"]["targets"]["path"])
    target.write_text("changed native labels after model evaluation")
    with pytest.raises(ValueError, match="changed since loading"):
        write_screen_bundle(data, **args)
    assert not args["output_directory"].exists()
    data.records[0].pop("case")
    with pytest.raises(ValueError, match="cannot be silently dropped"):
        write_screen_bundle(data, **args)


def test_missing_qualified_model_j_uses_runner_failure_reason(tmp_path):
    data, args = setup(tmp_path)
    row = args["candidate_evaluation"]["predictions"][0]
    row.update(jacobian_status="unavailable", jacobian=None, failure_reason="active distance tie")
    result = gate.load_and_screen(write_screen_bundle(data, **args))
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["availability"]["candidate"]["unavailable_qualified_jacobians"] == ["low"]
