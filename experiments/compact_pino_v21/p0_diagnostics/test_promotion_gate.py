"""Synthetic gate contracts; these are not evaluations of a real checkpoint."""
import copy
import json

import numpy as np
import pytest

from experiments.compact_pino_v21.p0_diagnostics import promotion_gate as gate


def inputs():
    protocol = dict(schema=gate.PROTOCOL_SCHEMA, protocol_id="synthetic-frozen-screen",
                    registered_protocol_sha256="a"*64,
                    primary_aggregation="source_macro_group_rmse",
                    jacobian_aggregation="source_macro_group_rmse",
                    primary_relative_improvement=.05, protection_max_ratio=1.05,
                    protected_retention_metric="raw_mass_rmse",
                    protected_jacobian_metric="raw_jacobian_rmse",
                    require_no_learning_baseline=True, expected_record_ids=["low", "middle", "high"])
    label_meta = dict(scope="synthetic", moment_order=list(gate.ORDER),
                      evaluation_manifest_sha256="b"*64, teacher_sha256="c"*64)
    records = []
    for name, fraction, kind, source in (("low", .05, "R3", "A"), ("middle", .5, "R8", "B"),
                                         ("high", .9, "Full-J", "C")):
        d = gate.DIMENSIONS[kind]
        records.append(dict(record_id=name, geometry_id="geometry_"+name, source_group_id=source,
                            retention_ratio=fraction, moment_valid=True, native_moments=[1.]*10,
                            derivative_valid=True, derivative_type=kind, chart_id="chart_"+kind,
                            qualification_receipt_sha256="d"*64,
                            native_jacobian=np.ones((10, d)).tolist()))
    labels = dict(schema=gate.LABEL_SCHEMA, metadata=label_meta, records=records)
    common = dict(**label_meta, run_id="synthetic-paired-run", training_manifest_sha256="e"*64,
                  config_sha256="f"*64, protocol_sha256="1"*64)
    predictions = []
    for role, delta in (("epoch0", 1.), ("candidate", .9), ("no_learning", .8)):
        meta = dict(common, role=role, checkpoint_sha256="2"*64 if role == "epoch0" else "3"*64)
        if role == "no_learning":
            meta.update(checkpoint_sha256=None, model_weights_used=False)
        rows = []
        for row in records:
            d = gate.DIMENSIONS[row["derivative_type"]]
            rows.append(dict(record_id=row["record_id"], status="available", moments=[1.+delta]*10,
                             jacobian_status="available", chart_id=row["chart_id"],
                             jacobian=np.full((10, d), 1.+delta).tolist()))
        predictions.append(dict(schema=gate.PREDICTION_SCHEMA, metadata=meta, records=rows))
    return protocol, labels, *predictions


def run(values):
    p, labels, e0, candidate, baseline = values
    return gate.screen(p, labels, e0, candidate, no_learning=baseline)


def test_passing_research_screen_never_authorizes_training_or_physical_success():
    result = run(inputs())
    assert result["decision"] == "GO"
    assert result["primary"]["current"]["value"] == pytest.approx(.9)
    assert result["candidate_relative_to_no_learning"] == pytest.approx(.125)
    assert not result["baseline_gap_is_registered_hard_gate"]
    assert not result["training_authorized"] and not result["eligible_for_long_training_review"]
    assert result["physical_acceptance"] == "NOT_EVALUATED" and not result["speedup_claim"]
    assert result["coverage"]["qualified_jacobian_by_type"] == {"R3": 1, "R8": 1, "Full-J": 1}


def test_development_go_is_only_eligible_for_separate_long_training_review():
    values = inputs()
    for document in values[1:]:
        document["metadata"]["scope"] = "development"
    result = run(values)
    assert result["decision"] == "GO" and result["eligible_for_long_training_review"]
    assert result["training_authorized"] is False


def test_reported_312_step_ratios_are_unambiguously_no_go():
    # Reproduce the supplied summary ratios with manufactured errors only.
    values = inputs()
    values[0]["protected_retention_metric"] = "scaled10_moment_rmse"
    for row in values[1]["records"]:
        row["moment_scale"] = np.arange(1., 11.).tolist()
    for row, factor in zip(values[3]["records"], (.78305, .78305, 1.2977)):
        row["moments"] = [1.+factor]*10
    for row in values[4]["records"]:
        row["moments"] = [1.+.9546/1.0773]*10
    result = run(values)
    assert result["primary"]["relative_change"] == pytest.approx(-.0454)
    assert result["protections"]["high"]["relative_change"] == pytest.approx(.2977)
    assert result["candidate_relative_to_no_learning"] == pytest.approx(.0773)
    assert result["decision"] == "NO_GO"
    assert set(result["failed_checks"]) == {"primary", "high"}


@pytest.mark.parametrize("kind", ["R3", "R8", "Full-J"])
def test_missing_qualified_derivative_denominator_cannot_pass(kind):
    values = inputs()
    for row in values[1]["records"]:
        if row["derivative_type"] == kind:
            row.update(derivative_valid=False, derivative_reason="no certified neighbourhood")
            row.pop("native_jacobian")
    result = run(values)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert kind in result["insufficient_evidence"]
    assert result["protections"][kind]["current"]["value"] is None


def test_missing_low_denominator_is_evidence_gap_not_pass():
    values = inputs()
    values[1]["records"][0]["retention_ratio"] = .2
    result = run(values)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "low_retention_denominator" in result["insufficient_evidence"]


def test_missing_model_j_is_null_and_does_not_shrink_the_denominator():
    values = inputs()
    values[3]["records"][0].update(jacobian_status="unavailable", jacobian=None,
                                   jacobian_reason="nonregular active distance")
    result = run(values)
    guard = result["protections"]["R3"]
    assert guard["status"] == "INSUFFICIENT_EVIDENCE"
    assert guard["current"]["queries"] == 1 and guard["current"]["scored_queries"] == 0
    assert guard["current"]["missing_record_ids"] == ["low"]
    values[3]["records"][0]["jacobian"] = np.zeros((10, 3)).tolist()
    with pytest.raises(ValueError, match="null"):
        run(values)


def test_candidate_failure_is_charged_and_cannot_be_dropped():
    values = inputs()
    values[3]["records"][0] = dict(record_id="low", status="unavailable", reason="inference failed")
    result = run(values)
    assert result["decision"] == "NO_GO"
    assert "candidate_prediction_failures" in result["failed_checks"]
    assert result["primary"]["current"]["value"] is None
    values[3]["records"].pop(0)
    with pytest.raises(ValueError, match="every frozen query"):
        run(values)


def test_macro_aggregation_order_is_explicit_and_not_sample_micro_average():
    values = inputs()
    for row, source in zip(values[1]["records"], ("A", "A", "B")):
        row["source_group_id"] = source
    for row, error in zip(values[2]["records"], (1., 3., 2.)):
        row["moments"] = [1.+error]*10
    result = run(values)
    expected = (np.sqrt(5.)+2.)/2
    assert result["source_macro_moment"]["epoch0"]["value"] == pytest.approx(expected)
    values[0]["primary_aggregation"] = "source_macro_mean_query_rmse"
    assert run(values)["source_macro_moment"]["epoch0"]["value"] == pytest.approx(2.)
    values[0].pop("primary_aggregation")
    with pytest.raises(ValueError, match="explicitly declare"):
        run(values)


def test_scaled_derivative_protection_uses_frozen_scales_and_preserves_masks():
    values = inputs()
    values[0]["protected_jacobian_metric"] = "scaled_jacobian_rmse"
    for row in values[1]["records"]:
        d = gate.DIMENSIONS[row["derivative_type"]]
        row["jacobian_scale"] = np.full((10, d), 2.).tolist()
    result = run(values)
    assert result["protections"]["R8"]["current"]["value"] == pytest.approx(.45)
    assert result["protections"]["R8"]["relative_change"] == pytest.approx(-.1)
    values[1]["records"][0]["jacobian_scale"][0][0] = 0.
    with pytest.raises(ValueError, match="positive"):
        run(values)


def test_scaled_retention_protection_does_not_change_the_raw_primary():
    values = inputs()
    values[0]["protected_retention_metric"] = "scaled10_moment_rmse"
    for row in values[1]["records"]:
        row["moment_scale"] = np.arange(1., 11.).tolist()
    result = run(values)
    assert result["primary"]["current"]["value"] == pytest.approx(.9)
    assert result["protections"]["high"]["current"]["value"] == pytest.approx(
        np.sqrt(np.mean((.9/np.arange(1., 11.))**2)))
    values[1]["records"][2].pop("moment_scale")
    with pytest.raises(ValueError, match="frozen per-query moment_scale"):
        run(values)


@pytest.mark.parametrize("field", ["run_id", "training_manifest_sha256", "config_sha256",
                                    "evaluation_manifest_sha256", "teacher_sha256", "protocol_sha256"])
def test_candidate_compares_only_to_its_own_matched_e0(field):
    values = inputs()
    values[3]["metadata"][field] = "different-run" if field == "run_id" else "9"*64
    with pytest.raises(ValueError, match="identity mismatch"):
        run(values)


def test_chart_mismatch_and_nonfinite_valid_arrays_are_rejected():
    values = inputs()
    values[3]["records"][0]["chart_id"] = "different-chart"
    with pytest.raises(ValueError, match="chart mismatch"):
        run(values)
    values = inputs()
    values[3]["records"][0]["moments"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run(values)


def test_zero_epoch0_primary_error_does_not_invent_relative_improvement():
    values = inputs()
    for role in (2, 3):
        for row in values[role]["records"]:
            row["moments"] = [1.]*10
    result = run(values)
    assert result["primary"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"


def test_registered_rules_cannot_be_relaxed_or_blind_data_opened():
    values = inputs()
    values[0]["protection_max_ratio"] = 1.3
    with pytest.raises(ValueError, match="may not be changed"):
        run(values)
    values = inputs()
    values[1]["metadata"]["scope"] = "blind"
    with pytest.raises(ValueError, match="development"):
        run(values)


def bundle(tmp_path):
    values = list(inputs())
    assets = {}
    for role in ("registered_protocol", "epoch0_checkpoint", "candidate_checkpoint"):
        path = tmp_path / (role + ".txt")
        path.write_text("explicit synthetic " + role)
        assets[role] = dict(path=path.name, sha256=gate.sha256(path))
    config = {"seed": 830251, "scope": "synthetic"}
    for role, data in (("config", config), ("training_manifest", {"scope":"synthetic"}),
                       ("evaluation_manifest", {"scope":"synthetic"})):
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(data))
        assets[role] = dict(path=path.name, sha256=gate.sha256(path))
    for document in values[1:]:
        document["metadata"]["evaluation_manifest_sha256"] = assets["evaluation_manifest"]["sha256"]
        if document["metadata"].get("role") in ("epoch0", "candidate"):
            document["metadata"]["training_manifest_sha256"] = assets["training_manifest"]["sha256"]
            document["metadata"]["config_sha256"] = gate.canonical_sha256(config)
    values[0]["registered_protocol_sha256"] = assets["registered_protocol"]["sha256"]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(values[0]))
    assets["protocol"] = dict(path=protocol.name, sha256=gate.sha256(protocol))
    for role, document in zip(("labels", "epoch0", "candidate", "no_learning"), values[1:]):
        if role in ("epoch0", "candidate"):
            document["metadata"]["protocol_sha256"] = assets["protocol"]["sha256"]
            document["metadata"]["checkpoint_sha256"] = assets[role + "_checkpoint"]["sha256"]
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(document))
        assets[role] = dict(path=path.name, sha256=gate.sha256(path))
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(dict(schema=gate.BUNDLE_SCHEMA, artifacts=assets)))
    return path


def test_file_loader_checks_checkpoint_protocol_and_prediction_bytes(tmp_path):
    path = bundle(tmp_path)
    result = gate.load_and_screen(path)
    assert result["decision"] == "GO" and len(result["verified_artifacts"]) == 11
    (tmp_path / "candidate_checkpoint.txt").write_text("changed checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        gate.load_and_screen(path)


def test_self_consistent_file_hashes_cannot_hide_mismatched_checkpoint_binding(tmp_path):
    path = bundle(tmp_path)
    prediction = tmp_path / "candidate.json"
    doc = json.loads(prediction.read_text())
    doc["metadata"]["checkpoint_sha256"] = "8"*64
    prediction.write_text(json.dumps(doc))
    spec = json.loads(path.read_text())
    spec["artifacts"]["candidate"]["sha256"] = gate.sha256(prediction)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="checkpoint identity"):
        gate.load_and_screen(path)


def test_absent_required_baseline_is_not_an_automatic_pass():
    p, labels, e0, candidate, _ = inputs()
    result = gate.screen(p, labels, e0, candidate)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["insufficient_evidence"] == ["no_learning_baseline"]
