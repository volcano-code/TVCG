from __future__ import annotations

import hashlib
import json

import pytest

from experiments.compact_pino_v21.runtime_efficiency.coverage import audit_manifest, main


def inventory(tmp_path):
    rows = []
    for i, role in enumerate(("train", "development")):
        case = tmp_path/f"case{i}.json"
        case.write_text(json.dumps({"meta": {
            "split": role, "source_group": f"source{i}", "geometry_id": f"g{i}",
            "geometry_sha256": str(i)*64, "query_id": f"q{i}",
            "derivative_valid": i == 0, "derivative_dimension": 3,
        }, "assets": {"targets": {"path": "MUST_NOT_BE_READ"}}}))
        rows.append(dict(record_id=f"r{i}", role=role, loader_eligible=True,
                         case_path=case.name, case_sha256=hashlib.sha256(case.read_bytes()).hexdigest()))
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(dict(schema="compact_pino.training-manifest.v1",
                                   snapshot_id="frozen", registered_train_count=1, records=rows)))
    return path


def test_metadata_counts_do_not_claim_loaded_labels_or_missing_stage(tmp_path):
    path = inventory(tmp_path)
    report = audit_manifest(path)
    assert report["verification_mode"] == "metadata_only"
    assert report["splits"]["train"]["declared_derivative_qualified"]["R3"] == 1
    assert report["splits"]["train"]["moment_loader_verified"] is None
    assert report["splits"]["train"]["jacobian_loader_qualified"] is None
    assert report["records"][0]["trajectory_stage"] == "UNKNOWN"
    assert report["records"][0]["retention_bin"] == "UNKNOWN"
    assert report["optimizer_journal"] is None
    assert report["source_split_consistent"] is True
    assert report["training_or_promotion_authorized"] is False


def test_case_checksum_is_verified_before_coverage(tmp_path):
    path = inventory(tmp_path)
    case = tmp_path/"case0.json"
    case.write_text(case.read_text()+" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit_manifest(path)


def test_source_and_trajectory_leakage_reported_without_relabeling(tmp_path):
    path = inventory(tmp_path)
    document = json.loads(path.read_text())
    case = tmp_path/"case1.json"
    data = json.loads(case.read_text())
    data["meta"]["source_group"] = "source0"
    case.write_text(json.dumps(data))
    document["records"][1]["case_sha256"] = hashlib.sha256(case.read_bytes()).hexdigest()
    document["records"][0]["trajectory_id"] = "same-trajectory"
    document["records"][1]["trajectory_id"] = "same-trajectory"
    path.write_text(json.dumps(document))
    report = audit_manifest(path)
    assert report["source_or_identity_overlap"]["source_group_id"] == ["source0"]
    assert report["source_or_identity_overlap"]["trajectory_id"] == ["same-trajectory"]
    assert not report["source_split_consistent"]


def test_actual_optimizer_exposure_is_distinct_from_declared_coverage(tmp_path):
    path = inventory(tmp_path)
    steps = tmp_path/"steps.jsonl"
    steps.write_text(json.dumps({
        "record_ids": ["r0", "r0"], "committed": True, "moment_mask": [True, True],
        "jacobian_mask": [True, False], "moment_denominator": 4.,
        "jacobian_denominator": 1.})+"\n"+json.dumps({
        "record_ids": ["r0"], "committed": False})+"\n")
    report = audit_manifest(path, steps=steps)
    journal = report["optimizer_journal"]
    assert journal["training_forward_exposures"] == 3
    assert journal["optimizer_committed_exposures"] == 2
    assert journal["unique_training_records_exposed"] == 1
    assert journal["objectives"]["moment"]["positive_memberships"] == 2
    assert journal["objectives"]["jacobian"]["positive_memberships"] == 1
    assert journal["objectives"]["moment"]["denominator_sum"] == 4.
    assert report["records"][1]["optimizer_committed_exposures"] == 0
    assert report["records"][1]["evaluation_exposures"] is None


def test_development_optimizer_exposure_is_never_silently_accepted(tmp_path):
    path = inventory(tmp_path)
    steps = tmp_path/"steps.jsonl"
    steps.write_text(json.dumps({"record_ids": ["r1"], "committed": False})+"\n")
    with pytest.raises(ValueError, match="development"):
        audit_manifest(path, steps=steps)


def test_cli_writes_a_readable_coverage_receipt(tmp_path):
    path = inventory(tmp_path)
    output = tmp_path/"out"/"coverage.json"
    assert main(["--manifest", str(path), "--output", str(output)]) == 0
    data = json.loads(output.read_text())
    assert data["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(data["cells"]) == 2


def test_verified_mode_uses_real_existing_synthetic_loader(tmp_path):
    from experiments.compact_pino_v21.training.smoke import make_smoke_manifest
    path = make_smoke_manifest(tmp_path/"fixture")
    report = audit_manifest(path, verify_load=True, mode="smoke", cache_directory=tmp_path/"cache")
    assert report["splits"]["train"]["moment_loader_verified"] == 2
    assert report["splits"]["train"]["jacobian_loader_qualified"]["R3"] == 2
    assert all(row["retention_bin"] == "middle" for row in report["records"])
    assert report["training_or_promotion_authorized"] is False
