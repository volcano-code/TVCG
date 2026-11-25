"""Connect the actual TrainingData/evaluate output to the research screen.

No predictions, derivative certificates or GO decisions are manufactured here.
Every declared development row must have a loaded baseline case; incomplete
inventories must be resolved explicitly rather than silently dropping rows.
The evaluation protocol is an existing caller-supplied frozen file.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from . import promotion_gate as gate
from .baseline_cases import moment_scale
from .export_cases import verify_export_qualification


def _receipt(path, expected=None):
    path = Path(path).resolve(strict=True)
    actual = gate.sha256(path)
    if expected is not None and actual != expected:
        raise ValueError("source asset changed since loading: " + str(path))
    return dict(path=str(path), sha256=actual)


def _prediction_rows(value, labels):
    if isinstance(value, dict):
        value = value.get("predictions", value.get("records", value.get("rows")))
    rows = copy.deepcopy(value)
    indexed = gate._rows(rows, "runner evaluation")
    if set(indexed) != {row["record_id"] for row in labels}:
        raise ValueError("runner evaluation does not cover the frozen development record IDs")
    for row in rows:
        if row.get("status") == "unavailable" and not row.get("reason"):
            row["reason"] = row.get("failure_reason")
        if row.get("jacobian_status") != "available" and not row.get("jacobian_reason"):
            row["jacobian_reason"] = row.get("failure_reason") or "runner did not provide a qualified derivative"
    return rows


def _labels_from_training(data):
    if data.scope not in ("synthetic", "development_training"):
        raise ValueError("unsupported training data evidence scope")
    _receipt(data.path, data.manifest_sha256)
    development = [item for item in data.records if item["role"] == "development"]
    if not development or any("case" not in item for item in development):
        raise ValueError("screen export requires all declared development rows; unloaded rows cannot be silently dropped")
    records, source_assets, evaluation_rows = [], [], []
    scope = "synthetic" if data.scope == "synthetic" else "development"
    for item in development:
        case, rid = item["case"], item["record_id"]
        meta = case["meta"]
        case_file = _receipt(item["case_path"], item["case_sha256"])
        target = case["asset_receipts"]["targets"]
        target_file = _receipt(target["path"], target["sha256"])
        source_assets.append(dict(record_id=rid, targets_sha256=target_file["sha256"],
                                  teacher_version=meta["teacher_version"]))
        evaluation_rows.append(dict(record_id=rid, geometry_id=meta["geometry_id"],
                                    source_group_id=meta["source_group"], case_sha256=case_file["sha256"],
                                    targets_sha256=target_file["sha256"]))
        m = np.asarray(case["native_moments"], dtype=float)
        retained = float((m[0] - case["fixed_moments"][0]) / case["native_full_editable_moments"][0])
        if not np.isfinite(retained) or not -1e-12 <= retained <= 1.+1e-12:
            raise ValueError("native retained fraction is inconsistent with fixed/editable material")
        declared = meta["derivative_valid"]
        # A file's existence alone cannot qualify a real derivative. Recheck
        # its query/event/teacher binding and the copied original evidence.
        qualification = (verify_export_qualification(case, rid)
                         if scope != "synthetic" and declared else None)
        certified = bool(declared and (scope == "synthetic" or qualification is not None))
        d = meta.get("derivative_dimension")
        row = dict(record_id=rid, geometry_id=meta["geometry_id"], source_group_id=meta["source_group"],
                   retention_ratio=float(np.clip(retained, 0., 1.)), moment_valid=True,
                   native_moments=m.tolist(), moment_scale=moment_scale(meta["editable_mass"], meta["length_scale"]).tolist(),
                   derivative_valid=certified, declared_derivative_valid=declared,
                   derivative_type={3:"R3", 8:"R8", 10:"Full-J"}.get(d, "none"),
                   chart_id=meta["chart_id"])
        if certified:
            row.update(native_jacobian=np.asarray(case["native_jacobian"]).tolist(),
                       jacobian_scale=(np.asarray(row["moment_scale"])[:, None] /
                                       np.asarray(case["parameter_scales"])[None, :]).tolist(),
                       qualification_receipt_sha256=(qualification or target_file)["sha256"],
                       qualification_receipt_kind="bound_external_declaration" if qualification else "synthetic_analytic_target_asset",
                       qualification_independently_reaudited=False)
        else:
            row["derivative_reason"] = ("declared derivative has no source qualification receipt"
                                        if declared else meta["derivative_reason"])
        records.append(row)
    source_assets.sort(key=lambda row: row["record_id"])
    evaluation_rows.sort(key=lambda row: row["record_id"])
    return records, dict(schema="tvcg.v22.screen-evaluation-manifest.v1", scope=scope,
                          training_manifest_sha256=data.manifest_sha256, records=evaluation_rows), {
                              "teacher_sha256": gate.canonical_sha256(source_assets),
                              "teacher_identity_kind": "canonical_sorted_record_targets_sha256_and_teacher_version_v1",
                              "teacher_source_assets": source_assets}


def write_screen_bundle(data, *, protocol_path, registered_protocol_path, epoch0_evaluation,
                        candidate_evaluation, run_id, config, epoch0_checkpoint_path,
                        candidate_checkpoint_path, output_directory, no_learning_evaluation=None):
    """Export verified labels and actual runner rows, then return bundle.json.

    ``data`` is TrainingData from load_training_manifest. Evaluations are the
    runner's records/rows/predictions list, or the list itself. ``config`` must
    be the same canonical configuration stored in the two paired checkpoints.
    This function binds supplied predictions to supplied checkpoint bytes;
    it does not regenerate predictions or infer their provenance from weights.
    """
    protocol_path = Path(protocol_path).resolve(strict=True)
    registered_protocol_path = Path(registered_protocol_path).resolve(strict=True)
    protocol = gate._read_json(protocol_path)
    expected_ids = gate._protocol(protocol)
    if protocol["registered_protocol_sha256"] != gate.sha256(registered_protocol_path):
        raise ValueError("registered protocol identity mismatch")
    records, evaluation_manifest, teacher = _labels_from_training(data)
    if expected_ids != {row["record_id"] for row in records}:
        raise ValueError("frozen screening protocol differs from the actual development set")
    gate._text(run_id, "run_id")
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError("screen export requires a new directory")
    assets = dict(protocol=_receipt(protocol_path), registered_protocol=_receipt(registered_protocol_path),
                  training_manifest=_receipt(data.path, data.manifest_sha256),
                  epoch0_checkpoint=_receipt(epoch0_checkpoint_path), candidate_checkpoint=_receipt(candidate_checkpoint_path))
    scope = evaluation_manifest["scope"]
    config_digest = gate.canonical_sha256(config)
    directory.mkdir(parents=True, exist_ok=False)
    def write(role, document):
        destination = directory / (role + ".json")
        destination.write_text(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf8")
        assets[role] = _receipt(destination)
    write("config", config)
    write("evaluation_manifest", evaluation_manifest)
    label_meta = dict(scope=scope, moment_order=list(gate.ORDER),
                      evaluation_manifest_sha256=assets["evaluation_manifest"]["sha256"], **teacher)
    labels = dict(schema=gate.LABEL_SCHEMA, metadata=label_meta, records=records)
    write("labels", labels)
    for role, evaluation in (("epoch0", epoch0_evaluation), ("candidate", candidate_evaluation),
                             ("no_learning", no_learning_evaluation)):
        if evaluation is None:
            if role != "no_learning":
                raise ValueError("both E0 and candidate evaluations are required")
            continue
        metadata = dict(label_meta, role=role, run_id=run_id,
                        training_manifest_sha256=data.manifest_sha256, config_sha256=config_digest,
                        protocol_sha256=assets["protocol"]["sha256"],
                        checkpoint_sha256=assets[role + "_checkpoint"]["sha256"] if role != "no_learning" else None)
        if role == "no_learning":
            metadata["model_weights_used"] = False
        document = dict(schema=gate.PREDICTION_SCHEMA, metadata=metadata,
                        records=_prediction_rows(evaluation, records))
        gate._predictions(document, role, gate._rows(records, "labels"), label_meta)
        write(role, document)
    bundle = directory / "bundle.json"
    bundle.write_text(json.dumps(dict(schema=gate.BUNDLE_SCHEMA, artifacts=assets), indent=2, allow_nan=False)+"\n", encoding="utf8")
    return bundle
