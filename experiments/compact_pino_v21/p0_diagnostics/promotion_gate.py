"""Recompute the frozen research screen from paired query-level predictions.

This is a research screen, not a training authorization or native deployment
acceptance. The aggregation order is explicit: the older prose protocol alone
does not distinguish a group RMSE from a mean of per-query RMSEs. Input assets
are checked by SHA256; their producer remains responsible for truthful labels,
qualified derivative charts, and freezing the protocol before evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ORDER = ("1", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
PROTOCOL_SCHEMA = "tvcg.v22.research-screen-protocol.v1"
LABEL_SCHEMA = "tvcg.v22.research-screen-labels.v1"
PREDICTION_SCHEMA = "tvcg.v22.research-screen-predictions.v1"
BUNDLE_SCHEMA = "tvcg.v22.research-screen-bundle.v1"
REPORT_SCHEMA = "tvcg.v22.research-screen.v1"
AGGREGATIONS = ("source_macro_group_rmse", "source_macro_mean_query_rmse")
DIMENSIONS = {"R3": 3, "R8": 8, "Full-J": 10}
IDENTITY = ("run_id", "training_manifest_sha256", "config_sha256",
            "evaluation_manifest_sha256", "teacher_sha256", "protocol_sha256")


def _hash(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


def _read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def _rows(records, name):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name} requires a nonempty records list")
    indexed = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"{name} record must be an object")
        key = _text(row.get("record_id"), "record_id")
        if key in indexed:
            raise ValueError(f"duplicate {name} record_id: {key}")
        indexed[key] = row
    return indexed


def retention_bin(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("retention_ratio must be finite in [0,1]")
    return "near_zero" if value < .01 else "low" if value < .1 else "middle" if value < .7 else "high"


def _protocol(protocol):
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported research screen protocol schema")
    _text(protocol.get("protocol_id"), "protocol_id")
    _hash(protocol.get("registered_protocol_sha256"), "registered_protocol_sha256")
    for field in ("primary_aggregation", "jacobian_aggregation"):
        if protocol.get(field) not in AGGREGATIONS:
            raise ValueError(f"{field} must explicitly declare a supported aggregation")
    if protocol.get("primary_relative_improvement") != .05 or protocol.get("protection_max_ratio") != 1.05:
        raise ValueError("the existing 5% improvement and 5% protection rules may not be changed here")
    if protocol.get("protected_retention_metric") not in ("raw_mass_rmse", "raw10_moment_rmse",
                                                          "scaled_mass_rmse", "scaled10_moment_rmse"):
        raise ValueError("declare the original raw/scaled middle/high mass or ten-moment protection metric")
    if protocol.get("protected_jacobian_metric") not in ("raw_jacobian_rmse", "scaled_jacobian_rmse"):
        raise ValueError("declare the original qualified-J protection metric")
    _boolean(protocol.get("require_no_learning_baseline"), "require_no_learning_baseline")
    ids = protocol.get("expected_record_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("expected_record_ids must freeze a nonempty unique evaluation set")
    for value in ids:
        _text(value, "expected_record_id")
    return set(ids)


def _labels(document, expected_ids):
    if document.get("schema") != LABEL_SCHEMA:
        raise ValueError("unsupported screen label schema")
    meta = document.get("metadata", {})
    if meta.get("scope") not in ("development", "synthetic"):
        raise ValueError("research screening accepts only frozen development or explicit synthetic data")
    if tuple(meta.get("moment_order", ())) != ORDER:
        raise ValueError("raw undoubled moment order mismatch")
    for field in ("evaluation_manifest_sha256", "teacher_sha256"):
        _hash(meta.get(field), field)
    rows = _rows(document.get("records"), "label")
    if set(rows) != expected_ids:
        raise ValueError("label record identities differ from the frozen evaluation set")
    for key, row in rows.items():
        for field in ("geometry_id", "source_group_id"):
            _text(row.get(field), field)
        row_bin = retention_bin(row.get("retention_ratio"))
        if "retention_bin" in row and row["retention_bin"] != row_bin:
            raise ValueError("retention_bin disagrees with the registered fraction boundaries")
        valid_m = _boolean(row.get("moment_valid"), "moment_valid")
        valid_j = _boolean(row.get("derivative_valid"), "derivative_valid")
        if valid_m:
            _array(row.get("native_moments"), (10,), key + " native moments")
            if "moment_scale" in row and np.any(_array(row["moment_scale"], (10,), "moment_scale") <= 0):
                raise ValueError("Moment scales must be positive")
        else:
            _text(row.get("moment_reason"), "invalid Moment reason")
        if valid_j:
            if not valid_m or row.get("derivative_type") not in DIMENSIONS:
                raise ValueError("qualified J requires valid Moment and an explicit Full-J/R3/R8 type")
            d = DIMENSIONS[row["derivative_type"]]
            _text(row.get("chart_id"), "chart_id")
            _hash(row.get("qualification_receipt_sha256"), "qualification receipt")
            _array(row.get("native_jacobian"), (10, d), key + " native Jacobian")
            if "jacobian_scale" in row:
                if np.any(_array(row["jacobian_scale"], (10, d), "jacobian_scale") <= 0):
                    raise ValueError("Jacobian scales must be positive")
        else:
            _text(row.get("derivative_reason"), "unqualified derivative reason")
    return meta, rows


def _predictions(document, role, labels, label_meta):
    if document.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("unsupported prediction schema")
    meta = document.get("metadata", {})
    if meta.get("role") != role or meta.get("scope") != label_meta["scope"]:
        raise ValueError("prediction role or evidence scope mismatch")
    if tuple(meta.get("moment_order", ())) != ORDER:
        raise ValueError("prediction moment order mismatch")
    for field in ("evaluation_manifest_sha256", "teacher_sha256"):
        if meta.get(field) != label_meta[field]:
            raise ValueError("prediction/label identity mismatch: " + field)
    if role != "no_learning":
        for field in IDENTITY:
            (_text if field == "run_id" else _hash)(meta.get(field), field)
        _hash(meta.get("checkpoint_sha256"), "checkpoint_sha256")
    elif meta.get("checkpoint_sha256") is not None or meta.get("model_weights_used") is not False:
        raise ValueError("no-learning response must explicitly use no checkpoint or learned weights")
    rows = _rows(document.get("records"), role)
    if set(rows) != set(labels):
        raise ValueError(role + " must include every frozen query, including explicit failures")
    errors, jac_errors, failures, missing_j = {}, {}, {}, []
    for key, target in labels.items():
        row = rows[key]
        if row.get("status") not in ("available", "unavailable"):
            raise ValueError("prediction status must be available or unavailable")
        if row["status"] == "unavailable":
            failures[key] = _text(row.get("reason"), "unavailable prediction reason")
            continue
        moments = _array(row.get("moments"), (10,), key + " predicted moments")
        if target["moment_valid"]:
            errors[key] = moments - np.asarray(target["native_moments"])
        if target["derivative_valid"]:
            if row.get("jacobian_status") != "available":
                if row.get("jacobian") is not None:
                    raise ValueError("unavailable model derivative must remain null, never a zero-filled label")
                _text(row.get("jacobian_reason"), "unavailable Jacobian reason")
                missing_j.append(key)
                continue
            if row.get("chart_id") != target["chart_id"]:
                raise ValueError("predicted Jacobian chart mismatch")
            d = DIMENSIONS[target["derivative_type"]]
            jac_errors[key] = _array(row.get("jacobian"), (10, d), key + " predicted Jacobian") - np.asarray(target["native_jacobian"])
    return meta, errors, jac_errors, dict(unavailable_predictions=failures, unavailable_qualified_jacobians=missing_j)


def _metric(errors, rows, ids, aggregation, *, component=None, scale_kind=None):
    selected = set(ids)
    missing = sorted(selected - set(errors))
    groups = {}
    for key in sorted(selected & set(errors)):
        value = np.asarray(errors[key])
        if scale_kind is not None:
            if scale_kind not in rows[key]:
                raise ValueError("scaled protection requires frozen per-query " + scale_kind)
            value = value / np.asarray(rows[key][scale_kind])
        if component is not None:
            value = value[component:component+1]
        # Query averaging is explicit: different derivative dimensions do not
        # change per-query weights, and each protected dimension is also separate.
        mse = float(np.mean(np.square(value)))
        if not np.isfinite(mse):
            raise ValueError("squared prediction error overflowed; no finite screen can be reported")
        groups.setdefault(rows[key]["source_group_id"], []).append(mse)
    source = {name: dict(queries=len(values), value=float(np.sqrt(np.mean(values)))
                        if aggregation == "source_macro_group_rmse" else float(np.mean(np.sqrt(values))))
              for name, values in groups.items()}
    value = float(np.mean([v["value"] for v in source.values()])) if source and not missing else None
    return dict(value=value, queries=len(selected), scored_queries=len(selected)-len(missing),
                source_groups=len(source), missing_record_ids=missing, by_source=source,
                aggregation=aggregation)


def _relative(current, initial):
    if current is None or initial is None or initial == 0:
        return None
    return float(current / initial - 1.)


def _comparison(current, initial, *, multiplier, primary=False):
    if current["value"] is None or initial["value"] is None:
        return dict(status="INSUFFICIENT_EVIDENCE", current=current, epoch0=initial,
                    relative_change=None, limit=None)
    if primary and initial["value"] == 0:
        return dict(status="INSUFFICIENT_EVIDENCE", reason="relative improvement is undefined at zero E0 error",
                    current=current, epoch0=initial, relative_change=None, limit=0.)
    limit = multiplier * initial["value"]
    return dict(status="PASS" if current["value"] <= limit else "FAIL", current=current,
                epoch0=initial, relative_change=_relative(current["value"], initial["value"]), limit=limit)


def screen(protocol, labels, epoch0, candidate, *, no_learning=None):
    """Compute a matched research screen; file authenticity is checked by the CLI."""
    expected = _protocol(protocol)
    label_meta, rows = _labels(labels, expected)
    e0_meta, e0_m, e0_j, e0_missing = _predictions(epoch0, "epoch0", rows, label_meta)
    pred_meta, pred_m, pred_j, pred_missing = _predictions(candidate, "candidate", rows, label_meta)
    for field in IDENTITY:
        if e0_meta[field] != pred_meta[field]:
            raise ValueError("candidate and its own E0 identity mismatch: " + field)
    models = {"epoch0": (e0_m, e0_j), "candidate": (pred_m, pred_j)}
    missing = {"epoch0": e0_missing, "candidate": pred_missing}
    if no_learning is not None:
        _, base_m, base_j, base_missing = _predictions(no_learning, "no_learning", rows, label_meta)
        models["no_learning"] = (base_m, base_j)
        missing["no_learning"] = base_missing
    main_ids = {key for key, row in rows.items() if row["moment_valid"]}
    moment_agg, j_agg = protocol["primary_aggregation"], protocol["jacobian_aggregation"]
    main = {name: _metric(m, rows, main_ids, moment_agg) for name, (m, _) in models.items()}
    primary = _comparison(main["candidate"], main["epoch0"], multiplier=.95, primary=True)
    protections = {}
    for name in ("middle", "high"):
        ids = {key for key in main_ids if retention_bin(rows[key]["retention_ratio"]) == name}
        retention_metric = protocol["protected_retention_metric"]
        component = 0 if retention_metric in ("raw_mass_rmse", "scaled_mass_rmse") else None
        scale_kind = "moment_scale" if retention_metric.startswith("scaled") else None
        values = {model: _metric(m, rows, ids, moment_agg, component=component, scale_kind=scale_kind)
                  for model, (m, _) in models.items()}
        protections[name] = _comparison(values["candidate"], values["epoch0"], multiplier=1.05)
        protections[name]["metric"] = protocol["protected_retention_metric"]
    for name in DIMENSIONS:
        ids = {key for key, row in rows.items() if row["derivative_valid"] and row["derivative_type"] == name}
        values = {model: _metric(j, rows, ids, j_agg,
                                scale_kind="jacobian_scale" if protocol["protected_jacobian_metric"] == "scaled_jacobian_rmse" else None)
                  for model, (_, j) in models.items()}
        protections[name] = _comparison(values["candidate"], values["epoch0"], multiplier=1.05)
        protections[name]["metric"] = protocol["protected_jacobian_metric"]
    strata = {}
    for name in ("near_zero", "low", "middle", "high"):
        ids = {key for key in main_ids if retention_bin(rows[key]["retention_ratio"]) == name}
        strata[name] = {model: _metric(m, rows, ids, moment_agg) for model, (m, _) in models.items()}
    components = {model: {component: _metric(m, rows, main_ids, moment_agg, component=i)
                          for i, component in enumerate(ORDER)} for model, (m, _) in models.items()}
    failed = (["primary"] if primary["status"] == "FAIL" else [])
    failed += [name for name, result in protections.items() if result["status"] == "FAIL"]
    insufficient = (["primary"] if primary["status"] == "INSUFFICIENT_EVIDENCE" else [])
    insufficient += [name for name, result in protections.items() if result["status"] == "INSUFFICIENT_EVIDENCE"]
    if not strata["low"]["candidate"]["queries"]:
        insufficient.append("low_retention_denominator")
    all_sources = {row["source_group_id"] for row in rows.values()}
    scored_sources = {rows[key]["source_group_id"] for key in main_ids}
    if all_sources - scored_sources:
        insufficient.append("source_groups_without_qualified_moments")
    if any(key in main_ids for key in pred_missing["unavailable_predictions"]):
        failed.append("candidate_prediction_failures")
    if protocol["require_no_learning_baseline"] and ("no_learning" not in main or main["no_learning"]["value"] is None):
        insufficient.append("no_learning_baseline")
    baseline_gap = (_relative(main["candidate"]["value"], main["no_learning"]["value"])
                    if "no_learning" in main else None)
    decision = "NO_GO" if failed else "INSUFFICIENT_EVIDENCE" if insufficient else "GO"
    coverage = dict(registered_queries=len(rows), qualified_moment_queries=len(main_ids),
                    rejected_moment_queries=len(rows)-len(main_ids),
                    qualified_jacobian_queries=sum(row["derivative_valid"] for row in rows.values()),
                    qualified_jacobian_by_type={name: sum(row["derivative_valid"] and row["derivative_type"] == name
                                                       for row in rows.values()) for name in DIMENSIONS},
                    declared_but_unreceipted_jacobians=sum(row.get("declared_derivative_valid") is True and not row["derivative_valid"]
                                                          for row in rows.values()),
                    source_groups_without_qualified_moments=sorted(all_sources-scored_sources),
                    geometry_ids=sorted({row["geometry_id"] for row in rows.values()}),
                    source_group_ids=sorted({row["source_group_id"] for row in rows.values()}))
    return dict(schema=REPORT_SCHEMA, mode="research_screen", scope=label_meta["scope"], decision=decision,
                training_authorized=False, eligible_for_long_training_review=decision == "GO" and label_meta["scope"] == "development",
                physical_acceptance="NOT_EVALUATED", speedup_claim=False,
                identity={**{key: pred_meta[key] for key in IDENTITY},
                          "candidate_checkpoint_sha256": pred_meta["checkpoint_sha256"],
                          "epoch0_checkpoint_sha256": e0_meta["checkpoint_sha256"]},
                protocol_id=protocol["protocol_id"], primary=primary, protections=protections,
                failed_checks=failed, insufficient_evidence=insufficient,
                source_macro_moment=main, candidate_relative_to_no_learning=baseline_gap,
                baseline_gap_is_registered_hard_gate=False,
                by_retention=strata, by_component=components, coverage=coverage, availability=missing,
                aggregation_definitions={"source_macro_group_rmse": "mean over sources of sqrt(mean over that source's queries and components of squared raw errors))",
                                         "source_macro_mean_query_rmse": "mean over sources of mean over queries of sqrt(mean over components of squared raw errors))"},
                evidence_boundary="Hash checks and declared qualifications do not replace teacher audits; this research screen is not native physical acceptance or permission to train.")


def load_and_screen(bundle_path):
    """Resolve and verify the exact assets used for a command-line decision."""
    path = Path(bundle_path).resolve(strict=True)
    bundle = _read_json(path)
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported screen bundle schema")
    entries = bundle.get("artifacts", {})
    required = {"protocol", "registered_protocol", "labels", "epoch0", "candidate",
                "epoch0_checkpoint", "candidate_checkpoint", "training_manifest", "config",
                "evaluation_manifest"}
    if not required <= set(entries) or set(entries) - required - {"no_learning"}:
        raise ValueError("screen bundle requires exact protocol, labels, predictions and checkpoint assets")
    receipts, paths = {}, {}
    for name, entry in entries.items():
        expected = _hash(entry.get("sha256"), name + " artifact")
        asset = Path(_text(entry.get("path"), name + " path"))
        asset = (asset if asset.is_absolute() else path.parent / asset).resolve(strict=True)
        actual = sha256(asset)
        if actual != expected:
            raise ValueError("artifact checksum mismatch: " + name)
        paths[name] = asset
        receipts[name] = dict(path=str(asset), sha256=actual, bytes=asset.stat().st_size)
    protocol, labels = _read_json(paths["protocol"]), _read_json(paths["labels"])
    epoch0, candidate = _read_json(paths["epoch0"]), _read_json(paths["candidate"])
    if protocol.get("registered_protocol_sha256") != receipts["registered_protocol"]["sha256"]:
        raise ValueError("registered protocol identity mismatch")
    for role, document in (("epoch0", epoch0), ("candidate", candidate)):
        if document.get("metadata", {}).get("protocol_sha256") != receipts["protocol"]["sha256"]:
            raise ValueError("prediction used a different frozen evaluation protocol")
        if document["metadata"].get("checkpoint_sha256") != receipts[role + "_checkpoint"]["sha256"]:
            raise ValueError("prediction/checkpoint identity mismatch: " + role)
        if document["metadata"].get("training_manifest_sha256") != receipts["training_manifest"]["sha256"]:
            raise ValueError("prediction/training manifest identity mismatch")
        if document["metadata"].get("config_sha256") != canonical_sha256(_read_json(paths["config"])):
            raise ValueError("prediction/canonical training config identity mismatch")
    if labels.get("metadata", {}).get("evaluation_manifest_sha256") != receipts["evaluation_manifest"]["sha256"]:
        raise ValueError("label/evaluation manifest identity mismatch")
    result = screen(protocol, labels, epoch0, candidate,
                    no_learning=_read_json(paths["no_learning"]) if "no_learning" in paths else None)
    result["verified_artifacts"] = receipts
    result["bundle_receipt"] = dict(path=str(path), sha256=sha256(path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing screening receipt")
    result = load_and_screen(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
