"""Audit the current training manifest without inventing coverage or exposure.

Default mode checks case JSON hashes and declared metadata only; it does not
call the teacher or grant label/training admission. --verify-load delegates to
the existing training loader for its authority, asset and qualification checks.
Optional steps JSONL is an optimizer journal, never a development evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "compact_pino.training-manifest.v1"
UNKNOWN = "UNKNOWN"
OBJECTIVES = ("moment", "jacobian", "residual", "constraint")


def _read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError("nonfinite JSON value: "+x)))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _known(*values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return UNKNOWN


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("sample weights must be finite nonnegative numbers")
    if not math.isfinite(value) or value < 0:
        raise ValueError("sample weights must be finite nonnegative numbers")
    return float(value)


def _bool_or_unknown(value):
    return value if type(value) is bool else None


def _read_records(path, *, verify_load=False, mode="trial", cache_directory=None):
    path = Path(path).resolve(strict=True)
    manifest = _read(path)
    if manifest.get("schema") != SCHEMA or not manifest.get("snapshot_id"):
        raise ValueError("expected a versioned training-manifest.v1")
    raw = manifest.get("records")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest records must be nonempty")
    records, seen = [], set()
    for row in raw:
        rid = row.get("record_id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("record IDs must be nonempty and unique")
        seen.add(rid)
        if row.get("role") not in ("train", "development"):
            raise ValueError("coverage audit only reads training/development cases")
        if type(row.get("loader_eligible")) is not bool:
            raise ValueError("loader eligibility must be explicit")
        case_path = Path(row["case_path"])
        case_path = case_path if case_path.is_absolute() else path.parent/case_path
        if _sha(case_path) != row.get("case_sha256"):
            raise ValueError("case JSON SHA256 mismatch: "+rid)
        spec = _read(case_path)
        meta = spec.get("meta", {})
        if meta.get("split") in ("test", "locked_test", "fixed_benchmark", "benchmark"):
            raise ValueError("locked test/benchmark cases are outside this audit")
        if meta.get("split") not in (None, "synthetic", row["role"]):
            raise ValueError("case split differs from declared role: "+rid)
        dimension = meta.get("derivative_dimension")
        declared = _bool_or_unknown(meta.get("derivative_valid"))
        derivative_type = {3: "R3", 8: "R8", 10: "Full-J"}.get(dimension, UNKNOWN)
        if declared and derivative_type == UNKNOWN:
            raise ValueError("declared qualified derivative has no valid dimension: "+rid)
        records.append(dict(
            record_id=rid, role=row["role"], loader_eligible=row["loader_eligible"],
            query_id=_known(meta.get("query_id")),
            geometry_id=_known(meta.get("geometry_id")),
            geometry_sha256=_known(meta.get("geometry_sha256")),
            source_group_id=_known(meta.get("source_group"), meta.get("source_group_id")),
            surface_family=_known(row.get("surface_family"), meta.get("surface_family"), meta.get("surface_type")),
            retention_bin=_known(row.get("retention_bin"), meta.get("retention_bin")),
            trajectory_stage=_known(row.get("trajectory_stage"), meta.get("trajectory_stage")),
            trajectory_id=_known(row.get("trajectory_id"), meta.get("trajectory_id")),
            task_id=_known(row.get("task_id"), meta.get("task_id")),
            derivative_type=derivative_type, derivative_declared_valid=declared,
            moment_declared_valid=_bool_or_unknown(row.get("moment_valid", meta.get("moment_valid"))),
            moment_loader_verified=None, jacobian_loader_qualified=None,
            residual_loader_qualified=None, retention_ratio_native=None,
            moment_weight=_weight(row.get("moment_weight", 1.)),
            jacobian_weight=_weight(row.get("jacobian_weight", 1.)),
            residual_weight=_weight(row.get("residual_weight", row.get("moment_weight", 1.))),
            case_path=str(case_path.resolve()), case_sha256=row["case_sha256"],
        ))
    if verify_load:
        from ..training.manifest import load_training_manifest
        from ..p0_diagnostics.promotion_gate import retention_bin
        data = load_training_manifest(path, mode=mode, cache_directory=cache_directory)
        loaded = {row["record_id"]: row for row in data.records}
        for record in records:
            row = loaded[record["record_id"]]
            case = row.get("case")
            if case is None:
                continue
            record["moment_loader_verified"] = True
            record["jacobian_loader_qualified"] = bool(case["meta"]["derivative_valid"])
            record["residual_loader_qualified"] = bool(row.get("residual_valid"))
            full = float(case["native_full_editable_moments"][0])
            if not math.isfinite(full) or full <= 0:
                raise ValueError("native editable mass must be finite and positive")
            ratio = float((case["native_moments"][0]-case["fixed_moments"][0])/full)
            if not math.isfinite(ratio) or ratio < -1e-12 or ratio > 1+1e-12:
                raise ValueError("native retained editable ratio outside [0,1]")
            ratio = min(1., max(0., ratio))
            actual_bin = retention_bin(ratio)
            if record["retention_bin"] != UNKNOWN and record["retention_bin"] != actual_bin:
                raise ValueError("declared retention bin differs from native label: "+record["record_id"])
            record["retention_bin"] = actual_bin
            record["retention_ratio_native"] = ratio
    return manifest, records


def _group_summary(rows, *, loader_verified):
    dimensions = ("source_group_id", "geometry_id", "trajectory_id", "task_id")
    result = dict(records=len(rows), loader_eligible=sum(r["loader_eligible"] for r in rows))
    for dimension in dimensions:
        result[dimension+"_distinct_known"] = len({r[dimension] for r in rows if r[dimension] != UNKNOWN})
        result[dimension+"_unknown_records"] = sum(r[dimension] == UNKNOWN for r in rows)
    result["declared_derivative_qualified"] = {
        kind: sum(r["derivative_declared_valid"] is True and r["derivative_type"] == kind for r in rows)
        for kind in ("R3", "R8", "Full-J")
    }
    result["derivative_qualification_unknown"] = sum(r["derivative_declared_valid"] is None for r in rows)
    result["moment_loader_verified"] = (sum(r["moment_loader_verified"] is True for r in rows)
                                         if loader_verified else None)
    result["jacobian_loader_qualified"] = ({
        kind: sum(r["jacobian_loader_qualified"] is True and r["derivative_type"] == kind for r in rows)
        for kind in ("R3", "R8", "Full-J")} if loader_verified else None)
    result["residual_loader_qualified"] = (
        sum(r["residual_loader_qualified"] is True for r in rows) if loader_verified else None)
    return result


def _audit_steps(path, records, manifest_sha256):
    """Count journaled attempts and updates; never infer missing loss masks."""
    by_id = {r["record_id"]: r for r in records}
    attempted, committed = Counter(), Counter()
    objectives = {name: dict(committed_steps_with_mask=0, positive_memberships=0,
                             denominator_sum=0., committed_steps_with_denominator=0)
                  for name in OBJECTIVES}
    attempts = updates = 0
    explicit_manifest_bindings = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        step = json.loads(line)
        ids = step.get("record_ids")
        if not isinstance(ids, list) or not ids or type(step.get("committed")) is not bool:
            raise ValueError("steps must be optimizer receipts with record_ids and committed")
        for rid in ids:
            if rid not in by_id:
                raise ValueError("optimizer journal contains unknown record: "+str(rid))
            if by_id[rid]["role"] != "train":
                raise ValueError("development record appears in optimizer journal: "+rid)
            if not by_id[rid]["loader_eligible"]:
                raise ValueError("ineligible record appears in optimizer journal: "+rid)
        binding = step.get("training_manifest_sha256")
        if binding is not None:
            if binding != manifest_sha256:
                raise ValueError("optimizer journal manifest binding mismatch")
            explicit_manifest_bindings += 1
        attempts += 1
        attempted.update(ids)
        if not step["committed"]:
            continue
        updates += 1
        committed.update(ids)
        for name in OBJECTIVES:
            mask = step.get(name+"_mask")
            if mask is not None:
                if not isinstance(mask, list) or len(mask) != len(ids) or any(type(v) is not bool for v in mask):
                    raise ValueError("objective mask must align with record_ids")
                objectives[name]["committed_steps_with_mask"] += 1
                objectives[name]["positive_memberships"] += sum(mask)
            denominator = step.get(name+"_denominator")
            if denominator is not None:
                objectives[name]["denominator_sum"] += _weight(denominator)
                objectives[name]["committed_steps_with_denominator"] += 1
    for record in records:
        rid = record["record_id"]
        record["training_forward_exposures"] = attempted[rid]
        record["optimizer_committed_exposures"] = committed[rid]
        record["evaluation_exposures"] = None
    return dict(
        path=str(Path(path).resolve()), sha256=_sha(path), attempts=attempts,
        committed_steps=updates, training_forward_exposures=sum(attempted.values()),
        optimizer_committed_exposures=sum(committed.values()),
        unique_training_records_exposed=len(attempted), objectives=objectives,
        manifest_hash_bound_steps=explicit_manifest_bindings,
        identity_scope="record IDs and train roles checked; only explicitly hash-bound rows bind manifest bytes",
        evaluation_exposures=None,
        evaluation_scope="evaluation exposures are not inferred from an optimizer journal",
    )


def audit_manifest(path, *, verify_load=False, mode="trial", cache_directory=None, steps=None):
    """Return source/surface/retention/stage coverage for the exact manifest."""
    path = Path(path).resolve(strict=True)
    manifest, records = _read_records(path, verify_load=verify_load, mode=mode, cache_directory=cache_directory)
    digest = _sha(path)
    steps_report = _audit_steps(steps, records, digest) if steps else None
    if steps is None:
        for record in records:
            record.update(training_forward_exposures=None, optimizer_committed_exposures=None,
                          evaluation_exposures=None)
    fields = ("role", "source_group_id", "surface_family", "retention_bin", "trajectory_stage")
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in fields)].append(row)
    cells = [dict(zip(fields, key), **_group_summary(value, loader_verified=verify_load))
             for key, value in sorted(groups.items())]
    leaks = {}
    for key in ("source_group_id", "geometry_id", "geometry_sha256", "trajectory_id"):
        train = {r[key] for r in records if r["role"] == "train" and r[key] != UNKNOWN}
        development = {r[key] for r in records if r["role"] == "development" and r[key] != UNKNOWN}
        leaks[key] = sorted(train & development)
    query_members = defaultdict(list)
    for r in records:
        if r["geometry_id"] != UNKNOWN and r["query_id"] != UNKNOWN:
            query_members[(r["geometry_id"], r["query_id"])].append(r["record_id"])
    duplicate_queries = [dict(geometry_id=k[0], query_id=k[1], record_ids=v)
                         for k, v in query_members.items() if len(v) > 1]
    return dict(
        schema="compact_pino.dataset-coverage.v1", manifest_path=str(path),
        manifest_sha256=digest, snapshot_id=manifest["snapshot_id"],
        registered_train_count=manifest.get("registered_train_count"),
        verification_mode="existing_training_loader" if verify_load else "metadata_only",
        qualification_scope=("existing loader authority/asset/qualification checks; teacher geometry is not rerun"
                             if verify_load else
                             "case JSON hashes and declared metadata only; loaded-label counts are null"),
        splits={role: _group_summary([r for r in records if r["role"] == role],
                                     loader_verified=verify_load)
                for role in ("train", "development")},
        cells=cells, records=records, source_or_identity_overlap=leaks,
        duplicate_query_identities=duplicate_queries, optimizer_journal=steps_report,
        source_split_consistent=not any(leaks.values()) and not duplicate_queries,
        training_or_promotion_authorized=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-load", action="store_true",
                        help="reuse existing loader checks; may load material and label assets")
    parser.add_argument("--mode", choices=("smoke", "trial", "long"), default="trial")
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--steps", type=Path, help="optimizer steps JSONL, not evaluation rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_manifest(args.manifest, verify_load=args.verify_load, mode=args.mode,
                            cache_directory=args.cache_directory, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(snapshot_id=report["snapshot_id"], verification_mode=report["verification_mode"],
                          splits=report["splits"], source_split_consistent=report["source_split_consistent"]),
                     allow_nan=False))
    return 0 if report["source_split_consistent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
