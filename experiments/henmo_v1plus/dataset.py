"""Reuse the admitted Compact data format, with HENMO-specific audit bindings.

This module never creates labels, upgrades derivative qualifications, or reads
the sealed benchmark as training data. Coverage counts are counts of loaded
evidence, not predictions of generalization or a training authorization.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json

import numpy as np

from experiments.compact_pino_v21.training.manifest import load_training_manifest
from experiments.compact_pino_v21.p0_diagnostics.promotion_gate import retention_bin


UNKNOWN = "UNKNOWN"


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def model_contract(model):
    """Architecture semantics are part of every checkpoint and data binding."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("a model must expose its explicit scientific config")
    if is_dataclass(config):
        config = asdict(config)
    elif hasattr(config, "as_dict"):
        config = config.as_dict()
    if not isinstance(config, dict):
        raise ValueError("model config must be a dataclass or JSON mapping")
    # JSON normalization removes tuple/list differences between runtime and disk.
    return json.loads(json.dumps(dict(
        model_class=type(model).__module__ + "." + type(model).__qualname__,
        config=config,
        query_semantics="same_finite_oriented_capped_operand",
        moment_order=["1", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy"],
        fixed_material="added_once",
        derivative="from_the_same_final_moment_response_on_valid_strata",
    ), allow_nan=False))


def _known(*values):
    return next((value for value in values if isinstance(value, str) and value.strip()), UNKNOWN)


def _metadata(row):
    return row.get("case", {}).get("meta", row.get("case_meta", {}))


def check_source_isolation(records):
    """Reject held-out use and cross-split aliases before learning or reporting."""
    membership = {role: {key: set() for key in ("source_group", "geometry_id", "geometry_sha256")}
                  for role in ("train", "development")}
    seen = set()
    for row in records:
        role = row.get("role")
        if role not in membership:
            raise ValueError("only train/development rows may enter a HENMO data inventory")
        rid = row.get("record_id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("nonempty unique record IDs required")
        seen.add(rid)
        meta = _metadata(row)
        if meta.get("split") in ("test", "locked_test", "fixed_benchmark", "benchmark"):
            raise ValueError("sealed/test/benchmark material is forbidden in training inventory")
        if meta.get("split") not in (None, "synthetic", role):
            raise ValueError("case and inventory role disagree")
        for key in membership[role]:
            value = meta.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError("source isolation requires known " + key)
            membership[role][key].add(value)
    for key in membership["train"]:
        if membership["train"][key] & membership["development"][key]:
            raise ValueError("cross-split leakage in " + key)
    return {role: {key: sorted(values) for key, values in fields.items()}
            for role, fields in membership.items()}


def coverage_report(records):
    """Report source × surface × retention × stage and actual J qualification."""
    isolation = check_source_isolation(records)
    groups = defaultdict(list)
    rows = []
    for record in records:
        meta = _metadata(record)
        eligible = record.get("loader_eligible") is True
        case = record.get("case")
        loaded = eligible and case is not None
        qualified = meta.get("derivative_valid")
        if type(qualified) is not bool:
            raise ValueError("derivative qualification must be an explicit Boolean")
        dimension = meta.get("derivative_dimension")
        kind = {3: "R3", 8: "R8", 10: "Full-J"}.get(dimension, UNKNOWN)
        if qualified and kind == UNKNOWN:
            raise ValueError("qualified derivative must have dimension 3, 8 or 10")
        fraction = None
        retention = _known(record.get("retention_bin"), meta.get("retention_bin"))
        if loaded:
            full = float(case["native_full_editable_moments"][0])
            if not np.isfinite(full) or full <= 0:
                raise ValueError("positive complete editable mass required")
            fraction = float((case["native_moments"][0] - case["fixed_moments"][0]) / full)
            if not np.isfinite(fraction) or not -1e-12 <= fraction <= 1 + 1e-12:
                raise ValueError("native retained editable ratio is outside [0,1]")
            fraction = min(1., max(0., fraction))
            actual = retention_bin(fraction)
            if retention != UNKNOWN and retention != actual:
                raise ValueError("declared retention bin differs from the native label")
            retention = actual
        row = dict(
            record_id=record["record_id"], role=record["role"],
            source_group=meta["source_group"], geometry_id=meta["geometry_id"],
            surface_family=_known(record.get("surface_family"), meta.get("surface_family"), meta.get("surface_type")),
            retention_bin=retention, retention_ratio_native=fraction,
            trajectory_stage=_known(record.get("trajectory_stage"), meta.get("trajectory_stage")),
            moment_loaded=loaded, derivative_declared_valid=qualified,
            derivative_loaded_qualified=loaded and qualified, derivative_type=kind,
            residual_loaded_qualified=loaded and record.get("residual_valid") is True,
            derivative_reason=meta.get("derivative_reason"),
            material_identity=meta.get("material_identity"),
            native_resolution=int(case["blocks"].side) if loaded else None,
        )
        rows.append(row)
        key = tuple(row[field] for field in ("role", "source_group", "surface_family", "retention_bin", "trajectory_stage"))
        groups[key].append(row)
    result = []
    for key, members in sorted(groups.items()):
        item = dict(zip(("role", "source_group", "surface_family", "retention_bin", "trajectory_stage"), key))
        item.update(records=len(members), moment_loaded=sum(row["moment_loaded"] for row in members),
                    geometry_count=len({row["geometry_id"] for row in members}),
                    residual_qualified=sum(row["residual_loaded_qualified"] for row in members),
                    jacobian_qualified={kind: sum(row["derivative_loaded_qualified"] and row["derivative_type"] == kind
                                                for row in members) for kind in ("R3", "R8", "Full-J")})
        result.append(item)
    missing = []
    for role in ("train", "development"):
        subset = [row for row in rows if row["role"] == role]
        for kind in ("R3", "R8", "Full-J"):
            if not any(row["derivative_loaded_qualified"] and row["derivative_type"] == kind for row in subset):
                missing.append(dict(role=role, stratum=kind, status="INSUFFICIENT_EVIDENCE"))
        for stage in ("near_convergence", "endpoint"):
            if not any(row["moment_loaded"] and row["trajectory_stage"] == stage for row in subset):
                missing.append(dict(role=role, trajectory_stage=stage, status="INSUFFICIENT_EVIDENCE"))
    return dict(schema="henmo.coverage.v1", groups=result, rows=rows, missing=missing,
                isolation=isolation, scope="loaded evidence only; no inferred training exposure or promotion")


def dataset_binding(data, architecture):
    """Bind resumes to the whole verified snapshot, material and query semantics."""
    identities = []
    for row in data.records:
        meta = _metadata(row)
        case = row.get("case")
        identities.append(dict(
            record_id=row["record_id"], role=row["role"], case_sha256=row["case_sha256"],
            source_group=meta["source_group"], geometry_sha256=meta["geometry_sha256"],
            material_identity=meta.get("material_identity"),
            operation=meta.get("operation"), query_sha256=meta.get("query_content_sha256"),
            operand_sha256=meta.get("operand_content_sha256"),
            native_resolution=int(case["blocks"].side) if case is not None else None,
            material_layout=case.get("material_layout", "blocks") if case is not None else None,
            derivative_dimension=meta.get("derivative_dimension"),
            derivative_valid=meta.get("derivative_valid"),
        ))
    binding = dict(schema="henmo.training-binding.v1", snapshot_id=data.snapshot_id,
                   manifest_sha256=data.manifest_sha256, scope=data.scope,
                   architecture=architecture, source_admission=data.source_admission,
                   record_identities=identities)
    # Normalize all JSON-compatible tuples and reject nonfinite metadata early.
    return json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))


def load_henmo_dataset(path, *, mode="trial", cache_directory=None,
                       allowed_native_resolutions=(256, 512, 1024), material_layout="prepared"):
    """Use the original receipt/teacher checks; never widen them silently.

    High-resolution loading requires the explicit compatible loader extension.
    An old loader can only service its original native256 scope.
    """
    allowed = tuple(allowed_native_resolutions)
    if not allowed or any(type(side) is not int or side not in (256, 512, 1024) for side in allowed):
        raise ValueError("explicit supported native material resolutions required")
    kwargs = dict(mode=mode, cache_directory=cache_directory)
    if material_layout not in ("blocks", "prepared"):
        raise ValueError("material_layout must be blocks or prepared")
    kwargs["material_layout"] = material_layout
    if "allowed_native_resolutions" in inspect.signature(load_training_manifest).parameters:
        kwargs["allowed_native_resolutions"] = allowed
    elif allowed != (256,):
        raise ValueError("this loader has not implemented explicit high-resolution admission")
    data = load_training_manifest(path, **kwargs)
    for record in data.train + data.development:
        if record["case"]["blocks"].side not in allowed:
            raise ValueError("loaded material resolution is outside the HENMO contract")
    return data, coverage_report(data.records)
