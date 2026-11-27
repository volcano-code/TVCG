"""Physical error and complete-solve comparisons; missing evidence stays missing."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import moment_scale
from experiments.compact_pino_v21.runtime_efficiency.metrics import (
    QueryTrace, reconcile_timing, summarize_queries, summarize_solve_pairs,
)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if value < 0 or positive and value == 0:
        raise ValueError(name + " must be " + ("positive" if positive else "nonnegative"))
    return float(value)


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("error statistics require finite nonnegative observations")
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None, p99=None, maximum=None,
                    status="INSUFFICIENT_EVIDENCE")
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)),
                maximum=float(values.max()), status="OBSERVED",
                quantile_scope="descriptive finite sample; no population guarantee")


def prediction_metrics(prediction, record):
    """Scale M by mL^degree and J/Jv by the declared parameter chart scales."""
    case, meta = record["case"], record["case"]["meta"]
    scale = moment_scale(meta["editable_mass"], meta["length_scale"])
    estimate, native = _array(prediction.moments), _array(case["native_moments"])
    if estimate.shape != (10,) or native.shape != (10,) or not np.isfinite([estimate, native]).all():
        raise ValueError("finite predicted and native ten moments required")
    raw = estimate - native
    error = raw / scale
    full_mass = float(case["native_full_editable_moments"][0])
    ratio = float((native[0] - case["fixed_moments"][0]) / full_mass)
    row = dict(record_id=record["record_id"], source_group=meta["source_group"],
               geometry_id=meta["geometry_id"], surface_family=record.get("surface_family", "UNKNOWN"),
               trajectory_stage=record.get("trajectory_stage", "UNKNOWN"), status="OK",
               derivative_label_qualified=meta["derivative_valid"],
               retention_ratio_native=ratio,
               moment=dict(raw_rmse=float(np.sqrt(np.mean(raw**2))),
                           scaled_rmse=float(np.sqrt(np.mean(error**2))),
                           worst_scaled_component=float(np.max(np.abs(error))),
                           scaled_components=error.tolist()),
               jacobian=None, direction=None)
    if not meta["derivative_valid"]:
        row["jacobian"] = dict(status="LABEL_UNQUALIFIED", reason=meta.get("derivative_reason"),
                               scaled_rmse=None, worst_scaled_component=None)
        return row
    if prediction.jacobian is None:
        row["jacobian"] = dict(status="RESPONSE_UNAVAILABLE", scaled_rmse=None, worst_scaled_component=None)
        return row
    expected = _array(case["native_jacobian"])
    actual = _array(prediction.jacobian)
    dim = meta["derivative_dimension"]
    pscale = _array(case["parameter_scales"])
    if (dim not in (3, 8, 10) or expected.shape != (10, dim) or actual.shape != expected.shape
            or pscale.shape != (dim,) or not np.isfinite(expected).all() or not np.isfinite(actual).all()
            or not np.isfinite(pscale).all() or np.any(pscale <= 0)):
        raise ValueError("invalid qualified Jacobian or chart scale")
    difference = (actual - expected) * pscale[None, :] / scale[:, None]
    row["jacobian"] = dict(status="QUALIFIED", kind={3: "R3", 8: "R8", 10: "Full-J"}[dim],
                           scaled_rmse=float(np.sqrt(np.mean(difference**2))),
                           worst_scaled_component=float(np.max(np.abs(difference))))
    if case.get("direction") is not None:
        direction = _array(case["direction"])
        if direction.shape != (dim,) or not np.isfinite(direction).all():
            raise ValueError("invalid native chart direction")
        normalized = direction / pscale
        length = float(np.linalg.norm(normalized))
        if length == 0:
            row["direction"] = dict(status="ZERO_DIRECTION", scaled_error=None)
        else:
            delta = difference @ (normalized / length)
            row["direction"] = dict(status="QUALIFIED", scaled_error=float(np.linalg.norm(delta)),
                                    worst_scaled_component=float(np.max(np.abs(delta))),
                                    source=meta.get("direction_source"))
    return row


def summarize_prediction_metrics(rows):
    rows = list(rows)
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evaluation query; do not count repeated rows as independent data")
    good = [row for row in rows if row["status"] == "OK"]
    failed = [row for row in rows if row["status"] != "OK"]
    sources = defaultdict(list)
    for row in good:
        sources[row["source_group"]].append(row)
    source_rows = []
    for name, members in sorted(sources.items()):
        # All ten components and all rows receive equal weight within a source.
        source_rows.append(dict(source_group=name, count=len(members),
            scaled_moment_rmse=math.sqrt(np.mean([row["moment"]["scaled_rmse"]**2 for row in members])),
            raw_moment_rmse=math.sqrt(np.mean([row["moment"]["raw_rmse"]**2 for row in members]))))
    # Never turn a failed query into an improvement by omitting it from a gate.
    complete = len(rows) > 0 and not failed
    macro = (float(np.mean([row["scaled_moment_rmse"] for row in source_rows]))
             if complete and source_rows else None)
    raw_macro = (float(np.mean([row["raw_moment_rmse"] for row in source_rows]))
                 if complete and source_rows else None)
    derivative = {}
    for kind in ("R3", "R8", "Full-J"):
        selected = [row["jacobian"]["scaled_rmse"] for row in good
                    if row["jacobian"] and row["jacobian"].get("status") == "QUALIFIED"
                    and row["jacobian"]["kind"] == kind]
        derivative[kind] = _stats(selected)
    requested = sum(row.get("derivative_label_qualified") is True for row in rows)
    available = sum(bool(row.get("jacobian")) and row["jacobian"].get("status") == "QUALIFIED" for row in rows)
    return dict(schema="henmo.evaluation.v1", records=len(rows), successful_predictions=len(good),
                failed_predictions=failed, complete=complete,
                qualified_derivative_labels=requested, available_derivative_predictions=available,
                derivative_complete=(available == requested) if requested else None,
                source_macro_scaled_moment_rmse=macro, source_macro_raw_moment_rmse=raw_macro,
                successful_only_moment=_stats([row["moment"]["worst_scaled_component"] for row in good]),
                qualified_jacobian=derivative, source_groups=source_rows, rows=rows,
                promotion="NOT_DECIDED_BY_THIS_REPORT")


def _paired_group(pairs, *, bootstrap_samples, seed, objective_relative_tolerance, bootstrap_unit):
    descriptive = summarize_solve_pairs(pairs)
    descriptive.pop("confidence_interval", None)
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be an integer >= 100")
    _number(objective_relative_tolerance, "objective_relative_tolerance")
    if bootstrap_unit not in ("source_group", "pair"):
        raise ValueError("bootstrap unit must be source_group or pair")
    seen, complete, groups = set(), True, defaultdict(list)
    missing, quality_failures = [], []
    paired_rows = []
    for pair in pairs:
        pid = pair.get("pair_id")
        if not isinstance(pid, str) or not pid or pid in seen:
            raise ValueError("distinct pair_id values required")
        seen.add(pid)
        source = pair.get("source_group")
        if not isinstance(source, str) or not source:
            raise ValueError("source group required for independent-unit accounting")
        for key in ("geometry_sha256", "initial_state_sha256", "task_contract_sha256"):
            value = pair.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("paired solve requires exact " + key)
        pair_complete = True
        for side in ("reference", "candidate"):
            arm = pair[side]
            if arm.get("timing_complete") is not True:
                missing.append(dict(pair_id=pid, side=side, reason="INCOMPLETE_END_TO_END_TIMING"))
                pair_complete = False
            if arm["accepted"] and arm.get("independent_native_acceptance") is not True:
                missing.append(dict(pair_id=pid, side=side, reason="NATIVE_ACCEPTANCE_NOT_PROVEN"))
                pair_complete = False
        complete = complete and pair_complete
        ratio, objective_error = None, None
        common = pair["reference"]["accepted"] and pair["candidate"]["accepted"]
        if common:
            reference, candidate = pair["reference"], pair["candidate"]
            r, c = reference.get("objective"), candidate.get("objective")
            if (isinstance(r, bool) or isinstance(c, bool) or not isinstance(r, (float, int))
                    or not isinstance(c, (float, int)) or not math.isfinite(r) or not math.isfinite(c)):
                missing.append(dict(pair_id=pid, reason="QUALIFIED_OBJECTIVE_UNAVAILABLE"))
                complete = False
            else:
                denominator = abs(r)
                if denominator == 0:
                    denominator = _number(pair.get("objective_scale"), "zero-reference objective scale", positive=True)
                objective_error = abs(c - r) / denominator
                if objective_error > objective_relative_tolerance:
                    quality_failures.append(pid)
            ratio = _number(candidate["total_s"], "candidate total_s", positive=True) / _number(
                reference["total_s"], "reference total_s", positive=True)
            key = source if bootstrap_unit == "source_group" else pid
            groups[key].append(math.log(ratio))
        paired_rows.append(dict(pair_id=pid, common_success=common, time_ratio=ratio,
                                relative_objective_difference=objective_error))
    interval = None
    mean_ratio = None
    if len(groups) >= 2:
        # Each independent source receives equal weight. Bootstrap complete
        # source clusters, never individual correlated Newton evaluations.
        samples = np.array([np.mean(values) for values in groups.values()])
        rng = np.random.default_rng(seed)
        selected = rng.integers(0, len(samples), size=(bootstrap_samples, len(samples)))
        estimates = np.exp(samples[selected].mean(axis=1))
        bounds = np.quantile(estimates, [.025, .975])
        interval = [float(bounds[0]), float(bounds[1])]
        mean_ratio = float(np.exp(samples.mean()))
    per_success = descriptive["all_attempt_time_per_accepted_s"]
    full_cost_improved = (per_success["candidate"] is not None and per_success["reference"] is not None
                         and per_success["candidate"] < per_success["reference"])
    evidence = (complete and full_cost_improved and not quality_failures
                and not descriptive["candidate_new_failures"] and interval is not None and interval[1] < 1)
    return dict(**descriptive, paired_rows=paired_rows,
                confidence_interval_95=interval, geometric_mean_time_ratio=mean_ratio,
                bootstrap_unit=bootstrap_unit, independent_units=len(groups),
                bootstrap_samples=bootstrap_samples, bootstrap_seed=seed,
                quality_failures=quality_failures, missing_evidence=missing,
                all_attempt_cost_per_accepted_improved=full_cost_improved,
                objective_relative_tolerance=objective_relative_tolerance,
                acceleration_evidence=evidence,
                interval_scope="common-success pairs only; all failures/costs remain in descriptive totals",
                generalization_scope=("source-cluster resampling; inspect actual source count"
                                      if bootstrap_unit == "source_group" else
                                      "conditional on the frozen geometries; no cross-geometry inference"))


def paired_solve_report(pairs, *, bootstrap_samples=2000, seed=830251,
                        objective_relative_tolerance=.01, bootstrap_unit="source_group"):
    """Complete physical solve evidence, separated into cold and reused-cache runs.

    Each pair binds geometry, initial state and task contract hashes. Each arm
    must count all preprocessing, queries, failed attempts, native correction
    and verification in total_s and explicitly declare timing_complete. These
    declarations are receipt fields, not a proof produced by this summarizer.
    """
    groups = defaultdict(list)
    for pair in pairs:
        state = pair.get("cache_state")
        if state not in ("cold", "warm"):
            raise ValueError("solve pairs must explicitly distinguish cold/warm cache state")
        groups[state].append(pair)
    result = {name: _paired_group(members, bootstrap_samples=bootstrap_samples, seed=seed,
                objective_relative_tolerance=objective_relative_tolerance, bootstrap_unit=bootstrap_unit)
              for name, members in sorted(groups.items())}
    return dict(schema="henmo.paired-solves.v1", cache_groups=result,
                missing_cache_groups=[name for name in ("cold", "warm") if name not in result],
                claim_scope="no mixing of cold/warm or synthetic/native solve evidence")
