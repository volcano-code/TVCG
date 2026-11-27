"""Lightweight runtime receipts for new-query and repeated-query measurements.

The trace never computes a physical label or grants solver acceptance. A caller
supplies a synchronization callback when completed CUDA work is being timed.
Stages are non-overlapping wall-clock intervals; external native work belongs
inside these intervals too. Detailed profiler traces remain a separate tool.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

TRACE_SCHEMA = "compact_pino.runtime-query.v1"
MODES = {"M", "joint"}
CACHE_STATES = {"new_query", "repeated_query", "unknown"}


def _seconds(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


class QueryTrace:
    """Record one callback without swallowing its exception.

    Use trace.stage(name) to measure sequential (not nested) phases. A
    synchronization callback, normally lambda: torch.cuda.synchronize(device),
    runs at timing boundaries; omit it only for CPU or explicitly host-only
    timing. The caller chooses labels and supplies cache statistics in metadata.
    Repeated q is not automatically considered a prepared-query cache hit.
    """

    def __init__(self, query_id, *, mode="joint", query=None, cache_state="unknown",
                 synchronize=None, metadata=None, clock=None):
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be a nonempty string")
        if mode not in MODES or cache_state not in CACHE_STATES:
            raise ValueError("invalid request mode or cache state")
        if synchronize is not None and not callable(synchronize):
            raise ValueError("synchronize must be callable or None")
        self.query_id = query_id
        self.mode = mode
        self.cache_state = cache_state
        self.synchronize = synchronize
        self.clock = clock or time.perf_counter
        self.metadata = _json_copy(metadata or {})
        self.query = None
        self.query_sha256 = None
        if query is not None:
            q = _array(query)
            if q.shape != (10,) or not np.isfinite(q).all():
                raise ValueError("query must contain ten finite coefficients")
            self.query = q.tolist()
            self.query_sha256 = hashlib.sha256(np.asarray(q, dtype="<f8").tobytes()).hexdigest()
        self.stages = []
        self._active = None
        self._start = None
        self._end = None
        self._error = None
        self._sync_errors = []

    def _sync(self, existing_error=None):
        if self.synchronize is None:
            return
        try:
            self.synchronize()
        except BaseException as exc:
            self._sync_errors.append({"type": type(exc).__name__, "message": str(exc)})
            if existing_error is None:
                raise

    def __enter__(self):
        if self._start is not None:
            raise RuntimeError("QueryTrace is single-use")
        self._sync()
        self._start = self.clock()
        return self

    @contextmanager
    def stage(self, name):
        if self._start is None or self._end is not None:
            raise RuntimeError("stages require a running query trace")
        if self._active is not None:
            raise RuntimeError("timed stages must not overlap or nest")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be nonempty")
        self._sync()
        start = self.clock()
        self._active = name
        error = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self._sync(error)
            except BaseException as exc:
                error = exc
                raise
            finally:
                end = self.clock()
                self._active = None
                self.stages.append({
                    "name": name,
                    "start_s": start - self._start,
                    "end_s": end - self._start,
                    "wall_s": end - start,
                    "failed": error is not None,
                    "error": None if error is None else
                        {"type": type(error).__name__, "message": str(error)},
                })

    def __exit__(self, kind, error, tb):
        try:
            self._sync(error)
        except BaseException as sync_error:
            self._error = {"type": type(sync_error).__name__, "message": str(sync_error)}
            raise
        finally:
            self._end = self.clock()
            if error is not None:
                self._error = {"type": kind.__name__, "message": str(error)}
        return False

    def as_dict(self):
        if self._end is None:
            raise RuntimeError("query trace is not complete")
        total = self._end - self._start
        measured = sum(stage["wall_s"] for stage in self.stages)
        if total < 0 or any(stage["wall_s"] < 0 for stage in self.stages):
            raise ValueError("timing clock moved backwards")
        row = dict(
            schema=TRACE_SCHEMA, query_id=self.query_id, query=self.query,
            query_sha256=self.query_sha256, mode=self.mode, cache_state=self.cache_state,
            timing_basis="synchronized_wall" if self.synchronize is not None else "host_wall",
            completed_work_timing=self.synchronize is not None,
            callback_wall_s=total, stage_wall_sum_s=measured,
            unattributed_wall_s=total-measured, stages=self.stages,
            defined=self._error is None, error=self._error,
            synchronization_errors=self._sync_errors, metadata=self.metadata,
        )
        return _json_copy(row)


def reconcile_timing(production_s, stage_seconds, *, job_s=None, diagnostic_s=0.,
                     diagnostic_in_production=False, tolerance_s=1e-6):
    """Account for disjoint production stages and an optional extra diagnostic.

    Unattributed positive time remains visible. Negative remainder means stages
    overlap or the boundary is inconsistent; it is never clipped into a speedup.
    A diagnostic used by the deployment decision belongs inside production.
    """
    production = _seconds(production_s, "production_s")
    diagnostic = _seconds(diagnostic_s, "diagnostic_s")
    tolerance = _seconds(tolerance_s, "tolerance_s")
    if type(diagnostic_in_production) is not bool:
        raise ValueError("diagnostic_in_production must be an explicit bool")
    stages = {str(k): _seconds(v, str(k)) for k, v in stage_seconds.items()}
    stage_sum = sum(stages.values())
    production_remainder = production-stage_sum
    job = None if job_s is None else _seconds(job_s, "job_s")
    extra = 0. if diagnostic_in_production else diagnostic
    job_remainder = None if job is None else job-production-extra
    consistent = production_remainder >= -tolerance and (
        job_remainder is None or job_remainder >= -tolerance)
    if diagnostic_in_production and diagnostic > production+tolerance:
        consistent = False
    return dict(
        production_s=production, stages_s=stages, stage_sum_s=stage_sum,
        production_unattributed_s=production_remainder, job_s=job,
        diagnostic_s=diagnostic, diagnostic_in_production=diagnostic_in_production,
        extra_diagnostic_s=extra, job_unattributed_s=job_remainder,
        consistent=consistent, tolerance_s=tolerance,
        assumption="production stages are disjoint; diagnostics used in deployment belong in production",
    )


def _prediction_value(prediction, primary, secondary):
    if isinstance(prediction, dict):
        return prediction.get(primary, prediction.get(secondary))
    return getattr(prediction, primary, None)


def _comparison(reference, candidate, scale=None):
    reference, candidate = _array(reference), _array(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("prediction shapes differ")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return dict(shape=list(reference.shape), finite=False, max_abs=None,
                    relative_l2=None, max_scaled_abs=None)
    difference = candidate-reference
    denominator = float(np.linalg.norm(reference))
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.
    result = dict(shape=list(reference.shape), finite=True, max_abs=maximum,
                  relative_l2=float(np.linalg.norm(difference))/denominator if denominator else None,
                  max_scaled_abs=None)
    if scale is not None:
        scale = _array(scale)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("comparison scales must be finite and positive")
        try:
            result["max_scaled_abs"] = float(np.max(np.abs(difference)/scale))
        except ValueError as exc:
            raise ValueError("comparison scales do not broadcast to output") from exc
    return result


def compare_outputs(reference, candidate, *, with_jacobian=True,
                    moment_scale=None, parameter_scales=None):
    """Describe numerical differences, without assigning a physics acceptance."""
    rm = _prediction_value(reference, "moments", "raw_moments")
    cm = _prediction_value(candidate, "moments", "raw_moments")
    if rm is None or cm is None or _array(rm).shape != (10,) or _array(cm).shape != (10,):
        raise ValueError("both predictions need ten moments")
    result = {"moments": _comparison(rm, cm, moment_scale), "jacobian": None}
    if not with_jacobian:
        result["jacobian"] = {"availability": "NOT_REQUESTED"}
        return result
    rj = _prediction_value(reference, "jacobian", "moment_jacobian")
    cj = _prediction_value(candidate, "jacobian", "moment_jacobian")
    if rj is None or cj is None:
        result["jacobian"] = {"availability": "BOTH_UNAVAILABLE" if rj is None and cj is None
                              else "REFERENCE_ONLY" if cj is None else "CANDIDATE_ONLY"}
        return result
    rj, cj = _array(rj), _array(cj)
    if rj.ndim != 2 or rj.shape[0] != 10 or rj.shape[1] not in (3, 8, 10):
        raise ValueError("Jacobian must have 10 rows and 3, 8 or 10 columns")
    scales = None
    if moment_scale is not None:
        m = _array(moment_scale)
        if m.shape != (10,):
            raise ValueError("moment_scale must contain ten entries")
        p = np.ones(rj.shape[1]) if parameter_scales is None else _array(parameter_scales)
        if p.shape != (rj.shape[1],) or not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError("parameter_scales must match the derivative chart")
        scales = m[:, None]/p[None, :]
    elif parameter_scales is not None:
        raise ValueError("parameter scales require physical moment scales")
    result["jacobian"] = dict(availability="BOTH_AVAILABLE", **_comparison(rj, cj, scales))
    return result


def _distribution(values):
    values = [_seconds(v, "query time") for v in values]
    return dict(count=len(values), sum_s=sum(values),
                median_s=statistics.median(values) if values else None,
                p95_s=float(np.quantile(values, .95)) if values else None,
                p99_s=float(np.quantile(values, .99)) if values else None,
                quantile_method="linear descriptive sample quantiles; inspect count, not an uncertainty estimate")


def summarize_queries(rows):
    """Report all attempts, including failures, separated by request/cache mode."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("schema") != TRACE_SCHEMA or type(row.get("defined")) is not bool:
            raise ValueError("expected completed runtime-query receipts")
        if row.get("mode") not in MODES or row.get("cache_state") not in CACHE_STATES:
            raise ValueError("invalid query receipt mode/cache state")
        groups[(row["mode"], row["cache_state"], row["timing_basis"])].append(row)
    summaries = []
    for (mode, cache, timing), members in sorted(groups.items()):
        stages = defaultdict(float)
        for row in members:
            for stage in row["stages"]:
                stages[stage["name"]] += _seconds(stage["wall_s"], "stage time")
        summaries.append(dict(
            mode=mode, cache_state=cache, timing_basis=timing,
            all_attempts=_distribution(r["callback_wall_s"] for r in members),
            successful_calls=_distribution(r["callback_wall_s"] for r in members if r["defined"]),
            failed_calls=[dict(query_id=r["query_id"], error=r["error"],
                               wall_s=r["callback_wall_s"]) for r in members if not r["defined"]],
            stage_totals_s=dict(stages),
            unattributed_s=sum(r["unattributed_wall_s"] for r in members),
        ))
    return dict(schema="compact_pino.runtime-summary.v1", groups=summaries,
                claim_scope="query execution only; no solver success or generalization is inferred")


def summarize_solve_pairs(pairs):
    """Keep failures and full attempt costs next to common-success ratios.

    Each item has reference/candidate mappings with accepted and total_s.
    This intentionally supplies no confidence interval from correlated queries.
    """
    totals = {"reference": 0., "candidate": 0.}
    success = {"reference": 0, "candidate": 0}
    ratios, new_failures = [], []
    for i, pair in enumerate(pairs):
        for side in totals:
            arm = pair[side]
            if type(arm.get("accepted")) is not bool:
                raise ValueError("accepted must be an explicit bool")
            seconds = _seconds(arm["total_s"], side+" total_s")
            totals[side] += seconds
            success[side] += arm["accepted"]
        if pair["reference"]["accepted"] and not pair["candidate"]["accepted"]:
            new_failures.append(pair.get("pair_id", i))
        if pair["reference"]["accepted"] and pair["candidate"]["accepted"]:
            denominator = float(pair["reference"]["total_s"])
            if denominator <= 0:
                raise ValueError("a reference accepted solve needs positive time")
            ratios.append(float(pair["candidate"]["total_s"])/denominator)
    return dict(
        pairs=len(pairs), accepted=success, all_attempt_time_s=totals,
        all_attempt_time_per_accepted_s={k: totals[k]/success[k] if success[k] else None for k in totals},
        candidate_new_failures=new_failures, common_success_pairs=len(ratios),
        common_success_time_ratio_median=statistics.median(ratios) if ratios else None,
        confidence_interval=None,
        claim_scope="descriptive paired outcomes; failed attempts are not excluded from total costs",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="QueryTrace JSONL")
    parser.add_argument("--solve-pairs", type=Path, help="JSON array of paired complete solve receipts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    report = summarize_queries(rows)
    if args.solve_pairs:
        report["solve_pairs"] = summarize_solve_pairs(json.loads(args.solve_pairs.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
