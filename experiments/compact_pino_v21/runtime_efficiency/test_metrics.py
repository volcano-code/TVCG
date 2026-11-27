from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.compact_pino_v21.runtime_efficiency.metrics import (
    QueryTrace, compare_outputs, reconcile_timing, summarize_queries, summarize_solve_pairs,
)


class Clock:
    def __init__(self):
        self.value = 0.
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds


def test_sequential_repeated_stages_reconcile_including_uninstrumented_time():
    clock = Clock()
    trace = QueryTrace("q1", query=np.arange(10), mode="joint",
                       cache_state="new_query", clock=clock)
    with trace:
        clock.advance(1)
        for _ in range(2):
            with trace.stage("response"):
                clock.advance(3)
        clock.advance(2)
    row = trace.as_dict()
    assert row["callback_wall_s"] == 9
    assert row["stage_wall_sum_s"] == 6
    assert row["unattributed_wall_s"] == 3
    assert row["timing_basis"] == "host_wall"
    summary = summarize_queries([row])
    assert summary["groups"][0]["stage_totals_s"] == {"response": 6}
    assert summary["groups"][0]["all_attempts"]["count"] == 1


def test_query_failure_is_logged_and_propagated_and_cost_is_counted():
    clock = Clock()
    trace = QueryTrace("rejected", clock=clock, mode="M")
    with pytest.raises(ValueError, match="native surface"):
        with trace:
            with trace.stage("source"):
                clock.advance(4)
                raise ValueError("native surface undefined")
    row = trace.as_dict()
    assert row["defined"] is False
    assert row["error"]["type"] == "ValueError"
    assert row["stages"][0]["failed"]
    group = summarize_queries([row])["groups"][0]
    assert group["all_attempts"]["sum_s"] == 4
    assert group["successful_calls"]["count"] == 0
    assert group["failed_calls"][0]["wall_s"] == 4


def test_sync_error_does_not_hide_original_failure():
    clock = Clock()
    state = {"n": 0}
    def sync():
        state["n"] += 1
        if state["n"] >= 3:
            raise RuntimeError("device queue failed")
    trace = QueryTrace("q", clock=clock, synchronize=sync)
    with pytest.raises(ValueError, match="original"):
        with trace:
            with trace.stage("compute"):
                clock.advance(1)
                raise ValueError("original")
    row = trace.as_dict()
    assert row["timing_basis"] == "synchronized_wall"
    assert row["error"]["type"] == "ValueError"
    assert len(row["synchronization_errors"]) == 2


def test_overlap_and_reuse_are_rejected():
    trace = QueryTrace("q")
    with pytest.raises(RuntimeError, match="overlap"):
        with trace:
            with trace.stage("a"):
                with trace.stage("b"):
                    pass
    with pytest.raises(RuntimeError, match="single-use"):
        with trace:
            pass


def test_production_and_optional_diagnostic_are_not_double_counted():
    reconciled = reconcile_timing(115.03888522298075,
        {"proposal": 34.256496452959254, "correction": 38.42311306600459,
         "verification": 20.998612897004932}, job_s=136.8475925740786,
        diagnostic_s=21.808699080953375)
    assert reconciled["consistent"]
    assert reconciled["production_unattributed_s"] == pytest.approx(21.360662807011977)
    assert reconciled["job_unattributed_s"] == pytest.approx(8.270144e-6)
    assert not reconcile_timing(10, {"a": 8, "b": 4})["consistent"]
    included = reconcile_timing(12, {"work": 10, "diagnostic": 2},
                               job_s=12, diagnostic_s=2, diagnostic_in_production=True)
    assert included["consistent"]
    assert included["extra_diagnostic_s"] == 0


def test_moment_and_chart_derivative_differences_use_physical_scales():
    reference = SimpleNamespace(moments=np.ones(10), jacobian=np.ones((10, 3)))
    candidate = SimpleNamespace(moments=np.ones(10)*3, jacobian=np.ones((10, 3))*2)
    result = compare_outputs(reference, candidate, moment_scale=np.ones(10)*2,
                             parameter_scales=np.array([1., 2., 4.]))
    assert result["moments"]["max_scaled_abs"] == 1
    assert result["jacobian"]["max_scaled_abs"] == 2
    missing = compare_outputs(reference, {"raw_moments": np.ones(10)})
    assert missing["jacobian"]["availability"] == "REFERENCE_ONLY"
    invalid = compare_outputs(reference, {"moments": np.full(10, np.nan)}, with_jacobian=False)
    assert not invalid["moments"]["finite"]
    json.dumps(invalid, allow_nan=False)


def test_pair_summary_keeps_failed_costs_and_does_not_fabricate_speedup():
    pairs = [
        {"pair_id": "s0", "reference": {"accepted": True, "total_s": 10},
         "candidate": {"accepted": False, "total_s": 8}},
        {"pair_id": "s1", "reference": {"accepted": True, "total_s": 12},
         "candidate": {"accepted": True, "total_s": 6}},
    ]
    report = summarize_solve_pairs(pairs)
    assert report["candidate_new_failures"] == ["s0"]
    assert report["all_attempt_time_s"]["candidate"] == 14
    assert report["all_attempt_time_per_accepted_s"]["candidate"] == 14
    assert report["common_success_time_ratio_median"] == .5
    assert report["confidence_interval"] is None
    failed = summarize_solve_pairs(pairs[:1])
    assert failed["all_attempt_time_per_accepted_s"]["candidate"] is None
