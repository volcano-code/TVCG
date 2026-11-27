import copy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.henmo_v1plus.metrics import (
    paired_solve_report, prediction_metrics, summarize_prediction_metrics,
)


def pair(pid, source, *, accepted=True, candidate_time=5., cache_state="cold"):
    return dict(pair_id=pid, source_group=source, cache_state=cache_state,
                geometry_sha256="a" * 64, initial_state_sha256="b" * 64, task_contract_sha256="c" * 64,
                reference=dict(accepted=True, total_s=10., objective=1., timing_complete=True,
                               independent_native_acceptance=True),
                candidate=dict(accepted=accepted, total_s=candidate_time, objective=1.005,
                               timing_complete=True, independent_native_acceptance=accepted))


def test_paired_success_includes_complete_cost_and_clusters_sources():
    report = paired_solve_report([pair("a", "s1"), pair("b", "s2")], bootstrap_samples=100)
    cold = report["cache_groups"]["cold"]
    assert cold["acceleration_evidence"]
    assert cold["confidence_interval_95"] == [.5, .5]
    assert cold["independent_units"] == 2
    assert cold["all_attempt_time_per_accepted_s"]["candidate"] == 5.
    assert report["missing_cache_groups"] == ["warm"]


def test_failed_fast_candidate_cannot_establish_acceleration_and_cost_is_counted():
    pairs = [pair("a", "s1"), pair("b", "s2"), pair("c", "s3", accepted=False, candidate_time=30.)]
    cold = paired_solve_report(pairs, bootstrap_samples=100)["cache_groups"]["cold"]
    assert not cold["acceleration_evidence"]
    assert cold["candidate_new_failures"] == ["c"]
    assert cold["all_attempt_time_s"]["candidate"] == 40.
    assert cold["all_attempt_time_per_accepted_s"]["candidate"] == 20.


def test_many_queries_on_one_source_do_not_create_independent_sources():
    pairs = [pair(str(index), "one-source") for index in range(20)]
    cold = paired_solve_report(pairs, bootstrap_samples=100)["cache_groups"]["cold"]
    assert cold["confidence_interval_95"] is None
    assert not cold["acceleration_evidence"]
    assert cold["independent_units"] == 1


def test_common_success_speedup_cannot_hide_cost_of_other_failed_attempts():
    pairs = [pair("a", "s1"), pair("b", "s2"), pair("c", "s3", accepted=False, candidate_time=100.)]
    pairs[-1]["reference"]["accepted"] = False
    cold = paired_solve_report(pairs, bootstrap_samples=100)["cache_groups"]["cold"]
    assert cold["confidence_interval_95"] == [.5, .5]
    assert not cold["all_attempt_cost_per_accepted_improved"]
    assert not cold["acceleration_evidence"]


def test_missing_native_acceptance_or_objective_and_incomplete_timing_fail_closed():
    pairs = [pair("a", "s1"), pair("b", "s2")]
    pairs[0]["candidate"]["independent_native_acceptance"] = False
    pairs[0]["candidate"]["timing_complete"] = False
    del pairs[1]["candidate"]["objective"]
    cold = paired_solve_report(pairs, bootstrap_samples=100)["cache_groups"]["cold"]
    assert not cold["acceleration_evidence"]
    assert len(cold["missing_evidence"]) == 3


def test_objective_degradation_blocks_speed_claim_and_caches_stay_separate():
    pairs = [pair("a", "s1"), pair("b", "s2"), pair("a", "s1", cache_state="warm")]
    pairs[1]["candidate"]["objective"] = 1.1
    report = paired_solve_report(pairs, bootstrap_samples=100)
    assert report["cache_groups"]["cold"]["quality_failures"] == ["b"]
    assert not report["cache_groups"]["cold"]["acceleration_evidence"]
    assert report["cache_groups"]["warm"]["pairs"] == 1


def test_pair_identity_and_duplicate_rejected():
    item = pair("a", "s1")
    with pytest.raises(ValueError, match="distinct pair_id"):
        paired_solve_report([item, copy.deepcopy(item)], bootstrap_samples=100)
    del item["initial_state_sha256"]
    with pytest.raises(ValueError, match="initial_state_sha256"):
        paired_solve_report([item], bootstrap_samples=100)


def metric_row(rid, source, value):
    return dict(record_id=rid, source_group=source, status="OK",
                moment=dict(raw_rmse=value, scaled_rmse=value, worst_scaled_component=value),
                jacobian=dict(status="LABEL_UNQUALIFIED"))


def test_source_macro_is_not_query_micro_average_and_failures_do_not_disappear():
    rows = [metric_row("a", "s1", 1.), metric_row("b", "s1", 1.), metric_row("c", "s2", 3.)]
    summary = summarize_prediction_metrics(rows)
    assert summary["source_macro_scaled_moment_rmse"] == 2.
    assert summary["qualified_jacobian"]["R8"]["mean"] is None
    rows.append(dict(record_id="d", source_group="s2", status="FAILED", moment=None, jacobian=None))
    summary = summarize_prediction_metrics(rows)
    assert summary["source_macro_scaled_moment_rmse"] is None
    assert not summary["complete"]


def test_direction_derivative_uses_physical_and_parameter_scales():
    expected = np.zeros((10, 3)); actual = np.ones((10, 3))
    record = dict(record_id="q", case=dict(
        meta=dict(editable_mass=2., length_scale=3., derivative_valid=True, derivative_dimension=3,
                  source_group="s", geometry_id="g", direction_source="native_newton"),
        native_moments=np.r_[1., np.zeros(9)], fixed_moments=np.zeros(10),
        native_full_editable_moments=np.r_[2., np.zeros(9)], native_jacobian=expected,
        parameter_scales=np.array([2., 1., .5]), direction=np.array([2., 0., 0.])))
    prediction = SimpleNamespace(moments=record["case"]["native_moments"], jacobian=actual)
    row = prediction_metrics(prediction, record)
    assert row["moment"]["worst_scaled_component"] == 0
    assert row["direction"]["worst_scaled_component"] == 1.
    assert row["jacobian"]["kind"] == "R3"
