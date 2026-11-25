"""Contract tests use a declared toy backend, never a claimed native solver."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.compact_pino_v21.p0_diagnostics.endpoint_attribution import (
    COEFFICIENT_MULTIPLIERS, ORDER, attribution, jsonable, sha256, verify_artifacts,
)


class ToyAdapter:
    """A stateful test double with moment-dependent endogenous multipliers."""
    def __init__(self):
        self.calls = 0

    def accept(self, c, m, j):
        self.calls += 1
        assert self.calls == 1, "an arm reused mutable solver state"
        multiplier = float(m[3] / m[0])
        s, b = abs(multiplier - 1.), abs(float(m[1] - 1.))
        return dict(stationarity_residual=s, balance_residual=b,
                    accepted=s <= 1e-12 and b <= 1e-6, multiplier=multiplier)

    def step(self, c, m, j):
        self.calls += 1
        assert self.calls == 1
        # Deliberately only a test double; this is not a Newton formula.
        direction = m[:j.shape[1]] + j.sum(axis=0)
        return dict(direction=direction, line_search_accepted=False,
                    reason="synthetic_trial_rejected", trial_coefficients=None,
                    trial_native_stationarity_residual=None, trial_native_balance_residual=None)


def fixture(dimension=10):
    meta = dict(geometry_id="synthetic", query_id="q", source_group_id="toy_group", split="synthetic",
                teacher_version="toy", solver_version="toy", coordinate_frame="Cartesian", units="toy",
                trajectory_stage="endpoint", chart_id="raw" if dimension == 10 else f"chart{dimension}",
                moment_order=ORDER, coefficient_basis="raw_coefficients",
                coefficient_multipliers=COEFFICIENT_MULTIPLIERS,
                thresholds=dict(stationarity=1e-12, balance=1e-6), editable_mass=2., length_scale=1.,
                coefficient_scale=1., derivative_valid=True, derivative_dimension=dimension,
                adapter_contract=dict(acceptance_reads_jacobian=False, multiplier_policy="native_recompute",
                                      external_state_sha256="b" * 64),
                replay_tolerances=dict(residual_atol=1e-14, residual_rtol=1e-12,
                                      direction_atol=1e-14, direction_rtol=1e-12,
                                      coefficient_atol=1e-14, coefficient_rtol=1e-12))
    for role in ("geometry", "operand", "manifest", "checkpoint", "solver"):
        meta[role + "_sha256"] = "a" * 64
    case = dict(meta=meta, coefficients=np.arange(1., 11.), predicted_moments=np.ones(10),
                native_moments=np.arange(1., 11.), predicted_jacobian=np.eye(10)[:, :dimension],
                native_jacobian=2 * np.eye(10)[:, :dimension], chart_basis=np.eye(10)[:, :dimension],
                direction=np.ones(dimension))
    c, n, j = case["coefficients"], case["native_moments"], case["native_jacobian"]
    meta["native_replay"] = dict(accept=ToyAdapter().accept(c, n, j), step=ToyAdapter().step(c, n, j))
    return case


def test_native_replays_before_any_hybrid_and_each_arm_gets_fresh_state():
    case, events = fixture(), []

    class Observed(ToyAdapter):
        def accept(self, c, m, j):
            events.append(("accept", m.copy(), j.copy()))
            return super().accept(c, m, j)

        def step(self, c, m, j):
            events.append(("step", m.copy(), j.copy()))
            return super().step(c, m, j)

    report = attribution(case, lambda c: Observed(), run_step=True)
    assert [x[0] for x in events[:2]] == ["accept", "step"]
    for _, m, j in events[:2]:
        np.testing.assert_array_equal(m, case["native_moments"])
        np.testing.assert_array_equal(j, case["native_jacobian"])
    assert report["fixed_endpoint"]["A_predM_predJ"]["accepted"]
    assert not report["fixed_endpoint"]["D_nativeM_nativeJ"]["accepted"]
    assert report["endpoint_J_invariance_observed"] is True
    assert report["fixed_endpoint_repeat_identical"] is True
    assert report["status"] == "DIAGNOSTIC_ONLY_NO_TRAINING_PROMOTION"
    assert len(report["single_component_replacements"]) == 10
    # Recomputed multipliers follow each M. Reusing native multipliers would fail.
    assert report["fixed_endpoint"]["A_predM_predJ"]["multiplier"] == 1.
    assert report["fixed_endpoint"]["D_nativeM_nativeJ"]["multiplier"] == 4.


def test_failed_native_replay_stops_before_substitutions():
    case, calls = fixture(), []
    case["meta"]["native_replay"]["accept"]["stationarity_residual"] += .1

    def factory(c):
        calls.append(1)
        return ToyAdapter()

    with pytest.raises(ValueError, match="native replay mismatch"):
        attribution(case, factory, run_step=True)
    assert len(calls) == 1


def test_same_moment_acceptance_allows_only_frozen_numerical_jitter():
    c = fixture()
    c["meta"]["replay_tolerances"]["residual_atol"] = 1e-13
    count = [0]

    class Jitter(ToyAdapter):
        def accept(self, c, m, j):
            r = super().accept(c, m, j)
            count[0] += 1
            r["stationarity_residual"] += (count[0] % 2) * 1e-14
            return r

    r = attribution(c, lambda c: Jitter())
    assert r["endpoint_J_invariance_observed"] is True
    assert r["fixed_endpoint_repeat_within_frozen_tolerance"] is True


def test_step_repetition_detects_shared_backend_state_between_new_adapters():
    c, counter = fixture(), [0]

    class Shared(ToyAdapter):
        def step(self, c, m, j):
            r = super().step(c, m, j)
            r["direction"] = r["direction"] + .1 * counter[0]
            counter[0] += 1
            return r

    with pytest.raises(ValueError, match="native replay mismatch: direction"):
        attribution(c, lambda c: Shared(), run_step=True)


def test_archived_prediction_anchor_detects_wrong_frozen_multiplier_policy():
    c = fixture()
    c["meta"]["predicted_replay"] = ToyAdapter().accept(c["coefficients"], c["predicted_moments"], c["predicted_jacobian"])

    class FrozenMultiplier(ToyAdapter):
        def accept(self, c, m, j):
            r = super().accept(c, m, j)
            r.update(multiplier=4., stationarity_residual=3., accepted=False)
            return r

    with pytest.raises(ValueError, match="native replay mismatch"):
        attribution(c, lambda c: FrozenMultiplier())


@pytest.mark.parametrize("dimension", [3, 8, 10])
def test_valid_charts_and_actual_direction_are_reported_separately(dimension):
    case = fixture(dimension)
    if dimension < 10:
        case["chart_basis"] *= np.arange(1, dimension + 1)[None, :]
    r = attribution(case, lambda c: ToyAdapter(), run_step=True)
    assert len(r["directional_derivative"]["raw_direction"]) == 10
    np.testing.assert_allclose(r["directional_derivative"]["scaled_unit_direction_error"],
        (case["predicted_jacobian"] - case["native_jacobian"]) @ case["direction"] /
        (2 * np.linalg.norm(case["chart_basis"] @ case["direction"])))
    assert "raw_coefficient" in r["step_direction_comparison"]["A_predM_predJ"]


def test_rank_deficient_but_qualified_physical_J_is_not_rejected():
    c = fixture()
    c["native_jacobian"][:] = 0.
    c["predicted_jacobian"][:] = 0.
    c["meta"]["native_replay"]["step"] = ToyAdapter().step(c["coefficients"], c["native_moments"], c["native_jacobian"])
    assert attribution(c, lambda c: ToyAdapter(), run_step=True)["derivative_status"] == "matched_valid"


def test_invalid_derivative_is_moment_only_not_zero_filled():
    c = fixture()
    c["meta"].update(derivative_valid=False, derivative_reason="topology_event")
    del c["native_jacobian"], c["predicted_jacobian"], c["chart_basis"]
    r = attribution(c, lambda c: ToyAdapter())
    assert len(r["fixed_endpoint"]) == 2
    assert r["one_step"] is None and r["endpoint_J_invariance_observed"] is None
    assert "directional_derivative" not in r
    with pytest.raises(ValueError, match="qualified derivatives"):
        attribution(c, lambda c: ToyAdapter(), run_step=True)


@pytest.mark.parametrize("direction", [None, np.zeros(10)])
def test_stopped_endpoint_direction_does_not_block_moment_attribution(direction):
    c = fixture()
    c["direction"] = direction
    r = attribution(c, lambda c: ToyAdapter())
    assert len(r["fixed_endpoint"]) == 4
    assert r["directional_derivative"]["status"] in ("direction_unavailable", "zero_direction_not_informative")


def test_native_and_initial_baselines_are_at_exactly_the_same_query():
    c = fixture()
    c["no_learning_moments"] = c["native_moments"].copy()
    c["initial_moments"] = 1.01 * c["predicted_moments"]
    r = attribution(c, lambda c: ToyAdapter(), run_step=True)
    baseline = r["alternative_responses"]["no_learning"]
    np.testing.assert_array_equal(baseline["scaled_moment_errors"], 0.)
    assert baseline["one_step"] is None and baseline["derivative_status"] == "unavailable"


def test_j_dependent_acceptance_cannot_masquerade_as_moment_only():
    c = fixture()

    class Bad(ToyAdapter):
        def accept(self, c, m, j):
            r = super().accept(c, m, j)
            r["stationarity_residual"] += j.sum()
            r["accepted"] = False
            return r

    c["meta"]["native_replay"]["accept"] = Bad().accept(c["coefficients"], c["native_moments"], c["native_jacobian"])
    with pytest.raises(ValueError, match="Moment-only"):
        attribution(c, lambda c: Bad())


def test_native_linear_solve_rejection_can_report_no_direction():
    c = fixture()

    class Refused(ToyAdapter):
        def step(self, c, m, j):
            r = super().step(c, m, j)
            r.update(direction=None, reason="synthetic_singular_bordered_system")
            return r

    c["meta"]["native_replay"]["step"] = Refused().step(c["coefficients"], c["native_moments"], c["native_jacobian"])
    r = attribution(c, lambda c: Refused(), run_step=True)
    assert all(v is None for v in r["step_direction_comparison"].values())


@pytest.mark.parametrize("mutation", ["moment_order", "coefficient_convention", "blind", "nan_J", "rank_chart", "threshold", "replay_missing"])
def test_fail_closed_on_ambiguous_semantics(mutation):
    c = fixture()
    if mutation == "moment_order": c["meta"]["moment_order"] = list(reversed(ORDER))
    elif mutation == "coefficient_convention": c["meta"]["coefficient_multipliers"] = [1.] * 10
    elif mutation == "blind": c["meta"]["split"] = "blind"
    elif mutation == "nan_J": c["native_jacobian"][0, 0] = np.nan
    elif mutation == "rank_chart": c["chart_basis"][0, 0] = 0.
    elif mutation == "threshold": c["meta"]["thresholds"]["stationarity"] = True
    else: c["meta"].pop("native_replay")
    with pytest.raises(ValueError): attribution(c, lambda c: ToyAdapter())


def test_real_artifact_bytes_are_checked_not_just_hash_format(tmp_path):
    meta = copy.deepcopy(fixture()["meta"])
    meta["artifacts"] = {}
    for role in ("geometry", "operand", "manifest", "checkpoint", "solver", "solver_state"):
        path = tmp_path / role
        path.write_text("actual bytes " + role)
        if role == "solver_state": meta["adapter_contract"]["external_state_sha256"] = sha256(path)
        else: meta[role + "_sha256"] = sha256(path)
        meta["artifacts"][role] = dict(path=role)
    assert len(verify_artifacts(meta, tmp_path)) == 6
    (tmp_path / "solver").write_text("changed equation")
    with pytest.raises(ValueError, match="solver artifact checksum"):
        verify_artifacts(meta, tmp_path)


def test_cli_passes_resolved_verified_assets_to_the_adapter(tmp_path, monkeypatch):
    from experiments.compact_pino_v21.p0_diagnostics import endpoint_attribution as module
    c = fixture()
    assets = tmp_path / "assets"
    assets.mkdir()
    c["meta"]["artifacts"] = {}
    for role in ("geometry", "operand", "manifest", "checkpoint", "solver", "solver_state"):
        path = assets / role
        path.write_text(role)
        if role == "solver_state": c["meta"]["adapter_contract"]["external_state_sha256"] = sha256(path)
        else: c["meta"][role + "_sha256"] = sha256(path)
        c["meta"]["artifacts"][role] = dict(path=role)
    meta_path, npz_path, output = tmp_path / "meta.json", tmp_path / "case.npz", tmp_path / "report.json"
    meta_path.write_text(json.dumps(jsonable(c["meta"])))
    np.savez(npz_path, **{k: v for k, v in c.items() if k != "meta"})

    def factory(case):
        assert case["asset_root"] == str(assets.resolve())
        for role, row in case["verified_artifacts"].items():
            assert Path(row["path"]) == assets / role
            expected = (c["meta"]["adapter_contract"]["external_state_sha256"] if role == "solver_state"
                        else c["meta"][role + "_sha256"])
            assert sha256(row["path"]) == expected
        return ToyAdapter()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "load_factory", lambda name: factory)
    monkeypatch.setattr(sys, "argv", ["attribution", "--case", str(npz_path), "--meta", str(meta_path),
                                     "--asset-root", str(assets), "--adapter", "toy:factory", "--output", str(output)])
    module.main()
    assert json.loads(output.read_text())["verified_artifacts"]["solver"]["sha256"] == c["meta"]["solver_sha256"]
