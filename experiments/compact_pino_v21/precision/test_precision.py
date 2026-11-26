"""Independent analytic response and common-query reporting contracts."""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import (
    load_baseline_case, sha256, write_synthetic_finite_case,
)
from experiments.compact_pino_v21.precision.analytic_cells import polynomial_reference, slanted_cell_report
from experiments.compact_pino_v21.precision.compare_rules import CASE_SCHEMA, SCHEMA, compare_case, compare_records, main


@pytest.fixture(scope="module")
def cell_report(tmp_path_factory):
    return slanted_cell_report(tmp_path_factory.mktemp("analytic_cell"))


@pytest.fixture(scope="module")
def case_path(tmp_path_factory):
    return write_synthetic_finite_case(tmp_path_factory.mktemp("precision_case") / "source")


@pytest.fixture(scope="module")
def case(case_path):
    return load_baseline_case(case_path)


def test_full_ten_moment_exactness_does_not_imply_response_exactness(cell_report):
    by_rule = {r["full_cell_rule"]: r for r in cell_report["records"]}
    assert [by_rule[name]["actual_nodes"] for name in ("tetra4", "gl2", "gl3")] == [4, 8, 27]
    for row in by_rule.values():
        assert row["minimum_weight"] > 0
        np.testing.assert_allclose(row["full_moment_scaled_error"], 0., atol=3e-14)
        np.testing.assert_allclose(row["complement_scaled_error"], 0., atol=3e-13)
        assert row["complement_jacobian_max"] < 3e-13
    # The independent reference is a degree-five polynomial, integrated by
    # antiderivatives rather than by another numerical quadrature routine.
    np.testing.assert_allclose(by_rule["gl3"]["scaled_moment_error"], 0., atol=3e-13)
    np.testing.assert_allclose(by_rule["gl3"]["chart_unit_Jv_scaled_error"], 0., atol=3e-12)
    assert np.linalg.norm(by_rule["tetra4"]["scaled_moment_error"]) > 1e-4
    assert np.linalg.norm(by_rule["gl2"]["scaled_moment_error"]) > 1e-5
    assert np.linalg.norm(by_rule["gl2"]["chart_unit_Jv_scaled_error"]) > 1e-5
    assert cell_report["fixed_grid"]["h_over_pitch"] == 1.
    assert cell_report["native_solver_acceptance"] == "NOT_EVALUATED"
    assert cell_report["reference_kind"] == "analytic_cubic_smoothed_finite_plane"


def test_polynomial_reference_has_independent_axis_plane_values():
    reference = polynomial_reference(np.full(3, -.5), np.full(3, .5), normal=[0, 0, 1],
                                     offset=0., bandwidth=1., density=1.)
    np.testing.assert_allclose(reference["moments"],
                               [.5, 0., 0., -19/320, 1/24, 1/24, 1/24, 0., 0., 0.], atol=3e-16)
    assert reference["jacobian"][0, 0] == pytest.approx(11/16, abs=3e-16)
    assert reference["jacobian"][6, 0] == pytest.approx(17/320, abs=3e-16)
    with pytest.raises(ValueError, match="strictly inside"):
        polynomial_reference(np.full(3, -.5), np.full(3, .5), normal=[1, 2, 3],
                             offset=.4, bandwidth=1.)


def test_analytic_response_preserves_physical_pitch_and_density(tmp_path):
    report = slanted_cell_report(tmp_path, pitch=.125, density=2.7, rules=("gl3",), chunk_size=2)
    assert report["fixed_grid"]["h"] == report["fixed_grid"]["pitch"] == .125
    row = report["records"][0]
    np.testing.assert_allclose(row["scaled_moment_error"], 0., atol=5e-13)
    np.testing.assert_allclose(row["chart_unit_Jv_scaled_error"], 0., atol=5e-12)
    assert row["diagnostics"]["band_nodes"] == 27


def test_common_queries_report_native_values_scales_directions_and_missing_profile(case):
    report = compare_case(case, chunk_size=113, repeats=2)
    assert report["schema"] == CASE_SCHEMA
    assert report["model_weights_used"] is False
    assert report["native_solver_acceptance"] == "NOT_EVALUATED"
    np.testing.assert_array_equal(report["native_moments"], case["native_moments"])
    assert report["fixed_grid"]["h"] == case["blocks"].pitch
    scale = np.asarray(report["moment_scale"])
    node_counts = []
    for row in report["records"]:
        assert row["h_over_pitch"] == 1.
        node_counts.append(row["actual_evaluated_band_nodes"])
        assert len(row["query_seconds"]) == 2 and all(t >= 0 for t in row["query_seconds"])
        assert row["physical_residual_error"]["availability"] == "UNAVAILABLE_PROFILE"
        predicted = np.asarray(row["moments"])
        np.testing.assert_allclose(row["moment_error"]["raw_delta"], predicted-case["native_moments"], rtol=0, atol=0)
        derivative = row["derivative_error"]
        assert derivative["availability"] == "AVAILABLE"
        assert derivative["dimension"] == 3 and derivative["derivative_type"] == "R3"
        directions = np.asarray(derivative["chart_unit_axes"]["directions"])
        parameters = np.asarray(derivative["parameter_scales"])
        np.testing.assert_allclose(np.linalg.norm(directions/parameters[:, None], axis=0), 1.)
        expected = ((np.asarray(derivative["predicted_jacobian"])-case["native_jacobian"])
                    @ directions/scale[:, None])
        np.testing.assert_allclose(derivative["chart_unit_axes"]["scaled_delta"], expected, atol=1e-14)
        assert derivative["native_direction"]["source"] == "synthetic_fixture"
    assert node_counts[0] < node_counts[1] < node_counts[2]


def test_unqualified_derivative_remains_missing_not_zero(case):
    withheld = dict(case, meta=dict(case["meta"]), reference_meta=dict(case["reference_meta"]))
    withheld["meta"].update(derivative_valid=False, derivative_reason="fixture deliberately withholds derivative")
    withheld["reference_meta"]["derivative_valid"] = False
    for name in ("native_jacobian", "chart_basis", "parameter_scales", "direction"):
        withheld.pop(name, None)
    report = compare_case(withheld, rules=("tetra4",))
    derivative = report["records"][0]["derivative_error"]
    assert derivative["availability"] == "UNAVAILABLE_NATIVE_REFERENCE"
    assert derivative["native_jacobian"] is None
    assert derivative["chart_unit_axes"] is None
    assert derivative["dimension"] == 3


def _linear_target(case):
    from experiments.compact_pino_v21.training.residual_matching import (
        LINEARIZED_PROFILE, MOMENT_ORDER, P_CONVENTION, TARGET_SCHEMA,
    )
    dual = case["coefficients"] * np.array([1.] * 7 + [2.] * 3)
    p = dual / np.linalg.norm(dual)
    hessian = np.eye(10)
    hessian[0, 3] = hessian[3, 0] = 2.
    return dict(schema=TARGET_SCHEMA, valid=True, profile=LINEARIZED_PROFILE,
                teacher_role="synthetic_fixture", binding=dict(case_sha256=case["manifest_receipt"]["sha256"]),
                formula_provenance=dict(source_profile="synthetic-linear-sensitivity-fixture",
                                        source_sha256={"test_precision.py": sha256(__file__)}),
                moment_order=list(MOMENT_ORDER), p_convention=P_CONVENTION,
                reference_moments=case["native_moments"].tolist(), p=p.tolist(),
                multipliers=[.2, -.1], multiplier_mu=.7, stationarity_scale=np.ones(10).tolist(),
                constraint_scale=[1., 1.], lagrangian_hessian=hessian.tolist(),
                constraint_jacobian=np.eye(2, 10).tolist(),
                reference_stationarity=np.full(10, .1).tolist(), reference_constraints=[.2, .3])


def test_fixed_reference_physical_error_is_labelled_linearized_and_preserves_joint_terms(case):
    target = _linear_target(case)
    report = compare_case(case, rules=("tetra4",), residual_target=target)
    row = report["records"][0]
    physics = row["physical_residual_error"]
    assert physics["availability"] == "AVAILABLE"
    assert physics["approximation"] == "first_order_at_native_moments"
    delta = np.asarray(row["moments"])-case["native_moments"]
    np.testing.assert_allclose(physics["stationarity_delta"], np.asarray(target["lagrangian_hessian"])@delta,
                               rtol=1e-12, atol=1e-14)
    assert physics["native_solver_acceptance"] == "NOT_EVALUATED"
    assert target == _linear_target(case)  # All multipliers, targets and scales remain frozen.


def test_invalid_or_unavailable_profile_cannot_be_silently_promoted(case):
    target = _linear_target(case)
    target["binding"]["case_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bound to this exact case"):
        compare_case(case, rules=("tetra4",), residual_target=target)
    target = _linear_target(case)
    target["profile"] = "pretend_deployment_profile"
    with pytest.raises(ValueError, match="unknown residual profile"):
        compare_case(case, rules=("tetra4",), residual_target=target)
    unavailable = {key: target[key] for key in ("schema", "binding")}
    unavailable.update(valid=False, reason="native fixed multipliers were not exported")
    report = compare_case(case, rules=("tetra4",), residual_target=unavailable)
    assert report["records"][0]["physical_residual_error"]["availability"] == "UNAVAILABLE_TARGET"
    # The strict manifest also accepts a bare explicit absence declaration.
    unbound = dict(valid=False, reason="multipliers were not exported")
    report = compare_case(case, rules=("tetra4",), residual_target=unbound)
    assert report["records"][0]["physical_residual_error"]["availability"] == "UNAVAILABLE_TARGET"


def test_full_material_gate_stops_all_query_comparisons(case):
    invalid = dict(case, native_full_editable_moments=case["native_full_editable_moments"].copy())
    invalid["native_full_editable_moments"][4] += .1
    report = compare_case(invalid)
    assert report["status"] == "STOP_FULL_MATERIAL_REFERENCE"
    assert report["records"] == []


def test_record_wrapper_keeps_identity_and_rejects_duplicates(case):
    record = dict(record_id="synthetic/plane", role="train", case=case)
    report = compare_records([record], rules=("tetra4",))
    assert report["schema"] == SCHEMA and report["selected_record_ids"] == ["synthetic/plane"]
    assert report["cases"][0]["material_identity"] == case["blocks"].identity
    assert report["cases"][0]["manifest_receipt"] == case["manifest_receipt"]
    assert report["runtime"]["device"] == "cpu"
    with pytest.raises(ValueError, match="unique"):
        compare_records([record, record], rules=("tetra4",))
    with pytest.raises(ValueError, match="eligible frozen case"):
        compare_records([])


def test_cli_direct_case_and_strict_manifest_are_runnable(case_path, tmp_path):
    direct = tmp_path/"direct.json"
    assert main(["--case", str(case_path), "--rules", "tetra4", "--output", str(direct)]) == 0
    result = json.loads(direct.read_text())
    assert result["source"]["source_authority"] == "NOT_RECHECKED_WITHOUT_MANIFEST"
    row = dict(record_id="fixture-train", role="train", loader_eligible=True,
               case_path=str(case_path), case_sha256=sha256(case_path))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(schema="compact_pino.training-manifest.v1",
                                        snapshot_id="synthetic-precision-cli", registered_train_count=1, records=[row])))
    output = tmp_path/"manifest-report.json"
    assert main(["--manifest", str(manifest), "--mode", "smoke", "--roles", "train",
                 "--rules", "gl3", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["source"]["sha256"] == sha256(manifest)
    assert result["selected_record_ids"] == ["fixture-train"]
    assert result["source"]["excluded_record_ids"] == []
    assert result["cases"][0]["records"][0]["full_cell_rule"] == "gl3"
    with pytest.raises(SystemExit):
        main(["--case", str(case_path), "--output", str(output)])


def test_locked_case_is_refused_before_target_read(case_path, tmp_path):
    spec = json.loads(case_path.read_text())
    spec["meta"]["split"] = "locked_test"
    spec["assets"]["targets"]["path"] = "missing-should-not-be-opened.npz"
    path = tmp_path/"locked.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="locked test is not inspected"):
        main(["--case", str(path), "--output", str(tmp_path/"forbidden.json")])
    assert not (tmp_path/"forbidden.json").exists()
