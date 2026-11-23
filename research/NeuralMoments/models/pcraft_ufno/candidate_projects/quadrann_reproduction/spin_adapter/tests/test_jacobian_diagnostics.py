import json
import warnings

import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.jacobian_diagnostics import (
    audit_jacobian_paths,
    compare_jacobians,
    directional_jacobian_error,
    finite_difference_jacobian,
)


def _moment(c):
    # Ten outputs with a nontrivial, exactly known raw-coefficient Jacobian.
    return torch.stack((
        c[:, 0] ** 2,
        c[:, 1] * c[:, 2],
        c[:, 3] + 2 * c[:, 7],
        c[:, 4] * c[:, 5],
        c[:, 6] ** 2 + c[:, 8],
        c[:, 9] ** 3,
        c[:, 0] * c[:, 9],
        c[:, 1] + c[:, 4],
        c[:, 2] - c[:, 6],
        c.sum(-1),
    ), dim=-1)


def _jacobian(c):
    rows = []
    for q in c:
        j = torch.zeros(10, 10, dtype=c.dtype, device=c.device)
        j[0, 0] = 2 * q[0]
        j[1, 1], j[1, 2] = q[2], q[1]
        j[2, 3], j[2, 7] = 1, 2
        j[3, 4], j[3, 5] = q[5], q[4]
        j[4, 6], j[4, 8] = 2 * q[6], 1
        j[5, 9] = 3 * q[9] ** 2
        j[6, 0], j[6, 9] = q[9], q[0]
        j[7, 1], j[7, 4] = 1, 1
        j[8, 2], j[8, 6] = 1, -1
        j[9, :] = 1
        rows.append(j)
    return torch.stack(rows)


def test_central_difference_matches_known_raw_jacobian():
    c = torch.tensor([[.3, -.2, .4, .5, -.6, .7, .8, -.9, .2, .1]],
                     dtype=torch.float64)
    expected = _jacobian(c)
    actual = finite_difference_jacobian(_moment, c, step=1e-5)
    assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-7)


def test_compare_jacobians_reports_per_column_and_scaled_metrics():
    c = torch.randn(3, 10, dtype=torch.float64)
    reference = _jacobian(c)
    report = compare_jacobians(reference, reference, component_scales=torch.ones(10))
    assert report["relative_frobenius"] == 0.0
    assert len(report["per_coefficient_column_relative"]) == 10
    assert report["component_scaled_rmse"] == 0.0


def test_directional_error_uses_actual_query_directions():
    c = torch.randn(2, 10, dtype=torch.float64)
    j = _jacobian(c)
    v = torch.randn(2, 10, dtype=torch.float64)
    report = directional_jacobian_error(j, j, v)
    assert report["relative_l2"] == 0.0
    assert abs(report["cosine_similarity"] - 1.0) < 1e-12


def test_audit_separates_self_consistency_from_native_accuracy():
    c = torch.tensor([[.2, .3, -.4, .5, .6, -.7, .8, .9, -.1, .15]],
                     dtype=torch.float64)
    j = _jacobian(c)
    v = torch.ones_like(c)
    report = audit_jacobian_paths(
        _moment, _jacobian, c,
        step_sizes=(1e-4, 1e-5),
        native_jacobian=j,
        directions=v,
    )
    assert set(report["model_self_consistency"]) == {"0.0001", "1e-05"}
    assert report["native_physical_accuracy"]["relative_frobenius"] < 1e-12
    assert report["native_physical_accuracy"]["newton_direction"]["relative_l2"] < 1e-12


def test_invalid_inputs_fail_closed():
    c = torch.zeros(10, dtype=torch.float64)
    try:
        finite_difference_jacobian(_moment, c, step=1e-4)
    except ValueError as exc:
        assert "nonzero" in str(exc)
    else:
        raise AssertionError("zero coefficient vector must be rejected")


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_step_and_metric_epsilon_must_be_positive_finite(value):
    c = torch.ones(1, 10, dtype=torch.float64)
    j = _jacobian(c)
    with pytest.raises(ValueError, match="positive and finite"):
        finite_difference_jacobian(_moment, c, step=value)
    with pytest.raises(ValueError, match="positive and finite"):
        compare_jacobians(j, j, eps=value)
    with pytest.raises(ValueError, match="positive and finite"):
        directional_jacobian_error(j, j, c, eps=value)


@pytest.mark.parametrize("shape", [(0, 10, 10), (1, 0, 10), (1, 10, 0)])
def test_empty_jacobians_are_rejected(shape):
    empty = torch.empty(shape)
    with pytest.raises(ValueError, match="nonempty"):
        compare_jacobians(empty, empty)
    with pytest.raises(ValueError, match="nonempty"):
        directional_jacobian_error(empty, empty, torch.empty(shape[0], shape[2]))


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex128])
def test_nonreal_or_nonfloating_jacobians_are_rejected(dtype):
    invalid = torch.ones(1, 2, 10, dtype=dtype)
    with pytest.raises(TypeError, match="floating-point"):
        compare_jacobians(invalid, invalid)


def test_small_aligned_vectors_and_zero_vectors_are_not_confused():
    small = torch.full((1, 2, 10), 1e-8, dtype=torch.float64)
    aligned = compare_jacobians(small, small)
    assert aligned["cosine_defined"]
    assert aligned["cosine_similarity"] == pytest.approx(1.0)
    zero = compare_jacobians(torch.zeros_like(small), torch.zeros_like(small))
    assert zero["cosine_similarity"] is None
    assert not zero["cosine_defined"]
    assert zero["relative_denominator_floored"]
    assert zero["near_zero_reference_columns"] == list(range(10))
    # Zero-signal reports remain valid JSON instead of emitting NaN.
    json.dumps(zero, allow_nan=False)


def test_metric_norm_overflow_fails_instead_of_reporting_cosine_zero():
    huge = torch.full((1, 2, 10), 1e200, dtype=torch.float64)
    with pytest.raises(ValueError, match="norms overflowed"):
        compare_jacobians(huge, huge)


def test_component_scale_lists_preserve_float64_range():
    predicted = torch.full((1, 2, 10), 1e-51, dtype=torch.float64)
    report = compare_jacobians(predicted, torch.zeros_like(predicted),
                              component_scales=[1e-50, 1e-50])
    assert report["component_scaled_rmse"] == pytest.approx(0.1)


def test_directional_products_align_dtype_before_matmul_and_detach():
    # A float32 matmul before promotion would overflow even though the FP64
    # result and all requested error metrics are representable.
    predicted = torch.full((2, 3, 10), 1e30, requires_grad=True)
    reference = predicted.detach().to(torch.float64).requires_grad_(True)
    directions = torch.full((2, 10), 1e10, dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = directional_jacobian_error(predicted, reference, directions)
    assert report["relative_l2"] == 0.0
    assert report["cosine_similarity"] == pytest.approx(1.0)
    assert len(report["per_query"]) == 2
    assert predicted.grad is reference.grad is directions.grad is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_directions_fail_before_projection(bad):
    j = torch.ones(1, 2, 10)
    directions = torch.ones(1, 10)
    directions[0, 2] = bad
    with pytest.raises(ValueError, match="directions must be finite"):
        directional_jacobian_error(j, j, directions)


def test_zero_directions_are_explicitly_uninformative():
    j = torch.ones(2, 2, 10)
    directions = torch.ones(2, 10)
    directions[1] = 0
    report = directional_jacobian_error(j, j, directions)
    assert report["near_zero_direction_queries"] == [1]
    assert report["per_query"][0]["cosine_similarity"] == pytest.approx(1.0)
    assert report["per_query"][1]["cosine_similarity"] is None
    assert report["per_query"][1]["near_zero_direction"]


def test_query_reports_locate_one_bad_query_in_a_synthetic_24_query_batch():
    c = torch.arange(1, 241, dtype=torch.float64).reshape(24, 10) / 100
    reference = _jacobian(c)
    predicted = reference.clone()
    predicted[17, 3, 4] += 0.5
    labels = [f"synthetic_query_{i:02d}" for i in range(24)]
    report = compare_jacobians(predicted, reference, query_ids=labels,
                              component_scales=torch.ones(10))
    assert [row["query_index"] for row in report["per_query"]] == list(range(24))
    assert [row["query_id"] for row in report["per_query"]] == labels
    nonzero = [row for row in report["per_query"] if row["absolute_frobenius"] > 0]
    assert len(nonzero) == 1
    assert nonzero[0]["query_index"] == 17
    assert nonzero[0]["per_coefficient_column_absolute"][4] == pytest.approx(0.5)
    assert nonzero[0]["component_scaled_rmse"] > 0


def test_finite_differences_default_to_batch_diagonal_blocks():
    c = torch.ones(2, 10, dtype=torch.float64)

    def coupled_moments(values):
        return values[:, :2] + values[:, :2].mean(dim=0, keepdim=True)

    diagonal = finite_difference_jacobian(coupled_moments, c, step=1e-5)
    expected = torch.zeros(2, 2, 10, dtype=c.dtype)
    expected[:, 0, 0] = expected[:, 1, 1] = 1.5
    torch.testing.assert_close(diagonal, expected)
    # The explicit fast path assumes a property this coupled function lacks.
    # Its result demonstrates why that assumption must never be implicit.
    simultaneous = finite_difference_jacobian(coupled_moments, c, step=1e-5,
                                              assume_query_independence=True)
    assert simultaneous[0, 0, 0] == pytest.approx(2.0)


def test_independent_query_fast_path_matches_default_and_leaves_no_graph():
    c = torch.ones(2, 10, dtype=torch.float64, requires_grad=True)

    def internally_differentiable(values):
        with torch.enable_grad():
            return _moment(values.requires_grad_(True))

    diagonal = finite_difference_jacobian(internally_differentiable, c, step=1e-5)
    independent = finite_difference_jacobian(internally_differentiable, c, step=1e-5,
                                             assume_query_independence=True)
    torch.testing.assert_close(diagonal, independent)
    assert not diagonal.requires_grad and not independent.requires_grad
    assert diagonal.grad_fn is independent.grad_fn is None


def test_unrepresentable_finite_difference_step_is_rejected():
    with pytest.raises(ValueError, match="representable"):
        finite_difference_jacobian(_moment, torch.ones(1, 10), step=1e-20)


@pytest.mark.parametrize("bad_output", [lambda c: torch.empty(len(c), 0),
                                      lambda c: torch.ones(len(c), 10, dtype=torch.int64),
                                      lambda c: torch.full((len(c), 10), float("nan"))])
def test_invalid_moment_outputs_are_rejected(bad_output):
    with pytest.raises((ValueError, TypeError)):
        finite_difference_jacobian(bad_output, torch.ones(1, 10), step=1e-4)


@pytest.mark.parametrize("steps", [(), (1e-4, 1e-4)])
def test_empty_or_duplicate_sweeps_are_rejected_before_model_calls(steps):
    def must_not_run(_):
        raise AssertionError("invalid sweep must not invoke the model")

    with pytest.raises(ValueError, match="nonempty and unique"):
        audit_jacobian_paths(must_not_run, must_not_run, torch.ones(1, 10), step_sizes=steps)


def test_audit_preserves_query_ids_and_explicit_fd_semantics():
    c = torch.ones(2, 10, dtype=torch.float64)
    report = audit_jacobian_paths(_moment, _jacobian, c, step_sizes=(1e-4,),
        native_jacobian=_jacobian(c), directions=c,
        query_ids=["synthetic_a", "synthetic_b"], assume_query_independence=True)
    assert report["finite_difference_semantics"] == "independent_queries_assumed"
    for comparison in (report["model_self_consistency"]["0.0001"],
                       report["native_physical_accuracy"],
                       report["native_physical_accuracy"]["newton_direction"]):
        assert [row["query_id"] for row in comparison["per_query"]] == ["synthetic_a", "synthetic_b"]
