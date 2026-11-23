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
