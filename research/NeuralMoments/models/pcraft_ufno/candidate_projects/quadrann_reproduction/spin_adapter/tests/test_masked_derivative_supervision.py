"""Analytic and padding regressions for frozen directional derivative labels."""
from __future__ import annotations

import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.masked_derivative_supervision import derivative_supervision_loss


def _mixed_batch():
    generator = torch.Generator().manual_seed(23119)
    prediction = torch.randn(4, 10, 10, generator=generator, dtype=torch.float64)
    prediction[3] = float("nan")
    full = torch.full_like(prediction, float("nan"))
    full[0] = torch.randn(10, 10, generator=generator, dtype=torch.float64)
    basis = torch.full((4, 10, 8), float("nan"), dtype=torch.float64)
    target = torch.full_like(basis, float("nan"))
    for row, dimension in ((1, 3), (2, 8)):
        basis[row, :, :dimension] = torch.randn(10, dimension, generator=generator, dtype=torch.float64)
        target[row, :, :dimension] = torch.randn(10, dimension, generator=generator, dtype=torch.float64)
    return prediction.requires_grad_(), full.requires_grad_(), basis.requires_grad_(), target.requires_grad_()


def test_mixed_full_plane_paraboloid_and_moment_only_has_analytic_gradient_and_query_weights():
    prediction, full, basis, target = _mixed_batch()
    full_scale = torch.linspace(.7, 2., 100, dtype=torch.float64).reshape(10, 10)
    output_scale = torch.linspace(.5, 2., 10, dtype=torch.float64)
    result = derivative_supervision_loss(prediction, full,
        derivative_valid=torch.tensor([True, False, False, False]),
        restricted_basis=basis, restricted_target=target,
        restricted_valid=torch.tensor([False, True, True, False]),
        restricted_dimension=torch.tensor([float("nan"), 3., 8., float("nan")]),
        jacobian_scale=full_scale, restricted_scale=output_scale)
    expected_gradient = torch.zeros_like(prediction)
    full_error = (prediction[0].detach() - full[0].detach()) / full_scale
    expected_gradient[0] = 2 * (prediction[0].detach() - full[0].detach()) / (full_scale.square() * 300)
    row_means = [full_error.square().sum() / 100]
    element_sum = full_error.square().sum()
    for row, dimension in ((1, 3), (2, 8)):
        b = basis[row, :, :dimension].detach()
        # The nonorthogonal teacher basis is used directly, without QR or normalization.
        assert not torch.allclose(b.T @ b, torch.eye(dimension, dtype=torch.float64))
        raw_error = prediction[row].detach() @ b - target[row, :, :dimension].detach()
        error = raw_error / output_scale[:, None]
        row_means.append(error.square().sum() / (10 * dimension))
        element_sum += error.square().sum()
        expected_gradient[row] = 2 * (raw_error / output_scale[:, None].square()) @ b.T / (3 * 10 * dimension)
    expected = torch.stack(row_means).mean()
    torch.testing.assert_close(result["loss"], expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(result["full_loss"], row_means[0])
    torch.testing.assert_close(result["restricted_loss"], torch.stack(row_means[1:]).mean())
    assert not torch.isclose(result["loss"], element_sum / (100 + 30 + 80))
    assert result["full_count"].item() == 1
    assert result["restricted_count"].item() == 2
    assert result["restricted_dimensions"].item() == 11
    for name in ("full_count", "restricted_count", "restricted_dimensions"):
        assert result[name].shape == () and result[name].dtype == torch.int64
    result["loss"].backward()
    torch.testing.assert_close(prediction.grad, expected_gradient, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[3], torch.zeros_like(prediction.grad[3]))
    assert full.grad is None and basis.grad is None and target.grad is None


def test_full_only_defaults_to_all_rows_and_preserves_standard_mse():
    prediction = torch.arange(200, dtype=torch.float64).reshape(2, 10, 10).requires_grad_()
    target = torch.ones_like(prediction, requires_grad=True)
    scale = torch.full((10, 10), 2., dtype=torch.float64, requires_grad=True)
    result = derivative_supervision_loss(prediction, target, jacobian_scale=scale)
    torch.testing.assert_close(result["loss"], ((prediction - 1.) / 2.).square().mean())
    assert result["full_count"].item() == 2
    assert result["restricted_count"].item() == result["restricted_dimensions"].item() == 0
    assert result["restricted_loss"].item() == 0 and not result["restricted_loss"].requires_grad
    result["loss"].backward()
    assert target.grad is None and scale.grad is None


@pytest.mark.parametrize("branch",["full", "restricted"])
def test_python_scalar_scales_retain_prediction_precision(branch):
    prediction = torch.ones(1, 10, 10, dtype=torch.float64)
    if branch == "full":
        result = derivative_supervision_loss(prediction, torch.zeros_like(prediction),
                                              jacobian_scale=.1)
    else:
        basis = torch.eye(10, dtype=torch.float64)[None, :, :3]
        result = derivative_supervision_loss(prediction, None, restricted_basis=basis,
            restricted_target=torch.zeros_like(basis), restricted_scale=.1)
    torch.testing.assert_close(result["loss"], torch.tensor(100., dtype=torch.float64),
                               rtol=1e-14, atol=0)


@pytest.mark.parametrize("scale_shape",[(), (10,), (10, 1)])
def test_restricted_only_defaults_and_scales_raw_directional_derivatives(scale_shape):
    prediction = torch.arange(100, dtype=torch.float64).reshape(1, 10, 10) / 10
    basis = torch.arange(30, dtype=torch.float64).reshape(1, 10, 3) / 15
    target = torch.ones_like(basis)
    scale = torch.full(scale_shape, 2., dtype=torch.float64)
    full_scale = torch.linspace(1., 20., 100).reshape(10, 10)
    result = derivative_supervision_loss(prediction, None, restricted_basis=basis,
        restricted_target=target, jacobian_scale=full_scale, restricted_scale=scale)
    torch.testing.assert_close(result["loss"], ((prediction @ basis - target) / 2.).square().mean())
    assert result["full_loss"].item() == 0
    assert result["restricted_count"].item() == 1 and result["restricted_dimensions"].item() == 3
    incorrectly_projected = ((prediction / full_scale) @ basis - target) / 2.
    assert not torch.isclose(result["loss"], incorrectly_projected.square().mean())


def test_all_derivative_masks_off_ignores_nan_padding_and_builds_no_graph():
    prediction = torch.full((2, 10, 10), float("nan"), dtype=torch.float64, requires_grad=True)
    target = torch.full_like(prediction, float("nan"), requires_grad=True)
    basis = torch.full((2, 10, 8), float("nan"), dtype=torch.float64, requires_grad=True)
    restricted = torch.full_like(basis, float("nan"), requires_grad=True)
    off = torch.zeros(2, dtype=torch.bool)
    result = derivative_supervision_loss(prediction, target, derivative_valid=off,
        restricted_basis=basis, restricted_target=restricted, restricted_valid=off,
        restricted_dimension=torch.full((2,), float("nan")),
        jacobian_scale=float("nan"), restricted_scale=float("nan"))
    assert all(value.item() == 0 and not value.requires_grad for value in result.values())
    assert result["loss"].dtype == prediction.dtype and result["loss"].device == prediction.device
    without_targets = derivative_supervision_loss(prediction, None,
        derivative_valid=off, restricted_valid=off)
    assert all(value.item() == 0 and not value.requires_grad for value in without_targets.values())


def test_overlapping_full_and_restricted_masks_are_rejected():
    prediction = torch.zeros(1, 10, 10)
    with pytest.raises(ValueError, match="disjoint"):
        derivative_supervision_loss(prediction, torch.zeros_like(prediction),
            restricted_basis=torch.ones(1, 10, 3), restricted_target=torch.zeros(1, 10, 3))


@pytest.mark.parametrize("argument,value,message",[
    ("derivative_valid", torch.tensor([1]), "boolean mask"),
    ("derivative_valid", torch.tensor([[True]]), "shape"),
    ("restricted_valid", torch.tensor([0.]), "boolean mask"),
    ("jacobian_scale", torch.ones(10), "shape"),
    ("restricted_scale", torch.ones(10, 3), "shape"),
    ("restricted_dimension", torch.ones(1, 1), "shape"),
    ("restricted_dimension", torch.tensor([True]), "real vector"),
])
def test_shapes_and_types_fail_with_meaningful_errors(argument,value,message):
    with pytest.raises(ValueError, match=message):
        derivative_supervision_loss(torch.zeros(1, 10, 10), None, **{argument:value})


@pytest.mark.parametrize("name",["full_target", "restricted_basis", "restricted_target"])
def test_complex_teacher_data_is_rejected_even_when_masked_out(name):
    prediction = torch.zeros(1, 10, 10)
    full = torch.zeros_like(prediction)
    options = {"derivative_valid":torch.tensor([False]),
        "restricted_valid":torch.tensor([False]),
        "restricted_basis":torch.zeros(1, 10, 3), "restricted_target":torch.zeros(1, 10, 3)}
    if name == "full_target":
        full = full.to(torch.complex64)
    else:
        options[name] = options[name].to(torch.complex64)
    with pytest.raises(ValueError, match="real numeric"):
        derivative_supervision_loss(prediction, full, **options)


@pytest.mark.parametrize("dimension",[0., 4., 1.5, float("nan"), float("inf")])
def test_active_restricted_dimensions_must_be_finite_exact_integers_in_range(dimension):
    with pytest.raises(ValueError, match="restricted_dimension"):
        derivative_supervision_loss(torch.zeros(1, 10, 10), None,
            restricted_basis=torch.ones(1, 10, 3), restricted_target=torch.zeros(1, 10, 3),
            restricted_dimension=torch.tensor([dimension]))


@pytest.mark.parametrize("invalid",["prediction", "target", "scale"])
def test_active_full_nonfinite_values_are_rejected(invalid):
    prediction = torch.zeros(1, 10, 10)
    target = torch.zeros_like(prediction)
    scale = 1.
    if invalid == "prediction": prediction[0, 1, 2] = float("nan")
    elif invalid == "target": target[0, 1, 2] = float("nan")
    else: scale = float("nan")
    with pytest.raises(ValueError, match="finite"):
        derivative_supervision_loss(prediction, target, jacobian_scale=scale)


@pytest.mark.parametrize("invalid",["prediction", "basis", "target", "scale"])
def test_active_restricted_nonfinite_values_are_rejected(invalid):
    prediction = torch.zeros(1, 10, 10)
    basis = torch.ones(1, 10, 3)
    target = torch.zeros_like(basis)
    scale = 1.
    if invalid == "prediction": prediction[0, 1, 2] = float("nan")
    elif invalid == "basis": basis[0, 1, 2] = float("nan")
    elif invalid == "target": target[0, 1, 2] = float("nan")
    else: scale = float("nan")
    with pytest.raises(ValueError, match="finite"):
        derivative_supervision_loss(prediction, None,
            restricted_basis=basis, restricted_target=target, restricted_scale=scale)


@pytest.mark.parametrize("branch",["full", "restricted"])
@pytest.mark.parametrize("scale",[0., -1.])
def test_active_normalization_scales_must_be_positive(branch,scale):
    prediction = torch.zeros(1, 10, 10)
    if branch == "full":
        options = {"jacobian_scale":scale}
        target = torch.zeros_like(prediction)
    else:
        options = {"restricted_basis":torch.ones(1, 10, 3),
                   "restricted_target":torch.zeros(1, 10, 3), "restricted_scale":scale}
        target = None
    with pytest.raises(ValueError, match="positive"):
        derivative_supervision_loss(prediction, target, **options)


@pytest.mark.parametrize("options,message",[
    ({"derivative_valid":torch.tensor([True])}, "full_target is required"),
    ({"restricted_valid":torch.tensor([True])}, "required"),
    ({"restricted_basis":torch.zeros(1,10,3)}, "provided together"),
    ({"restricted_basis":torch.zeros(1,10,3), "restricted_target":torch.zeros(1,10,8)}, "matching shape"),
])
def test_required_teacher_components_and_shapes_are_validated(options,message):
    with pytest.raises(ValueError, match=message):
        derivative_supervision_loss(torch.zeros(1,10,10), None, **options)


@pytest.mark.parametrize("prediction",[torch.zeros(10,10), torch.zeros(1,10,9),
                                        torch.zeros(1,10,10,dtype=torch.complex64)])
def test_prediction_requires_real_batched_full_jacobian(prediction):
    with pytest.raises(ValueError, match="prediction"):
        derivative_supervision_loss(prediction, None)


def test_full_target_shape_is_validated_before_arithmetic():
    with pytest.raises(ValueError, match="full_target.*shape"):
        derivative_supervision_loss(torch.zeros(1,10,10), torch.zeros(1,10))
