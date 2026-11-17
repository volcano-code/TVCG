import numpy as np
import pytest
import torch

from spin_adapter.spin_quadrann import SpinQuadrANN
from spin_adapter.rle_quadrature import rle_gauss_quadrature

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    geometry_from_columns,
)


def make_geometry():
    columns = np.array([
        [1, 1, 1, 5],
        [1, 2, 2, 7],
        [2, 1, 0, 6],
        [3, 3, 1, 8],
        [4, 3, 0, 7],
    ], dtype=np.int32)

    return geometry_from_columns({
        "columns": columns,
        "origin": np.array([-1., -2., .5]),
        "h": np.array(.25),
        "side": np.array(8, dtype=np.int32),
        "density": np.array(1.7),
    })


def make_model():
    return SpinQuadrANN(
        hidden_dim=16,
        final_hidden_dim=32,
        num_layers=3,
        neighbors=4,
        temperature=.04,
        quadrature_order=2,
        chunk_nodes=16,
    )


def test_integrated_forward_shapes_and_mass():
    geometry = make_geometry()
    model = make_model().eval()

    coefficients = torch.randn(10)
    coefficients = coefficients / coefficients.norm()

    result = model(
        geometry=geometry,
        coefficients=coefficients,
        with_jacobian=True,
    )

    assert result["moments"].shape == (10,)
    assert result["jacobian_q"].shape == (10, 10)
    assert result["jacobian_c"].shape == (10, 10)

    assert torch.isfinite(result["moments"]).all()
    assert torch.isfinite(result["jacobian_q"]).all()

    assert (result["node_weights"] > 0).all()

    assert result["total_mass_error"].item() < 1e-6


def test_zero_logits_recover_base_quadrature_weights():
    geometry = make_geometry()
    model = make_model().eval()

    # Force exactly zero GNN logits.
    final_layer = model.encoder.output[-1]

    torch.nn.init.zeros_(final_layer.weight)
    torch.nn.init.zeros_(final_layer.bias)

    coefficients = torch.randn(10)
    coefficients = coefficients / coefficients.norm()

    result = model(
        geometry=geometry,
        coefficients=coefficients,
        with_jacobian=False,
    )

    base = rle_gauss_quadrature(
        geometry, order=2
    )

    expected = torch.as_tensor(
        base["weights"],
        dtype=result["node_weights"].dtype,
    )

    assert torch.allclose(
        result["node_weights"],
        expected,
        rtol=1e-5,
        atol=1e-7,
    )


def test_moment_loss_reaches_gnn_parameters():
    geometry = make_geometry()
    model = make_model().train()

    coefficients = torch.randn(10)
    coefficients = coefficients / coefficients.norm()

    result = model(
        geometry=geometry,
        coefficients=coefficients,
        with_jacobian=True,
    )

    loss = result["moments"].square().mean()
    loss.backward()

    grads = [
        p.grad for p in model.encoder.parameters()
        if p.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(g).all() for g in grads)

    grad_norm = torch.sqrt(
        sum(g.square().sum() for g in grads)
    )

    assert grad_norm.item() > 0


def test_q_jacobian_is_symmetric_psd():
    geometry = make_geometry()
    model = make_model().eval()

    coefficients = torch.randn(10)
    coefficients = coefficients / coefficients.norm()

    result = model(
        geometry=geometry,
        coefficients=coefficients,
    )

    J = result["jacobian_q"]

    assert torch.allclose(
        J, J.T, rtol=1e-5, atol=1e-6
    )

    assert torch.linalg.eigvalsh(
        0.5*(J+J.T)
    ).min() >= -1e-6


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_integrated_cuda_forward_backward():
    geometry = make_geometry()
    model = make_model().cuda().train()

    coefficients = torch.randn(
        10, device="cuda"
    )
    coefficients = coefficients / coefficients.norm()

    result = model(
        geometry=geometry,
        coefficients=coefficients,
    )

    assert result["moments"].is_cuda

    loss = result["moments"].square().mean()
    loss.backward()

    grads = [
        p.grad for p in model.encoder.parameters()
        if p.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
