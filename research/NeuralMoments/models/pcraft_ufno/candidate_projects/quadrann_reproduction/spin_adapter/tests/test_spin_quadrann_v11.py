import numpy as np
import pytest
import torch

from spin_adapter.spin_quadrann_v11 import SpinQuadrANNV11
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
        "side": np.array(512, dtype=np.int32),
        "density": np.array(1.7),
    })


def make_model(device="cpu"):
    return SpinQuadrANNV11(
        preset="large_ms",
        hidden_dim=16,
        final_hidden_dim=64,
        num_layers=3,
        neighbors=4,
        temperature=.04,
        quadrature_order=2,
        chunk_intervals=2,
        chunk_nodes=16,
    ).to(device)


def make_coefficients(device="cpu"):
    torch.manual_seed(830231)
    c = torch.randn(10, device=device)
    return c / c.norm()


def test_v11_forward_and_physics_contract():
    geometry = make_geometry()
    model = make_model().eval()

    with torch.no_grad():
        result = model(
            geometry=geometry,
            coefficients=make_coefficients(),
            with_jacobian=True,
        )

    assert result["moments"].shape == (10,)
    assert result["jacobian_q"].shape == (10, 10)
    assert result["jacobian_c"].shape == (10, 10)

    assert result["logits"].shape == (5,)
    assert result["probabilities"].shape == (5,)
    assert result["learned_interval_mass"].shape == (5,)

    assert torch.isfinite(result["moments"]).all()
    assert torch.isfinite(result["jacobian_q"]).all()

    assert (result["node_weights"] > 0).all()

    assert result["total_mass_error"].item() < 1e-6

    assert torch.allclose(
        result["probabilities"].sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )

    assert torch.allclose(
        result["jacobian_q"],
        result["jacobian_q"].T,
        atol=1e-5,
        rtol=1e-5,
    )


def test_zero_fusion_logits_recover_base_rle_weights():
    geometry = make_geometry()
    model = make_model().eval()

    # Zero logits make the mass-prior Softmax reproduce
    # the original RLE interval mass distribution.
    torch.nn.init.zeros_(model.weight_head[-1].weight)
    torch.nn.init.zeros_(model.weight_head[-1].bias)

    coefficients = make_coefficients()

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


def test_both_branches_receive_finite_nonzero_gradients():
    geometry = make_geometry()
    model = make_model().train()

    result = model(
        geometry=geometry,
        coefficients=make_coefficients(),
        with_jacobian=False,
    )

    loss = (
        result["moments"].square().mean()
        + 0.1 * result["logits"].square().mean()
    )

    loss.backward()

    branches = {
        "pino_encoder": model.pcraft.operator.pino_encoder,
        "pino_scale_score": model.pcraft.operator.pino_scale_score,
        "quadrann_initial": model.gnn.initial,
        "quadrann_layers": model.gnn.layers,
        "fusion": model.fusion,
        "weight_head": model.weight_head,
    }

    for name, module in branches.items():
        grads = [
            p.grad for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]

        assert grads, f"{name}: no gradients"

        assert all(
            torch.isfinite(g).all() for g in grads
        ), f"{name}: nonfinite gradients"

        norm = torch.sqrt(
            sum(g.abs().square().sum() for g in grads)
        )

        assert norm.item() > 0, f"{name}: zero gradient"


def test_unused_quadrann_head_is_removed():
    geometry = make_geometry()
    model = make_model()

    assert isinstance(model.gnn.output, torch.nn.Identity)

    result = model(
        geometry=geometry,
        coefficients=make_coefficients(),
        with_jacobian=False,
    )

    assert result["logits"].shape == (5,)


def test_original_pcraft_query_head_is_not_called():
    geometry = make_geometry()
    model = make_model().eval()

    calls = []

    handle = model.pcraft.operator.query_head.register_forward_hook(
        lambda module, inputs, output: calls.append(True)
    )

    try:
        with torch.no_grad():
            result = model(
                geometry=geometry,
                coefficients=make_coefficients(),
                with_jacobian=False,
            )
    finally:
        handle.remove()

    assert result["moments"].shape == (10,)
    assert calls == []


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_v11_cuda_forward_backward():
    geometry = make_geometry()
    model = make_model("cuda").train()

    result = model(
        geometry=geometry,
        coefficients=make_coefficients("cuda"),
        with_jacobian=True,
    )

    assert result["moments"].is_cuda
    assert result["jacobian_q"].is_cuda

    loss = result["moments"].square().mean()
    loss.backward()

    for module in (
        model.pcraft.operator.pino_encoder,
        model.gnn.initial,
        model.fusion,
        model.weight_head,
    ):
        grads = [
            p.grad for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]

        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
