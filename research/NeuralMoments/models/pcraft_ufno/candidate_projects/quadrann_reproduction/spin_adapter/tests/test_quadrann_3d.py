import numpy as np
import pytest
import torch

from spin_adapter.quadrann_3d import (
    SpinQuadrANN3D,
    make_topology_graph,
)


def fixture_data(device="cpu"):
    torch.manual_seed(830221)

    xyz = torch.rand(24, 3, dtype=torch.float32)

    # Two disconnected geometric components.
    xyz[:12] *= 0.3
    xyz[12:] = 0.7 + xyz[12:] * 0.3

    components = np.array(
        [0] * 12 + [1] * 12,
        dtype=np.int32,
    )

    mass = torch.exp(
        torch.linspace(-3, 3, 24, dtype=torch.float32)
    )

    return (
        xyz.to(device),
        mass.to(device),
        components,
    )


def make_model():
    return SpinQuadrANN3D(
        hidden_dim=16,
        final_hidden_dim=32,
        num_layers=3,
        neighbors=4,
        pe_abs=2,
        pe_rel=2,
    )


def test_topology_graph_has_no_cross_component_edges():
    xyz, _, components = fixture_data()

    graph = make_topology_graph(
        xyz.numpy(), components, neighbors=4
    )

    source, target = graph

    assert np.array_equal(
        components[source],
        components[target],
    )

    assert graph.shape[0] == 2


def test_forward_positive_weights_and_mass_conservation():
    xyz, mass, components = fixture_data()

    model = make_model().eval()

    with torch.no_grad():
        result = model(xyz, mass, components)

    weights = result["weights"]

    assert weights.shape == mass.shape
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()

    assert torch.allclose(
        weights.sum(),
        mass.sum(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_backward_has_finite_gradients():
    xyz, mass, components = fixture_data()

    model = make_model().train()

    result = model(xyz, mass, components)

    loss = (
        result["weights"].square().mean()
        + result["node_features"].square().mean()
    )

    loss.backward()

    grads = [
        p.grad for p in model.parameters()
        if p.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_permutation_equivariance():
    xyz, mass, components = fixture_data()

    model = make_model().eval()

    # Disable dropout through eval mode.
    with torch.no_grad():
        original = model(
            xyz, mass, components
        )["weights"]

        permutation = torch.randperm(len(xyz))

        permuted = model(
            xyz[permutation],
            mass[permutation],
            components[permutation.numpy()],
        )["weights"]

    restored = torch.empty_like(permuted)
    restored[permutation] = permuted

    assert torch.allclose(
        original, restored,
        rtol=2e-5,
        atol=2e-6,
    )


def test_cuda_forward_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    xyz, mass, components = fixture_data("cuda")

    model = make_model().cuda().train()

    result = model(xyz, mass, components)

    assert result["weights"].is_cuda
    assert torch.isfinite(result["weights"]).all()

    loss = result["weights"].square().mean()
    loss.backward()

    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
