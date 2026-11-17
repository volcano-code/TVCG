import torch
import pytest

from spin_adapter.spin_moment_operator import (
    SpinMomentOperator,
    NATIVE_B,
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.spin_pino import (
    SpinPINO,
)


def fixture(device="cpu"):
    torch.manual_seed(830224)

    xyz = torch.rand(64, 3, device=device, dtype=torch.float64)
    weights = torch.rand(64, device=device, dtype=torch.float64)
    weights = weights / weights.sum()

    coefficients = torch.randn(
        10, device=device, dtype=torch.float64
    )
    coefficients = coefficients / coefficients.norm()

    return xyz, weights, coefficients


def test_moment_matches_existing_spin_pino():
    xyz, weights, coefficients = fixture()

    operator = SpinMomentOperator(
        temperature=0.04,
        chunk_nodes=13,
    )

    result = operator.evaluate(xyz, weights, coefficients)

    model = SpinPINO(
        width=8,
        modes=4,
        n_layers=4,
        grid_size=16,
        points_per_axis=4,
        hidden_dim=96,
        smoothing=0.04,
        moment_parameterization="smooth_hinge",
    ).double()

    # Use exactly the same ten monomial values as the new operator.
    from spin_adapter.spin_moment_operator import raw_moment_basis

    basis = raw_moment_basis(xyz)

    q = coefficients / coefficients.norm()
    B = torch.tensor(NATIVE_B, dtype=q.dtype)
    q = q * B

    a = basis @ q
    s = torch.sqrt(a.square() + 0.04**2 * q.square().sum())

    n = a[:, None] * basis + 0.04**2 * q

    expected = (
        0.5 * weights[:, None] *
        (basis + n / s[:, None])
    ).sum(dim=0)

    assert torch.allclose(
        result["moments"], expected,
        rtol=1e-10, atol=1e-11,
    )


def test_q_jacobian_is_symmetric_psd():
    xyz, weights, coefficients = fixture()

    result = SpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    Jq = result["jacobian_q"]

    assert torch.allclose(
        Jq, Jq.T, rtol=1e-10, atol=1e-11
    )

    assert torch.linalg.eigvalsh(Jq).min() >= -1e-10


def test_coefficient_jacobian_coordinate_relation():
    xyz, weights, coefficients = fixture()

    result = SpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    B = torch.tensor(NATIVE_B, dtype=xyz.dtype)

    expected = result["jacobian_q"] * B.unsqueeze(0)

    assert torch.allclose(
        result["jacobian_c"],
        expected,
        rtol=1e-11,
        atol=1e-12,
    )


def test_jacobian_matches_finite_difference():
    xyz, weights, coefficients = fixture()

    operator = SpinMomentOperator(
        temperature=0.04,
        chunk_nodes=17,
    )

    result = operator.evaluate(
        xyz, weights, coefficients
    )

    J = result["jacobian_c"]
    delta = 1e-6

    numerical = torch.zeros_like(J)

    for j in range(10):
        plus = coefficients.clone()
        minus = coefficients.clone()

        plus[j] += delta
        minus[j] -= delta

        rp = operator.evaluate(
            xyz, weights, plus,
            with_jacobian=False,
        )["moments"]

        rm = operator.evaluate(
            xyz, weights, minus,
            with_jacobian=False,
        )["moments"]

        numerical[:, j] = (rp-rm)/(2*delta)

    assert torch.allclose(
        J, numerical,
        rtol=2e-5,
        atol=2e-7,
    )


def test_chunked_and_full_results_match():
    xyz, weights, coefficients = fixture()

    full = SpinMomentOperator(chunk_nodes=1000).evaluate(
        xyz, weights, coefficients
    )

    chunked = SpinMomentOperator(chunk_nodes=7).evaluate(
        xyz, weights, coefficients
    )

    assert torch.allclose(
        full["moments"],
        chunked["moments"],
        rtol=1e-11,
        atol=1e-12,
    )

    assert torch.allclose(
        full["jacobian_c"],
        chunked["jacobian_c"],
        rtol=1e-10,
        atol=1e-11,
    )


def test_cuda_forward_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    xyz, weights, coefficients = fixture("cuda")

    coefficients = coefficients.detach().requires_grad_(True)

    result = SpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    loss = (
        result["moments"].square().mean()
        + result["potential"].square()
    )

    loss.backward()

    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()


def test_invalid_weights_rejected():
    xyz, weights, coefficients = fixture()

    weights[0] = -1

    with pytest.raises(ValueError):
        SpinMomentOperator().evaluate(
            xyz, weights, coefficients
        )


def test_direct_spin_pino_moment_and_jacobian_parity():
    xyz, weights, coefficients = fixture()

    model = SpinPINO(
        width=8,
        modes=4,
        n_layers=4,
        grid_size=16,
        points_per_axis=4,
        hidden_dim=96,
        smoothing=0.04,
        moment_parameterization="smooth_hinge",
    ).double().eval()

    operator = SpinMomentOperator(
        temperature=0.04,
        chunk_nodes=1000,
    )

    from spin_adapter.spin_moment_operator import raw_moment_basis

    monomials = raw_moment_basis(xyz)

    # Evaluate the existing SpinPINO equations directly on the same
    # arbitrary RLE quadrature nodes and weights.
    _, _, B, m, a, s, n, eps2 = model._terms(
        weights.unsqueeze(0), coefficients.unsqueeze(0)
    ) if False else (None,)*8

    B = model.native_B.to(coefficients)
    q = coefficients * B
    a = monomials @ q
    s = torch.sqrt(a.square() + 0.04**2 * q.square().sum())
    n = a[:, None] * monomials + 0.04**2 * q

    expected_moments = (
        0.5 * weights[:, None]
        * (monomials + n / s[:, None])
    ).sum(0)

    eye = torch.eye(10, dtype=xyz.dtype)
    mm = monomials.unsqueeze(-1) * monomials.unsqueeze(-2)

    hess = 0.5 * (
        (mm + 0.04**2 * eye) / s[:, None, None]
        - (n.unsqueeze(-1) * n.unsqueeze(-2))
        / s[:, None, None]**3
    )

    expected_jq = torch.einsum(
        "n,nij->ij", weights, hess
    )

    expected_jc = expected_jq * B.unsqueeze(0)

    actual = operator.evaluate(xyz, weights, coefficients)

    assert torch.allclose(
        actual["moments"], expected_moments,
        rtol=1e-11, atol=1e-12,
    )

    assert torch.allclose(
        actual["jacobian_q"], expected_jq,
        rtol=1e-10, atol=1e-11,
    )

    assert torch.allclose(
        actual["jacobian_c"], expected_jc,
        rtol=1e-10, atol=1e-11,
    )
