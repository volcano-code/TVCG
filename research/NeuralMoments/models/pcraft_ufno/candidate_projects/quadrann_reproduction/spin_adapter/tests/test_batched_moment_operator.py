import torch
import pytest

from spin_adapter.spin_moment_operator import (
    SpinMomentOperator,
)
from spin_adapter.batched_moment_operator import (
    BatchedSpinMomentOperator,
)


def fixture(device="cpu"):
    torch.manual_seed(830241)

    dtype = torch.float64

    xyz = torch.randn(
        257, 3, device=device, dtype=dtype
    )

    weights = torch.rand(
        257, device=device, dtype=dtype
    ) + 0.1

    coefficients = torch.randn(
        7, 10, device=device, dtype=dtype
    )

    return xyz, weights, coefficients


def test_batch_shapes():
    xyz, weights, coefficients = fixture()

    operator = BatchedSpinMomentOperator(
        chunk_nodes=64,
        chunk_queries=3,
    )

    result = operator.evaluate(
        xyz, weights, coefficients
    )

    assert result["moments"].shape == (7, 10)
    assert result["jacobian_q"].shape == (7, 10, 10)
    assert result["jacobian_c"].shape == (7, 10, 10)
    assert result["potential"].shape == (7,)


def test_batch_matches_original_single_query():
    xyz, weights, coefficients = fixture()

    single = SpinMomentOperator(
        chunk_nodes=64
    )

    batch = BatchedSpinMomentOperator(
        chunk_nodes=64,
        chunk_queries=3,
    )

    result = batch.evaluate(
        xyz, weights, coefficients
    )

    for i in range(len(coefficients)):
        reference = single.evaluate(
            xyz,
            weights,
            coefficients[i],
        )

        assert torch.allclose(
            result["moments"][i],
            reference["moments"],
            rtol=1e-10,
            atol=1e-11,
        )

        assert torch.allclose(
            result["jacobian_q"][i],
            reference["jacobian_q"],
            rtol=1e-10,
            atol=1e-11,
        )

        assert torch.allclose(
            result["jacobian_c"][i],
            reference["jacobian_c"],
            rtol=1e-10,
            atol=1e-11,
        )


def test_native_jacobian_coordinate_transform():
    xyz, weights, coefficients = fixture()

    result = BatchedSpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    B = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
        dtype=torch.float64,
    )

    expected = (
        result["jacobian_q"]
        * B[None, None, :]
    )

    assert torch.allclose(
        result["jacobian_c"],
        expected,
    )


def test_q_jacobian_symmetric_psd():
    xyz, weights, coefficients = fixture()

    result = BatchedSpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    J = result["jacobian_q"]

    assert torch.allclose(
        J, J.transpose(-1, -2),
        atol=1e-10,
        rtol=1e-10,
    )

    eigenvalues = torch.linalg.eigvalsh(J)

    assert eigenvalues.min() >= -1e-9


def test_batched_moment_gradients():
    xyz, weights, coefficients = fixture()

    weights.requires_grad_(True)
    coefficients.requires_grad_(True)

    result = BatchedSpinMomentOperator(
        chunk_nodes=64,
        chunk_queries=3,
    ).evaluate(xyz, weights, coefficients)

    loss = (
        result["moments"].square().mean()
        + result["jacobian_q"].square().mean()
    )

    loss.backward()

    assert weights.grad is not None
    assert coefficients.grad is not None

    assert torch.isfinite(weights.grad).all()
    assert torch.isfinite(coefficients.grad).all()

    assert weights.grad.abs().sum() > 0
    assert coefficients.grad.abs().sum() > 0


def test_single_query_input_compatibility():
    xyz, weights, coefficients = fixture()

    result = BatchedSpinMomentOperator().evaluate(
        xyz, weights, coefficients[0]
    )

    assert result["moments"].shape == (1, 10)
    assert result["jacobian_q"].shape == (1, 10, 10)


def test_no_jacobian_mode():
    xyz, weights, coefficients = fixture()

    result = BatchedSpinMomentOperator().evaluate(
        xyz,
        weights,
        coefficients,
        with_jacobian=False,
    )

    assert result["moments"].shape == (7, 10)
    assert result["jacobian_q"].shape == (7, 10, 10)
    assert torch.count_nonzero(result["jacobian_q"]) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_batch_matches_cpu():
    xyz, weights, coefficients = fixture()

    cpu = BatchedSpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    cuda = BatchedSpinMomentOperator().evaluate(
        xyz.cuda(),
        weights.cuda(),
        coefficients.cuda(),
    )

    assert torch.allclose(
        cpu["moments"],
        cuda["moments"].cpu(),
        rtol=1e-9,
        atol=1e-10,
    )
