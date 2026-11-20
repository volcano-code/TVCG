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



def test_jacobian_checkpoint_node_chunk_equivalence():
    xyz, weights, coefficients = fixture()
    reference = BatchedSpinMomentOperator(
        chunk_nodes=128, chunk_queries=2
    ).evaluate(xyz, weights, coefficients)
    chunked = BatchedSpinMomentOperator(
        chunk_nodes=17, chunk_queries=1
    ).evaluate(xyz, weights, coefficients)
    for key in ("potential", "moments", "jacobian_q", "jacobian_c"):
        assert torch.allclose(
            reference[key], chunked[key], rtol=1e-10, atol=1e-11
        )


def test_jacobian_checkpoint_backward_matches_finite_gradients():
    xyz, weights, coefficients = fixture()
    weights.requires_grad_(True)
    coefficients.requires_grad_(True)
    result = BatchedSpinMomentOperator(
        chunk_nodes=19, chunk_queries=1
    ).evaluate(xyz, weights, coefficients, with_jacobian=True)
    loss = result["moments"].square().mean() + result["jacobian_q"].square().mean()
    loss.backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert coefficients.grad is not None and torch.isfinite(coefficients.grad).all()
    assert weights.grad.abs().sum() > 0
    assert coefficients.grad.abs().sum() > 0

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


@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float64, 2e-9, 3e-10),
    (torch.float32, 4e-4, 8e-5),
])
def test_contracted_hessian_and_full_derivatives(dtype, rtol, atol):
    """Independent unchunked outer-product reference, including second derivatives."""
    torch.manual_seed(211)
    xyz = torch.randn(19, 3, dtype=dtype, requires_grad=True)
    weights = torch.rand(19, dtype=dtype, requires_grad=True)
    queries = torch.randn(3, 10, dtype=dtype, requires_grad=True)
    inputs = (xyz, weights, queries)
    directions = [torch.randn_like(v) for v in inputs]
    probe = torch.randn(3, 10, 10, dtype=dtype)

    def evaluate(reference):
        if reference:
            rows = [SpinMomentOperator().evaluate(xyz, weights, q) for q in queries]
            moments = torch.stack([r["moments"] for r in rows])
            jacobian = torch.stack([r["jacobian_q"] for r in rows])
        else:
            out = BatchedSpinMomentOperator(chunk_nodes=6, chunk_queries=2).evaluate(*inputs)
            moments, jacobian = out["moments"], out["jacobian_q"]
        objective = (moments.square().mean() + (probe * jacobian).mean())
        first = torch.autograd.grad(objective, inputs, create_graph=True)
        second = torch.autograd.grad(sum((g * v).sum() for g, v in zip(first, directions)), inputs)
        return (moments, jacobian, *first, *second)

    for actual, expected in zip(evaluate(False), evaluate(True)):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
