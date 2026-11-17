import torch

from spin_adapter.batched_moment_operator import (
    BatchedSpinMomentOperator,
)


def run_operator(chunk_nodes):
    torch.manual_seed(830315)

    xyz = torch.randn(
        37, 3, dtype=torch.float64
    ).requires_grad_(True)

    weights = (
        torch.rand(37, dtype=torch.float64) + 0.1
    ).requires_grad_(True)

    coefficients = torch.randn(
        5, 10, dtype=torch.float64
    ).requires_grad_(True)

    operator = BatchedSpinMomentOperator(
        temperature=0.04,
        chunk_nodes=chunk_nodes,
        chunk_queries=2,
    )

    result = operator.evaluate(
        xyz,
        weights,
        coefficients,
        with_jacobian=False,
    )

    loss = result["moments"].square().mean()

    loss.backward()

    return (
        result["moments"].detach(),
        xyz.grad.detach(),
        weights.grad.detach(),
        coefficients.grad.detach(),
    )


def test_checkpointed_node_chunks_match_full_forward_and_gradients():
    full = run_operator(128)
    chunked = run_operator(7)

    for a, b in zip(full, chunked):
        torch.testing.assert_close(
            a, b,
            rtol=1e-9,
            atol=1e-10,
        )


def test_checkpointed_node_chunk_has_finite_gradients():
    result = run_operator(5)

    for value in result:
        assert torch.isfinite(value).all()
