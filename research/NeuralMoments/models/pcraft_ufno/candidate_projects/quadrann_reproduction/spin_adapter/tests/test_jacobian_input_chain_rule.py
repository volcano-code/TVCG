import torch

from spin_adapter.batched_moment_operator import (
    BatchedSpinMomentOperator,
)


def fixture():
    torch.manual_seed(830251)

    xyz = torch.randn(91, 3, dtype=torch.float64)
    weights = torch.rand(91, dtype=torch.float64) + .1
    coefficients = torch.randn(
        4, 10, dtype=torch.float64
    )

    return xyz, weights, coefficients


def test_raw_input_jacobian_matches_autograd():
    xyz, weights, coefficients = fixture()

    operator = BatchedSpinMomentOperator(
        chunk_nodes=32,
        chunk_queries=2,
    )

    result = operator.evaluate(
        xyz, weights, coefficients
    )

    for i in range(len(coefficients)):
        c = coefficients[i].clone().requires_grad_(True)

        def evaluate_moment(value):
            return operator.evaluate(
                xyz, weights, value,
                with_jacobian=False,
            )["moments"][0]

        jacobian = torch.autograd.functional.jacobian(
            evaluate_moment, c
        )

        assert torch.allclose(
            result["jacobian_input"][i],
            jacobian,
            atol=1e-9,
            rtol=1e-8,
        )


def test_radial_direction_is_null():
    xyz, weights, coefficients = fixture()

    result = BatchedSpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    radial = torch.einsum(
        "bij,bj->bi",
        result["jacobian_input"],
        coefficients,
    )

    assert radial.abs().max() < 1e-9


def test_normalized_jacobian_compatibility():
    xyz, weights, coefficients = fixture()

    result = BatchedSpinMomentOperator().evaluate(
        xyz, weights, coefficients
    )

    B = torch.tensor(
        [1,1,1,1,1,1,1,2,2,2],
        dtype=torch.float64,
    )

    expected = result["jacobian_q"] * B[None,None,:]

    assert torch.allclose(
        result["jacobian_c"],
        expected,
        atol=1e-10,
        rtol=1e-10,
    )

    assert torch.allclose(
        result["jacobian_c"],
        result["jacobian_normalized"],
    )
