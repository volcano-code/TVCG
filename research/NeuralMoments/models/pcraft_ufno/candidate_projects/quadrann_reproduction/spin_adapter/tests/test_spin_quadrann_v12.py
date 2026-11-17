import numpy as np
import pytest
import torch

from spin_adapter.spin_quadrann_v12 import SpinQuadrANNV12
from spin_adapter.spin_quadrann_v11 import SpinQuadrANNV11
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


def make_model():
    return SpinQuadrANNV12(
        preset="large_ms",
        hidden_dim=16,
        final_hidden_dim=64,
        num_layers=3,
        neighbors=4,
        temperature=.04,
        quadrature_order=2,
        chunk_intervals=2,
        chunk_nodes=64,
        query_chunk=2,
    )


def make_queries(dtype=torch.float32):
    torch.manual_seed(830242)
    q = torch.randn(8, 10, dtype=dtype)
    return q / q.norm(dim=-1, keepdim=True)


def test_geometry_encoding_occurs_once_for_multiple_query_batches():
    model = make_model().eval()
    geometry = make_geometry()

    cache = model.encode_geometry(geometry)

    assert model.geometry_encode_calls == 1

    first = model.evaluate_queries(
        cache, make_queries()[:4]
    )

    second = model.evaluate_queries(
        cache, make_queries()[4:]
    )

    assert first["moments"].shape == (4, 10)
    assert second["moments"].shape == (4, 10)

    assert model.geometry_encode_calls == 1
    assert model.query_evaluation_calls == 2


def test_geometry_cache_preserves_mass():
    model = make_model().eval()
    cache = model.encode_geometry(make_geometry())

    assert cache.node_weights.shape == (cache.n_nodes,)

    assert torch.all(cache.node_weights > 0)

    assert torch.allclose(
        cache.node_weights.sum(),
        cache.interval_mass.sum(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_batched_queries_match_separate_query_evaluation():
    model = make_model().eval()
    cache = model.encode_geometry(make_geometry())

    queries = make_queries()

    batch = model.evaluate_queries(
        cache, queries, with_jacobian=True
    )

    singles = [
        model.evaluate_queries(
            cache, queries[i], with_jacobian=True
        )
        for i in range(len(queries))
    ]

    expected_moments = torch.cat(
        [x["moments"] for x in singles], dim=0
    )

    expected_jacobians = torch.cat(
        [x["jacobian_q"] for x in singles], dim=0
    )

    assert torch.allclose(
        batch["moments"],
        expected_moments,
        atol=1e-6,
        rtol=1e-5,
    )

    assert torch.allclose(
        batch["jacobian_q"],
        expected_jacobians,
        atol=1e-6,
        rtol=1e-5,
    )

    assert model.geometry_encode_calls == 1


def test_geometry_cache_retains_pcraft_and_gnn_gradients():
    model = make_model().train()
    cache = model.encode_geometry(make_geometry())

    result = model.evaluate_queries(
        cache, make_queries(), with_jacobian=True
    )

    loss = (
        result["moments"].square().mean()
        + 0.1 * result["jacobian_q"].square().mean()
    )

    loss.backward()

    for name, module in (
        ("P-CRAFT", model.pcraft.operator.pino_encoder),
        ("QuadrANN", model.gnn.layers),
        ("Fusion", model.fusion),
        ("Weight Head", model.weight_head),
    ):
        grads = [
            p.grad for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]

        assert grads, f"{name}: no gradients"

        norm = torch.sqrt(
            sum(g.abs().square().sum() for g in grads)
        )

        assert torch.isfinite(norm)
        assert norm.item() > 0, f"{name}: zero gradient"


def test_teacher_float64_queries_are_supported():
    model = make_model().eval()
    cache = model.encode_geometry(make_geometry())

    result = model.evaluate_queries(
        cache,
        make_queries(dtype=torch.float64),
    )

    assert result["moments"].dtype == torch.float64
    assert result["jacobian_q"].dtype == torch.float64
    assert torch.isfinite(result["moments"]).all()


def test_eval_cache_is_deterministic():
    model = make_model().eval()
    geometry = make_geometry()
    queries = make_queries()

    with torch.no_grad():
        cache1 = model.encode_geometry(geometry)
        result1 = model.evaluate_queries(cache1, queries)

        cache2 = model.encode_geometry(geometry)
        result2 = model.evaluate_queries(cache2, queries)

    assert torch.allclose(
        result1["moments"],
        result2["moments"],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_geometry_cache_and_batch():
    model = make_model().cuda().eval()
    cache = model.encode_geometry(make_geometry())

    result = model.evaluate_queries(
        cache,
        make_queries().cuda(),
    )

    assert result["moments"].is_cuda
    assert result["jacobian_q"].is_cuda
    assert result["moments"].shape == (8, 10)




def test_fixed_moments_are_restored_exactly_once():
    model = make_model().eval()
    geometry = make_geometry()
    queries = make_queries()[:2]

    with torch.no_grad():
        cache = model.encode_geometry(geometry)

        result = model.evaluate_queries(
            cache,
            queries,
            with_jacobian=True,
        )

        fixed = cache.fixed_moments.to(
            device=result["moments"].device,
            dtype=result["moments"].dtype,
        )

        # Fixed moments must be added exactly once.
        assert torch.allclose(
            result["moments_total"],
            result["moments_editable"] + fixed.unsqueeze(0),
            atol=1e-7,
            rtol=1e-7,
        )

        assert torch.allclose(
            result["moments"],
            result["moments_total"],
            atol=1e-7,
            rtol=1e-7,
        )

        # Independent recomputation from the same cache:
        # fixed moments must not alter the query Jacobian.
        operator_result = model.batched_moment_operator.evaluate(
            cache.node_xyz.to(queries),
            cache.node_weights.to(queries),
            queries,
            with_jacobian=True,
        )

        assert torch.allclose(
            result["jacobian_q"],
            operator_result["jacobian_q"],
            atol=1e-7,
            rtol=1e-7,
        )

        assert torch.allclose(
            result["jacobian_c"],
            operator_result["jacobian_c"],
            atol=1e-7,
            rtol=1e-7,
        )

        assert torch.isfinite(result["moments_total"]).all()
        assert torch.isfinite(result["jacobian_q"]).all()
