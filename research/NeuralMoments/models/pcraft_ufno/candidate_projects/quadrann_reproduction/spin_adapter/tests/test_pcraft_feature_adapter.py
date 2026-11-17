import numpy as np
import torch

from spin_adapter.pcraft_feature_adapter import PcraftFeatureAdapter
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


def make_adapter():
    return PcraftFeatureAdapter(
        preset="large_ms",
        min_native_resolution=512,
        chunk_intervals=2,
    )


def test_multiscale_interval_feature_shapes():
    geometry = make_geometry()
    adapter = make_adapter().eval()

    with torch.no_grad():
        result = adapter(geometry)

    assert result["local_features"].shape == (5, 96)
    assert result["global_context"].shape == (96,)
    assert result["descriptor"].shape == (5, 13)

    assert result["global_scale_weights"].shape == (3, 1)

    weights = result["global_scale_weights"].flatten()

    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_adapter_is_deterministic_in_eval_mode():
    geometry = make_geometry()
    adapter = make_adapter().eval()

    with torch.no_grad():
        first = adapter(geometry)
        second = adapter(geometry)

    assert torch.allclose(
        first["local_features"],
        second["local_features"],
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        first["global_context"],
        second["global_context"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_pcraft_encoder_receives_gradients():
    geometry = make_geometry()
    adapter = make_adapter().train()

    result = adapter(geometry)

    loss = (
        result["local_features"].square().mean()
        + result["global_context"].square().mean()
    )

    loss.backward()

    grads = [
        p.grad for p in adapter.operator.pino_encoder.parameters()
        if p.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(g).all() for g in grads)

    norm = torch.sqrt(
        sum(g.abs().square().sum() for g in grads)
    )
    assert torch.isfinite(norm)
    assert norm.item() > 0


def test_original_query_head_is_never_invoked():
    geometry = make_geometry()
    adapter = make_adapter().eval()

    calls = []

    handle = adapter.operator.query_head.register_forward_hook(
        lambda module, inputs, output: calls.append(True)
    )

    with torch.no_grad():
        result = adapter(geometry)

    handle.remove()

    assert result["local_features"].shape[0] == 5
    assert calls == []



def test_three_scales_are_used():
    geometry = make_geometry()
    adapter = make_adapter().eval()

    observed = []
    encoder = adapter.operator.pino_encoder
    original = encoder.forward_with_map

    def tracked(grid, *, preserve_resolution=False):
        observed.append(tuple(grid.shape[2:]))
        return original(
            grid,
            preserve_resolution=preserve_resolution,
        )

    encoder.forward_with_map = tracked

    try:
        with torch.no_grad():
            adapter(geometry)
    finally:
        encoder.forward_with_map = original

    assert observed == [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ]
