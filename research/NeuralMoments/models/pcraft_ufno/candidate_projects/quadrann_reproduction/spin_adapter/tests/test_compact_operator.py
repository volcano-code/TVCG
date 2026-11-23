"""Factory-to-update integration on native RLE with two real tiny scale grids.

Grid/width reductions are test-only. This is an engineering contract check,
not a production-resolution benchmark or a physical-accuracy experiment.
"""
from dataclasses import replace
import math

import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import unified_factory as factory
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.jacobian_diagnostics import (
    compare_jacobians, finite_difference_jacobian,
)
from spin_adapter.tests.test_unified_m80_sqno import make_geometry, queries


def test_factory_compact_two_scale_operator_has_raw_derivatives_and_updates(monkeypatch):
    real_native_config = factory.make_pcraft_config
    # Every context grid must divide the fixture's native RLE side (512).
    # Powers of two preserve that physical aggregation contract; grid 12
    # would not be valid even though it meets the neural layer size rules.
    grid_map = {32: 8, 64: 16, 128: 32}

    def tiny_native_config(preset, **overrides):
        resolved = real_native_config(preset, **overrides)
        return replace(resolved,
            grid_size=grid_map[resolved.grid_size],
            multiscale_grids=tuple(grid_map[size] for size in resolved.multiscale_grids),
            pino_width=2, pino_modes=1, pino_layers=2,
            interval_width=8, interval_depth=1, chunk_intervals=16,
            min_native_resolution=1)

    monkeypatch.setattr(factory, "make_pcraft_config", tiny_native_config)
    overrides = dict(sqno_hidden_dim=4, sqno_final_hidden_dim=16, sqno_num_layers=2,
        sqno_neighbors=4, sqno_chunk_intervals=2, sqno_chunk_nodes=64, query_chunk=2,
        query_fusion_hidden=8, geometry_attention_chunk=2)
    torch.manual_seed(48123)
    reference = factory.build_model("ablation_g1_t1", config_overrides=overrides).eval()
    compact = factory.build_model("compact_v1_candidate", config_overrides=overrides).eval()
    assert reference.shared_encoder.config.gno_enabled
    assert reference.shared_encoder.config.transolver_enabled
    assert reference.cross_scale_attention is not None
    assert reference.attention_mixing == "gated"
    assert compact.shared_encoder.config.multiscale_grids == (16, 32)
    for name in ("gno_kernel", "gno_value", "gno_update", "transolver", "uq_head", "event_validity_head"):
        assert not hasattr(compact.shared_encoder, name)
    assert compact.sqno.pcraft is None
    assert compact.cross_scale_attention is None and compact.parallel_interaction_logit is None
    assert compact.attention_mixing == "sum"

    # Activate learned corrections before pairing initialization: zero-output
    # attention or a constant initial query gate would conceal derivative bugs.
    torch.manual_seed(1842)
    torch.nn.init.normal_(reference.geometry_physics_attention.output[-1].weight, std=.02)
    torch.nn.init.normal_(reference.geometry_physics_attention.output[-1].bias, std=.001)
    torch.nn.init.normal_(reference.cross_scale_attention.out[-1].weight, std=.02)
    torch.nn.init.normal_(reference.query_fusion_gate[-1].weight, std=.4)
    source = reference.state_dict()
    target = compact.state_dict()
    assert set(target) < set(source)
    assert all(source[name].shape == value.shape for name, value in target.items())
    compact.load_state_dict({name: source[name].clone() for name in target}, strict=True)
    assert all(torch.equal(value, source[name]) for name, value in compact.state_dict().items())

    geometry = make_geometry()
    raw = queries()[1:2] * .7
    with torch.no_grad():
        reference_cache = reference.encode_geometry(geometry, inference=True)
        reference_output = reference.evaluate_queries(reference_cache, raw)
    assert reference_cache.shared_cache.scale_ids.tolist() == [8, 16, 32]
    assert reference_output["cross_scale_valid_count"].tolist() == [3]
    assert torch.isfinite(reference_output["jacobian_coefficients"]).all()

    # This cache actually contains two independently encoded feature grids.
    # Keep it differentiable for independent AD of the returned Moments.
    cache = compact.encode_geometry(geometry)
    assert cache.shared_cache.scale_ids.tolist() == [16, 32]
    assert cache.shared_cache.scale_features.shape == (2, 8)
    assert cache.shared_cache.global_scale_weights.shape == (2, 1)
    assert cache.native_intervals == len(geometry["columns"])
    assert compact.sqno.quadrature_order == 2
    assert len(cache.sqno_cache.node_xyz) == len(geometry["columns"]) * 8
    torch.testing.assert_close(cache.sqno_cache.node_weights.sum(),
        torch.as_tensor(geometry["mass"].sum(), dtype=cache.sqno_cache.node_weights.dtype),
        rtol=1e-6, atol=1e-8)
    with torch.no_grad():
        output = compact.evaluate_queries(cache, raw)
        correction, _, _ = compact.geometry_physics_attention.forward_with_query_jacobian(
            raw, cache.shared_cache.interval_features, cache.shared_cache.interval_mass)
    assert output["moments"].shape == (1, 10)
    assert output["jacobian_coefficients"].shape == (1, 10, 10)
    assert torch.isfinite(output["jacobian_coefficients"]).all()
    assert correction.norm() > 0 and output["fusion_gate_jacobian_norm"].min() > 0
    independent_ad = torch.autograd.functional.jacobian(
        lambda c: compact.evaluate_queries(cache, c, create_graph=True)["moments"], raw)
    torch.testing.assert_close(output["jacobian_coefficients"], independent_ad[:, :, 0, :],
                               rtol=2e-6, atol=2e-8)
    for step in (2e-4, 1e-4):
        finite_difference = finite_difference_jacobian(
            lambda c: compact.evaluate_queries(cache, c)["moments"], raw, step=step)
        comparison = compare_jacobians(output["jacobian_coefficients"], finite_difference)
        assert comparison["relative_frobenius"] < 3e-4
        assert comparison["max_absolute"] < 2e-3

    # Synthetic zero targets exercise the real joint training path. They are
    # not native teacher labels and establish no scientific training benefit.
    compact.train()
    training_cache = compact.encode_geometry(geometry)
    before = {name: parameter.detach().clone() for name, parameter in compact.named_parameters()}
    optimizer = compact.build_optimizer(learning_rate=1e-4)
    metrics = compact.train_step(optimizer, training_cache, raw,
        torch.zeros_like(output["moments"]), torch.zeros_like(output["jacobian_coefficients"]),
        jacobian_weight=.1, branch_aux_weight=.25)
    assert math.isfinite(metrics["loss"]) and metrics["loss"] > 0
    assert math.isfinite(metrics["gradient_norm"]) and metrics["gradient_norm"] > 0
    parameters = dict(compact.named_parameters())
    for prefix in ("shared_encoder.", "sqno.", "geometry_physics_attention.", "query_fusion_gate."):
        names = [name for name in parameters if name.startswith(prefix)]
        assert any(not torch.equal(before[name], parameters[name]) for name in names), prefix
        gradients = [parameters[name].grad for name in names if parameters[name].grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    with pytest.raises(ValueError, match="stale"):
        compact.evaluate_queries(training_cache, raw)
    with pytest.raises(ValueError, match="stale"):
        compact.evaluate_queries(cache, raw)
    compact.eval()
    with torch.no_grad():
        refreshed = compact.encode_geometry(geometry, inference=True)
        updated = compact.evaluate_queries(refreshed, raw)
    assert refreshed.shared_cache.scale_ids.tolist() == [16, 32]
    assert torch.isfinite(updated["moments"]).all()
    assert torch.isfinite(updated["jacobian_coefficients"]).all()
