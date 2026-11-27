"""V2.3 uses the released encoder and E16-compatible response, not a tiny substitute."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spin_adapter.compact_pino_v21 import CompactPINOv21
from spin_adapter.compact_pino_v23 import CompactPINOv23, RuntimeConfig
from spin_adapter.tests.test_v22_compact_integration import _case
from spin_adapter.v21_response import (LocalRetention, prepare_measure, prepare_stencil,
                                      prepare_surface, evaluate_stencil)


def test_same_checkpoint_new_queries_sides_and_requested_moment_only():
    torch.manual_seed(830251)
    old = CompactPINOv21(response_type="monotone_warp", training_scope="response",
                        query_cache_bytes=0).eval()
    new = CompactPINOv23(runtime=RuntimeConfig(surface_workers=1)).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    blocks, operand, fixed = _case()
    try:
        with torch.no_grad():
            old_features = old.encode_geometry(blocks, fixed, inference=True)
            features = new.encode_geometry(blocks, fixed, inference=True)
            for shift in (0., .0002):
                vertices = operand.vertices.copy()
                vertices[vertices[:, 0] == .5, 0] += shift
                for operation in ("intersection", "difference"):
                    q = replace(operand, vertices=vertices, operation=operation)
                    reference = old.evaluate_operand(old_features, q, chunk_size=2)
                    observed = new.evaluate_operand(features, q, chunk_size=2)
                    torch.testing.assert_close(observed.moments, reference.moments,
                                               rtol=3e-13, atol=2e-22)
                    torch.testing.assert_close(observed.jacobian, reference.jacobian,
                                               rtol=3e-13, atol=2e-20)
                    value_only = new.evaluate_operand(features, replace(q, vertex_velocity=None),
                                                      with_jacobian=False, chunk_size=2)
                    torch.testing.assert_close(value_only.moments, observed.moments,
                                               rtol=3e-13, atol=2e-22)
                    assert value_only.jacobian is None
            stats = new.efficiency_statistics()
            assert stats["encoder_calls"] == 1
            assert stats["inference_projection"]["misses"] == 1
            assert stats["inference_projection"]["hits"] == 7
            assert stats["prepared_queries"]["builds"] == 0
    finally:
        new.close()
    assert new.efficiency_statistics()["runtime_closed"]
    with pytest.raises(RuntimeError, match="closed"):
        new.evaluate_operand(features, operand)


def test_inference_projection_rebuilds_after_weight_change_and_training_keeps_j_gradient():
    torch.manual_seed(830252)
    with CompactPINOv23(runtime=RuntimeConfig(surface_workers=1)) as model:
        model.eval()
        blocks, operand, fixed = _case()
        with torch.inference_mode():
            cache = model.encode_geometry(blocks, fixed, inference=True)
            a = model.evaluate_operand(cache, operand)
        with torch.no_grad():
            model.response.local[0].weight.add_(.025)
            b = model.evaluate_operand(cache, operand)
        assert model.efficiency_statistics()["inference_projection"]["misses"] == 2
        assert not torch.equal(a.moments, b.moments)
        model.train()
        assert model.efficiency_statistics()["inference_projection"]["entries"] == 0
        optimizer = torch.optim.Adam(model.response.parameters(), lr=3e-5)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            features = model.encode_geometry(blocks, fixed)
            result = model.evaluate_operand(features, operand)
            loss = (result.jacobian * blocks.pitch / blocks.moments[:, 0].sum()).square().mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.response.parameters())
            assert all(p.grad is None for p in model.shared_encoder.parameters())
            assert model.response.local[0].weight.grad.norm() > 0
            optimizer.step()
        assert model.encoder_calls == 1


@pytest.mark.parametrize("profile", ["bounded_cubic", "monotone_warp"])
@pytest.mark.parametrize("analytic", [True, False])
def test_fused_reduction_preserves_jacobian_training_gradients(profile, analytic):
    torch.manual_seed(23)
    blocks, operand, _ = _case()
    measure = prepare_measure(blocks.nodes[:, :3], blocks.nodes[:, 3])
    stencil = prepare_stencil(measure, prepare_surface(operand.vertices, operand.faces),
                              bandwidth=blocks.pitch)
    model = LocalRetention(context_dim=2, hidden_dim=8, response_type=profile).double()
    model.use_analytic_slope = analytic
    context = torch.randn(len(measure.nodes), 2, dtype=torch.float64, requires_grad=True)
    parameters = list(model.parameters()) + [context]
    outputs, gradients = [], []
    for fused in (False, True):
        result = evaluate_stencil(stencil, model=model, context=context,
                                  vertex_velocity=operand.vertex_velocity,
                                  fused_contraction=fused, chunk_size=2)
        loss = (result.moments / measure.total_moments[0]).square().mean()
        loss = loss + (result.jacobian * blocks.pitch / measure.total_moments[0]).square().mean()
        gradients.append(torch.autograd.grad(loss, parameters))
        outputs.append(result)
    torch.testing.assert_close(outputs[0].moments, outputs[1].moments, rtol=1e-13, atol=1e-22)
    torch.testing.assert_close(outputs[0].jacobian, outputs[1].jacobian, rtol=1e-13, atol=1e-20)
    for before, after in zip(*gradients):
        torch.testing.assert_close(before, after, rtol=2e-12, atol=1e-15)
    assert any(x.abs().max() > 0 for x in gradients[1])


def test_projection_budget_and_table_mutation_do_not_reuse_old_values():
    with CompactPINOv23(runtime=RuntimeConfig(surface_workers=1, projection_cache_bytes=1)) as model:
        model.eval()
        dim = model.response.context_dim
        table = torch.randn(4, dim)
        cache = SimpleNamespace(inference_only=True, table=table, frozen_encoder=True,
                                projected_context=None)
        with torch.no_grad():
            first = model._project_query_context(cache)
            table.add_(1.)
            second = model._project_query_context(cache)
        assert not torch.equal(first.values, second.values)
        assert model.efficiency_statistics()["inference_projection"]["entries"] == 0
        assert model.efficiency_statistics()["inference_projection"]["bypasses"] == 2


@pytest.mark.parametrize("kwargs", [
    {"surface_workers": 0}, {"surface_workers": True},
    {"projection_cache_bytes": -1}, {"fused_contraction": 1},
    {"cache_inference_projection": "true"},
])
def test_runtime_options_are_explicit(kwargs):
    with pytest.raises(ValueError):
        RuntimeConfig(**kwargs)


def test_fused_operator_joint_training_updates_encoder_attention_and_response():
    from spin_adapter.tests.test_v22_compact_integration import _joint_backward
    torch.manual_seed(830253)
    old = CompactPINOv21(response_type="monotone_warp", training_scope="all", query_cache_bytes=0)
    with CompactPINOv23(training_scope="all", runtime=RuntimeConfig(surface_workers=1)) as model:
        model.load_state_dict(old.state_dict(), strict=True)
        case = _case()
        expected_m, expected_j, _, expected_gradients, _ = _joint_backward(old, case)
        actual_m, actual_j, _, actual_gradients, encoded = _joint_backward(model, case)
        torch.testing.assert_close(actual_m, expected_m, rtol=2e-6, atol=2e-16)
        torch.testing.assert_close(actual_j, expected_j, rtol=2e-6, atol=2e-14)
        nonzero = dict.fromkeys(("shared_encoder.pino_encoder.", "geometry_attention.", "response."), False)
        for name, expected in expected_gradients.items():
            actual = actual_gradients[name]
            assert (actual is None) == (expected is None), name
            if actual is not None:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-8, msg=name)
                for prefix in nonzero:
                    if name.startswith(prefix) and torch.linalg.vector_norm(actual) > 0:
                        nonzero[prefix] = True
        assert all(nonzero.values())
        assert model.efficiency_statistics()["inference_projection"]["entries"] == 0
        torch.optim.SGD(model.parameters(), lr=1e-4).step()
        with pytest.raises(ValueError, match="stale learned geometry"):
            model.evaluate_operand(encoded, case[1])
