"""Production geometry freezing, cross-update caching and graph ownership.

These checks use the actual shared 32^3/64^3 U-FNO and Token16 operator. The
small positive material is a software fixture, not evidence of mesh accuracy.
"""
from dataclasses import replace

import pytest
import torch

from spin_adapter.compact_pino_v21 import CompactPINOv21
from spin_adapter.tests.test_v22_compact_integration import _case
from spin_adapter.v22_frozen_features import FrozenFeatureCache


def _model(**kwargs):
    torch.manual_seed(830251)
    return CompactPINOv21(training_scope='response', response_type='monotone_warp', **kwargs)


def _frozen_state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            if not name.startswith('response.')}


def _loss_and_gradient(model, case):
    blocks, operand, fixed = case
    model.train()
    model.zero_grad(set_to_none=True)
    features = model.encode_geometry(blocks, fixed)
    prediction = model.evaluate_operand(features, operand, chunk_size=2)
    mass = blocks.moments[:, 0].sum()
    # J-only backpropagation is deliberate: a detached derivative must fail.
    loss = (prediction.jacobian * blocks.pitch / mass).square().mean()
    loss.backward()
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters() if parameter.requires_grad}
    return prediction, features, gradients


def test_response_updates_reuse_real_encoder_and_match_uncached_values_and_gradients():
    cached, uncached = _model(), _model(frozen_feature_cache_bytes=0)
    uncached.load_state_dict(cached.state_dict())
    case = _case()
    before = _frozen_state(cached)
    optimizers = [torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
                  for model in (cached, uncached)]
    first_table, first_m = None, None
    for update in range(2):
        observed, features, gradients = _loss_and_gradient(cached, case)
        expected, _, references = _loss_and_gradient(uncached, case)
        torch.testing.assert_close(observed.moments, expected.moments, rtol=0, atol=0)
        torch.testing.assert_close(observed.jacobian, expected.jacobian, rtol=0, atol=0)
        assert gradients.keys() == references.keys()
        for name, value in gradients.items():
            assert torch.isfinite(value).all() and value.norm() > 0, name
            torch.testing.assert_close(value, references[name], rtol=0, atol=0)
        assert features.frozen_encoder and not features.inference_only
        assert features.projected_context is None
        assert not features.table.requires_grad and features.table.grad_fn is None
        assert cached.training and cached.response.training
        assert not cached.shared_encoder.training and not cached.geometry_attention.training
        assert all(not m.training for m in cached.shared_encoder.modules())
        assert all(p.grad is None for p in cached.shared_encoder.parameters())
        if update:
            assert features.table is first_table
            assert not torch.equal(observed.moments.detach(), first_m)
        first_table, first_m = features.table, observed.moments.detach().clone()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    for name, old in before.items():
        torch.testing.assert_close(cached.state_dict()[name], old, rtol=0, atol=0)
    stats = cached.efficiency_statistics()
    assert stats['encoder_calls'] == 1 and stats['encoder_grid_calls'] == 2
    assert uncached.encoder_calls == 2
    assert stats['frozen_features']['hits'] == 1
    assert stats['frozen_state_digest_builds'] == 1
    assert all(not name.startswith(('frozen_feature', '_frozen_state')) for name in cached.state_dict())


def test_frozen_payload_survives_response_updates_but_rejects_producer_and_payload_changes():
    model = _model()
    model.shared_encoder.register_buffer('_cache_test_buffer', torch.zeros(1))
    blocks, operand, fixed = _case()
    first = model.encode_geometry(blocks, fixed)
    with torch.no_grad():
        model.response.local[0].weight.add_(.001)
    # Response projection is fresh and therefore no stale W_h cache is used.
    assert torch.isfinite(model.evaluate_operand(first, operand).moments).all()
    again = model.encode_geometry(blocks, fixed)
    assert again.table is first.table
    with torch.no_grad():
        model.shared_encoder._cache_test_buffer.add_(1)
    with pytest.raises(ValueError, match='stale learned geometry'):
        model.evaluate_operand(first, operand)
    after_buffer = model.encode_geometry(blocks, fixed)
    assert after_buffer.table is not first.table
    assert model.encoder_calls == 2
    assert model.efficiency_statistics()['frozen_state_digest_builds'] == 2
    with torch.no_grad():
        next(model.geometry_attention.parameters()).add_(.001)
    with pytest.raises(ValueError, match='stale learned geometry'):
        model.evaluate_operand(after_buffer, operand)
    after_weight = model.encode_geometry(blocks, fixed)
    model.shared_encoder.train()
    with pytest.raises(ValueError, match='remain in eval'):
        model.encode_geometry(blocks, fixed)
    model.train()
    assert model.encode_geometry(blocks, fixed).table is after_weight.table
    with torch.no_grad():
        after_weight.table.add_(1)
    with pytest.raises(ValueError, match='table was mutated'):
        model.evaluate_operand(after_weight, operand)
    with pytest.raises(ValueError, match='table was mutated'):
        model.encode_geometry(blocks, fixed)


def test_material_fixed_density_scope_and_inference_identity_are_separate():
    model = _model()
    blocks, operand, fixed = _case()
    model.eval()
    with torch.inference_mode():
        inference = model.encode_geometry(blocks, fixed, inference=True)
        output = model.evaluate_operand(inference, operand)
    assert not output.moments.requires_grad
    assert not inference.table.is_inference()
    model.train()
    with pytest.raises(ValueError, match='inference-only'):
        model.evaluate_operand(inference, operand)
    training = model.encode_geometry(blocks, fixed)
    assert training.table is inference.table and not training.inference_only
    model.evaluate_operand(training, operand).moments.sum().backward()
    assert any(p.grad is not None for p in model.response.parameters())
    assert model.encode_geometry(blocks, fixed, density=1.2).table is not training.table
    changed_fixed = fixed.copy(); changed_fixed[0] *= 1.01
    assert model.encode_geometry(blocks, changed_fixed).table is not training.table
    original = blocks.nodes.copy()
    blocks.nodes[0, 0] += 1e-8
    with pytest.raises(ValueError, match='material changed'):
        model.evaluate_operand(training, operand)
    blocks.nodes[:] = original
    with pytest.raises(ValueError, match='foreign or stale'):
        model.evaluate_operand(replace(training, owner=-1), operand)
    model.set_training_scope('all')
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match='response-only'):
        model.evaluate_operand(training, operand)


def test_dtype_load_and_cache_budget_clear_frozen_entries():
    model = _model()
    blocks, operand, fixed = _case()
    features = model.encode_geometry(blocks, fixed)
    assert model.frozen_feature_cache.snapshot()['entries'] == 1
    model.load_state_dict(model.state_dict(), strict=True)
    assert model.frozen_feature_cache.snapshot()['entries'] == 0
    with pytest.raises(ValueError, match='stale learned geometry'):
        model.evaluate_operand(features, operand)
    fresh = model.encode_geometry(blocks, fixed)
    # Keep the dtype here: Module.to() is still an explicit cache lifecycle.
    model.to(device='cpu')
    assert model.frozen_feature_cache.snapshot()['entries'] == 0
    model.double()
    with pytest.raises(ValueError, match='stale learned geometry'):
        model.evaluate_operand(fresh, operand)
    model.clear_efficiency_caches()
    assert model.frozen_feature_cache.snapshot()['bytes'] == 0
    cache = FrozenFeatureCache(max_bytes=features.table.untyped_storage().nbytes()+fixed.nbytes,
                               max_entries=1)
    cache.put('first', features)
    cache.put('second', fresh)
    assert cache.snapshot()['entries'] == 1 and cache.snapshot()['evictions'] == 1
    assert cache.get('first') is None and cache.get('second') is fresh
    bypass = FrozenFeatureCache(max_bytes=1)
    bypass.put('x', fresh)
    assert bypass.snapshot()['entries'] == 0 and bypass.snapshot()['bypasses'] == 1


def test_all_parameter_training_rejects_a_consumed_geometry_graph_before_reuse():
    torch.manual_seed(830251)
    model = CompactPINOv21()
    blocks, operand, fixed = _case()
    features = model.encode_geometry(blocks, fixed)
    output = model.evaluate_operand(features, operand)
    (output.moments / blocks.moments[:, 0].sum()).square().mean().backward()
    assert features.graph_lifetime.consumed
    with pytest.raises(ValueError, match='graph was consumed'):
        model.evaluate_operand(features, operand)


@pytest.mark.parametrize('kwargs', [dict(training_scope='unexpected'),
    dict(frozen_feature_cache_bytes=-1), dict(frozen_feature_cache_bytes=True)])
def test_invalid_frozen_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CompactPINOv21(**kwargs)
