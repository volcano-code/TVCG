"""Actual default U-FNO integration checks for the five execution optimizations.

The positive measure is deliberately small but uses native256 coordinates and
the real 32^3/64^3 shared encoder, Token16 attention and local physical response.
It is a software fixture, not an independently meshed material-accuracy test.
"""
import gc
from dataclasses import replace

import numpy as np
import pytest
import torch

from spin_adapter.compact_pino_v21 import CompactPINOv21
from spin_adapter.v21_blocks import BlockMeasure, basis_at
from spin_adapter.v21_native import FiniteOperand


def _case():
    points = np.array([[.4995, .213, .317], [.5005, .413, .517], [.4998, .613, .717]])
    weights = np.array([.2, .35, .45]) * (1 / 256)**3
    keys = np.ravel_multi_index(np.floor(points * 32).astype(int).T, (32,) * 3)
    order = np.argsort(keys, kind='stable')
    points, weights, keys = points[order], weights[order], keys[order]
    ids, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    moments = np.stack([(basis_at(points[i:i+c]) * weights[i:i+c, None]).sum(0)
                        for i, c in zip(starts, counts)])
    centers = (np.array(np.unravel_index(ids, (32,) * 3)).T + .5) / 32
    blocks = BlockMeasure(np.column_stack((points, weights)), ids,
                          np.r_[starts, len(points)], moments, centers,
                          np.sqrt(3) / 64, np.zeros(3), 1 / 256, 256, 8,
                          'analytic-positive-native256-integration-fixture')
    lo, hi = [-1., -1., -1.], [.5, 2., 2.]
    vertices = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                         [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                         [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                         [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    velocity = np.zeros((8, 3, 3))
    velocity[vertices[:, 0] == .5, 0, :] = [1., -.2, .5]
    operand = FiniteOperand(vertices, faces, np.arange(len(faces)) % 2 == 0,
                            'intersection', velocity)
    fixed = basis_at(np.array([[.1, .2, .3]]))[0] * weights.sum() * .1
    return blocks, operand, fixed


@pytest.fixture(scope='module')
def actual_models():
    # Construction remains the released default model, not a reduced/mock net.
    torch.manual_seed(830251)
    reference = CompactPINOv21(efficiency=False)
    optimized = CompactPINOv21(efficiency=True)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    yield reference, optimized, _case()
    del reference, optimized
    gc.collect()


def _joint_backward(model, case):
    blocks, operand, fixed = case
    model.train()
    model.zero_grad(set_to_none=True)
    encoded = model.encode_geometry(blocks, fixed)
    output = model.evaluate_operand(encoded, operand, with_jacobian=True, chunk_size=2)
    mass = blocks.moments[:, 0].sum()
    moment_weights = torch.linspace(.1, 1., 10, dtype=torch.float64)
    moment = output.moments / mass
    derivative = output.jacobian * blocks.pitch / mass
    loss = (moment * moment_weights).square().mean() + derivative.square().mean()
    loss.backward()
    gradients = {name: None if p.grad is None else p.grad.detach().clone()
                 for name, p in model.named_parameters()}
    return (output.moments.detach(), output.jacobian.detach(), output.diagnostics,
            gradients, encoded)


def test_default_operator_efficiency_preserves_values_jacobian_parameter_gradients_and_weights(actual_models):
    reference, optimized, case = actual_models
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    assert all(a.shape == optimized.state_dict()[name].shape
               for name, a in reference.state_dict().items())
    expected_m, expected_j, expected_diag, expected_gradients, old_reference = _joint_backward(reference, case)
    del old_reference
    observed_m, observed_j, observed_diag, observed_gradients, old_optimized = _joint_backward(optimized, case)
    torch.testing.assert_close(observed_m, expected_m, rtol=2e-6, atol=2e-16)
    torch.testing.assert_close(observed_j, expected_j, rtol=2e-6, atol=2e-14)
    assert observed_diag == expected_diag and observed_diag['active_nodes'] == 3
    nonzero = {'shared_encoder.pino_encoder.': False,
               'geometry_attention.': False, 'response.': False}
    for name, expected in expected_gradients.items():
        observed = observed_gradients[name]
        assert (expected is None) == (observed is None), name
        if expected is not None:
            assert torch.isfinite(expected).all() and torch.isfinite(observed).all(), name
            torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-8, msg=name)
            for prefix in nonzero:
                if name.startswith(prefix) and torch.linalg.vector_norm(observed) > 0:
                    nonzero[prefix] = True
    assert all(nonzero.values()), nonzero
    # A real optimizer update invalidates the learned cache; the static query
    # preparation survives, but an entirely fresh encoder graph is required.
    optimizer = torch.optim.SGD(optimized.parameters(), lr=1e-4)
    optimizer.step()
    blocks, operand, fixed = case
    with pytest.raises(ValueError, match='stale learned geometry'):
        optimized.evaluate_operand(old_optimized, operand)
    optimized.zero_grad(set_to_none=True)
    optimized.eval()
    with torch.no_grad():
        fresh = optimized.encode_geometry(blocks, fixed)
        fresh_result = optimized.evaluate_operand(fresh, operand, chunk_size=2)
    assert torch.isfinite(fresh_result.moments).all()
    assert torch.isfinite(fresh_result.jacobian).all()
    assert optimized.query_preparation_cache.info()['hits'] >= 1
    assert not any('cache' in key or 'projected_context' in key for key in optimized.state_dict())


def test_default_operator_no_grad_inference_reuses_queries_and_rejects_training_use(actual_models):
    _, model, (blocks, operand, fixed) = actual_models
    model.eval()
    with torch.no_grad():
        cache = model.encode_geometry(blocks, fixed, inference=True)
        first = model.evaluate_operand(cache, operand, chunk_size=2)
        second = model.evaluate_operand(cache, operand, chunk_size=2)
        opposite = model.evaluate_operand(cache, replace(operand, operation='difference'), chunk_size=2)
    torch.testing.assert_close(first.moments, second.moments, rtol=0, atol=0)
    torch.testing.assert_close(first.jacobian, second.jacobian, rtol=0, atol=0)
    assert not first.moments.requires_grad and not first.jacobian.requires_grad
    torch.testing.assert_close(first.moments + opposite.moments,
        torch.as_tensor(2 * fixed + blocks.moments.sum(0)), rtol=2e-14, atol=1e-22)
    torch.testing.assert_close(first.jacobian + opposite.jacobian,
                               torch.zeros_like(first.jacobian), rtol=0, atol=1e-18)
    changed_vertices = operand.vertices.copy()
    changed_vertices[changed_vertices[:, 0] == .5, 0] += .0004
    changed = replace(operand, vertices=changed_vertices)
    misses = model.query_preparation_cache.info()['misses']
    moved = model.evaluate_operand(cache, changed, chunk_size=2)
    assert model.query_preparation_cache.info()['misses'] == misses + 1
    assert not torch.equal(first.moments, moved.moments)
    model.train()
    with pytest.raises(ValueError, match='inference-only'):
        model.evaluate_operand(cache, operand)


def test_default_operator_rejects_material_changes_after_encoding(actual_models):
    _, model, (blocks, operand, fixed) = actual_models
    model.eval()
    with torch.no_grad():
        cache = model.encode_geometry(blocks, fixed, inference=True)
    for name in ('nodes', 'moments', 'origin'):
        original = getattr(blocks, name).copy()
        getattr(blocks, name).flat[0] += 1e-6
        try:
            with pytest.raises(ValueError, match='material changed'):
                model.evaluate_operand(cache, operand)
        finally:
            getattr(blocks, name)[:] = original
    with torch.no_grad():
        unchanged = model.evaluate_operand(cache, operand)
    assert torch.isfinite(unchanged.moments).all()


def test_default_operator_inference_mode_returns_graph_free_moment_and_jacobian(actual_models):
    _, model, (blocks, operand, fixed) = actual_models
    model.eval()
    with torch.inference_mode():
        cache = model.encode_geometry(blocks, fixed, inference=True)
        output = model.evaluate_operand(cache, operand)
    assert torch.isfinite(output.moments).all() and torch.isfinite(output.jacobian).all()
    assert not output.moments.requires_grad and not output.jacobian.requires_grad
