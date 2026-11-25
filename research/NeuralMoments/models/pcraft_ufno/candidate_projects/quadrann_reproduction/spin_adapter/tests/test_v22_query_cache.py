"""Frozen-query reuse must preserve response semantics, motion, and graph lifetime."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spin_adapter import v22_query_cache as cached
from spin_adapter.compact_pino_v21 import evaluate_packed
from spin_adapter.v21_blocks import BlockMeasure, basis_at
from spin_adapter.v21_native import FiniteOperand
from spin_adapter.v21_response import LocalRetention, evaluate_stencil, prepare_surface


def _box(lo, hi):
    vertices = np.asarray([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                           [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                           [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                           [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


def _material(points=None):
    if points is None:
        points = [[.495, .2, .3], [.505, .25, .35], [.25, .3, .4], [.8, .35, .45]]
    points = np.asarray(points, dtype=float)
    weights = np.arange(1, len(points) + 1, dtype=float) / 100
    n = 32
    keys = np.ravel_multi_index(np.floor(points * n).astype(int).T, (n,) * 3)
    order = np.argsort(keys, kind='stable')
    points, weights, keys = points[order], weights[order], keys[order]
    ids, first, counts = np.unique(keys, return_index=True, return_counts=True)
    moments = np.stack([(basis_at(points[i:i + c]) * weights[i:i + c, None]).sum(0)
                        for i, c in zip(first, counts)])
    centers = (np.array(np.unravel_index(ids, (n,) * 3)).T + .5) / n
    return BlockMeasure(np.column_stack((points, weights)), ids,
                        np.r_[first, len(points)], moments, centers,
                        np.sqrt(3) / (2 * n), np.zeros(3), 1 / 256, 256, 8,
                        'same-material-metadata-not-a-content-substitute')


def _operand(operation='intersection', cut=.5):
    vertices, faces = _box([-1.] * 3, [cut, 2., 2.])
    velocity = np.zeros((8, 3, 3))
    velocity[vertices[:, 0] == cut, 0, :] = [1., -.5, 2.]
    return FiniteOperand(vertices, faces, np.arange(len(faces)) % 2 == 0,
                         operation, velocity)


def _evaluate(prepared, operand, *, fixed=None, model=None, context=None):
    fixed = np.zeros(10) if fixed is None else fixed
    device = next(model.parameters()).device if model is not None else 'cpu'
    value = torch.as_tensor(np.asarray(fixed) + prepared.saturated_moments,
                            dtype=torch.float64, device=device)
    jac = value.new_zeros((10, operand.vertex_velocity.shape[2]))
    for stencil in prepared.stencils:
        out = evaluate_stencil(stencil, operation=operand.operation,
                               vertex_velocity=operand.vertex_velocity,
                               model=model, context=context,
                               chunk_size=prepared.chunk_size)
        value = value + out.moments
        if jac is not None:
            jac = None if out.jacobian is None else jac + out.jacobian
    return value, jac


@pytest.mark.parametrize('operation', ['intersection', 'difference'])
def test_cache_matches_original_streaming_moments_jacobian_and_fixed_once(operation):
    blocks, operand = _material(), _operand(operation)
    cache = cached.QueryPreparationCache()
    prepared = cache.get_or_prepare(blocks, operand, .04, chunk_size=2)
    fixed = basis_at(np.array([[.1, .2, .3]]))[0] * .012
    value, jac = _evaluate(prepared, operand, fixed=fixed)
    expected = evaluate_packed(blocks, prepare_surface(operand.vertices, operand.faces), .04,
                               operation, fixed, operand.vertex_velocity, chunk_size=2)
    torch.testing.assert_close(value, expected.moments, rtol=0, atol=0)
    torch.testing.assert_close(jac, expected.jacobian, rtol=0, atol=0)
    assert prepared.band_nodes == sum(stencil.total_nodes for stencil in prepared.stencils)
    assert prepared.full.shape == prepared.band.shape == blocks.ids.shape
    assert prepared.byte_size <= cache.max_bytes


def test_same_frozen_query_reuses_surface_and_stencils_without_repreparing(monkeypatch):
    blocks, operand = _material(), _operand()
    calls = {'surface': 0, 'stencil': 0}
    original_surface, original_stencil = cached.prepare_surface, cached.prepare_stencil
    def surface(*args, **kwargs):
        calls['surface'] += 1
        return original_surface(*args, **kwargs)
    def stencil(*args, **kwargs):
        calls['stencil'] += 1
        return original_stencil(*args, **kwargs)
    monkeypatch.setattr(cached, 'prepare_surface', surface)
    monkeypatch.setattr(cached, 'prepare_stencil', stencil)
    cache = cached.QueryPreparationCache()
    first = cache.get_or_prepare(blocks, operand, .04, chunk_size=1)
    recorded = dict(calls)
    second = cache.get_or_prepare(blocks, operand, .04, chunk_size=1)
    assert second is first
    assert calls == recorded and calls['surface'] == 1 and calls['stencil'] > 0
    assert cache.info()['hits'] == 1 and cache.info()['builds'] == 1


@pytest.mark.parametrize('change', ['vertices', 'faces', 'caps', 'operation', 'bandwidth', 'chunk'])
def test_each_cut_semantic_or_chunk_change_gets_a_distinct_preparation(change):
    blocks, operand = _material(), _operand()
    cache = cached.QueryPreparationCache()
    first = cache.get_or_prepare(blocks, operand, .04, chunk_size=2)
    width, chunk = .04, 2
    if change == 'vertices':
        operand.vertices[operand.vertices[:, 0] == .5, 0] += .01
    elif change == 'faces':
        operand.faces[[0, 1]] = operand.faces[[1, 0]]
    elif change == 'caps':
        operand.face_is_cap[:] = ~operand.face_is_cap
    elif change == 'operation':
        operand = replace(operand, operation='difference')
    elif change == 'bandwidth':
        width *= 1.1
    else:
        chunk += 1
    second = cache.get_or_prepare(blocks, operand, width, chunk_size=chunk)
    assert second is not first and second.key != first.key
    assert cache.info()['misses'] == 2 and cache.info()['hits'] == 0


@pytest.mark.parametrize('field', ['nodes', 'moments', 'centers', 'origin', 'pitch', 'radius', 'identity'])
def test_material_mutations_cannot_hit_using_only_old_metadata_identity(field):
    blocks, operand = _material(), _operand()
    cache = cached.QueryPreparationCache()
    first = cache.get_or_prepare(blocks, operand, .04)
    if field in ('nodes', 'moments', 'centers', 'origin'):
        array = getattr(blocks, field)
        array.flat[0] += 1e-4
    elif field == 'identity':
        blocks.identity += '-new'
    else:
        setattr(blocks, field, getattr(blocks, field) * 1.001)
    second = cache.get_or_prepare(blocks, operand, .04)
    assert second is not first and second.key != first.key


def test_motion_is_read_fresh_and_not_stored_in_geometry_cache():
    blocks, operand = _material(), _operand()
    cache = cached.QueryPreparationCache()
    prepared = cache.get_or_prepare(blocks, operand, .04)
    first_value, first_j = _evaluate(prepared, operand)
    operand.vertex_velocity[:] *= 3
    again = cache.get_or_prepare(blocks, operand, .04)
    value, jac = _evaluate(again, operand)
    assert again is prepared and not hasattr(prepared, 'vertex_velocity')
    torch.testing.assert_close(value, first_value, rtol=0, atol=0)
    torch.testing.assert_close(jac, first_j * 3, rtol=1e-14, atol=1e-16)


def test_preparation_preserves_nonregular_distance_jacobian_unavailability():
    blocks = _material([[.5, .5, .5]])
    vertices, faces = _box([0.] * 3, [1.] * 3)
    operand = FiniteOperand(vertices, faces, np.zeros(len(faces), bool), 'intersection',
                            np.ones((8, 3, 3)))
    prepared = cached.QueryPreparationCache().get_or_prepare(blocks, operand, .8)
    value, jac = _evaluate(prepared, operand)
    assert torch.isfinite(value).all() and jac is None
    assert sum(st.ambiguous_nodes for st in prepared.stencils) == 1


def test_cache_retains_no_autograd_graph_and_supports_two_independent_updates():
    torch.manual_seed(13)
    blocks, operand = _material(), _operand()
    cache = cached.QueryPreparationCache()
    model = LocalRetention(context_dim=2, hidden_dim=8).double()
    encoder = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(encoder.parameters()), lr=.001)
    for step in range(2):
        prepared = cache.get_or_prepare(blocks, operand, .04)
        optimizer.zero_grad()
        value, jac = _evaluate(prepared, operand, model=model, context=encoder)
        (value.square().sum() + jac.square().sum()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in list(model.parameters()) + list(encoder.parameters()))
        assert encoder.weight.grad.norm() > 0
        optimizer.step()
        assert all(not isinstance(getattr(stencil, item.name), torch.Tensor)
                   for stencil in prepared.stencils for item in fields(stencil))
    assert cache.info()['hits'] == 1


def test_live_vertex_graph_bypasses_preparation_cache():
    blocks, operand = _material(), _operand()
    live = torch.from_numpy(operand.vertices.copy()).requires_grad_()
    operand = SimpleNamespace(vertices=live, faces=operand.faces,
                              face_is_cap=operand.face_is_cap, operation=operand.operation,
                              vertex_velocity=operand.vertex_velocity)
    cache = cached.QueryPreparationCache()
    assert cache.get_or_prepare(blocks, operand, .04) is None
    assert cache.info()['live_vertex_bypasses'] == 1 and cache.info()['entries'] == 0


def test_oversize_negative_cache_is_bounded_and_cleared_on_resize(monkeypatch):
    blocks, operand = _material(), _operand()
    calls = 0
    original = cached.prepare_surface
    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(cached, 'prepare_surface', count)
    cache = cached.QueryPreparationCache(max_bytes=1, negative_capacity=2)
    assert cache.get_or_prepare(blocks, operand, .04) is None
    assert cache.get_or_prepare(blocks, operand, .04) is None
    assert calls == 1 and cache.info()['negative_hits'] == 1
    for width in (.041, .042, .043):
        assert cache.get_or_prepare(blocks, operand, width) is None
    assert cache.info()['negative_entries'] == 2
    assert cache.info()['bytes'] == 0
    cache.resize(1_000_000)
    assert cache.info()['negative_entries'] == 0
    assert cache.get_or_prepare(blocks, operand, .04) is not None


def test_oversize_query_stops_collecting_chunks_and_repeated_call_bypasses(monkeypatch):
    blocks = _material([[.495, y, .3] for y in np.linspace(.1, .9, 100)])
    operand = _operand()
    complete = cached.QueryPreparationCache().get_or_prepare(blocks, operand, .04, chunk_size=1)
    calls = 0
    original = cached.prepare_stencil
    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(cached, 'prepare_stencil', count)
    cache = cached.QueryPreparationCache(max_bytes=complete.byte_size // 2)
    assert cache.get_or_prepare(blocks, operand, .04, chunk_size=1) is None
    assert 0 < calls < len(complete.stencils)
    stopped_at = calls
    assert cache.get_or_prepare(blocks, operand, .04, chunk_size=1) is None
    assert calls == stopped_at and cache.info()['negative_hits'] == 1
    assert cache.info()['entries'] == cache.info()['bytes'] == 0


def test_lru_eviction_and_clear_obey_retained_byte_limit():
    blocks, operand = _material(), _operand()
    cache = cached.QueryPreparationCache()
    first = cache.get_or_prepare(blocks, operand, .04)
    cache.resize(first.byte_size + 1)
    operand.face_is_cap[:] = ~operand.face_is_cap
    second = cache.get_or_prepare(blocks, operand, .04)
    assert second is not first
    assert cache.info()['entries'] == 1 and cache.info()['evictions'] == 1
    assert cache.info()['bytes'] <= cache.max_bytes
    cache.clear(reset_stats=True)
    assert cache.info()['entries'] == cache.info()['bytes'] == cache.info()['hits'] == 0


def test_cached_arrays_are_readonly_copies_and_source_mutation_cannot_corrupt_old_pack():
    blocks, operand = _material(), _operand()
    cache = cached.QueryPreparationCache()
    prepared = cache.get_or_prepare(blocks, operand, .04)
    old_vertices = prepared.surface.vertices.copy()
    operand.vertices[:] += .1
    np.testing.assert_array_equal(prepared.surface.vertices, old_vertices)
    assert not np.shares_memory(prepared.surface.vertices, operand.vertices)
    arrays = [prepared.full, prepared.band, prepared.saturated_moments, prepared.surface.vertices]
    arrays.extend(getattr(stencil, field.name) for stencil in prepared.stencils for field in fields(stencil)
                  if isinstance(getattr(stencil, field.name), np.ndarray))
    assert all(not array.flags.writeable for array in arrays)


def test_zero_budget_does_not_prepare_or_hash_large_material(monkeypatch):
    blocks, operand = _material(), _operand()
    def fail(*args, **kwargs):
        raise AssertionError('disabled cache must not prepare/hash')
    monkeypatch.setattr(cached, 'prepare_surface', fail)
    monkeypatch.setattr(cached, 'material_signature', fail)
    cache = cached.QueryPreparationCache(0)
    assert cache.get_or_prepare(blocks, operand, .04) is None
    assert cache.info()['disabled_bypasses'] == 1


def test_input_mutation_during_preparation_is_not_admitted(monkeypatch):
    blocks, operand = _material(), _operand()
    original = cached.prepare_stencil
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        operand.face_is_cap[:] = ~operand.face_is_cap
        return result
    monkeypatch.setattr(cached, 'prepare_stencil', mutate)
    cache = cached.QueryPreparationCache()
    with pytest.raises(RuntimeError, match='changed during'):
        cache.get_or_prepare(blocks, operand, .04, chunk_size=100)
    assert cache.info()['entries'] == 0


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_cache_rejects_invalid_budgets(value):
    with pytest.raises(ValueError):
        cached.QueryPreparationCache(value)
