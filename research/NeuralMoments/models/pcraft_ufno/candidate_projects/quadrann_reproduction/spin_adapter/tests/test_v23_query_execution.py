"""Prepared-query geometry must borrow the operator's execution resources."""
from dataclasses import replace

import pytest
import torch

from spin_adapter import v21_response_surface as surface
from spin_adapter import v23_surface_runtime as runtime
from spin_adapter.compact_pino_v23 import CompactPINOv23, RuntimeConfig
from spin_adapter.tests.test_v22_compact_integration import _case
from spin_adapter.v22_query_cache import QueryPreparationCache, query_preparation_key


def _forbid_default():
    raise AssertionError('an explicitly owned query preparation must not use the default executor')


@pytest.mark.parametrize('workers', [1, 2])
def test_v23_nonzero_query_cache_uses_owned_executor_and_closes_it(monkeypatch, workers):
    monkeypatch.setattr(surface, 'get_surface_execution_context', _forbid_default)
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: (2, ('affinity', (0, 1))))
    torch.manual_seed(830254)
    blocks, operand, fixed = _case()
    model = CompactPINOv23(query_cache_bytes=1024**2,
                          runtime=RuntimeConfig(surface_workers=workers)).eval()
    execution = model.surface_execution
    # Force parallel scheduling on the tiny physical fixture, without creating
    # a large synthetic material merely to exercise resource lifetime.
    execution.points_per_worker = 1
    try:
        with torch.no_grad():
            features = model.encode_geometry(blocks, fixed, inference=True)
            first = model.evaluate_operand(features, operand, chunk_size=3)
            prepared = next(iter(model.query_preparation_cache._entries.values()))
            assert prepared.surface.execution_context is execution
            assert all(stencil.surface is prepared.surface for stencil in prepared.stencils)
            assert not hasattr(prepared, 'vertex_velocity')
            assert prepared.key == query_preparation_key(blocks, operand, blocks.pitch, 3)
            before_hit = execution.snapshot()
            assert before_hit['queries'] > 0
            assert before_hit['last_workers'] == workers
            assert before_hit['pool_creations'] == int(workers > 1)

            moved = replace(operand, vertex_velocity=operand.vertex_velocity * 3.)
            second = model.evaluate_operand(features, moved, chunk_size=3)
            assert execution.snapshot()['queries'] == before_hit['queries']
            assert model.query_preparation_cache.info()['builds'] == 1
            assert model.query_preparation_cache.info()['hits'] == 1
            torch.testing.assert_close(first.moments, second.moments, rtol=0, atol=0)
            torch.testing.assert_close(second.jacobian, first.jacobian * 3., rtol=2e-14, atol=1e-20)
            assert model.encoder_calls == 1
    finally:
        model.close()
    after = execution.snapshot()
    assert after['closed'] and after['pool_capacity'] == 0
    assert after['pool_shutdowns'] == int(workers > 1)
    assert model.query_preparation_cache.info()['entries'] == 0
    with pytest.raises(RuntimeError, match='closed'):
        prepared.surface.signed_distance([[.5, .3, .4]])
    with pytest.raises(RuntimeError, match='closed'):
        model.evaluate_operand(features, operand)


def test_execution_binding_is_not_a_physical_key_and_cache_does_not_own_context(monkeypatch):
    monkeypatch.setattr(surface, 'get_surface_execution_context', _forbid_default)
    blocks, operand, _ = _case()
    with runtime.SurfaceExecutionContext(max_workers=1) as first_context:
        with runtime.SurfaceExecutionContext(max_workers=2) as second_context:
            first_cache, second_cache = QueryPreparationCache(), QueryPreparationCache()
            first = first_cache.get_or_prepare(blocks, operand, blocks.pitch,
                                                execution_context=first_context)
            second = second_cache.get_or_prepare(blocks, operand, blocks.pitch,
                                                  execution_context=second_context)
            assert first.key == second.key
            assert first.surface.execution_context is first_context
            assert second.surface.execution_context is second_context
            first_cache.clear()
            second_cache.clear()
            assert not first_context.snapshot()['closed']
            assert not second_context.snapshot()['closed']
