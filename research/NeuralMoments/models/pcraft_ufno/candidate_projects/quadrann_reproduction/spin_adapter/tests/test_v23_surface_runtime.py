"""Scheduling changes must preserve witnesses, finite-cut M/J and resources."""
from threading import Barrier, Lock

import numpy as np
import pytest
import torch

from spin_adapter import v21_response as response
from spin_adapter import v21_response_surface as surface
from spin_adapter import v23_surface_runtime as runtime


def box(lo=(-1., -1., -1.), hi=(1., 1., 1.)):
    vertices = np.asarray([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                           [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                           [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                           [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


def array_kernel(points, offset=0.):
    return points + offset, points.sum(axis=1)


def test_cpu_budget_prefers_affinity_and_has_portable_fallback(monkeypatch):
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda _: {2, 9}, raising=False)
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: 128)
    assert runtime._cpu_budget() == (2, ('affinity', (2, 9)))

    def unavailable(_):
        raise OSError('affinity unavailable')

    monkeypatch.setattr(runtime.os, 'sched_getaffinity', unavailable)
    monkeypatch.setattr(runtime.os, 'process_cpu_count', lambda: 3, raising=False)
    assert runtime._cpu_budget() == (3, ('cpu_count', 3))
    monkeypatch.setattr(runtime.os, 'process_cpu_count', lambda: None)
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: None)
    assert runtime._cpu_budget() == (1, ('cpu_count', 1))
    with runtime.SurfaceExecutionContext(max_workers=None) as automatic:
        assert automatic.snapshot()['max_workers'] == 8


@pytest.mark.parametrize('kwargs', [dict(max_workers=0), dict(max_workers=True),
                                    dict(points_per_worker=0), dict(points_per_worker=2.5)])
def test_invalid_resource_budgets_rejected(kwargs):
    with pytest.raises(ValueError, match='positive integer'):
        runtime.SurfaceExecutionContext(**kwargs)


def test_affinity_caps_workers_and_pool_is_reused_across_chunks(monkeypatch):
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: (2, ('affinity', (6, 7))))
    points = np.arange(192, dtype=float).reshape(-1, 3)
    with runtime.SurfaceExecutionContext(max_workers=8, points_per_worker=4) as execution:
        for chunk in np.array_split(points, 4):
            actual = execution.query(array_kernel, chunk, 7.)
            for value, expected in zip(actual, array_kernel(chunk, 7.)):
                np.testing.assert_array_equal(value, expected)
        facts = execution.snapshot()
        assert facts['pool_creations'] == 1
        assert facts['parallel_queries'] == 4
        assert facts['pool_capacity'] == facts['last_workers'] == 2
        assert facts['submitted_partitions'] == 8
    assert execution.snapshot()['closed']
    assert execution.snapshot()['pool_shutdowns'] == 1
    assert execution.snapshot()['pool_capacity'] == 0
    execution.close()
    with pytest.raises(RuntimeError, match='closed'):
        execution.query(array_kernel, points)


def test_affinity_set_change_retires_inherited_worker_affinity(monkeypatch):
    allowed = [(2, ('affinity', (2, 3)))]
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: allowed[0])
    points = np.arange(96, dtype=float).reshape(-1, 3)
    with runtime.SurfaceExecutionContext(max_workers=4, points_per_worker=2) as execution:
        execution.query(array_kernel, points)
        # Same count, different set: old worker threads may still be on 2/3.
        allowed[0] = (2, ('affinity', (6, 7)))
        execution.query(array_kernel, points)
        assert execution.snapshot()['pool_creations'] == 2
        assert execution.snapshot()['pool_shutdowns'] == 1
        allowed[0] = (1, ('affinity', (6,)))
        execution.query(array_kernel, points)
        assert execution.snapshot()['serial_queries'] == 1
        assert execution.snapshot()['pool_capacity'] == 0
        assert execution.snapshot()['pool_shutdowns'] == 2


def test_serial_reference_and_small_or_empty_queries_do_not_create_pool(monkeypatch):
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: (4, ('affinity', (0, 1, 2, 3))))
    for options in (dict(max_workers=1, points_per_worker=1),
                    dict(max_workers=4, points_per_worker=8192)):
        with runtime.SurfaceExecutionContext(**options) as execution:
            for points in (np.empty((0, 3)), np.ones((16, 3))):
                result = execution.query(array_kernel, points)
                for actual, expected in zip(result, array_kernel(points)):
                    np.testing.assert_array_equal(actual, expected)
            assert execution.snapshot()['pool_creations'] == 0
            assert execution.snapshot()['serial_queries'] == 2


def test_worker_failure_drains_other_partitions_and_context_remains_usable(monkeypatch):
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: (2, ('affinity', (0, 1))))
    barrier = Barrier(2, timeout=5)
    lock = Lock()
    active = [0]
    finished = []
    points = np.arange(24, dtype=float).reshape(-1, 3)

    def failing(chunk):
        with lock:
            active[0] += 1
        try:
            barrier.wait()
            if chunk[0, 0] == 0:
                raise ValueError('manufactured kernel failure')
            return array_kernel(chunk)
        finally:
            with lock:
                active[0] -= 1
                finished.append(float(chunk[0, 0]))

    with runtime.SurfaceExecutionContext(max_workers=2, points_per_worker=1) as execution:
        with pytest.raises(ValueError, match='manufactured kernel failure'):
            execution.query(failing, points)
        assert active[0] == 0 and len(finished) == 2
        np.testing.assert_array_equal(execution.query(array_kernel, points)[0], points)
        assert execution.snapshot()['failed_queries'] == 1
        assert execution.snapshot()['pool_creations'] == 1


def test_scopes_restore_parent_and_explicit_surface_never_falls_back():
    vertices, faces = box()
    with runtime.SurfaceExecutionContext(max_workers=1) as outer:
        unbound = surface.prepare_surface(vertices, faces)
        with runtime.SurfaceExecutionContext(max_workers=1) as inner:
            bound = surface.prepare_surface(vertices, faces, execution_context=inner)
            assert unbound.signed_distance([[.2, .3, .4]])[0] > 0
            assert inner.snapshot()['queries'] == 1
        assert runtime.get_surface_execution_context() is outer
        assert unbound.signed_distance([[.2, .3, .4]])[0] > 0
        with pytest.raises(RuntimeError, match='closed'):
            bound.signed_distance([[.2, .3, .4]])
        assert outer.snapshot()['queries'] == 1
        with pytest.raises(RuntimeError, match='already active'):
            outer.__enter__()
    with runtime.SurfaceExecutionContext(max_workers=1) as later:
        # Geometric caches constructed without a bound context survive scopes.
        unbound.signed_distance([[.2, .3, .4]])
        assert later.snapshot()['queries'] == 1
    with pytest.raises(TypeError, match='SurfaceExecutionContext'):
        surface.prepare_surface(vertices, faces, execution_context=object())


def test_scope_closes_on_outer_failure_and_default_can_be_explicitly_released():
    execution = runtime.SurfaceExecutionContext(max_workers=1)
    with pytest.raises(ArithmeticError, match='caller failed'):
        with execution:
            raise ArithmeticError('caller failed')
    assert execution.snapshot()['closed']
    runtime.shutdown_default_surface_execution()
    first = runtime.get_surface_execution_context()
    assert runtime.get_surface_execution_context() is first
    runtime.shutdown_default_surface_execution()
    assert first.snapshot()['closed']
    second = runtime.get_surface_execution_context()
    assert second is not first
    runtime.shutdown_default_surface_execution()


def test_inherited_explicit_context_rejected_before_locking(monkeypatch):
    execution = runtime.SurfaceExecutionContext(max_workers=1)
    pid = runtime.os.getpid()
    with monkeypatch.context() as patch:
        patch.setattr(runtime.os, 'getpid', lambda: pid + 1)
        with pytest.raises(RuntimeError, match='another process'):
            execution.query(array_kernel, np.zeros((1, 3)))
    execution.close()


def test_parallel_witnesses_are_bitwise_equal_to_serial_reference(monkeypatch):
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: (4, ('affinity', (0, 1, 2, 3))))
    rng = np.random.default_rng(182)
    points = np.vstack((rng.uniform(-1.4, 1.4, (257, 3)),
                        [[0., 0., 0.], [1., 1., .3], [.2, .3, 1.],
                         [1.-2e-8, .3, 1.03], [1.03, 1.02, 1.04]]))
    vertices, faces = box()
    with runtime.SurfaceExecutionContext(max_workers=1) as serial:
        reference = surface.prepare_surface(vertices, faces, execution_context=serial)
        expected = reference._query(points)
    with runtime.SurfaceExecutionContext(max_workers=4, points_per_worker=8) as parallel:
        candidate = surface.prepare_surface(vertices, faces, execution_context=parallel)
        for _ in range(3):
            for actual, wanted in zip(candidate._query(points), expected):
                np.testing.assert_array_equal(actual, wanted)
        # Reusing the executor is independent of which surface owns the BVH.
        shifted = surface.prepare_surface(vertices + [.01, 0., 0.], faces, execution_context=parallel)
        shifted._query(points)
        assert parallel.snapshot()['pool_creations'] == 1
        assert parallel.snapshot()['parallel_queries'] == 4


def test_moving_finite_face_edge_cap_moments_j_and_parameter_gradients_preserved(monkeypatch):
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: (3, ('affinity', (0, 1, 2))))
    vertices, faces = box(hi=(.2, 1., 1.))
    base = np.array([[.18, -.3, .31], [.23, .12, -.4],
                     [.25, 1.03, .2], [.22, 1.03, 1.04]])
    rng = np.random.default_rng(222)
    nodes = np.tile(base, (16, 1)) + rng.normal(0., .001, (64, 3))
    measure = response.prepare_measure(nodes, np.full(64, 1/64))
    velocity = rng.normal(0., .1, (8, 3, 3))
    torch.manual_seed(223)
    encoder = torch.nn.Linear(3, 4).double()
    model = response.LocalRetention(context_dim=4, hidden_dim=8, response_type='monotone_warp').double()
    parameters = tuple(encoder.parameters()) + tuple(model.parameters())
    common = dict(bandwidth=.14, model=model, context=encoder, chunk_size=17)
    fixed = np.arange(10) / 37.

    def predict(v, execution, **options):
        operand = surface.prepare_surface(v, faces, execution_context=execution)
        return response.evaluate_response(measure, operand, **common,
                                          fixed_moments=fixed, **options)

    with runtime.SurfaceExecutionContext(max_workers=1) as serial:
        reference = predict(vertices, serial, vertex_velocity=velocity)
        ref_gradient = torch.autograd.grad(reference.moments.square().sum()
                                          + reference.jacobian.square().sum(), parameters)
    with runtime.SurfaceExecutionContext(max_workers=3, points_per_worker=2) as parallel:
        candidate = predict(vertices, parallel, vertex_velocity=velocity)
        gradient = torch.autograd.grad(candidate.moments.square().sum()
                                      + candidate.jacobian.square().sum(), parameters)
        torch.testing.assert_close(candidate.moments, reference.moments, rtol=0, atol=0)
        torch.testing.assert_close(candidate.jacobian, reference.jacobian, rtol=0, atol=0)
        for actual, expected in zip(gradient, ref_gradient):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finite_difference = []
        for direction in velocity.transpose(2, 0, 1):
            plus = predict(vertices + 1e-5*direction, parallel).moments
            minus = predict(vertices - 1e-5*direction, parallel).moments
            finite_difference.append((plus-minus)/2e-5)
        torch.testing.assert_close(candidate.jacobian, torch.stack(finite_difference, 1),
                                   rtol=8e-7, atol=3e-10)
        opposite = predict(vertices, parallel, operation='difference', vertex_velocity=velocity)
        torch.testing.assert_close(candidate.moments + opposite.moments,
                                   torch.tensor(2*fixed + measure.total_moments), rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(candidate.jacobian, -opposite.jacobian, rtol=1e-14, atol=1e-14)
        assert parallel.snapshot()['pool_creations'] == 1


def test_parallel_exact_ties_continue_to_withhold_undefined_jacobian(monkeypatch):
    monkeypatch.setattr(runtime, '_cpu_budget', lambda: (2, ('affinity', (0, 1))))
    gap = 1e-14
    left, faces = box(hi=(-gap, 1., 1.))
    right, other = box(lo=(gap, -1., -1.))
    vertices = np.vstack((left, right))
    faces = np.vstack((faces, other + 8))
    measure = response.prepare_measure(np.tile([0., .13, .21], (16, 1)), np.full(16, 1/16))
    values = []
    for workers in (1, 2):
        with runtime.SurfaceExecutionContext(max_workers=workers, points_per_worker=2) as execution:
            out = response.evaluate_response(measure,
                surface.prepare_surface(vertices, faces, execution_context=execution),
                bandwidth=.01, vertex_velocity=np.ones((16, 3, 1)))
            assert out.jacobian is None
            assert out.diagnostics['ambiguous_nodes'] == 16
            values.append(out.moments)
    torch.testing.assert_close(*values, rtol=0, atol=0)
