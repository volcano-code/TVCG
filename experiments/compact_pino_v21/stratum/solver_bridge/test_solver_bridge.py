"""Coordinate-contract regressions for the public solver callback."""
from types import SimpleNamespace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_stratum import (
    RawChart, supported_stratum_callback, unit_to_raw_jacobian,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import v21_stratum


class UnitMomentMap:
    """A differentiable sphere-valued input map with an analytic directional derivative."""

    identity = 'test-map'
    backend = SimpleNamespace(backend_identity='test-backend')

    def __init__(self, changes=None):
        self.changes = changes or {}
        self.matrix = np.sin(np.arange(100).reshape(10, 10) + .4)
        self.calls = 0

    def solver(self, q, chart):
        self.calls += 1
        moments = 2. + self.matrix @ (q / np.linalg.norm(q))
        raw = dict(raw_moments=moments, variable_raw_moments=moments - .1,
            evaluation_map_identity=self.identity, boolean_backend_identity=self.backend.backend_identity,
            geometry_hash='a' * 64, topology_hash='b' * 64, derivative_defined=False,
            stratum_derivative_defined=True, stratum_coordinate_convention='RAW_SUPPLIED_CHART_V1',
            stratum_coefficient_basis=chart.basis.copy(),
            stratum_moment_jacobian=self.matrix @ unit_to_raw_jacobian(q) @ chart.basis)
        raw.update(self.changes)
        return raw


def plane_chart(q):
    basis = np.zeros((10, 3))
    basis[:4] = np.linalg.svd(q[:4][None, :], full_matrices=True)[2][1:].T @ np.array(
        [[2., -.7, 1.], [0., .5, .22], [0., 0., 2.5]])
    # Raw charts need only span the normalized tangent after changing coordinates.
    basis += np.outer(q, [.08, -.51, 1.])
    return RawChart('PLANE', basis)


def test_current_query_chart_qr_and_radius_for_nonsymmetric_directional_map():
    service = UnitMomentMap()
    observed = []

    def provider(q):
        observed.append(q.copy())
        return plane_chart(q)

    support = np.eye(10)[:, :4].copy()
    callback = supported_stratum_callback(service, provider, support_basis=support)
    support[:] = 0.  # The adapter freezes the same support that native freezes.
    queries = [np.r_[np.array([1., -.5, .8, -.2]) * scale, np.zeros(6)] for scale in (.25, 4.)]
    queries.append(np.r_[.4, -.6, .23, 1.3, np.zeros(6)])
    for q in queries:
        result = callback(q)
        u, j = result['native_tangent_basis'], result['native_directional_jacobian']
        np.testing.assert_allclose(u.T @ u, np.eye(3), rtol=0, atol=2e-15)
        np.testing.assert_allclose(j, service.matrix @ u, rtol=0, atol=2e-14)
        np.testing.assert_allclose(q @ u, 0., rtol=0, atol=2e-14)
        assert result['derivative_defined'] is True and 'moment_jacobian' not in result
    np.testing.assert_array_equal(observed, queries)


@pytest.mark.parametrize('changes, match', [
    ({'stratum_derivative_defined': False, 'stratum_derivative_diagnostic': 'M only'}, 'M only'),
    ({'evaluation_map_identity': 'changed'}, 'identity'),
    ({'boolean_backend_identity': 'changed'}, 'identity'),
    ({'topology_hash': None}, 'topology_hash'),
    ({'geometry_hash': ''}, 'geometry_hash'),
    ({'stratum_coefficient_basis': np.zeros((10, 3))}, 'chart coordinates'),
    ({'stratum_coordinate_convention': 'AMBIENT_UNIT'}, 'chart coordinates'),
    ({'stratum_moment_jacobian': np.ones((10, 10))}, 'directional Jacobian'),
    ({'raw_moments': [np.nan] * 10}, 'raw_moments'),
])
def test_undefined_mismatched_or_unidentified_packet_is_rejected(changes, match):
    callback = supported_stratum_callback(UnitMomentMap(changes), plane_chart, support_basis=np.eye(10)[:, :4])
    with pytest.raises(ValueError, match=match):
        callback(np.r_[.4, .7, 1.4, -.2, np.zeros(6)])


def test_fixed_dual_support_rejects_excluded_tangent_with_current_point_inside():
    service = UnitMomentMap()
    q = np.r_[1., np.zeros(9)]
    # At this synthetic point even a support containing q can exclude chart directions.
    def wrong_tangent(_):
        return RawChart('PARABOLOID', np.eye(10)[:, [1, 2, 3, 4, 5, 6, 7, 9]])

    callback = supported_stratum_callback(service, wrong_tangent, support_basis=np.eye(10)[:, :9])
    with pytest.raises(ValueError, match='current chart tangent'):
        callback(q)
    assert service.calls == 0


def test_r8_dual_cross_terms_and_new_point_on_a_rotating_rank_two_leaf():
    def coefficients(n):
        n = n / np.linalg.norm(n)
        p = np.cross(n, [0., 0., 1.])
        p /= np.linalg.norm(p)
        w = np.cross(n, p)
        a = 1.1 * np.outer(p, p) + .55 * np.outer(w, w)
        return np.r_[.14, -.2, -.11, .34, a.diagonal(), a[1, 2], a[0, 2], a[0, 1]]

    def chart(q):
        a = np.array([[q[4], q[9], q[8]], [q[9], q[5], q[7]], [q[8], q[7], q[6]]])
        eigen, axes = np.linalg.eigh(a)
        x, y, z = axes[:, np.argmin(abs(eigen))]
        normal = np.r_[np.zeros(4), x*x, y*y, z*z, 2*y*z, 2*x*z, 2*x*y]
        return RawChart('PARABOLOID', np.linalg.svd(np.vstack((q, normal)), full_matrices=True)[2][2:].T)

    q = coefficients(np.array([1., 2., -1.]))
    q_next = coefficients(np.array([1.01, 2., -1.]))
    geometry_span = np.column_stack((q, chart(q).basis))
    scale = np.array([1.] * 7 + [2.] * 3)
    dual_support = np.linalg.qr(scale[:, None] * geometry_span, mode='reduced')[0]
    service = UnitMomentMap()
    callback = supported_stratum_callback(service, chart, support_basis=dual_support)
    first = callback(q)
    np.testing.assert_allclose(first['native_directional_jacobian'],
                               service.matrix @ first['native_tangent_basis'], rtol=0, atol=2e-14)
    with pytest.raises(ValueError, match='fixed dual support excludes'):
        callback(q_next)
    unscaled = np.linalg.qr(geometry_span, mode='reduced')[0]
    with pytest.raises(ValueError, match='fixed dual support excludes'):
        supported_stratum_callback(service, chart, support_basis=unscaled)(q)
    assert service.calls == 1


def test_radial_column_and_incompatible_dimensions_fail_before_evaluation():
    service = UnitMomentMap()
    q = np.eye(10)[:, 0]
    radial = RawChart('PLANE', np.eye(10)[:, :3])
    callback = supported_stratum_callback(service, lambda q: radial, support_basis=np.eye(10)[:, :4])
    with pytest.raises(ValueError, match='normalized chart rank'):
        callback(q)
    wide = supported_stratum_callback(service, plane_chart, support_basis=np.eye(10)[:, :9])
    with pytest.raises(ValueError, match='dimension'):
        wide(q)
    assert service.calls == 0


def test_archived_compiled_entry_captures_two_linearizations_and_one_rejection():
    here = Path(__file__).resolve().parent
    record = json.loads((here / 'receipt.json').read_bytes())
    frozen = json.loads((here.parent / 'native_backend/build_receipt.json').read_bytes())
    assert record['native_module_sha256'] == frozen['upstream_module_sha256']
    assert record['finite_source_module_sha256'] == frozen['module_sha256']
    assert record['public_facade_sha256'] == sha256(Path(v21_stratum.__file__).read_bytes()).hexdigest()
    assert record['replay_sha256'] == sha256((here / 'replay.py').read_bytes()).hexdigest()
    for key, dim in [('plane_tetra', 3), ('registered_R8', 8), ('incompatible_fidget_objective', 3)]:
        run = record[key]
        assert run['settings']['maximum_function_evaluations_per_restart'] == run['function_evaluations'] == 1
        evaluation, = run['supported_space']['evaluations']
        packet = evaluation['evaluation']
        assert packet['derivative_defined'] is True and 'moment_jacobian' not in packet
        assert np.shape(packet['native_directional_jacobian']) == (10, dim)
        assert np.shape(packet['native_tangent_basis']) == (10, dim)
        if key == 'incompatible_fidget_objective':
            assert evaluation['linearization'] is None and 'support excludes objective' in evaluation['error']
        else:
            assert not evaluation['error']
            assert np.shape(evaluation['linearization']['matrix']) == (dim + 3, dim + 3)
            assert np.shape(evaluation['linearization']['moment_tangent_jacobian']) == (10, dim)
    assert record['registered_R8']['registered_query_bitwise_equal']
    assert record['registered_R8']['registered_topology_hash_equal']
    assert record['registered_R8']['registered_geometry_hash_equal']
