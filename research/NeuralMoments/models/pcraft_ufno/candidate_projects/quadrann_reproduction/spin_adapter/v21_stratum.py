"""Opt-in finite-stratum teacher and supported-space Newton callback."""
import re

import numpy as np

from experiments.compact_pino_v21.stratum.shared_operator import SharedStratumOperator
from .v21_native import RawChart, unit_to_raw_jacobian


_CONTRACT_TOLERANCE = 1e-12  # linearizeSpinMomentSubspace in the pinned solver
_DUAL_SCALE = np.array([1.] * 7 + [2.] * 3)


def supported_stratum_callback(operator: SharedStratumOperator, chart_provider, *, support_basis):
    """Adapt a current-query raw chart to the compiled directional evaluator.

    Pass the same orthonormal dual-monomial ``support_basis`` to
    ``solve_spin_it_faster_with_external_moment_evaluator``. The native solver
    checks the omitted objective/constraint equations during each linearization.
    A rank-two leaf generally has a different tangent at its next query.
    """
    if not callable(chart_provider):
        raise TypeError('a callable chart provider for the current query is required')
    support = np.array(support_basis, dtype=np.float64, copy=True)
    if (support.shape not in ((10, 4), (10, 9)) or not np.isfinite(support).all()
            or np.linalg.norm(support.T @ support - np.eye(support.shape[1])) > _CONTRACT_TOLERANCE):
        raise ValueError('expected four or nine orthonormal columns in fixed dual support')
    map_id, backend_id = operator.identity, operator.backend.backend_identity
    if any(not isinstance(label, str) or not label or ';' in label for label in (map_id, backend_id)):
        raise ValueError('material map and backend need valid solver identities')

    def evaluate(query):
        q = np.array(query, dtype=np.float64, copy=True)
        normalization = unit_to_raw_jacobian(q)
        u = q / np.linalg.norm(q)
        supplied = chart_provider(q.copy())
        if not hasattr(supplied, 'basis') or not hasattr(supplied, 'surface'):
            raise TypeError('the current-query chart provider must return a typed raw chart')
        chart = RawChart(supplied.surface, np.array(supplied.basis, dtype=np.float64, copy=True))
        dim = chart.basis.shape[1]
        tangent = normalization @ chart.basis
        if support.shape[1] != dim + 1 or not np.isfinite(tangent).all() or np.linalg.matrix_rank(tangent) != dim:
            raise ValueError('normalized chart rank or dimension differs from fixed dual support')
        unit_basis, change = np.linalg.qr(tangent, mode='reduced')
        if np.linalg.norm(u @ unit_basis) > _CONTRACT_TOLERANCE:
            raise ValueError('normalized chart does not give unit-coefficient tangents')

        # Native dual monomials double the three mixed coefficients. Transport
        # this *fixed* support back to the unit geometry sphere at the actual q.
        dual = _DUAL_SCALE * u
        dual /= np.linalg.norm(dual)
        native_motion = (np.eye(10) - np.outer(u, u)) @ (support / _DUAL_SCALE[:, None])
        native_motion /= np.linalg.norm(dual / _DUAL_SCALE)
        if (np.linalg.norm(dual - support @ (support.T @ dual)) > _CONTRACT_TOLERANCE
                or np.linalg.norm(native_motion - unit_basis @ (unit_basis.T @ native_motion)) > _CONTRACT_TOLERANCE):
            raise ValueError('fixed dual support excludes the query or differs from its current chart tangent')

        raw = operator.solver(q, chart)
        if raw.get('evaluation_map_identity') != map_id or raw.get('boolean_backend_identity') != backend_id:
            raise ValueError('typed moment and derivative have a different map or backend identity')
        if raw.get('stratum_derivative_defined') is not True:
            raise ValueError(f"stratum derivative unavailable: {raw.get('stratum_derivative_diagnostic')}")
        if (raw.get('stratum_coordinate_convention') != 'RAW_SUPPLIED_CHART_V1'
                or not np.array_equal(raw.get('stratum_coefficient_basis'), chart.basis)):
            raise ValueError('returned derivative does not match the supplied raw chart coordinates')
        for key in ('geometry_hash', 'topology_hash'):
            if not isinstance(raw.get(key), str) or not re.fullmatch('[0-9a-f]{64}', raw[key]):
                raise ValueError(f'missing or invalid physical {key}')
        raw_j = np.asarray(raw.get('stratum_moment_jacobian'), dtype=np.float64)
        if raw_j.shape != (10, dim) or not np.isfinite(raw_j).all():
            raise ValueError('expected a finite physical directional Jacobian for the current chart')
        directional = np.linalg.solve(change.T, raw_j.T).T
        if not np.isfinite(directional).all():
            raise ValueError('raw-to-unit directional Jacobian overflowed')
        packet = {key: raw[key] for key in ('geometry_hash', 'topology_hash', 'evaluation_map_identity',
                                             'boolean_backend_identity')}
        for key in ('raw_moments', 'variable_raw_moments'):
            values = np.asarray(raw.get(key), dtype=np.float64)
            if values.shape != (10,) or not np.isfinite(values).all():
                raise ValueError(f'expected ten finite physical {key}')
            packet[key] = values
        return dict(packet, native_tangent_basis=unit_basis, native_directional_jacobian=directional,
                    derivative_defined=True, derivative_diagnostic='',
                    derivative_identity=f'TVCG_FINITE_STRATUM_UNIT_QR_V1:{chart.surface}:{map_id}')

    return evaluate


__all__ = ['RawChart', 'SharedStratumOperator', 'supported_stratum_callback']
