import numpy as np
import pytest

from experiments.compact_pino_v21.diagnostics import moment_errors, jacobian_errors, accepted_directions


def test_scales_use_full_editable_mass_across_moment_degrees():
    y = np.arange(10, dtype=float)
    scale = 2 * np.array([1, 3, 3, 3, 9, 9, 9, 9, 9, 9])
    errors = moment_errors(y + 0.01 * scale, y, 2., 3.)
    assert errors['scaled_max'] == pytest.approx(.01)
    assert errors['absolute'][7] == pytest.approx(.18)


def test_full_j_scaled_to_unit_coefficients_and_actual_direction():
    true = np.eye(10)
    pred = true.copy()
    pred[0, 1] += .08
    v = np.eye(10)[1]
    out = jacobian_errors(pred, true, 2., 1., 3., direction=v)
    assert out['unit_direction_scaled_max'] == pytest.approx(.12)
    assert out['Jv']['predicted'][0] == pytest.approx(.08)
    assert out['Jv']['reference'] == pytest.approx(v.tolist())
    assert jacobian_errors(true * 0, true * 0, 2., 1., 1.)['relative_frobenius'] is None


def test_accepted_native_directions_skip_backtracking_and_terminal():
    a, b = np.eye(10)[:2]
    direction = (a + b) / np.sqrt(2)
    rows = [
        {'function_evaluations': 1, 'quadric_coefficients': a.tolist(), 'restart': 0},
        {'function_evaluations': 4, 'quadric_coefficients': direction.tolist(), 'restart': 0},
    ]
    dirs = accepted_directions(rows)
    assert set(dirs) == {1}
    assert dirs[1] == pytest.approx(b)


def test_deficient_chart_requires_its_native_basis():
    with pytest.raises(ValueError, match='basis'):
        jacobian_errors(np.zeros((10, 3)), np.zeros((10, 3)), 2., 1., 1.)
    b = np.eye(10)[:, :3] * 2
    j = np.zeros((10, 3)); j[0, 2] = 2.
    out = jacobian_errors(j, j * 0, 1., 1., 1., basis=b)
    assert out['unit_direction_scaled_max'] == pytest.approx(1.)


def test_scoring_typed_prediction_uses_registered_chart_even_when_full_is_available():
    from experiments.compact_pino_v21.replay import score
    basis = np.eye(10)[:, :3] * 2
    row = dict(M=np.zeros(10), J_raw=np.eye(10), full_j_valid=True, restricted_valid=True,
               restricted_J=np.ones((10,3)), requested_chart={'dimension':3,'basis_raw':basis.tolist()},
               query=np.eye(10)[0])
    results = {'local':{'M':np.zeros(10), 'J':np.ones((10,3))}}
    score(results, row, {'reference_editable_mesh':[1.], 'length':1.}, None)
    assert results['local']['jacobian_error']['unit_direction_scaled_max'] == 0.


def test_comparison_uses_common_rows_separately_for_value_and_derivative():
    from experiments.compact_pino_v21.summarize import METHODS, matched_comparison
    def measured(value):
        return {'moment_error': {'scaled_max':value, 'relative_l2':value},
                'jacobian_error': {'unit_direction_scaled_max':value, 'relative_frobenius':value}}
    records = {k:{m:measured(i) for m in METHODS} for i,k in enumerate(('paired','one_missing_j','control_only'), 1)}
    del records['one_missing_j'][METHODS[0]]['jacobian_error']
    del records['control_only'][METHODS[0]]
    table, ids = matched_comparison(records)
    assert ids == {'moment_ids':['one_missing_j','paired'], 'jacobian_ids':['paired']}
    for result in table.values():
        assert result['scaled_moment']['n'] == 2
        assert result['scaled_jacobian']['p95'] == 1.
