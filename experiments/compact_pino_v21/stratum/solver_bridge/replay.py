"""Capture three single-call supported-space linearizations with the registered native runtime."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(HERE.parents[1] / 'native')]
from registry import Registry, save_json, sha
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import v21_stratum
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import unit_to_raw_jacobian
from experiments.compact_pino_v21.stratum.physical_replay.manufactured_plane import analytic


def one_call(spin, request, operator, query, dual_support, obtain_chart):
    observations = []

    def current_chart(q):
        chart = obtain_chart(q)
        observations.append(dict(query=q.copy(), raw_basis=chart.basis.copy()))
        return chart

    settings = replace(request.settings, equation_profile='DUAL_MONOMIAL_RAW_V1', maximum_iterations=1,
                       maximum_restarts=1, maximum_function_evaluations_per_restart=1,
                       capture_evaluation_payload=True)
    request = replace(request, settings=settings)
    callback = v21_stratum.supported_stratum_callback(operator, current_chart, support_basis=dual_support)
    result = spin.solve_spin_it_faster_with_external_moment_evaluator(
        request, callback, initial_quadrics=[query], support_basis=dual_support)
    attachment = result['supported_space']
    records = attachment['evaluations']
    if len(records) != 1 or len(observations) != 1 or result['function_evaluations'] != 1:
        raise RuntimeError('expected one compiled evaluation with one current-query chart')
    np.testing.assert_array_equal(records[0]['quadric_coefficients'], observations[0]['query'])
    np.testing.assert_array_equal(attachment['support_basis'], dual_support)
    return dict(mode=request.mode, static_objective=request.static_objective if request.mode == 'STATIC_BALANCE' else None,
                settings=asdict(settings), initial_query=query, current_chart=observations[0],
                status=result['status'], diagnostic=result['diagnostic'], function_evaluations=result['function_evaluations'],
                restart_count=result['restart_count'], supported_space=attachment)


def verify_positive(run, raw_q, raw_basis, expected_m, expected_j):
    record = run['supported_space']['evaluations'][0]
    packet, linearization = record['evaluation'], record['linearization']
    if record['error'] or not packet['derivative_defined'] or linearization is None:
        raise RuntimeError(f'first native linearization failed: {record["error"]}')
    unit_basis = np.asarray(packet['native_tangent_basis'])
    unit_j = np.asarray(packet['native_directional_jacobian'])
    if unit_basis.shape != unit_j.shape or unit_j.shape != raw_basis.shape or 'moment_jacobian' in packet:
        raise ValueError('wrong compiled directional layout')
    unit_transform = unit_to_raw_jacobian(raw_q)
    pulled_back = unit_j @ (unit_basis.T @ unit_transform @ raw_basis)
    np.testing.assert_allclose(packet['raw_moments'], expected_m, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pulled_back, expected_j, rtol=1e-9, atol=1e-12)
    # Check the actual pinned C++ dual-to-unit chain against its saved Jacobian.
    q = np.asarray(record['quadric_coefficients'])
    scale = np.array([1.] * 7 + [2.] * 3)
    dual = scale * q / np.linalg.norm(scale * q)
    native_motion = ((np.eye(10) - np.outer(q, q)) @ (linearization['solver_tangent'] / scale[:, None])
                     / np.linalg.norm(dual / scale))
    expected_newton_j = unit_j @ (unit_basis.T @ native_motion)
    np.testing.assert_allclose(linearization['moment_tangent_jacobian'], expected_newton_j, rtol=1e-10, atol=1e-12)
    for key in ('matrix', 'right_hand_side', 'full_residual', 'raw_residual'):
        if not np.isfinite(linearization[key]).all():
            raise ValueError(f'nonfinite native linearization {key}')
    return dict(moment_max_abs=float(np.max(np.abs(expected_m - packet['raw_moments']))),
                pulled_back_raw_jacobian_max_abs=float(np.max(np.abs(expected_j - pulled_back))),
                native_dual_jacobian_max_abs=float(np.max(np.abs(expected_newton_j - linearization['moment_tangent_jacobian']))),
                attachment_unit_tangent_shape=unit_j.shape, native_eq14_matrix_shape=linearization['matrix'].shape)


def plane_chart(q):
    basis = np.zeros((10, 3))
    # Nonorthogonal, differently scaled raw chart columns exercise the QR chain.
    basis[:4] = np.linalg.svd(q[:4][None, :], full_matrices=True)[2][1:].T @ np.array(
        [[1.5, .7, .2], [0., .55, -.3], [0., 0., 2.1]])
    return v21_stratum.RawChart('PLANE', basis)


def run(output):
    reg = Registry(DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS)
    spin = reg.panel.spin
    design = np.eye(4, 3, k=-1)
    faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    norm = dict(origin=[.125, -1.75, .375], L=3.25)
    fixed = np.array([.063, 0., .002, -.001, .005, .009, .007, 0., 0., 0.])
    density = 1.7
    request = spin.SpinItFasterRequest(design.tolist(), faces, design.tolist(), faces, material_density=density,
        fixed_shell_moments=tuple(fixed), original_units_per_normalized_unit=norm['L'])
    backend = reg.panel.direct(request, norm, (design * norm['L'] + norm['origin']).tolist())
    plane_op = v21_stratum.SharedStratumOperator.from_evaluator(backend)
    plane_q = np.r_[2.1 * np.array([-.18, 1.25, .83, .52]), np.zeros(6)]
    plane_support = np.eye(10)[:, :4]
    plane = one_call(spin, request, plane_op, plane_q, plane_support, plane_chart)
    reference_chart = plane_chart(plane_q)
    moments, raw_j = analytic(plane_q, reference_chart.basis, density, fixed)
    plane['closed_form_comparison'] = verify_positive(plane, plane_q, reference_chart.basis, moments, raw_j)
    print('Captured the plane linearization and checked the analytic derivative.', flush=True)
    incompatible = one_call(spin, replace(request, mode='FIDGET_SPINNER'), plane_op,
                            plane_q, plane_support, plane_chart)
    refused = incompatible['supported_space']['evaluations'][0]
    if (refused['evaluation'] is None or not refused['evaluation']['derivative_defined']
            or refused['linearization'] is not None or 'support excludes objective or constraint gradient' not in refused['error']):
        raise RuntimeError('the pinned solver did not reject the incompatible fidget objective')
    print('Captured the native objective-support rejection.', flush=True)

    geom = reg.example('R02')
    probe = next(item for item in geom.probes if item.id == 'R02_R8_032')
    parab_op = v21_stratum.SharedStratumOperator.from_evaluator(
        reg.panel.direct(geom.request, geom.normalization, np.asarray(geom.physical).tolist()))
    r8_q = np.asarray(probe.query)
    scaled_columns = np.column_stack((r8_q, probe.chart.basis))
    scaled_columns[7:] *= 2
    r8_support = np.linalg.qr(scaled_columns, mode='reduced')[0]

    def parab_chart(q):
        backend = parab_op.backend
        motion = parab_op.extension.source_motion(q, backend.target_design, backend.triangulation)
        if not motion['defined'] or motion['surface_type'] != 'PARABOLOID':
            raise ValueError('current native query is outside the R8 graph source')
        return v21_stratum.RawChart('PARABOLOID', np.asarray(motion['coefficient_basis']))

    static = replace(geom.request, mode='STATIC_BALANCE', static_objective='MAXIMUM_MASS')
    print('Evaluating the registered R02_R8_032 physical boundary.', flush=True)
    parab = one_call(spin, static, parab_op, r8_q, r8_support, parab_chart)
    archive = HERE.parent / 'qualification/native_admission/evidence/replay_inputs.zip'
    with ZipFile(archive) as source:
        center_bytes = source.read('centers/R02_R8_032.json')
    center = json.loads(center_bytes)
    if center['context'] != parab_op.identity or center['row_id'] != probe.id:
        raise ValueError('published physical reference differs from callback context')
    np.testing.assert_array_equal(r8_q, center['query'])
    np.testing.assert_array_equal(probe.chart.basis, center['raw_basis'])
    reference = center['material']
    parab['registered_tetra_comparison'] = verify_positive(parab, r8_q, probe.chart.basis,
        np.asarray(reference['M_independent_tetra']), np.asarray(reference['J_independent_tetra']))
    actual = parab['supported_space']['evaluations'][0]
    parab['registered_query_bitwise_equal'] = bool(np.array_equal(actual['quadric_coefficients'], r8_q))
    parab['registered_geometry_hash_equal'] = actual['evaluation']['geometry_hash'] == reference['material_geometry_hash']
    parab['registered_topology_hash_equal'] = actual['evaluation']['topology_hash'] == reference['material_topology_hash']
    parab['registered_reference'] = dict(row_id=probe.id, center_sha256=sha256(center_bytes).hexdigest(),
                                        query_and_chart=probe.origin)
    if not all(parab[key] for key in ('registered_query_bitwise_equal', 'registered_geometry_hash_equal',
                                     'registered_topology_hash_equal')):
        raise ValueError('compiled initial evaluation differs from the registered center')
    native_path = Path(backend.native.__file__).resolve()
    source = plane_op.extension.backend_identity
    if sha(native_path) != source['upstream_module_sha256']:
        raise ValueError('unexpected solver module binary')
    save_json(output, dict(schema='tvcg.stratum.supported-newton-first-evaluation.v1',
        scope='Single initial Eq.14 linearization for a compatible fixed dual support and static balance.',
        claims=dict(plane_initial_assembly=True, registered_R8_initial_assembly=True,
                    incompatible_objective_rejected=True, general_R8_trajectory_verified=False,
                    spinning_top_convergence_verified=False, performance_measured=False),
        native_entry='solve_spin_it_faster_with_external_moment_evaluator', native_module=str(native_path),
        native_module_sha256=sha(native_path), finite_source_module_sha256=source['module_sha256'],
        public_facade_sha256=sha(v21_stratum.__file__), replay_sha256=sha(__file__),
        plane_tetra=plane, registered_R8=parab, incompatible_fidget_objective=incompatible))
    print(f'Checked the registered raw derivative and wrote {output}.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
