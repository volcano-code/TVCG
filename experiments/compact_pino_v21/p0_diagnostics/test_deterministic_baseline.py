"""Synthetic geometry, actual native256 material/block/finite-response code path.

These tests use an independent analytic cuboid/plane reference and rebuild the
moving closed operand for finite differences. They are not noodles, deployed
Spin-It teacher replay, or evidence of solver acceptance.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import (
    COEFFICIENT_MULTIPLIERS, LABEL_IDENTITY_FIELDS, ORDER, PREFIX, SCHEMA, load_baseline_case,
    operand_fingerprint, query_fingerprint, sha256,
)
from experiments.compact_pino_v21.p0_diagnostics.deterministic_quadrature import (
    basis, full_moments, v21_band_sweep,
)


def box_mesh(lo, hi):
    vertices = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                         [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                         [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                         [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


def face_derivatives(cut, lo, hi, density):
    # Independent moving-boundary integration: 2-point Gauss is exact for
    # quadratic moment bases times {1,y,z} on a rectangular planar section.
    g, w = np.polynomial.legendre.leggauss(2)
    ys = (lo[1]+hi[1])/2 + g*(hi[1]-lo[1])/2
    zs = (lo[2]+hi[2])/2 + g*(hi[2]-lo[2])/2
    y, z = np.meshgrid(ys, zs, indexing='ij')
    wy, wz = np.meshgrid(w*(hi[1]-lo[1])/2, w*(hi[2]-lo[2])/2, indexing='ij')
    points = np.column_stack((np.full(y.size, cut), y.ravel(), z.ravel()))
    motion = np.column_stack((np.ones(y.size), y.ravel(), z.ravel()))
    return basis(points).T @ ((density*wy*wz).ravel()[:,None] * motion)


@pytest.fixture(scope='module')
def prepared():
    material_module = importlib.import_module(PREFIX + '.v21_material')
    lo, hi, density = np.array([-.21, -.17, -.13]), np.array([.29, .23, .18]), 1.3
    vertices, faces = box_mesh(lo, hi)
    # Non-grid-aligned faces force the actual conservative partial-cell path.
    material = material_module.prepare_material(vertices, faces, [-8.]*3, 1/16, side=256, density=density)
    assert material.partial_cell_count > 0
    return material, vertices, faces, lo, hi, density


def write_case(directory, prepared, *, mode='plane', qualified=True):
    directory.mkdir(parents=True, exist_ok=True)
    material, geometry_vertices, geometry_faces, lo, hi, density = prepared
    operand_type = importlib.import_module(PREFIX + '.v21_native').FiniteOperand
    cut = .037
    if mode == 'ambiguous':
        material_module = importlib.import_module(PREFIX + '.v21_material')
        columns = np.array([[x, y, 127, 129] for x in (127,128) for y in (127,128)], dtype=np.int32)
        material = material_module.prepare_rle_material(columns, [-8.]*3, 1/16, side=256, density=density)
        lo, hi = np.full(3, -1/16), np.full(3, 1/16)
        geometry_vertices, geometry_faces = box_mesh(lo, hi)
        vertices, faces = box_mesh([-1/8]*3, [1/8]*3)
    else:
        bounds = {'plane': ([-3.]*3, [cut, 3., 3.]),
                  'full': ([-3.]*3, [3.]*3), 'empty': ([3.]*3, [4.]*3)}[mode]
        vertices, faces = box_mesh(*bounds)
    velocity = np.zeros((8,3,3))
    moving = vertices[:,0] == vertices[:,0].max()
    velocity[moving,0,:] = np.column_stack((np.ones(moving.sum()), vertices[moving,1:3]))
    caps = np.ones(len(faces), dtype=bool)
    caps[6:8] = False
    operand = operand_type(vertices, faces, caps, 'intersection', velocity)
    coefficients = np.r_[-cut, 1., np.zeros(8)]
    full = density*full_moments([[lo, hi]])
    outer = density*full_moments([[lo-.1, hi+.1]])
    fixed = outer-full  # Positive, nonzero immutable surrounding shell.
    if mode == 'plane':
        retained_hi = hi.copy(); retained_hi[0] = cut
        native = fixed + density*full_moments([[lo, retained_hi]])
        jacobian = face_derivatives(cut, lo, hi, density)
    else:
        native = fixed if mode == 'empty' else fixed+full
        jacobian = np.zeros((10,3))
    geometry_path, material_path = directory/'geometry.npz', directory/'material.npz'
    operand_path, targets_path = directory/'operand.npz', directory/'targets.npz'
    np.savez(geometry_path, vertices=geometry_vertices, faces=geometry_faces)
    material.save(material_path)
    operand.save(operand_path)
    meta = dict(geometry_id='closed_box_fixture', query_id=mode, source_group='analytic_fixture',
                split='synthetic', teacher_version='independent-box-plane-integral-v1',
                coordinate_frame='fixture_physical_frame', units='m_kg', moment_order=list(ORDER),
                coefficient_basis='raw_coefficients', geometry_sha256=sha256(geometry_path),
                coefficient_multipliers=list(COEFFICIENT_MULTIPLIERS),
                material_identity=material.identity, operand_content_sha256=operand_fingerprint(operand),
                query_content_sha256=query_fingerprint(coefficients), operation=operand.operation,
                reference_kind='analytic_fixture', moments_include_fixed=True,
                derivative_valid=qualified, derivative_dimension=3, chart_id='raw-plane-offset-tilts',
                derivative_reason='' if qualified else 'test_explicitly_withholds_label_qualification',
                editable_mass=float(full[0]), length_scale=float(np.max(hi-lo)), density=density,
                length_scale_definition='editable_bbox_max_extent',
                direction_source='synthetic_fixture')
    chart = np.zeros((10,3)); chart[0,0] = chart[2,1] = chart[3,2] = -1.
    targets = dict(metadata=np.asarray(json.dumps({key: meta[key] for key in LABEL_IDENTITY_FIELDS})),
                   coefficients=coefficients, fixed_moments=fixed, native_moments=native,
                   native_full_editable_moments=full, native_jacobian=jacobian,
                   chart_basis=chart, parameter_scales=np.array([meta['length_scale'],1.,1.]),
                   direction=np.array([.02, .03, -.01]))
    if not qualified:
        targets['native_jacobian'][:] = np.nan  # Must not read/use an unqualified label.
    np.savez(targets_path, **targets)
    spec = dict(schema=SCHEMA, meta=meta, block_voxels=8,
                assets={name: dict(path=path.name, sha256=sha256(path)) for name,path in
                        [('geometry',geometry_path),('material',material_path),
                         ('operand',operand_path),('targets',targets_path)]})
    path = directory/'case.json'
    path.write_text(json.dumps(spec))
    return path


def test_actual_mesh_material_blocks_and_finite_cap_response(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    report = v21_band_sweep(case, chunk_size=113)
    assert report['scope'] == 'synthetic_geometry_actual_v21_code_path'
    assert report['native_solver_acceptance'] == 'NOT_EVALUATED'
    assert report['material_audit']['native_reference_moment_status'] == 'PASS'
    assert report['material_audit']['maximum_native_reference_scaled_error'] < 1e-8
    assert case['fixed_moments'][0] > 0
    nodes, weights = case['blocks'].nodes[:,:3], case['blocks'].nodes[:,3]
    weighted = basis(nodes)*weights[:,None]
    for item in report['records']:
        width = item['bandwidth']
        signed = (.037-nodes[:,0])/width
        t = (np.clip(signed,-1,1)+1)/2
        retention = t*t*(3-2*t)
        derivative = np.where(np.abs(signed)<1, .75*(1-signed*signed)/width, 0.)
        expected_m = case['fixed_moments'] + weighted.T@retention
        expected_j = weighted.T@(derivative[:,None]*np.column_stack((np.ones(len(nodes)),nodes[:,1:3])))
        np.testing.assert_allclose(item['moments'], expected_m, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(item['jacobian'], expected_j, rtol=1e-10, atol=1e-13)
        assert item['derivative_valid'] and np.asarray(item['derivative_mask']).all()
        assert item['maximum_complement_scaled_error'] < 2e-12
        assert np.max(np.abs(item['complement_jacobian_scaled_error'])) < 2e-12
        combined = np.asarray(item['moments']) + np.asarray(item['opposite_side']['moments'])
        np.testing.assert_allclose(combined, 2*case['fixed_moments']+case['native_full_editable_moments'], atol=1e-12)
        # Explicitly expose smoothing/quadrature error, rather than calling the
        # hard native reference and this response numerically identical.
        assert item['maximum_scaled_moment_error'] > 0


def test_motion_derivative_rebuilds_closed_operand_at_two_steps(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    operator = importlib.import_module(PREFIX + '.compact_pino_v21')
    response = importlib.import_module(PREFIX + '.v21_response')
    operand, blocks = case['operand'], case['blocks']
    surface = response.prepare_surface(operand.vertices, operand.faces)
    out = operator.evaluate_packed(blocks, surface, blocks.pitch, operand.operation,
                                  case['fixed_moments'], operand.vertex_velocity, model=None)
    for step in (1e-4, 2e-5):
        columns = []
        for motion in operand.vertex_velocity.transpose(2,0,1):
            values = []
            for sign in (1., -1.):
                rebuilt = response.prepare_surface(operand.vertices + sign*step*motion, operand.faces)
                values.append(operator.evaluate_packed(blocks, rebuilt, blocks.pitch, operand.operation,
                                                       case['fixed_moments'], model=None).moments.numpy())
            columns.append((values[0]-values[1])/(2*step))
        np.testing.assert_allclose(out.jacobian.numpy(), np.array(columns).T, rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize('mode', ['full','empty'])
def test_full_and_empty_keep_positive_fixed_shell_once(prepared, tmp_path, mode):
    case = load_baseline_case(write_case(tmp_path, prepared, mode=mode))
    report = v21_band_sweep(case)
    for item in report['records']:
        np.testing.assert_allclose(item['moments'], case['native_moments'], atol=1e-12)
        np.testing.assert_array_equal(item['jacobian'], np.zeros((10,3)))
        assert item['maximum_complement_scaled_error'] < 2e-12


def test_unqualified_derivative_is_not_fabricated_or_read(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared, qualified=False))
    assert 'native_jacobian' not in case
    for item in v21_band_sweep(case)['records']:
        assert item['jacobian'] is None
        assert not item['derivative_valid'] and not np.asarray(item['derivative_mask']).any()
        assert item['derivative_status'] == 'UNAVAILABLE_NATIVE_REFERENCE'
        assert item['jacobian_dimensionless_error'] is None


def test_nonregular_active_distance_withholds_an_otherwise_qualified_jacobian(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared, mode='ambiguous'))
    rows = v21_band_sweep(case)['records']
    ambiguous = [row for row in rows if row['diagnostics']['ambiguous_nodes']]
    assert ambiguous
    for row in ambiguous:
        assert row['jacobian'] is None and not np.asarray(row['derivative_mask']).any()
        assert row['derivative_status'] == 'UNAVAILABLE_ACTIVE_DISTANCE_NONREGULAR'


def test_refuses_locked_split_before_opening_target_files(prepared, tmp_path):
    path = write_case(tmp_path, prepared)
    spec = json.loads(path.read_text())
    spec['meta']['split'] = 'locked_test'
    spec['assets']['targets']['path'] = 'DO_NOT_OPEN.npz'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='locked test'):
        load_baseline_case(path)


def test_mismatched_asset_digest_rejected(prepared, tmp_path):
    path = write_case(tmp_path, prepared)
    spec = json.loads(path.read_text())
    spec['assets']['targets']['sha256'] = '0'*64
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='targets asset SHA256'):
        load_baseline_case(path)


@pytest.mark.parametrize('field', ['face_is_cap','vertex_velocity','operation'])
def test_operand_caps_motion_and_side_are_content_bound(prepared, tmp_path, field):
    path = write_case(tmp_path, prepared)
    spec = json.loads(path.read_text())
    op_path = tmp_path/'operand.npz'
    with np.load(op_path, allow_pickle=False) as data:
        arrays = {key:data[key].copy() for key in data.files}
    if field == 'operation':
        arrays[field] = np.asarray('difference')
    elif field == 'face_is_cap':
        arrays[field][0] = not arrays[field][0]
    else:
        arrays[field][0,0,0] = .5
    np.savez(op_path, **arrays)
    spec['assets']['operand']['sha256'] = sha256(op_path)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='operand content identity'):
        load_baseline_case(path)


def test_reference_query_mismatch_rejected(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['reference_meta']['query_id'] = 'another_query'
    with pytest.raises(ValueError, match='reference identity mismatch: query_id'):
        v21_band_sweep(case)


def test_tampered_length_scale_cannot_change_material_admission_threshold(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['meta']['length_scale'] *= 10.
    with pytest.raises(ValueError, match='reference identity mismatch: length_scale'):
        v21_band_sweep(case)


def test_corrupt_block_aggregates_rejected(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['blocks'].moments[0,1] += .01
    with pytest.raises(ValueError, match='per-block moments'):
        v21_band_sweep(case)


def test_full_material_reference_failure_stops_before_query_comparison(prepared, tmp_path, monkeypatch):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['native_full_editable_moments'][4] += .01
    operator = importlib.import_module(PREFIX + '.compact_pino_v21')
    def forbidden(*args, **kwargs):
        pytest.fail('query operator must not be run after the full-material gate fails')
    monkeypatch.setattr(operator, 'evaluate_packed', forbidden)
    report = v21_band_sweep(case)
    assert report['status'] == 'STOP_FULL_MATERIAL_REFERENCE'
    assert report['records'] == []


@pytest.mark.parametrize('direction', [None, np.zeros(3)])
def test_missing_or_zero_direction_preserves_moments_and_jacobian(prepared, tmp_path, direction):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['direction'] = direction
    for row in v21_band_sweep(case)['records']:
        assert row['derivative_valid'] and row['jacobian'] is not None
        assert row['unit_direction_Jv_scaled_error'] is None
        assert row['direction_status'] == ('MISSING' if direction is None else 'ZERO_DIRECTION_NOT_INFORMATIVE')


def test_rejects_rank_deficient_chart(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['chart_basis'][:,2] = case['chart_basis'][:,1]
    with pytest.raises(ValueError, match='chart must be full rank'):
        v21_band_sweep(case)


def test_rejects_undoubled_coefficient_cross_terms(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['meta']['coefficient_multipliers'] = [1]*10
    with pytest.raises(ValueError, match='coefficient cross-term multipliers'):
        v21_band_sweep(case)


def test_nonphysical_fixed_moments_rejected(prepared, tmp_path):
    case = load_baseline_case(write_case(tmp_path, prepared))
    case['fixed_moments'][4] = -1.
    with pytest.raises(ValueError, match='nonnegative mass measure'):
        v21_band_sweep(case)


def test_cli_replays_existing_asset_case(prepared, tmp_path):
    path = write_case(tmp_path, prepared, qualified=False)
    output = tmp_path/'report.json'
    result = subprocess.run([sys.executable, '-m',
        'experiments.compact_pino_v21.p0_diagnostics.deterministic_quadrature',
        '--case', str(path), '--output', str(output)],
        cwd=Path(__file__).resolve().parents[3], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert len(report['records']) == 3
    assert report['asset_receipts']['operand']['sha256'] == sha256(tmp_path/'operand.npz')
    assert report['model_weights_used'] is False


def test_cli_exports_and_runs_synthetic_finite_case(tmp_path):
    output = tmp_path/'report.json'
    fixture = tmp_path/'finite-fixture'
    result = subprocess.run([sys.executable, '-m',
        'experiments.compact_pino_v21.p0_diagnostics.deterministic_quadrature',
        '--write-synthetic-case', str(fixture), '--output', str(output)],
        cwd=Path(__file__).resolve().parents[3], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['status'] == 'COMPARISON_COMPLETE_NOT_PHYSICS_ACCEPTANCE'
    assert report['scope'] == 'synthetic_geometry_actual_v21_code_path'
    assert report['provenance']['reference_kind'] == 'analytic_fixture'
    assert (fixture/'case.json').is_file()
