"""Exercise both native entries on frozen queries, raw chart motion, polarity, and full rank."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'native'))
from registry import Registry, save_json
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS
from v21_native import audit_physical_source, operand_from_source


def matrix(q):
    return np.array([[q[4], q[9], q[8]], [q[9], q[5], q[7]], [q[8], q[7], q[6]]])


def retract(q, d, t, surface):
    if t == 0:
        return q.copy()
    trial = q + t*d
    if surface == 'PARABOLOID':
        a = matrix(trial)
        lam, axes = np.linalg.eigh(a)
        base = np.linalg.eigvalsh(matrix(q))
        k = np.argmin(abs(lam))
        a -= (lam[k] - base[np.argmin(abs(base))]) * np.outer(axes[:, k], axes[:, k])
        trial[4:] = [a[0,0], a[1,1], a[2,2], a[1,2], a[0,2], a[0,1]]
    return trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--historical', action='store_true')
    args = parser.parse_args()
    registry = Registry(DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS)
    extension = None
    if not args.historical:
        from bridge import load_native
        extension = load_native()
    report = dict(rows=[], identity=extension.backend_identity if extension else 'historical')
    for row_id in ('R01_R3_019', 'R02_R8_048', 'T2_ellipsoid_plus'):
        geom = registry.example(row_id.split('_')[0])
        probe = next(p for p in geom.probes if p.id == row_id)
        backend = registry.panel.direct(geom.request, geom.normalization, np.asarray(geom.physical).tolist())
        design, settings = backend.target_design, backend.triangulation
        q = probe.query if probe.chart is None else 2.75 * probe.query
        if args.historical:
            fn = (backend.native.evaluate_spin_it_faster_stratum_source_motion if probe.chart else
                  backend.native.evaluate_spin_it_faster_source_motion)
            try:
                packet = fn(q, design, settings)
            except (ValueError, RuntimeError) as exc:
                packet = dict(defined=False, diagnostic=str(exc))
            report['rows'].append(dict(row=row_id, defined=packet['defined'], diagnostic=packet['diagnostic']))
            continue
        if probe.chart is None:
            before = backend.native.evaluate_spin_it_faster_source_motion(q, design, settings)
            after = extension.full_rank_control(q, design, settings)
            assert before['defined'] and after['defined']
            for key in ('scientific_hash', 'triangulation_hash', 'vertices', 'faces', 'face_is_cap',
                        'vertex_velocity', 'interior_is_positive'):
                np.testing.assert_array_equal(before[key], after[key])
            report['rows'].append(dict(row=row_id, full_rank_source_bitwise=True, source_hash=after['scientific_hash']))
            continue
        value = extension.source_value(q, design, settings)
        moving = extension.source_motion(q, design, settings)
        assert value['defined'] and moving['defined'], (value['diagnostic'], moving['diagnostic'])
        assert value['algebraic_finite_map'] and value['side_selection_defined']
        assert 'vertex_velocity' not in value and 'coefficient_basis' not in value
        for key in ('vertices','faces','face_is_cap','triangulation_hash','interior_is_positive'):
            np.testing.assert_array_equal(value[key], moving[key])
        assert np.count_nonzero(value['face_is_cap']) == value['cap_face_count'] > 0
        opposite = extension.source_value(-q, design, settings)
        for key in ('vertices', 'faces', 'face_is_cap'):
            np.testing.assert_array_equal(value[key], opposite[key])
        assert opposite['interior_is_positive'] != value['interior_is_positive']
        operand = operand_from_source(moving, q, chart=probe.chart)
        facts = audit_physical_source(backend.native, operand, geom.request.target_axes,
                                     geom.request.target_origin, backend.length, backend.normalization_origin)
        errors = []
        for d, direction in enumerate(probe.chart.basis.T):
            sides = [extension.source_value(retract(q, direction, t, probe.chart.surface), design, settings)
                     for t in (-1.e-5, 1.e-5)]
            for side in sides:
                np.testing.assert_array_equal(side['faces'], value['faces'])
                assert side['interior_is_positive'] == value['interior_is_positive']
            fd = (sides[1]['vertices']-sides[0]['vertices'])/2.e-5
            errors.append(float(np.linalg.norm(fd-operand.vertex_velocity[...,d])/max(1.e-14,np.linalg.norm(fd))))
        assert max(errors) < 5.e-6, errors
        report['rows'].append(dict(row=row_id, raw_norm=float(np.linalg.norm(q)), source_hash=value['triangulation_hash'],
            motion_hash=moving['scientific_hash'], vertices=len(operand.vertices), faces=len(operand.faces),
            cap_faces=value['cap_face_count'], operation=operand.operation, polarity_geometry_equal=True,
            physical_boundary_digest=facts['input_digest'], raw_chart_fd_relative_per_column=errors,
            source_witness=value['construction_witness']))
    save_json(args.out, report)
    print(json.dumps({r['row']: r.get('defined', r.get('raw_chart_fd_relative_per_column',r.get('full_rank_source_bitwise'))) for r in report['rows']}))


if __name__ == '__main__':
    main()
