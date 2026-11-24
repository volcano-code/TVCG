"""Isolated diagnostic of the binary's alternative, non-algebraic R3/R8 map.

No output of this command is a deployed training or evaluation row. It changes
one native triangulation setting explicitly and gives it a separate evaluator ID.
"""
import argparse
from pathlib import Path

import numpy as np

from registry import Registry, RawChart, save_json
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS, evaluate_material_routes, fingerprint
from v21_native import FiniteOperand, operand_from_source


def as_boundary(source):
    if not source['defined'] or not source.get('side_selection_defined'):
        raise ValueError('alternative chart source rejected: ' + str(source['diagnostic']))
    return FiniteOperand(source['vertices'],source['faces'],np.asarray(source['face_is_cap']).reshape(-1),
                         'intersection' if source['interior_is_positive'] else 'difference')


def retract(q, direction, t, surface):
    """Retract to the rank-two quadratic block; its differential on a tangent is identity."""
    trial = q + t*direction
    if surface == 'PLANE':
        if np.linalg.norm(trial[4:]) > 1e-14:
            raise ValueError('requested plane path has quadratic curvature')
        return np.r_[trial[:4], np.zeros(6)]
    a = np.array([[trial[4],trial[9],trial[8]],[trial[9],trial[5],trial[7]],[trial[8],trial[7],trial[6]]])
    eigen, axes = np.linalg.eigh(a)
    null = np.argmin(np.abs(eigen))
    a -= eigen[null]*np.outer(axes[:,null],axes[:,null])
    return np.r_[trial[:4],a[0,0],a[1,1],a[2,2],a[1,2],a[0,2],a[0,1]]


def cut(backend, operand, *, chart=None):
    from v21_native_physics import evaluate_operand_physics
    targets = evaluate_operand_physics(backend, operand, chart=chart)
    if not targets.moment_valid:
        raise ValueError(targets.moment_diagnostic)
    if chart is not None and not targets.restricted_valid:
        raise ValueError(targets.derivative_diagnostic)
    return targets.moments, targets.restricted_jacobian


def run(args):
    r=Registry(args.alignment,args.corpus,args.physics)
    geom=r.example(r.panel.cross_ids[0])
    req=geom.request
    original=r.panel.profile
    changed=dict(original,algebraic_finite_map=False)
    base=r.panel.direct(req,geom.normalization,np.asarray(geom.physical).tolist())
    from make_it_stand.verification.cherchi_inprocess import InProcessCherchiEvaluator
    alt=InProcessCherchiEvaluator(req,r.panel.spin.SpinEvaluatorConfig(**changed),
        normalization=geom.normalization,design_vertices_physical=geom.physical,
        rational_arithmetic_backend='NATIVE_GMP_SURFACE_V1',fixed_boundary_reuse=True,
        integral_profile='CERTIFIED_MPFR256_GMP_FALLBACK_V1')
    if not original['algebraic_finite_map'] or base.context_identity==alt.context_identity:
        raise ValueError('alternative map was not version-separated')
    output=dict(scope='Alternative native non-algebraic instrument; no deployment admission',
                geometry_id=geom.id,source_geometry=geom.hashes,production_context=base.context_identity,
                alternative_context=alt.context_identity,production_profile=original,alternative_profile=changed,
                rows=[])
    def source(q):
        return alt.native.evaluate_spin_it_faster_stratum_source_motion(q,alt.target_design,alt.triangulation)
    for probe in geom.probes:
        if probe.chart is None: continue
        # A non-unit base query makes a missing normalization pullback observable.
        q=2.75*probe.query
        chart=RawChart(probe.chart.surface,probe.chart.basis)
        row=dict(probe_id=probe.id,query=q,chart=chart.surface,chart_basis_raw=chart.basis,source=probe.origin,
                 deployment_admissible=False)
        control=source(q)
        operand=operand_from_source(control,q,chart=chart)
        row.update(operand_faces=len(operand.faces),cap_faces=int(operand.face_is_cap.sum()),op=operand.operation,
                   positions_sha256=fingerprint(operand.vertices,'<f8'),velocity_shape=operand.vertex_velocity.shape)
        from v21_native_physics import evaluate_native_physics
        deployed = evaluate_native_physics(base,q,chart=chart)
        typed=dict(moment_valid=False,full_j_valid=False,restricted_valid=False,J_raw=None,
                   requested_chart=dict(surface=chart.surface,dimension=chart.basis.shape[1],basis_raw=chart.basis))
        evaluate_material_routes(typed,alt,q,operand,chart,r.panel.signed_moments)
        if not typed['moment_valid'] or not typed['restricted_valid'] or typed['full_j_valid']:
            raise ValueError('alternative typed replay did not return exclusively chart-local motion')
        np.testing.assert_allclose(typed['M'],typed['M_independent_tetra'],rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(typed['restricted_J'],typed['restricted_J_independent_tetra'],rtol=1e-9,atol=1e-11)
        row['deployed_same_query_status']=dict(moment_valid=deployed.moment_valid,
                restricted_valid=deployed.restricted_valid,diagnostic=deployed.moment_diagnostic)
        m,j=typed['M'],typed['restricted_J']
        row.update(M=m,chart_J=j,difference_steps=[])
        for h in (2e-5,1e-5):
            columns,velocity_errors=[],[]
            for k,direction in enumerate(chart.basis.T):
                sides=[]
                for sign in (-1,1):
                    q1=retract(q,direction,sign*h,chart.surface)
                    moving=as_boundary(source(q1))
                    if not np.array_equal(moving.faces,operand.faces) or moving.operation!=operand.operation:
                        raise ValueError('alternative chart source triangulation/side changed')
                    side,_=cut(alt,moving)
                    sides.append((side,moving.vertices))
                delta=(sides[1][0]-sides[0][0])/(2*h)
                columns.append(delta)
                dv=(sides[1][1]-sides[0][1])/(2*h)
                velocity_errors.append(float(np.linalg.norm(dv-operand.vertex_velocity[...,k])/
                                             max(np.linalg.norm(dv),1e-14)))
            fd=np.stack(columns,axis=1)
            row['difference_steps'].append(dict(h=h,FD=fd,absolute_max=float(np.max(abs(fd-j))),
                            moment_jacobian_relative_fro=float(np.linalg.norm(fd-j)/max(np.linalg.norm(j),1e-14)),
                            vertex_velocity_relative_per_column=velocity_errors))
        output['rows'].append(row)
        path=Path(args.out)/'diagnostics/alternative_chart.json'
        save_json(path,output)
        print(probe.id,row['difference_steps'][-1]['moment_jacobian_relative_fro'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment',default=DEFAULT_ALIGNMENT)
    p.add_argument('--physics',default=DEFAULT_PHYSICS)
    p.add_argument('--corpus',default=DEFAULT_CORPUS)
    p.add_argument('--out',required=True)
    run(p.parse_args())
