"""Closed-form moments and derivatives for the positive side of an oblique plane cutting a tetrahedron."""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[3]),str(HERE),str(HERE.parents[1]/"native")]
from experiments.compact_pino_v21.stratum.shared_operator import SharedStratumOperator
from experiments.compact_pino_v21.stratum.primal import retained_value
from registry import Registry, RawChart, save_json
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS
from checks import FD, error_metric, tetra_pair

POWERS=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,0,0],[0,2,0],[0,0,2],[0,1,1],[1,0,1],[1,1,0]])
DENOM=np.array([1,4,4,4,10,10,10,20,20,20])


def analytic(q,basis,density,fixed):
    sides=-q[0]/q[1:4]
    if np.any(sides<=0) or np.any(sides>=1):
        raise ValueError("the three axial intersections must be in the design tetrahedron")
    # Fraction of the unit tetrahedron, its first/second moment monomials, and their true product derivatives.
    cut=(density*np.prod(sides)/6)*np.prod(sides**POWERS,axis=1)/DENOM
    dlog_sides=basis[0]/q[0]-basis[1:4]/q[1:4,None]
    differential=-cut[:,None]*((1+POWERS)@dlog_sides)
    return density/(6*DENOM)-cut+fixed,differential


def run(out):
    reg=Registry(DEFAULT_ALIGNMENT,DEFAULT_CORPUS,DEFAULT_PHYSICS)
    design=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    norm=dict(origin=[.125,-1.75,.375],L=3.25)
    fixed=np.array([.063,0,.002,-.001,.005,.009,.007,0,0,0])
    density=1.7
    req=reg.panel.spin.SpinItFasterRequest(design.tolist(),faces.tolist(),design.tolist(),faces.tolist(),
         fixed_shell_moments=tuple(fixed),material_density=density,original_units_per_normalized_unit=norm["L"])
    physical=design*norm["L"]+norm["origin"]
    backend=reg.panel.direct(req,norm,physical.tolist())
    op=SharedStratumOperator.from_evaluator(backend)
    q=np.r_[2.1*np.array([-.18,1.25,.83,.52]),np.zeros(6)]
    basis=np.zeros((10,3))
    basis[:4]=np.linalg.svd(q[:4][None,:],full_matrices=True)[2][1:].T
    chart=RawChart("PLANE",basis)
    expected_m,expected_j=analytic(q,basis,density,fixed)
    target=op.teacher(q,chart,capture_boundary=True)
    if not target.moment_valid or not target.restricted_valid:
        raise ValueError((target.moment_diagnostic,target.derivative_diagnostic))
    np.testing.assert_allclose(target.moments,expected_m,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(target.restricted_jacobian,expected_j,atol=1e-12,rtol=1e-10)
    solver=op.solver(q,chart)
    if (not np.array_equal(solver["raw_moments"],target.moments) or
            not np.array_equal(solver["stratum_moment_jacobian"],target.restricted_jacobian)):
        raise ValueError("solver and teacher differ on closed-form control")
    w=target.boundary_witness
    independent_m,independent_j=tetra_pair(w.vertices,w.faces,w.vertex_velocity,density,fixed)
    np.testing.assert_allclose(independent_j,expected_j,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(independent_m,expected_m,atol=1e-12,rtol=1e-10)
    steps=[]
    for h in FD["raw_steps"]:
        cols=[]
        for k in range(3):
            values=[]
            for sign in (-1,1):
                varied=q+sign*h*basis[:,k]
                cut=retained_value(backend,op.condition(varied,chart,with_motion=False))
                expected,_=analytic(varied,basis,density,fixed)
                np.testing.assert_allclose(cut.moments,expected,atol=1e-12,rtol=1e-10)
                values.append(cut.moments)
            cols.append((values[1]-values[0])/(2*h))
        matrix=np.stack(cols,axis=1)
        differences=[error_metric(matrix[:,k],expected_j[:,k],"material") for k in range(3)]
        if not all(d["passed"] for d in differences):
            raise ValueError("closed-form finite-difference check failed")
        steps.append(dict(h=h,raw_FD=matrix,errors=differences))
    # For this second raw plane the whole tetrahedron is in its negative region.
    # The physical reference historically rejects an empty retained component.
    rejected=q.copy()
    rejected[0]=-8.
    reject_chart=np.zeros((10,3))
    reject_chart[:4]=np.linalg.svd(rejected[:4][None,:],full_matrices=True)[2][1:].T
    empty=op.teacher(rejected,RawChart("PLANE",reject_chart),capture_boundary=True)
    if empty.moment_valid or empty.restricted_valid or not empty.moment_diagnostic:
        raise ValueError("empty cut failed to preserve physical rejection")
    save_json(out,dict(scope="manufactured independent geometry and closed-form positive-side R3 control",identity=op.identity,
         formula="Editable positive solid = unit coordinate tetrahedron minus the origin tetrahedron with axial intercepts -q0/qi.",
         q=q,basis=basis,density=density,fixed_moments=fixed,normalization=norm,vertices=design,faces=faces,
         analytic_moments=expected_m,analytic_jacobian=expected_j,true_moments=target.moments,
         native_restricted_j=target.restricted_jacobian,M_independent_tetra=independent_m,
         J_independent_tetra=independent_j,steps=steps,solver_teacher_m_j_bitwise_equal=True,
         empty=dict(query=rejected,moment_valid=empty.moment_valid,restricted_valid=empty.restricted_valid,
                    diagnostic=empty.moment_diagnostic)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)
