"""Ambient and typed replay share one material only after establishing matching geometry."""
from pathlib import Path
import sys

import numpy as np
import pytest

repo=next(p for p in Path(__file__).resolve().parents if (p/'experiments/compact_pino_v21/native').is_dir())
sys.path.insert(0,str(repo/'experiments/compact_pino_v21/native'))
from registry import RawChart
from replay import record_typed_targets, tetra_motion
from v21_native_physics import NativePhysicalTargets, BoundaryWitness


@pytest.mark.parametrize('d,ambient', [(3,False),(8,True)])
def test_typed_measurement_uses_its_own_motion_and_keeps_ambient_eligibility(d,ambient):
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    basis=np.eye(10)[:,1:d+1]
    chart=RawChart('PLANE' if d==3 else 'PARABOLOID',basis)
    moving=np.zeros((4,3,d))
    moving[1,0]=np.arange(1,d+1)
    j=tetra_motion(vertices,faces,moving,1.)
    m=np.array([1/6,1/24,1/24,1/24,1/60,1/60,1/60,1/120,1/120,1/120])
    fixed=m*.125
    full=np.arange(100.).reshape(10,10) if ambient else None
    row=dict(moment_valid=ambient,full_j_valid=ambient,J_raw=full,restricted_valid=False,
             geometry_hash='retained' if ambient else None,M=m+fixed if ambient else None,
             M_editable=m if ambient else None,ambient_event_valid=-1,restricted_event_valid=-1,
             material_error=None if ambient else dict(diagnostic='ambient undefined'),
             requested_chart=dict(basis_raw=basis,dimension=d,surface=chart.surface))
    targets=NativePhysicalTargets(moments=m+fixed,editable_moments=m,restricted_jacobian=j,
        chart=chart,geometry_hash='retained',evaluation_map_identity='same-map',
        boundary_witness=BoundaryWitness(vertices,faces,moving))
    def signed_integral(v,f,rho):
        np.testing.assert_array_equal(v,vertices)
        np.testing.assert_array_equal(f,faces)
        return m*rho
    record_typed_targets(row,targets,fixed,1.,signed_integral)
    assert row['moment_valid'] and row['restricted_valid'] and row['full_j_valid']==ambient
    assert row['restricted_material_geometry_hash']=='retained'
    assert row['ambient_event_valid']==row['restricted_event_valid']==-1
    np.testing.assert_array_equal(row['M_independent_tetra'],m+fixed)
    np.testing.assert_array_equal(row['restricted_J'],j)
    np.testing.assert_array_equal(row['restricted_J_independent_tetra'],j)
    np.testing.assert_array_equal(row['restricted_J'][0],np.arange(1,d+1)/6)
    assert row['J_raw'] is full
    if ambient:
        targets.geometry_hash='another-retained-body'
        with pytest.raises(ValueError,match='geometr'):
            record_typed_targets(row,targets,fixed,1.,signed_integral)
    else:
        assert row['material_error']['diagnostic']=='ambient undefined'


@pytest.mark.parametrize('ambient_available',[False,True])
def test_replay_dispatches_ambient_and_typed_routes_independently(monkeypatch,ambient_available):
    from types import SimpleNamespace
    import replay
    from v21_native import FiniteOperand
    q=np.eye(10)[0]
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    caps=np.array([0,0,1,0],dtype=bool)
    basis=np.eye(10)[:,[3,1,2]]
    chart=RawChart('PLANE',basis)
    full_motion=np.zeros((4,3,10))
    full_motion[1,0]=np.arange(10.)
    typed=FiniteOperand(vertices,faces,caps,'intersection',full_motion@basis)
    m=np.array([1/6,1/24,1/24,1/24,1/60,1/60,1/60,1/120,1/120,1/120])
    full_j=tetra_motion(vertices,faces,full_motion,1.)
    source=dict(defined=True,side_selection_defined=True,interior_is_positive=True,vertices=vertices,
                faces=faces,face_is_cap=caps,cap_face_count=1,vertex_velocity=full_motion)

    class Measurement:
        context_identity='same'
        request=SimpleNamespace(material_density=1.)
        fixed=np.zeros(10)
        last=dict(vertices=vertices,faces=faces,source=source,operation='intersection')

        def __call__(self,query):
            if not ambient_available:
                raise ValueError('ambient rank is undefined')
            return dict(raw_moments=m,variable_raw_moments=m,moment_jacobian=full_j,derivative_defined=True,
                        geometry_hash='retained',topology_hash='branch',evaluation_map_identity='same')

    # The two native ABIs cannot be simultaneously positive on this frozen binary.
    # Supply their mathematically compatible tetrahedron observations to the real producer.
    def ambient_witness(backend,last,operand):
        return tetra_motion(last['vertices'],last['faces'],operand.vertex_velocity,1.)

    def chart_measurement(backend,operand,*,chart,capture_boundary):
        assert capture_boundary and operand is typed
        return NativePhysicalTargets(moments=m,editable_moments=m,chart=chart,geometry_hash='retained',
                evaluation_map_identity=backend.context_identity,restricted_jacobian=full_j@basis,
                boundary_witness=BoundaryWitness(vertices,faces,operand.vertex_velocity))

    monkeypatch.setattr(replay,'audit_jacobian',ambient_witness)
    monkeypatch.setattr(replay,'evaluate_operand_physics',chart_measurement)
    row=dict(moment_valid=False,full_j_valid=False,restricted_valid=False,J_raw=None,
             requested_chart=dict(surface=chart.surface,dimension=3,basis_raw=basis),
             ambient_event_valid=-1,restricted_event_valid=-1)
    replay.evaluate_material_routes(row,Measurement(),q,typed,chart,lambda v,f,d:m*d)
    assert row['moment_valid'] and row['restricted_valid'] and row['full_j_valid']==ambient_available
    assert row['ambient_event_valid']==row['restricted_event_valid']==-1
    np.testing.assert_array_equal(row['restricted_J_independent_tetra'],full_j@basis)
    if ambient_available:
        np.testing.assert_array_equal(row['J_raw'],full_j)
        np.testing.assert_array_equal(row['J_independent_tetra'],full_j)
        assert row['material_error'] is None
    else:
        assert row['J_raw'] is None and row['material_error']['diagnostic']=='ambient rank is undefined'
