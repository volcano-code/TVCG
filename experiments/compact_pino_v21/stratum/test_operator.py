import numpy as np
import pytest
from types import SimpleNamespace

from experiments.compact_pino_v21.stratum import shared_operator as op


Q = np.array([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.])
B = np.eye(10)[:,[0,2,3]]
V = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
F = np.array([[0,2,1],[0,1,3],[1,2,3],[0,3,2]])


def setup(monkeypatch):
    calls = []
    class Native:
        backend_identity = {'module_sha256':'a'*64, 'map':'test'}
        def source_value(self,q,design,settings):
            calls.append('value')
            return dict(defined=True, side_selection_defined=True, interior_is_positive=True,
                        surface_type='PLANE', vertices=V, faces=F, cap_face_count=1,face_is_cap=[1,0,0,0])
        def source_motion(self,q,design,settings):
            row = self.source_value(q,design,settings)
            calls[-1] = 'motion'
            return dict(row,coefficient_basis=B,vertex_velocity=np.ones((4,3,3)))
    class Backend:
        target_design = V
        native = None
        triangulation = SimpleNamespace(algebraic_finite_map=True)
        request = SimpleNamespace(target_axes=np.eye(3),target_origin=[0,0,0])
        length = 1.
        normalization_origin = np.zeros(3)
        context_identity = 'historical'
        backend_identity = 'boolean'
        def __call__(self,q):
            calls.append('ambient')
            return {'evaluation_map_identity':'historical','derivative_defined':True}
    monkeypatch.setattr(op, 'audit_physical_source',lambda *a:calls.append('audit'))
    def eval_phys(evaluator, operand, *, chart, capture_boundary, capture_topology=False):
        calls.append('cut')
        return op.NativePhysicalTargets(chart=chart,moments=np.arange(10.),editable_moments=np.arange(10.),
               restricted_jacobian=np.ones((10,3)) if operand.vertex_velocity is not None else None,
               geometry_hash='retained',topology_hash='physical_features' if capture_topology else None,
               evaluation_map_identity='historical')
    monkeypatch.setattr(op,'evaluate_operand_physics',eval_phys)
    return op.SharedStratumOperator(Backend(), Native()), calls


def test_typed_teacher_and_solver_share_moving_finite_geometry(monkeypatch):
    use, calls = setup(monkeypatch)
    chart = op.RawChart('PLANE',B)
    t = use.teacher(2.*Q,chart)
    packet = use.solver(2.*Q,chart)
    assert calls==['motion','audit','cut']*2
    assert t.evaluation_map_identity==packet['evaluation_map_identity']==use.identity
    assert use.identity!=use.backend.context_identity
    assert t.full_jacobian is None and not packet['derivative_defined']
    np.testing.assert_array_equal(t.restricted_jacobian,packet['stratum_moment_jacobian'])
    np.testing.assert_array_equal(t.moments,packet['raw_moments'])
    assert packet['stratum_derivative_defined'] and t.restricted_event_valid==-1
    assert packet['topology_hash']=='physical_features'
    value = use.condition(2.*Q,chart,with_motion=False)
    assert value.vertex_velocity is None and calls[-2:]==['value','audit']
    assert use.solver(Q)['evaluation_map_identity']=='historical' and calls[-1]=='ambient'


def test_report_value_separately_when_native_motion_rejects(monkeypatch):
    use, calls = setup(monkeypatch)
    def reject(*args):
        raise ValueError('chart pivot on seam')
    monkeypatch.setattr(use.extension,'source_motion',reject)
    result = use.teacher(Q,op.RawChart('PLANE',B))
    assert result.moment_valid and not result.restricted_valid and not result.full_j_valid
    assert 'chart pivot on seam' in result.derivative_diagnostic
    assert calls==['value','audit','cut']


def test_reject_wrong_surface_and_supplied_nontangent_basis(monkeypatch):
    use, _ = setup(monkeypatch)
    with pytest.raises(ValueError,match='surface'):
        use.condition(Q,op.RawChart('PARABOLOID',np.eye(10)[:,:8]),with_motion=False)
    with pytest.raises(ValueError,match='span'):
        use.condition(Q,op.RawChart('PLANE',np.eye(10)[:,[0,2,5]]))
    with pytest.raises(ValueError):
        use.condition(Q*0,op.RawChart('PLANE',B))

def test_project_entry_uses_the_audited_implementation():
    from experiments.compact_pino_v21.stratum.shared_operator import SharedStratumOperator
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import v21_stratum
    assert v21_stratum.SharedStratumOperator is SharedStratumOperator
