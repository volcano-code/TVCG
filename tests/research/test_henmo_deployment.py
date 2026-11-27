"""Binding and orchestration software tests; native arithmetic is a replay stub."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import pytest
import torch

from experiments.henmo_v1plus import deployment as d
from experiments.henmo_v1plus.dataset import canonical_digest,model_contract
from experiments.compact_pino_v21.runtime_efficiency import deployment as old_runtime
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_rle_material


REPLAY=json.loads((Path(__file__).resolve().parents[2]/'experiments/compact_pino_v21/runtime_efficiency/fixtures/work_acceptance_replay.json').read_text())


class NativeReplay:
    """Archived vectors test dispatch/formula only; this is not mesh evidence."""
    integral_profile=d.acceptance.INTEGRAL_PROFILE
    rational_arithmetic_backend='NATIVE_GMP_SURFACE_V1'
    moment_only_capabilities=dict(schema='cherchi-native-moment-only-v1',independent_mesh_evaluation=True,
        material_jacobian_avoided=True,material_result_cache_used=False,integral_profile=integral_profile)

    def __init__(self):
        self.fixed=np.zeros(10)
        self.request=SimpleNamespace(design_vertices=np.eye(3),design_faces=np.array([[0,1,2]]),
             fixed_shell_moments=self.fixed.copy(),target_axes=np.eye(3),target_origin=np.zeros(3),
             material_density=1.,center_of_mass_weight=REPLAY['beta'],mode='SPINNING_TOP',
             spinning_top_objective='INERTIA_AND_LOWERING',settings=SimpleNamespace(
                  equation_profile='DUAL_MONOMIAL_RAW_V1',equation_tolerance=1e-12,
                  physical_relative_tolerance=1e-6))
        self.joint_calls=self.value_calls=0
        self.last_integral_evidence=None

    def _packet(self,q,derivative):
        name='corrected' if np.allclose(q,REPLAY['cases']['corrected']['q'],rtol=0,atol=1e-14) else 'direct'
        self.last_integral_evidence=dict(enclosure_certified=True,precision_bits=256,fixture_only=True)
        return dict(raw_moments=np.asarray(REPLAY['cases'][name]['moments']),
             moment_jacobian=np.eye(10) if derivative else None,derivative_defined=derivative,
             boolean_backend_identity='cherchi-reference-fixture-only',geometry_hash='fixture-geometry',
             topology_hash='fixture-topology')

    def __call__(self,q):
        self.joint_calls+=1
        return self._packet(q,True)

    def evaluate_moments(self,q):
        self.value_calls+=1
        return self._packet(q,False)


class ModelFixture(torch.nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.weight=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
        self.status='PROPOSAL_REQUIRES_NATIVE_ACCEPTANCE';self.closed=False;self.encodes=0

    def encode_geometry(self,source,fixed,density,inference):
        self.encodes+=1
        return SimpleNamespace(blocks=SimpleNamespace(identity=source.identity))

    def evaluate_operand(self,cache,operand,with_jacobian,trace):
        return SimpleNamespace(moments=torch.ones(10,dtype=torch.float64),
             jacobian=torch.eye(10,dtype=torch.float64)*self.weight if with_jacobian else None,
             diagnostics={'solver_status':self.status})

    def efficiency_statistics(self):
        return {'encoder_calls':self.encodes}

    def clear_efficiency_caches(self):
        pass

    def close(self):
        self.closed=True


def setup_runtime(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'_new_model',ModelFixture)
    monkeypatch.setattr(old_runtime.v21_native,'native_operand',lambda *a,**kw:object())
    material=prepare_rle_material(np.array([[0,0,0,16]],dtype=np.int32),[0,0,0],1/16,16)
    backend=NativeReplay();config=d.HENMOConfig.cpu_smoke();model=ModelFixture(config)
    architecture=model_contract(model)
    binding=dict(schema='henmo.training-binding.v1',architecture=architecture,fixture_only=True)
    state=dict(schema='henmo.training-state.v1',architecture=architecture,binding=binding,
               binding_sha256=canonical_digest(binding),model=model.state_dict(),step_count=2)
    checkpoint=tmp_path/'checkpoint.pt';torch.save(state,checkpoint)
    registration=d.candidate_registration(backend,material,checkpoint,config,admission_id='software-fixture-only')
    registration['status']='ADMITTED'
    kwargs=dict(expected_registration_sha256=d.registration_digest(registration),config=config,device='cpu')
    return backend,material,checkpoint,registration,kwargs


def test_factory_and_runtime_keep_one_raw_to_unit_conversion_no_boolean_and_close(tmp_path,monkeypatch):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    traces=[]
    with d.make_predictor(backend,source,checkpoint,registration,trace_sink=traces.append,**kwargs) as runtime:
        q=np.arange(1.,11.)
        joint=runtime(q);value=runtime.evaluate(q,with_jacobian=False)
        np.testing.assert_allclose(joint['moment_jacobian'],np.eye(10)*.3*np.linalg.norm(q))
        assert joint['requires_native_acceptance']
        assert value['moment_jacobian'] is None
        assert backend.joint_calls==backend.value_calls==0
        assert runtime.operator.encodes==1
        assert all(not t['metadata']['material_boolean_requested'] for t in traces)
        assert not runtime.metadata['original_material_preprocessing_included']
    assert runtime.operator.closed
    with pytest.raises(RuntimeError,match='closed'):
        runtime(q)


@pytest.mark.parametrize('kind',['draft','registry','checkpoint','config','profile','material','geometry','fixed','density'])
def test_factory_rejects_wrong_frozen_bindings(tmp_path,monkeypatch,kind):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    if kind=='draft':registration['status']='DRAFT_NOT_ADMITTED'
    elif kind=='registry':kwargs['expected_registration_sha256']='0'*64
    elif kind=='checkpoint':checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
    elif kind=='config':kwargs['config']=replace(kwargs['config'],cut_attention=False)
    elif kind=='profile':kwargs['config']=replace(kwargs['config'],response_profile='bounded_residual')
    elif kind=='material':source.full_runs=source.full_runs.copy();source.full_runs[0,3]=15
    elif kind=='geometry':backend.request.design_vertices=backend.request.design_vertices+.1
    elif kind=='fixed':registration['fixed_moments'][0]=.2;kwargs['expected_registration_sha256']=d.registration_digest(registration)
    elif kind=='density':registration['density']=2.;kwargs['expected_registration_sha256']=d.registration_digest(registration)
    with pytest.raises(ValueError):
        d.make_predictor(backend,source,checkpoint,registration,**kwargs)
    assert backend.joint_calls==backend.value_calls==0


@pytest.mark.parametrize('kind',['weights','material','task'])
def test_runtime_refuses_stale_cached_state_before_native_source(tmp_path,monkeypatch,kind):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    runtime=d.make_predictor(backend,source,checkpoint,registration,**kwargs)
    source_calls=[]
    monkeypatch.setattr(old_runtime.v21_native,'native_operand',lambda *a,**kw:source_calls.append(1))
    if kind=='weights':
        with torch.no_grad():runtime.operator.weight.add_(.01)
    elif kind=='material':source.full_runs=source.full_runs.copy()
    else:backend.request.center_of_mass_weight+=.01
    with pytest.raises(ValueError,match='changed'):
        runtime(np.ones(10))
    assert source_calls==[]
    runtime.close()


@pytest.mark.parametrize('status',['NATIVE_REQUIRED_DERIVATIVE','NATIVE_REQUIRED_REFINEMENT'])
def test_finite_numbers_cannot_override_henmo_native_required_status(tmp_path,monkeypatch,status):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    with d.make_predictor(backend,source,checkpoint,registration,**kwargs) as runtime:
        runtime.operator.status=status
        with pytest.raises(ValueError,match=status):
            runtime(np.ones(10))
    assert backend.joint_calls==backend.value_calls==0


def _same_registered_solver(evaluator,initial):
    packet=evaluator(initial)
    name='direct' if packet.get('requires_native_acceptance') else 'corrected'
    return dict(query=np.array(REPLAY['cases'][name]['q']),status='CONVERGED',
                objective=float(REPLAY['cases'][name]['expected']['objective']),accepted=True)


def _bad_solver(evaluator,initial):
    evaluator(initial)
    return dict(query=np.array(REPLAY['cases']['direct']['q']),status='CONVERGED',objective=0.,accepted=True)


def registered(callback,request):
    return d.RegisteredSolver(callback,source_path=__file__,expected_source_sha256=d.sha256(__file__),
                               expected_task_sha256=d.task_digest(request))


def test_hybrid_uses_same_solver_then_unchanged_fresh_native_acceptance(tmp_path,monkeypatch):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    with d.make_predictor(backend,source,checkpoint,registration,**kwargs) as runtime:
        out=d.solve_hybrid(runtime,registered(_same_registered_solver,backend.request),
                           [REPLAY['cases']['direct']['q']],max_native_attempts=1)
    assert out['accepted']
    assert out['verifications'][0]['accepted'] is False
    assert out['verifications'][1]['accepted'] is True
    assert out['native_verification']['reference_verify_sha256']==d.acceptance.WORK_VERIFY_SHA256
    assert out['native_verification']['certified_integral_evidence']['fixture_only']
    assert out['native_verification_calls']==2 and out['native_solver_evaluator_calls']==1
    assert backend.value_calls==2 and backend.joint_calls==1
    assert out['neural_query_calls']==1 and len(out['native_attempts'])==1
    assert not out['gpu_parallel_multistart'] and out['guaranteed_polish_steps'] is None
    assert out['timing_scope']=='PARTIAL_WORKFLOW_ONLY'
    assert out['end_to_end_seconds'] is None and not out['eligible_for_end_to_end_comparison']


def test_neural_or_solver_accepted_claim_cannot_bypass_native_rejection(tmp_path,monkeypatch):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    with d.make_predictor(backend,source,checkpoint,registration,**kwargs) as runtime:
        out=d.solve_hybrid(runtime,registered(_bad_solver,backend.request),
                           [REPLAY['cases']['direct']['q']],max_native_attempts=1)
    assert not out['accepted'] and out['query'] is None
    assert len(out['native_attempts'])==1 and out['native_verification_calls']==2


def test_failed_proposal_is_timed_and_native_fallback_remains_explicit(tmp_path,monkeypatch):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    with d.make_predictor(backend,source,checkpoint,registration,**kwargs) as runtime:
        runtime.operator.status='NATIVE_REQUIRED_REFINEMENT'
        out=d.solve_hybrid(runtime,registered(_same_registered_solver,backend.request),
                           [REPLAY['cases']['direct']['q']])
    assert out['accepted']
    assert out['proposals'][0]['status']=='PROPOSAL_FAILED'
    assert 'NATIVE_REQUIRED_REFINEMENT' in out['proposals'][0]['error']
    assert out['native_attempts'][0]['warm_start']==REPLAY['cases']['direct']['q']
    assert len(out['native_attempts'])==1 and out['native_verification_calls']==1


def test_registered_solver_rejects_source_and_task_mismatch(tmp_path,monkeypatch):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='source identity'):
        d.RegisteredSolver(_same_registered_solver,source_path=__file__,expected_source_sha256='0'*64,
                           expected_task_sha256=d.task_digest(backend.request))
    with d.make_predictor(backend,source,checkpoint,registration,**kwargs) as runtime:
        solver=registered(_same_registered_solver,backend.request);solver.task_sha256='0'*64
        with pytest.raises(ValueError,match='different native tasks'):
            d.solve_hybrid(runtime,solver,[np.ones(10)])


def test_complete_solver_settings_bind_iteration_and_nested_line_search_controls(tmp_path,monkeypatch):
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    settings=backend.request.settings
    settings.max_newton_iterations=80
    settings.line_search={'shrink':.5,'step_max':.5,'attempts':20}
    # Admit the complete, explicit native settings, then change only a field
    # outside the acceptance formula. The value/J callback must still refuse.
    registration['task_sha256']=d.task_digest(backend.request)
    kwargs['expected_registration_sha256']=d.registration_digest(registration)
    with d.make_predictor(backend,source,checkpoint,registration,**kwargs) as runtime:
        bound=d.task_digest(backend.request)
        settings.line_search['shrink']=.3
        assert d.task_digest(backend.request)!=bound
        with pytest.raises(ValueError,match='native geometry/task changed'):
            runtime(np.ones(10))
        settings.line_search['shrink']=.5
        settings.max_newton_iterations=81
        assert d.task_digest(backend.request)!=bound
        with pytest.raises(ValueError,match='native geometry/task changed'):
            runtime(np.ones(10))
    assert backend.joint_calls==backend.value_calls==0


def test_nonserializable_settings_are_not_silently_excluded_from_task_binding():
    backend=NativeReplay()
    backend.request.settings.hidden_control=object()
    with pytest.raises(ValueError,match='unsupported configuration value'):
        d.task_digest(backend.request)


def test_complete_read_start_receipt_records_wall_clock_but_fixture_proves_no_acceleration(tmp_path,monkeypatch):
    started=perf_counter()
    backend,source,checkpoint,registration,kwargs=setup_runtime(tmp_path,monkeypatch)
    timing=dict(regime='cold',read_start_perf_counter=started,includes_original_mesh_read=True,
                includes_material_preparation_or_cache_open=True,includes_backend_initialization=True,
                includes_checkpoint_and_geometry_encoding=True)
    with d.make_predictor(backend,source,checkpoint,registration,**kwargs) as runtime:
        out=d.solve_hybrid(runtime,registered(_same_registered_solver,backend.request),
                           [REPLAY['cases']['direct']['q']],timing_context=timing)
    assert out['timing_scope']=='caller_observed_end_to_end'
    assert out['end_to_end_seconds']>=out['workflow_seconds']+out['factory_setup_seconds']
    assert out['original_material_preprocessing_included']
    assert not out['eligible_for_end_to_end_comparison']  # Recorded native data are a test fixture.
    assert not out['acceleration_proven']


def test_real_henmo_checkpoint_factory_and_callback_integration(tmp_path,monkeypatch):
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
    backend=NativeReplay()
    material=prepare_rle_material(np.array([[0,0,0,16]],dtype=np.int32),[0,0,0],1/16,16)
    config=d.HENMOConfig.cpu_smoke(response_path='deterministic',execution='leaf',leaf_rule='tetra4',
                                    training_scope='response',cut_attention=False)
    source_model=HENMOOperator(config)
    architecture=model_contract(source_model)
    original_state={name:value.detach().clone() for name,value in source_model.state_dict().items()}
    binding=dict(schema='henmo.training-binding.v1',architecture=architecture,fixture_only=True)
    checkpoint=tmp_path/'real-henmo.pt'
    torch.save(dict(schema='henmo.training-state.v1',architecture=architecture,binding=binding,
                    binding_sha256=canonical_digest(binding),model=source_model.state_dict(),step_count=0),checkpoint)
    source_model.close()
    registration=d.candidate_registration(backend,material,checkpoint,config,admission_id='real-operator-software-fixture')
    registration['status']='ADMITTED'
    v=np.array([[-1,-1,-1],[2,-1,-1],[2,2,-1],[-1,2,-1],[-1,-1,.47],[2,-1,.47],[2,2,.47],[-1,2,.47]],dtype=float)
    f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                 [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    motion=np.zeros((8,3,10));motion[4:,2,0]=1.
    operand=FiniteOperand(v,f,np.zeros(len(f),dtype=bool),'intersection',motion)
    monkeypatch.setattr(old_runtime.v21_native,'native_operand',
                        lambda *a,with_jacobian:operand if with_jacobian else replace(operand,vertex_velocity=None))
    with d.make_predictor(backend,material,checkpoint,registration,
              expected_registration_sha256=d.registration_digest(registration),device='cpu') as runtime:
        loaded_state=runtime.operator.state_dict()
        for name,value in original_state.items():
            assert loaded_state[name].dtype==value.dtype
            assert torch.equal(loaded_state[name],value)
        complex_weights=[value for value in loaded_state.values() if value.is_complex()]
        assert complex_weights and any(torch.count_nonzero(value.imag)>0 for value in complex_weights)
        query=np.arange(1.,11.)
        packet=runtime(query)
        value_only=runtime.evaluate(query,with_jacobian=False)
        with torch.no_grad():
            direct=runtime.operator.evaluate_operand(runtime.cache,operand)
        np.testing.assert_allclose(packet['raw_moments'],direct.moments.numpy(),rtol=3e-13)
        np.testing.assert_allclose(value_only['raw_moments'],direct.moments.numpy(),rtol=3e-13)
        assert value_only['moment_jacobian'] is None and not value_only['derivative_defined']
        assert value_only['requires_native_acceptance']
        np.testing.assert_allclose(packet['moment_jacobian'],direct.jacobian.numpy()*np.linalg.norm(query),rtol=3e-13)
        assert packet['requires_native_acceptance'] and backend.joint_calls==backend.value_calls==0


@pytest.mark.parametrize('precision',['float64','complex128'])
def test_real_factory_rejects_registered_precision_conversion_before_loading(tmp_path,precision):
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator
    backend=NativeReplay()
    material=prepare_rle_material(np.array([[0,0,0,16]],dtype=np.int32),[0,0,0],1/16,16)
    config=d.HENMOConfig.cpu_smoke(response_path='deterministic',execution='leaf',training_scope='response')
    source_model=HENMOOperator(config)
    architecture=model_contract(source_model)
    state=source_model.state_dict()
    changed=[]
    for name,value in state.items():
        if precision=='float64' and value.is_floating_point() and value.dtype!=torch.float64:
            state[name]=value.double();changed.append(name)
        elif precision=='complex128' and value.is_complex() and value.dtype!=torch.complex128:
            state[name]=value.to(torch.complex128);changed.append(name)
    assert changed
    binding=dict(schema='henmo.training-binding.v1',architecture=architecture,fixture_only=True)
    checkpoint=tmp_path/('registered-'+precision+'.pt')
    torch.save(dict(schema='henmo.training-state.v1',architecture=architecture,binding=binding,
                    binding_sha256=canonical_digest(binding),model=state,step_count=0),checkpoint)
    source_model.close()
    # All registration hashes legitimately describe the supplied precision.
    # Nevertheless, the default model constructor may not silently cast it.
    registration=d.candidate_registration(backend,material,checkpoint,config,admission_id='precision-fixture')
    registration['status']='ADMITTED'
    with pytest.raises(ValueError,match='checkpoint tensor mismatch:'):
        d.make_predictor(backend,material,checkpoint,registration,
                        expected_registration_sha256=d.registration_digest(registration),device='cpu')
    assert backend.joint_calls==backend.value_calls==0
