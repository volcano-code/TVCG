"""Software contracts for the new runner; no native physical-quality claims."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.compact_pino_v21.training import manifest as manifest_module
from experiments.compact_pino_v21.training.admission import SCHEMA as AUTH_SCHEMA, authorize, canonical_digest
from experiments.compact_pino_v21.training.losses import physical_loss
from experiments.compact_pino_v21.training.manifest import SCHEMA, load_training_manifest
from experiments.compact_pino_v21.training.runner import Trainer, TrainConfig, RESUME_SCHEMA, _read_resume, _evaluation_steps
from experiments.compact_pino_v21.training.smoke import make_smoke_manifest
from experiments.compact_pino_v21.training.state import StatefulShuffle, observed_optimizer_step
from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import (
    LABEL_IDENTITY_FIELDS, operand_fingerprint, query_fingerprint, sha256,
)


def _record(dimension=3,qualified=True,moment_weight=1.,jacobian_weight=1.):
    return dict(moment_weight=moment_weight,jacobian_weight=jacobian_weight,
                case=dict(meta=dict(editable_mass=1.,length_scale=1.,derivative_valid=qualified),
                          native_moments=np.zeros(10),native_jacobian=np.zeros((10,dimension)),
                          parameter_scales=np.ones(dimension)))


def _prediction(moment,jacobian,dimension=3):
    return SimpleNamespace(moments=torch.full((10,),float(moment),dtype=torch.float64,requires_grad=True),
                           jacobian=None if jacobian is None else torch.full((10,dimension),float(jacobian),
                                                    dtype=torch.float64,requires_grad=True))


def test_independent_loss_denominators_and_qualified_j_membership():
    records=[_record(moment_weight=1.,jacobian_weight=2.),
             _record(dimension=8,moment_weight=3.,jacobian_weight=1.),
             _record(qualified=False,moment_weight=0.,jacobian_weight=100.),
             _record(moment_weight=0.,jacobian_weight=100.)]
    predictions=[_prediction(1,2),_prediction(3,4,8),_prediction(99,100),_prediction(99,None)]
    loss,info=physical_loss(predictions,records,jacobian_loss_weight=.5)
    assert info['moment_denominator']==4
    assert info['jacobian_denominator']==3
    assert info['moment_loss']==pytest.approx(7.)
    assert info['jacobian_loss']==pytest.approx(8.)
    assert float(loss.detach())==pytest.approx(11.)
    assert info['moment_weight']==pytest.approx([.25,.75,0,0])
    assert info['jacobian_weight']==pytest.approx([1/3,1/6,0,0])
    assert info['jacobian_mask']==[True,True,False,False]
    loss.backward()
    assert predictions[0].jacobian.grad.abs().sum()>0
    assert predictions[1].jacobian.grad.abs().sum()>0
    assert predictions[2].jacobian.grad is None


def test_valid_zero_jacobian_is_a_label_not_a_missing_label():
    prediction=_prediction(0,0)
    loss,info=physical_loss([prediction],[_record()])
    assert info['jacobian_mask']==[True]
    assert info['jacobian_denominator']==1.
    assert info['jacobian_loss']==0.
    loss.backward()
    assert torch.equal(prediction.jacobian.grad,torch.zeros((10,3),dtype=torch.float64))


def test_no_qualified_j_has_zero_denominator_without_fabricated_targets():
    record=_record(qualified=False)
    del record['case']['native_jacobian']; del record['case']['parameter_scales']
    loss,info=physical_loss([_prediction(2,None)],[record])
    assert float(loss.detach())==4.
    assert info['jacobian_denominator']==0.
    assert info['jacobian_mask']==[False]
    assert info['jacobian_reasons']==['LABEL_UNQUALIFIED']


def test_sampler_visits_all_records_and_restores_cross_epoch_batches():
    sampler=StatefulShuffle(7,830251)
    first=sampler.next_batch(5)
    state=sampler.state_dict()
    next_values=sampler.next_batch(9)
    restored=StatefulShuffle(7,1); restored.load_state_dict(state)
    assert restored.next_batch(9)==next_values
    assert sorted(first+next_values[:2])==list(range(7))
    assert sorted(next_values[2:])==list(range(7))
    assert restored.state_dict()==sampler.state_dict()


def test_actual_gradscaler_skip_observed_from_optimizer_hook():
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.SGD([parameter],lr=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    scaler.scale((parameter*float('inf')).sum()).backward()
    assert not observed_optimizer_step(optimizer,scaler)
    assert parameter.item()==1.
    assert scaler.get_scale()==4.
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square().sum()).backward()
    assert observed_optimizer_step(optimizer,scaler)
    assert parameter.item()==pytest.approx(.8)


def _metadata_manifest(tmp_path,*,duplicate='source',forbidden=False):
    rows=[]
    for i,role in enumerate(('train','development')):
        meta=dict(split='synthetic',reference_kind='analytic_fixture',geometry_id=f'g{i}',query_id=f'q{i}',
                  source_group=f's{i}',geometry_sha256=str(i)*64,derivative_valid=False,
                  derivative_dimension=0,chart_id='none')
        if duplicate=='source': meta['source_group']='same'
        if duplicate=='geometry': meta['geometry_id']='same'
        if duplicate=='hash': meta['geometry_sha256']='a'*64
        if forbidden and i==1: meta['split']='fixed_benchmark'
        path=tmp_path/f'case{i}.json'; path.write_text(json.dumps(dict(meta=meta)))
        rows.append(dict(record_id=str(i),role=role,loader_eligible=True,
                         case_path=path.name,case_sha256=sha256(path)))
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema=SCHEMA,snapshot_id='metadata-check',registered_train_count=1,records=rows)))
    return path


@pytest.mark.parametrize('duplicate',['source','geometry','hash'])
def test_split_leakage_rejected_before_any_target_read(tmp_path,monkeypatch,duplicate):
    path=_metadata_manifest(tmp_path,duplicate=duplicate)
    def forbidden_read(*args,**kwargs):
        raise AssertionError('target read before whole-manifest split checks')
    monkeypatch.setattr(manifest_module,'load_baseline_case',forbidden_read)
    with pytest.raises(ValueError,match='must be disjoint'):
        load_training_manifest(path,mode='smoke')


def test_locked_benchmark_refused_before_target_read(tmp_path,monkeypatch):
    path=_metadata_manifest(tmp_path,duplicate=None,forbidden=True)
    monkeypatch.setattr(manifest_module,'load_baseline_case',lambda *a,**k:pytest.fail('labels opened'))
    with pytest.raises(ValueError,match='test/benchmark'):
        load_training_manifest(path,mode='smoke')


def test_export_receipt_hash_is_checked(tmp_path):
    path=_metadata_manifest(tmp_path,duplicate=None)
    value=json.loads(path.read_text()); value['export_receipt_sha256']='0'*64
    path.write_text(json.dumps(value))
    (tmp_path/'export_receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='export receipt SHA256'):
        load_training_manifest(path,mode='smoke')


def test_real_manifest_without_frozen_admission_refused_before_labels(tmp_path,monkeypatch):
    path=_metadata_manifest(tmp_path,duplicate=None)
    monkeypatch.setattr(manifest_module,'load_baseline_case',lambda *a,**k:pytest.fail('labels opened'))
    with pytest.raises(ValueError,match='source_authority'):
        load_training_manifest(path,mode='trial')


def _authorization(tmp_path,mode='trial'):
    value=dict(schema=AUTH_SCHEMA,mode=mode,training_authorized=True,authorized_by='software test fixture',
        training_manifest_sha256='a'*64,config_sha256='b'*64,max_optimizer_steps=4,max_attempts=8,
        max_wall_seconds=120.,evaluation_protocol_sha256='c'*64,
        protocol=dict(correctness_gates={key:'PASS' for key in (
            'teacher_and_operand_identity','source_split_isolation','material_ten_moments',
            'derivative_qualification_masks','finite_joint_update_and_resume')},
            screening_thresholds=dict(primary_improvement_fraction=.05)))
    path=tmp_path/'authorization.json'; path.write_text(json.dumps(value))
    return path,value


def _admit(**kwargs):
    return authorize(manifest_sha256='a'*64,config_sha256='b'*64,max_steps=4,max_attempts=8,**kwargs)


def test_bounded_trial_needs_explicit_authorization_but_not_final_model_physics(tmp_path):
    with pytest.raises(ValueError,match='explicit budget'):
        _admit(mode='trial')
    path,_=_authorization(tmp_path)
    result=_admit(mode='trial',authorization_path=path)
    assert result['max_wall_seconds']==120.
    assert 'screen_report' not in result


@pytest.mark.parametrize('wall',[None,0,float('inf'),6*3600+1])
def test_real_trial_requires_finite_bounded_wall_budget(tmp_path,wall):
    path,value=_authorization(tmp_path); value['max_wall_seconds']=wall
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='wall|six hours'):
        _admit(mode='trial',authorization_path=path)


def test_long_requires_development_go_bound_to_same_candidate_and_protocol(tmp_path):
    path,value=_authorization(tmp_path,'long')
    report=dict(schema='tvcg.v22.research-screen.v1',mode='research_screen',decision='GO',scope='development',
        eligible_for_long_training_review=True,training_authorized=False,
        identity=dict(training_manifest_sha256='a'*64,config_sha256='b'*64,
                      candidate_checkpoint_sha256='d'*64,protocol_sha256='c'*64))
    report_path=tmp_path/'screen.json'; report_path.write_text(json.dumps(report))
    value['screen_report']=dict(path=report_path.name,sha256=sha256(report_path))
    path.write_text(json.dumps(value))
    assert _admit(mode='long',authorization_path=path,resume_checkpoint_sha256='d'*64)['mode']=='long'
    with pytest.raises(ValueError,match='identity mismatch'):
        _admit(mode='long',authorization_path=path,resume_checkpoint_sha256='e'*64)
    report['scope']='synthetic'; report_path.write_text(json.dumps(report))
    value['screen_report']['sha256']=sha256(report_path); path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='development research-screen'):
        _admit(mode='long',authorization_path=path,resume_checkpoint_sha256='d'*64)


def test_execution_budget_is_not_part_of_scientific_config():
    config=asdict(TrainConfig())
    assert config['token_grid']==16 and config['seed']==830251 and config['learning_rate']==3e-5
    assert not {'mode','max_steps','max_attempts','output'} & config.keys()
    assert canonical_digest(config)==canonical_digest(dict(reversed(list(config.items()))))


def test_resume_rejects_checkpoint_from_other_exposure_lineage_before_journal_read(tmp_path):
    checkpoint=tmp_path/'state.pt'; checkpoint.write_bytes(b'explicit non-checkpoint test bytes')
    receipt=dict(schema=RESUME_SCHEMA,checkpoint=dict(path=checkpoint.name,sha256=sha256(checkpoint)),
                 exposure_receipt=dict(checkpoint_sha256='0'*64),journal_paths=[])
    path=tmp_path/'resume.json'; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='exposure lineage'):
        _read_resume(path)


def _saturated_manifest(directory):
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
    path=make_smoke_manifest(directory)
    raw=json.loads(path.read_text()); raw['records']=raw['records'][:1]; raw['registered_train_count']=1
    case_path=directory/raw['records'][0]['case_path']; spec=json.loads(case_path.read_text())
    operand_path=case_path.parent/'operand.npz'; operand=FiniteOperand.load(operand_path)
    vertices=operand.vertices.copy(); vertices[vertices[:,0]==.037,0]=.8
    moved=FiniteOperand(vertices,operand.faces,operand.face_is_cap,operand.operation,operand.vertex_velocity)
    moved.save(operand_path)
    coefficients=np.r_[-.8,1.,np.zeros(8)]
    spec['meta']['query_id']='fully_retained_synthetic'
    spec['meta']['operand_content_sha256']=operand_fingerprint(moved)
    spec['meta']['query_content_sha256']=query_fingerprint(coefficients)
    target_path=case_path.parent/'targets.npz'
    with np.load(target_path,allow_pickle=False) as data:
        values={key:data[key].copy() for key in data.files}
    values['native_moments']=values['fixed_moments']+values['native_full_editable_moments']
    values['native_jacobian']=np.zeros((10,3)); values['coefficients']=coefficients
    values['metadata']=np.asarray(json.dumps({key:spec['meta'][key] for key in LABEL_IDENTITY_FIELDS}))
    np.savez(target_path,**values)
    spec['assets']['operand']['sha256']=sha256(operand_path)
    spec['assets']['targets']['sha256']=sha256(target_path)
    case_path.write_text(json.dumps(spec))
    raw['records'][0]['case_sha256']=sha256(case_path); path.write_text(json.dumps(raw))
    return path


def test_actual_fully_retained_case_consumes_attempt_without_fake_update(tmp_path):
    manifest=_saturated_manifest(tmp_path/'assets')
    data=load_training_manifest(manifest,mode='smoke',cache_directory=tmp_path/'cache')
    trainer=Trainer(data,tmp_path/'run',max_steps=1,max_attempts=1)
    try:
        result=trainer.step()
        assert result['skip_reason']=='NO_PARAMETER_DEPENDENCE'
        assert not result['committed'] and not result['amp_skip']
        assert result['attempt']==1 and result['global_step']==0
        assert result['jacobian_mask']==[True]
        assert result['jacobian_loss']==0.
        assert trainer.optimizer.state=={}
        checkpoint=trainer.save_checkpoint()
        weights=Trainer(data,tmp_path/'fresh',max_steps=1,max_attempts=1,
                        initial_weights=checkpoint['path'],initial_weights_sha256=checkpoint['sha256'])
        try:
            assert weights.global_step==weights.attempts==0
            assert weights.optimizer.state=={}
            assert weights.sampler.cursor==0
            assert weights.run_id!=trainer.run_id
            assert all(torch.equal(value,weights.model.state_dict()[name])
                       for name,value in trainer.model.state_dict().items())
            state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
            state['execution_sources']={'changed_loss.py':'0'*64}
            with pytest.raises(ValueError,match='execution_sources'):
                weights._restore(state,None)
        finally: weights.close()
    finally: trainer.close()


def test_selection_uses_raw_frozen_metric_not_normalized_diagnostic(tmp_path):
    protocol_path=tmp_path/'protocol.json'; protocol_path.write_text('{}')
    trainer=object.__new__(Trainer)
    trainer.protocol=dict(primary_aggregation='source_macro_group_rmse')
    trainer.protocol_path=protocol_path; trainer.protocol_sha256=sha256(protocol_path)
    trainer.data=SimpleNamespace(development=[dict(record_id='a',case=dict(
        meta=dict(source_group='s',editable_mass=100.),native_moments=np.zeros(10)))])
    result=trainer._raw_selection_score(dict(rows=[dict(record_id='a',moments=[2.]*10)],
        normalized_source_macro_mean_query_rmse=.0001))
    assert result['value']==2.


def test_epoch0_is_baseline_not_candidate_and_equal_scores_keep_earliest(tmp_path):
    trainer=object.__new__(Trainer)
    trainer.output=tmp_path; trainer.global_step=0; trainer.attempts=0
    trainer.evaluation_steps=None
    trainer.protocol=dict(primary_aggregation='source_macro_group_rmse')
    trainer.evaluate_development=lambda:dict(rows=[])
    trainer.evaluate_no_learning=lambda:dict(rows=[])
    trainer._raw_selection_score=lambda evaluation:dict(value=1.)
    trainer._selection_evaluate(initialize=True)
    assert trainer.selection_state['best_checkpoint'] is None
    trainer.attempts=1
    trainer._selection_evaluate()
    assert trainer.selection_state['best_checkpoint'] is None
    assert trainer._export_screen()['reason']=='NO_POST_UPDATE_CANDIDATE'
    trainer.global_step=1; trainer.attempts=2
    trainer._raw_selection_score=lambda evaluation:dict(value=2.)
    trainer._selection_evaluate()
    assert trainer.selection_state['best_score']==2.  # worse than E0 but actual trained candidate
    assert trainer.selection_state['best_global_step']==1
    selected=trainer.selection_state['best_evaluation']
    trainer.global_step=2; trainer.attempts=3
    trainer._selection_evaluate()
    assert trainer.selection_state['best_evaluation']==selected


def test_frozen_candidate_schedule_refuses_unregistered_tail_and_is_extendable(tmp_path):
    assert _evaluation_steps(dict(evaluation_steps=[104,208,312,624]),312,required=True)==(104,208,312,624)
    with pytest.raises(ValueError,match='predeclared'):
        _evaluation_steps({},312,required=True)
    with pytest.raises(ValueError,match='increasing'):
        _evaluation_steps(dict(evaluation_steps=[208,104]),312,required=True)
    with pytest.raises(ValueError,match='at least one'):
        _evaluation_steps(dict(evaluation_steps=[104]),100,required=True)
    trainer=object.__new__(Trainer); trainer.evaluation_steps=(104,208,312); trainer.global_step=109
    with pytest.raises(ValueError,match='unregistered'):
        trainer._selection_evaluate()


def _manifest_with_distinct_development_geometry(directory):
    """Different closed cuboid, analytic labels; no renamed duplicate geometry."""
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_material
    from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import write_synthetic_finite_case
    manifest=make_smoke_manifest(directory)
    path=write_synthetic_finite_case(directory/'development')
    spec=json.loads(path.read_text()); meta=spec['meta']
    geometry_path=path.parent/'geometry.npz'
    with np.load(geometry_path,allow_pickle=False) as values:
        vertices,faces=values['vertices'].copy(),values['faces'].copy()
    vertices[vertices[:,1]==.23,1]=.31
    np.savez(geometry_path,vertices=vertices,faces=faces)
    material=prepare_material(vertices,faces,[-8.]*3,1/16,side=256,density=1.3)
    material_path=path.parent/'material.npz'; material.save(material_path)
    lo,hi=np.array([-.21,-.17,-.13]),np.array([.29,.31,.18]); cut=.037
    def integral(low,high):
        center=(low+high)/2; extent=high-low; x,y,z=center
        phi=np.array([1.,x,y,z,x*x,y*y,z*z,y*z,x*z,x*y]); phi[4:7]+=extent**2/12
        return 1.3*extent.prod()*phi
    full=integral(lo,hi); fixed=integral(lo-.1,hi+.1)-full
    retained=hi.copy(); retained[0]=cut
    def one(axis,power):
        return (hi[axis]**(power+1)-lo[axis]**(power+1))/(power+1)
    powers=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(0,1,1),(1,0,1),(1,1,0)]
    native_j=np.array([[cut**px*1.3*one(1,py+dy)*one(2,pz+dz)
                       for dy,dz in ((0,0),(1,0),(0,1))] for px,py,pz in powers])
    meta.update(geometry_id='different_synthetic_box',source_group='different_synthetic_analytic_group',
                geometry_sha256=sha256(geometry_path),material_identity=material.identity,editable_mass=float(full[0]))
    targets_path=path.parent/'targets.npz'
    with np.load(targets_path,allow_pickle=False) as values:
        targets={key:values[key].copy() for key in values.files}
    targets.update(native_moments=fixed+integral(lo,retained),fixed_moments=fixed,
                   native_full_editable_moments=full,native_jacobian=native_j,
                   metadata=np.asarray(json.dumps({key:meta[key] for key in LABEL_IDENTITY_FIELDS})))
    np.savez(targets_path,**targets)
    for key,asset in [('geometry',geometry_path),('material',material_path),('targets',targets_path)]:
        spec['assets'][key]['sha256']=sha256(asset)
    path.write_text(json.dumps(spec))
    raw=json.loads(manifest.read_text())
    raw['records'].append(dict(record_id='different-development-box',case_path=str(path.relative_to(directory)),
        case_sha256=sha256(path),role='development',loader_eligible=True,moment_weight=1.,jacobian_weight=1.,
        surface_family='PLANE',retention_bin='middle',trajectory_stage='synthetic_fixture'))
    manifest.write_text(json.dumps(raw))
    return manifest


def test_actual_run_exports_e0_best_last_and_frozen_screen(tmp_path):
    from experiments.compact_pino_v21.p0_diagnostics import promotion_gate
    manifest=_manifest_with_distinct_development_geometry(tmp_path/'assets')
    data=load_training_manifest(manifest,mode='smoke',cache_directory=tmp_path/'cache')
    registered=tmp_path/'registered.json'; registered.write_text('{"scope":"software-only"}')
    protocol=dict(schema=promotion_gate.PROTOCOL_SCHEMA,protocol_id='synthetic-software-run',
        registered_protocol_sha256=sha256(registered),primary_aggregation='source_macro_group_rmse',
        jacobian_aggregation='source_macro_group_rmse',primary_relative_improvement=.05,protection_max_ratio=1.05,
        protected_retention_metric='scaled10_moment_rmse',protected_jacobian_metric='scaled_jacobian_rmse',
        require_no_learning_baseline=True,expected_record_ids=['different-development-box'])
    protocol['evaluation_steps']=[1,2]
    protocol_path=tmp_path/'protocol.json'; protocol_path.write_text(json.dumps(protocol))
    trainer=Trainer(data,tmp_path/'run',max_steps=1,max_attempts=1,
                    evaluation_protocol_path=protocol_path,registered_protocol_path=registered)
    result=trainer.run(checkpoint_every=1,evaluate_every=1)
    assert result['global_step']==1 and result['segment_committed_updates']==1
    assert result['selection_status']=='FROZEN_RAW_SOURCE_MACRO'
    assert Path(result['epoch0_checkpoint']['path']).is_file()
    assert Path(result['best_checkpoint']['path']).is_file()
    assert (tmp_path/'run'/'last.json').is_file() and (tmp_path/'run'/'best.json').is_file()
    report=json.loads(Path(result['screen']['report_path']).read_text())
    assert report['scope']=='synthetic'
    assert report['decision']!='GO'  # no R8/Full-J/high-retention evidence in this fixture
    assert not report['eligible_for_long_training_review']
    state=torch.load(result['last_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state['evaluation_protocol_sha256']==sha256(protocol_path)
    assert state['selection']['primary_units']=='raw_undoubled_ten_moments'
