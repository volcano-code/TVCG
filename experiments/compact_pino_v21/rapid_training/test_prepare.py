from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.compact_pino_v21.rapid_training.prepare import (COHORT_SCHEMA, STAGES, build_schedule, file_receipt,
                      native_retention, prepare_pair, verify_cohort)
from experiments.compact_pino_v21.rapid_training.authorize import bind_authorization, GATES, _verify_software_smoke
from experiments.compact_pino_v21.rapid_training.runtime import estimate
from experiments.compact_pino_v21.p0_diagnostics.promotion_gate import PROTOCOL_SCHEMA
from experiments.compact_pino_v21.training.runner import TrainConfig, _execution_sources, model_kwargs
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def _smoke_fixture(tmp_path, config):
    # Explicit synthetic unit receipt: no hardware or real dataset was tested.
    from experiments.compact_pino_v21.rapid_training import authorize as module
    return _write(tmp_path/'unit-smoke.json', dict(
        schema='compact_pino.rapid-response-software-smoke.v1', scope='unit_fixture_only',
        status='PASS', checks={'unit_fixture_only': True}, actual_device='cuda:0',
        real_optimizer_calls_executed=6,
        supported_model_profiles=[{key: getattr(config, key) for key in
            ('response_type', 'training_scope', 'response_zero_init', 'token_grid', 'full_cell_rule', 'amp')}],
        execution_sources=_execution_sources(CompactPINOv21(**model_kwargs(config))),
        smoke_source_sha256=file_receipt(Path(module.__file__).with_name('smoke.py'))['sha256']))


def _data():
    records = []
    for geometry in range(4):
        for stage, count in STAGES.items():
            for i in range(count):
                rid = f'g{geometry}-{stage}-{i}'
                case = dict(meta=dict(geometry_id=f'g{geometry}', source_group=f's{geometry}',
                                      derivative_valid=True),
                            native_moments=np.r_[.85 if stage == 'high' else .5, np.zeros(9)],
                            fixed_moments=np.zeros(10), native_full_editable_moments=np.r_[1., np.zeros(9)])
                records.append(dict(record_id=rid, case=case, case_sha256='d'*64,
                    moment_weight=1., jacobian_weight=1., residual_weight=1., residual_valid=True,
                    residual_target=dict(valid=True, multipliers=[1.]), trajectory_stage=stage))
    return SimpleNamespace(train=records, development=[dict(record_id='dev0')],
                           manifest_sha256='a'*64, scope='development_training')


def _cohort(tmp_path, data):
    task = file_receipt(_write(tmp_path/'task.json', {'scope': 'test_fixture_task'}))
    source = file_receipt(_write(tmp_path/'source.json', {'scope': 'test_fixture_checks'}))
    rows = []
    for row in data.train:
        rid = row['record_id']
        evidence = _write(tmp_path/(rid+'.json'), dict(schema='compact_pino.task-legality.v1',
            record_id=rid, case_sha256=row['case_sha256'], geometry_legal=True, pivot_retained=True,
            native_terminal_accepted=row['trajectory_stage'] == 'terminal',
            task_asset=task, original_checks=[source]))
        rows.append(dict(record_id=rid, trajectory_stage=row['trajectory_stage'],
                         legality_evidence=file_receipt(evidence)))
    return _write(tmp_path/'cohort.json', dict(schema=COHORT_SCHEMA,
        training_manifest_sha256=data.manifest_sha256, records=rows))


def test_cohort_checks_task_evidence_and_native_high_retention(tmp_path):
    data = _data(); path = _cohort(tmp_path, data)
    report = verify_cohort(data, path)
    assert report['unique_queries'] == 24 and len(report['source_groups']) == 4
    assert len(report['legality_receipts']) == 24
    row = next(row for row in data.train if row['trajectory_stage'] == 'high')
    row['case']['native_moments'][0] = .1
    with pytest.raises(ValueError, match='nontrivial native retention'):
        verify_cohort(data, path)


def test_cohort_refuses_misbound_evidence_and_development_ids(tmp_path):
    data = _data(); path = _cohort(tmp_path, data)
    raw = json.loads(path.read_text())
    raw['records'][0]['record_id'] = 'dev0'
    _write(path, raw)
    with pytest.raises(ValueError, match='admitted train'):
        verify_cohort(data, path)
    raw['records'][0]['record_id'] = data.train[0]['record_id']
    raw['records'][0]['legality_evidence']['sha256'] = '0'*64
    _write(path, raw)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify_cohort(data, path)


def test_schedule_prefix_is_bounded_and_long_tail_is_reproducible():
    data = _data(); ids = [row['record_id'] for row in data.train]
    schedule = build_schedule(data, ids, total_steps=10000)
    assert schedule == build_schedule(data, ids, total_steps=10000)
    assert len(schedule['record_ids']) == 10000
    assert set(schedule['record_ids']) == set(ids)
    assert all(Counter(schedule['record_ids'][:52])[rid] >= 2 for rid in ids)
    assert schedule['report']['first104_high_occurrences'] >= 26
    assert len(set(schedule['record_ids'][:26])) == 24
    with pytest.raises(ValueError, match='admitted train'):
        build_schedule(data, ids+['dev0'])


def test_prepare_binds_full_schedule_and_rejects_mismatched_calibration(tmp_path, monkeypatch):
    from experiments.compact_pino_v21.rapid_training import prepare as module
    data = _data()
    manifest = _write(tmp_path/'manifest.json', {'scope': 'explicit_unit_fixture'})
    data.manifest_sha256 = file_receipt(manifest)['sha256']
    cohort = _cohort(tmp_path, data)
    registered = _write(tmp_path/'registered.json', {'scope': 'frozen_unit_protocol'})
    protocol = _write(tmp_path/'old_protocol.json', dict(schema=PROTOCOL_SCHEMA, protocol_id='fixture',
        registered_protocol_sha256=file_receipt(registered)['sha256'], expected_record_ids=['dev0'],
        primary_aggregation='source_macro_group_rmse', jacobian_aggregation='source_macro_group_rmse',
        primary_relative_improvement=.05, protection_max_ratio=1.05,
        protected_retention_metric='scaled10_moment_rmse', protected_jacobian_metric='scaled_jacobian_rmse',
        require_no_learning_baseline=True))
    calibration = _write(tmp_path/'calibration.json', dict(schema='compact_pino.rapid-calibration.v1',
        training_manifest_sha256=data.manifest_sha256, cohort_sha256=file_receipt(cohort)['sha256'],
        full_cell_rule='gl2', training_scope='response', response_zero_init=True, seed=830251,
        status='CALIBRATED_NOT_ACCEPTANCE', parameter_updates=0, initial_weights_sha256=None,
        weights=dict(jacobian_loss_weight=1., residual_loss_weight=.2, constraint_loss_weight=.3)))
    monkeypatch.setattr(module, 'load_training_manifest', lambda *args, **kwargs: data)
    output = tmp_path/'paired'
    bundle = prepare_pair(manifest, protocol, registered, cohort, calibration, output)
    a = json.loads(Path(bundle['configs']['A_old_response']['path']).read_text())
    b = json.loads(Path(bundle['configs']['B_monotone_response']['path']).read_text())
    assert a.pop('response_type') == 'bounded_cubic'
    assert b.pop('response_type') == 'monotone_warp'
    assert a == b
    assert a['exposure_schedule_sha256'] == bundle['schedule']['sha256']
    new_protocol = json.loads((output/'protocol.json').read_text())
    assert 26 in new_protocol['evaluation_steps'] and 26 not in new_protocol['selection_steps']
    assert new_protocol['selection_steps'][-1] == 10000
    assert new_protocol['selection_minimum_physics_exposure']['minimum_per_record'] == 2
    checks = _write(tmp_path/'correctness.json', dict(schema='compact_pino.rapid-correctness.v1',
        training_manifest_sha256=data.manifest_sha256,
        checks={key: dict(status='PASS', evidence=[file_receipt(
            _smoke_fixture(tmp_path, TrainConfig(**json.loads(Path(bundle['configs']['A_old_response']['path']).read_text())))
            if key == 'finite_joint_update_and_resume' else registered)]) for key in GATES}))
    authorization = bind_authorization(output/'bundle.json', 'A_old_response', checks,
        tmp_path/'authorization.json', authorized_by='explicit software test fixture')
    assert authorization['max_wall_seconds'] == 21600
    from experiments.compact_pino_v21.rapid_training.run import load_bundle
    mislabeled = copy.deepcopy(bundle)
    mislabeled['configs']['B_monotone_response'] = mislabeled['configs']['A_old_response']
    mislabeled_path = _write(output/'mislabeled_bundle.json', mislabeled)
    with pytest.raises(ValueError, match='exact old/new response type'):
        load_bundle(mislabeled_path, 'B_monotone_response')
    broken = json.loads(checks.read_text()); broken['checks'][GATES[0]]['status'] = 'UNAVAILABLE'
    _write(checks, broken)
    with pytest.raises(ValueError, match='external PASS'):
        bind_authorization(output/'bundle.json', 'A_old_response', checks,
            tmp_path/'refused.json', authorized_by='explicit software test fixture')
    values = json.loads(calibration.read_text()); values['full_cell_rule'] = 'tetra4'; _write(calibration, values)
    with pytest.raises(ValueError, match='calibration must match'):
        prepare_pair(manifest, protocol, registered, cohort, calibration, tmp_path/'refused-pair')


@pytest.mark.parametrize('mismatch', ['cpu', 'old_head', 'stale_sources', 'stale_smoke', 'checks_fail'])
def test_software_smoke_cannot_be_borrowed_from_another_profile_or_runtime(tmp_path, mismatch):
    config = TrainConfig(response_type='monotone_warp', training_scope='response', response_zero_init=True)
    path = _smoke_fixture(tmp_path, config)
    report = json.loads(path.read_text())
    if mismatch == 'cpu': report['actual_device'] = 'cpu'
    elif mismatch == 'old_head': report['supported_model_profiles'][0]['response_type'] = 'bounded_cubic'
    elif mismatch == 'stale_sources': report['execution_sources'].pop(next(iter(report['execution_sources'])))
    elif mismatch == 'stale_smoke': report['smoke_source_sha256'] = '0'*64
    else: report['checks']['unit_fixture_only'] = False
    _write(path, report)
    with pytest.raises(ValueError, match='CUDA with this model profile'):
        _verify_software_smoke([file_receipt(path)], config)


def test_runtime_projection_uses_measurements_and_reports_omissions(tmp_path):
    rows = [dict(committed=True, global_step=i+1, seconds=float(i+1), cuda_memory_bytes=None) for i in range(6)]
    (tmp_path/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    _write(tmp_path/'development-step-6-attempt-6.json', dict(evaluation_seconds=10.))
    protocol = _write(tmp_path/'protocol.json', dict(evaluation_steps=[6, 10, 20]))
    report = estimate(tmp_path, total_steps=20, warmup=3, protocol=protocol)
    assert report['step_seconds']['p50'] == 5.
    assert report['remaining_scheduled_evaluations'] == 2
    assert report['estimated_remaining_evaluation_hours'] == pytest.approx(20/3600)
    assert report['peak_measured_cuda_allocated_bytes'] is None
