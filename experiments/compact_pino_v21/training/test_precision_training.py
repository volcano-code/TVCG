"""Bindings and exposure semantics for optional physical supervision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.compact_pino_v21.training.attach_residual_targets import REQUEST_SCHEMA, attach_targets
from experiments.compact_pino_v21.training.manifest import load_training_manifest
from experiments.compact_pino_v21.training.precision_smoke import make_precision_manifest, fixture_target
from experiments.compact_pino_v21.training.runner import TrainConfig, Trainer, _physical_exposure_at_checkpoint
from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import sha256
from experiments.compact_pino_v21.p0_diagnostics.exposure_audit import ExposureRecorder, AuditError, read_events, summarize_events


@pytest.fixture(scope='module')
def bound_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('precision_binding')
    manifest = make_precision_manifest(root/'assets', cache_directory=root/'cache')
    return root, manifest


def test_attachment_keeps_original_labels_and_all_record_identities(bound_fixture):
    root, manifest = bound_fixture
    base = json.loads((root/'assets'/'training_manifest.json').read_text())
    attached = json.loads(manifest.read_text())
    data = load_training_manifest(manifest, mode='smoke', cache_directory=root/'cache')
    assert attached['residual_attachment']['source_manifest_sha256'] == sha256(root/'assets'/'training_manifest.json')
    assert [row['record_id'] for row in base['records']] == [row['record_id'] for row in attached['records']]
    assert [row['case_sha256'] for row in base['records']] == [row['case_sha256'] for row in attached['records']]
    assert len(data.train) == 2 and all(row['residual_valid'] for row in data.train)
    assert all(row['residual_valid'] for row in data.inventory['records'])
    for row in data.train:
        assert row['residual_target']['binding']['case_sha256'] == row['case_sha256']
        assert row['case']['meta']['split'] == 'synthetic'


@pytest.mark.parametrize('mutation,match', [
    ('binding', 'exact case manifest'), ('native', 'reference_moments'),
    ('role', 'cannot be interchanged'), ('p', 'unit dual')])
def test_target_binding_rejected_before_model_construction(tmp_path, bound_fixture, mutation, match):
    root, manifest = bound_fixture
    spec = json.loads(manifest.read_text())
    target = spec['records'][0]['residual_target']
    if mutation == 'binding': target['binding']['case_sha256'] = 'e'*64
    if mutation == 'native': target['reference_moments'][0] *= 1.01
    if mutation == 'role': target['teacher_role'] = 'native_teacher'
    if mutation == 'p': target['p'][0] += .01
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        load_training_manifest(path, mode='smoke', cache_directory=root/'cache')


def test_sidecar_is_hashed_and_explicitly_unavailable_rows_remain_moment_only(tmp_path, bound_fixture):
    root, manifest = bound_fixture
    spec = json.loads(manifest.read_text())
    row = spec['records'][0]
    sidecar = tmp_path/'target.json'
    sidecar.write_text(json.dumps(row.pop('residual_target')))
    row['residual_target_asset'] = dict(path=sidecar.name, sha256=sha256(sidecar))
    spec['records'][1]['residual_target'] = dict(valid=False, reason='missing native multipliers')
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(spec))
    data = load_training_manifest(path, mode='smoke', cache_directory=root/'cache')
    assert [row['residual_valid'] for row in data.train] == [True, False]
    sidecar.write_text(sidecar.read_text()+' ')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_training_manifest(path, mode='smoke', cache_directory=root/'cache')


def test_enabled_physics_without_targets_fails_before_output_or_model(tmp_path, bound_fixture):
    root, _ = bound_fixture
    data = load_training_manifest(root/'assets'/'training_manifest.json', mode='smoke', cache_directory=root/'cache')
    with pytest.raises(ValueError, match='no admitted positive-weight training targets'):
        Trainer(data, tmp_path/'run', config=TrainConfig(residual_loss_weight=1.), max_steps=1, max_attempts=1)
    assert not (tmp_path/'run').exists()


def test_attachment_rejects_cross_record_binding_and_never_overwrites(tmp_path, bound_fixture):
    root, manifest = bound_fixture
    data = load_training_manifest(manifest, mode='smoke', cache_directory=root/'cache')
    target = fixture_target(data.train[0]['case'])
    target['binding'] = dict(case_sha256=data.train[1]['case_sha256'])
    request = tmp_path/'request.json'
    request.write_text(json.dumps(dict(schema=REQUEST_SCHEMA, records=[dict(record_id='fixture-0', target=target)])))
    with pytest.raises(ValueError, match='binding contradicts'):
        attach_targets(manifest, request, tmp_path/'out.json', mode='smoke', cache_directory=root/'cache')
    assert not (tmp_path/'out.json').exists()
    with pytest.raises(FileExistsError, match='overwrite'):
        attach_targets(manifest, request, manifest, mode='smoke', cache_directory=root/'cache')


@pytest.mark.parametrize('rule', ['fit_compressed', 'sweep_compressed', 'sweep_raw'])
def test_export_binds_partial_rule_without_changing_teacher_labels(tmp_path, bound_fixture, rule):
    from experiments.compact_pino_v21.p0_diagnostics.export_cases import export_cases
    from experiments.compact_pino_v21.p0_diagnostics.test_export_cases import request
    root, _ = bound_fixture
    source = root/'assets'/'case0'/'case.json'
    request_path, payload = request(tmp_path, source)
    payload['records'][0]['grid']['partial_rule'] = rule
    request_path.write_text(json.dumps(payload))
    exported = export_cases(request_path, tmp_path/'export', synthetic=True)
    data = load_training_manifest(exported, mode='smoke')
    case = data.train[0]['case']
    assert case['blocks'].material.partial_rule == rule
    with np.load(source.parent/'targets.npz', allow_pickle=False) as original:
        assert np.array_equal(case['native_moments'], original['native_moments'])
        assert np.array_equal(case['native_jacobian'], original['native_jacobian'])
    receipt = json.loads((exported.parent/'export_receipt.json').read_text())
    assert receipt['records'][0]['partial_cell_rule'] == rule


@pytest.mark.parametrize('config', [TrainConfig(full_cell_rule='unknown'),
    TrainConfig(residual_loss_weight=float('nan')), TrainConfig(constraint_loss_weight=-1.),
    TrainConfig(residual_loss_weight=True)])
def test_invalid_new_config_rejected(config):
    with pytest.raises(ValueError): config.validate()


def test_checkpoint_exposure_cannot_be_borrowed_from_later_updates(tmp_path):
    inventory = dict(snapshot_id='fixture', manifest_sha256='a'*64, registered_count=1,
        records=[dict(record_id='r', query_id='q', geometry_id='g', source_group_id='s',
                      moment_valid=True, jacobian_valid=False, residual_valid=True,
                      loader_eligible=True, derivative_type=None, derivative_dimension=None, chart_id='none')])
    journal = tmp_path/'exposure.jsonl'
    recorder = ExposureRecorder(journal, inventory=inventory, run_id='physics-lineage')
    checkpoints = []
    for i in (0, 1):
        batch = str(i)
        recorder.forward(batch, batch, ['r'])
        recorder.loss(batch, moment_mask=[i == 0], moment_weight=[1. if i == 0 else 0.],
                      jacobian_mask=[False], jacobian_weight=[0.],
                      residual_mask=[i == 1], residual_weight=[1. if i == 1 else 0.],
                      constraint_mask=[False], constraint_weight=[0.])
        recorder.backward(batch, success=True)
        recorder.optimizer_step(batch, committed=True)
        digest = ('b' if i == 0 else 'c')*64
        event = recorder.checkpoint('c'+batch, digest, optimizer_step=i+1)
        receipt = tmp_path/('c'+batch+'.json')
        receipt.write_text(json.dumps(dict(checkpoint=dict(sha256=digest), exposure_receipt=event)))
        checkpoints.append(dict(sha256=digest, receipt_path=str(receipt)))
    recorder.close()
    events = read_events([journal])
    config = TrainConfig(residual_loss_weight=1.)
    first = _physical_exposure_at_checkpoint(events, checkpoints[0], config)
    last = _physical_exposure_at_checkpoint(events, checkpoints[1], config)
    assert not first['enabled_objectives_exercised'] and last['enabled_objectives_exercised']
    assert first['objectives']['residual']['committed_occurrences'] == 0
    assert last['objectives']['residual']['committed_occurrences'] == 1
    report = summarize_events(events, selected_checkpoint='c1')
    assert report['selected_lineage']['counts']['commits_without_supervision'] == 0
    receipt = json.loads(Path(checkpoints[0]['receipt_path']).read_text())
    receipt['exposure_receipt']['checkpoint_id'] = 'c1'
    Path(checkpoints[0]['receipt_path']).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='absent from the recorded'):
        _physical_exposure_at_checkpoint(events, checkpoints[0], config)


def test_unqualified_physics_cannot_claim_loss_exposure(tmp_path):
    inventory = dict(snapshot_id='fixture', manifest_sha256='a'*64, registered_count=1,
        records=[dict(record_id='r', query_id='q', geometry_id='g', source_group_id='s',
                      moment_valid=True, jacobian_valid=False, residual_valid=False,
                      loader_eligible=True, derivative_type=None, derivative_dimension=None, chart_id='none')])
    recorder = ExposureRecorder(tmp_path/'exposure.jsonl', inventory=inventory, run_id='bad-physics')
    recorder.forward('b', 'w', ['r'])
    with pytest.raises(AuditError, match='invalid residual target'):
        recorder.loss('b', moment_mask=[True], moment_weight=[1.], jacobian_mask=[False], jacobian_weight=[0.],
                      residual_mask=[True], residual_weight=[1.])
    recorder.close(interrupted=True)
