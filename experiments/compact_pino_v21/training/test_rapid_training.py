"""Finite exposure, frozen-parameter updates and checkpoint-specific selection."""
from __future__ import annotations

import json

import pytest
import torch

from experiments.compact_pino_v21.p0_diagnostics.baseline_cases import sha256
from experiments.compact_pino_v21.p0_diagnostics.exposure_audit import ExposureRecorder, read_events
from experiments.compact_pino_v21.training.manifest import load_training_manifest
from experiments.compact_pino_v21.training.precision_smoke import make_precision_manifest
from experiments.compact_pino_v21.training.runner import (
    TrainConfig, Trainer, _physical_exposure_at_checkpoint, _selection_steps, _early_stop_policy,
)
from experiments.compact_pino_v21.training.state import SCHEDULE_SCHEMA, ScheduleExhausted, load_exposure_schedule


def _schedule(path, ids, manifest_sha='a'*64):
    path.write_text(json.dumps(dict(schema=SCHEDULE_SCHEMA,
                                   training_manifest_sha256=manifest_sha, record_ids=ids)))
    return path, sha256(path)


def test_fixed_schedule_binds_manifest_ids_and_restores_without_cycling(tmp_path):
    path, digest = _schedule(tmp_path/'schedule.json', ['b', 'a', 'b', 'b'])
    first, receipt = load_exposure_schedule(path, digest, 'a'*64, ['a', 'b'])
    assert first.next_batch(2) == [1, 0]
    saved = first.state_dict()
    second, _ = load_exposure_schedule(path, digest, 'a'*64, ['a', 'b'])
    second.load_state_dict(saved)
    assert first.next_batch(2) == second.next_batch(2) == [1, 1]
    with pytest.raises(ScheduleExhausted, match='no partial batch'):
        second.next_batch(1)
    assert second.cursor == 4 and second.epoch == 0
    assert receipt['occurrences'] == 4 and receipt['unique_records'] == 2
    with pytest.raises(ValueError, match='manifest identity'):
        load_exposure_schedule(path, digest, 'c'*64, ['a', 'b'])
    with pytest.raises(ValueError, match='admitted training'):
        load_exposure_schedule(path, digest, 'a'*64, ['a'])
    reordered, _ = load_exposure_schedule(path, digest, 'a'*64, ['b', 'a'])
    with pytest.raises(ValueError, match='foreign fixed'):
        reordered.load_state_dict(saved)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='SHA256'):
        load_exposure_schedule(path, digest, 'a'*64, ['a', 'b'])


def test_schedule_batch_failure_does_not_consume_partial_tail(tmp_path):
    path, digest = _schedule(tmp_path/'schedule.json', ['a', 'b', 'a'])
    sampler, _ = load_exposure_schedule(path, digest, 'a'*64, ['a', 'b'])
    sampler.next_batch(2)
    with pytest.raises(ScheduleExhausted): sampler.next_batch(2)
    assert sampler.remaining == 1 and sampler.next_batch(1) == [0]


@pytest.mark.parametrize('config', [
    TrainConfig(sampler='fixed_schedule'), TrainConfig(exposure_schedule_sha256='a'*64),
    TrainConfig(training_scope='response_and_encoder'), TrainConfig(response_type='unknown'),
    TrainConfig(response_zero_init=1), TrainConfig(frozen_feature_cache_bytes=-1),
])
def test_new_config_rejects_implicit_or_inconsistent_scope(config):
    with pytest.raises(ValueError): config.validate()


def test_monitor_checkpoint_cannot_win_even_with_best_score(tmp_path):
    assert _selection_steps({'selection_steps': [52, 78, 104, 500]}, (26, 52, 78, 104, 500)) == (52, 78, 104, 500)
    with pytest.raises(ValueError, match='subset'):
        _selection_steps({'selection_steps': [25]}, (26, 52))
    trainer = object.__new__(Trainer)
    trainer.output = tmp_path; trainer.global_step = trainer.attempts = 0
    trainer.evaluation_steps = (1, 2, 3); trainer.selection_steps = (2, 3)
    trainer.protocol = dict(primary_aggregation='source_macro_group_rmse')
    trainer.evaluate_development = lambda: dict(rows=[])
    trainer.evaluate_no_learning = lambda: dict(rows=[])
    trainer._raw_selection_score = lambda _: dict(value=1.)
    trainer._selection_evaluate(initialize=True)
    trainer.global_step = trainer.attempts = 1
    trainer._raw_selection_score = lambda _: dict(value=.01)
    trainer._selection_evaluate()
    assert trainer.selection_state['best_checkpoint'] is None
    observed = json.loads((tmp_path/'development-step-1-attempt-1.json').read_text())
    assert observed['selection_qualification']['reason'] == 'MONITOR_ONLY_STEP'
    trainer.global_step = trainer.attempts = 2
    trainer._raw_selection_score = lambda _: dict(value=.5)
    trainer._selection_evaluate()
    assert trainer.selection_state['best_score'] == .5


def _inventory():
    return dict(snapshot_id='rapid-test', manifest_sha256='a'*64, registered_count=2,
        records=[dict(record_id=rid, query_id=rid, geometry_id='g', source_group_id='s',
                      moment_valid=True, jacobian_valid=False, residual_valid=True,
                      loader_eligible=True, derivative_type=None, derivative_dimension=None, chart_id='none')
                 for rid in ['a', 'b']])


def _commit(recorder, batch, rid):
    recorder.forward(batch, batch, [rid])
    recorder.loss(batch, moment_mask=[True], moment_weight=[1.],
                  jacobian_mask=[False], jacobian_weight=[0.],
                  residual_mask=[True], residual_weight=[.1],
                  constraint_mask=[True], constraint_weight=[.1])
    recorder.backward(batch, success=True)
    recorder.optimizer_step(batch, committed=True)


def _checkpoint(recorder, root, name, step, digest):
    event = recorder.checkpoint(name, digest, optimizer_step=step)
    path = root/(name+'.json')
    path.write_text(json.dumps(dict(checkpoint=dict(sha256=digest), exposure_receipt=event)))
    return dict(sha256=digest, receipt_path=str(path)), event


def test_per_record_requirement_rejects_borrowing_later_or_rolled_back_exposure(tmp_path):
    journal = tmp_path/'first.jsonl'
    recorder = ExposureRecorder(journal, inventory=_inventory(), run_id='same-run')
    _commit(recorder, 'a1', 'a')
    _commit(recorder, 'b1', 'b')
    first, parent = _checkpoint(recorder, tmp_path, 'first', 2, 'b'*64)
    _commit(recorder, 'a2', 'a')
    _commit(recorder, 'b2', 'b')
    late, _ = _checkpoint(recorder, tmp_path, 'late', 4, 'c'*64)
    recorder.close()
    branch_journal = tmp_path/'branch.jsonl'
    branch = ExposureRecorder(branch_journal, inventory=_inventory(), run_id='same-run', resume_from=parent)
    _commit(branch, 'branch-a2', 'a')
    rolled, _ = _checkpoint(branch, tmp_path, 'rolled', 3, 'd'*64)
    branch.close()
    events = read_events([journal, branch_journal])
    rule = dict(record_ids=['a', 'b'], minimum_per_record=2, objectives=['residual', 'constraint'])
    config = TrainConfig(residual_loss_weight=.1, constraint_loss_weight=.1)
    early = _physical_exposure_at_checkpoint(events, first, config, rule)['minimum_physics_exposure']
    complete = _physical_exposure_at_checkpoint(events, late, config, rule)['minimum_physics_exposure']
    rollback = _physical_exposure_at_checkpoint(events, rolled, config, rule)['minimum_physics_exposure']
    assert not early['satisfied'] and complete['satisfied'] and not rollback['satisfied']
    assert rollback['per_record']['a']['residual'] == 2
    assert rollback['per_record']['b']['residual'] == 1


def test_early_stop_uses_original_high_metric_and_needs_consecutive_available_checks():
    trainer = object.__new__(Trainer)
    trainer.early_stop_rule = _early_stop_policy(dict(early_stop=dict(
        consecutive_primary_and_high_regressions=2, after_step=52)))
    trainer.evaluation_steps = (26, 52, 78, 104, 500)
    trainer.selection_state = dict(epoch0_score=1.)
    trainer.global_step = trainer.attempts = 0
    trainer._protected_high_score = lambda _: dict(value=1., queries=2)
    trainer._update_early_stop({}, dict(value=1.), initialize=True)
    for step in (26, 52):
        trainer.global_step = trainer.attempts = step
        trainer._protected_high_score = lambda _: dict(value=1.051, queries=2)
        state = trainer._update_early_stop({}, dict(value=1.01), initialize=False)
    assert state['consecutive'] == 1 and not state['stop_requested']
    trainer.global_step = trainer.attempts = 78
    trainer._protected_high_score = lambda _: dict(value=None, queries=0)
    state = trainer._update_early_stop({}, dict(value=1.01), initialize=False)
    assert state['consecutive'] == 0 and not state['stop_requested']
    for step in (104, 500):
        trainer.global_step = trainer.attempts = step
        trainer._protected_high_score = lambda _: dict(value=1.051, queries=2)
        state = trainer._update_early_stop({}, dict(value=1.01), initialize=False)
    assert state['consecutive'] == 2 and state['stop_requested']
    assert state['checks'][2]['high_status'] == 'INSUFFICIENT_EVIDENCE'


@pytest.fixture(scope='module')
def rapid_data(tmp_path_factory):
    root = tmp_path_factory.mktemp('rapid-real-updates')
    manifest = make_precision_manifest(root/'assets', cache_directory=root/'cache')
    return root, load_training_manifest(manifest, mode='smoke', cache_directory=root/'cache')


def test_actual_response_only_update_resume_and_frozen_loader(tmp_path, rapid_data):
    from experiments.compact_pino_v21.precision.frozen_checkpoint import load_frozen_checkpoint
    _, data = rapid_data
    ids = [row['record_id'] for row in data.train]
    schedule, digest = _schedule(tmp_path/'schedule.json', ids*2, data.manifest_sha256)
    config = TrainConfig(response_type='monotone_warp', response_zero_init=True, training_scope='response',
        sampler='fixed_schedule', exposure_schedule_sha256=digest,
        residual_loss_weight=.1, constraint_loss_weight=.1)
    trainer = Trainer(data, tmp_path/'first', config=config, exposure_schedule=schedule,
                      max_steps=2, max_attempts=2)
    resumed = None
    try:
        initial_frozen = trainer.frozen_state_sha256
        optimized = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group['params']}
        assert optimized == {id(parameter) for parameter in trainer.model.response.parameters()}
        zero = trainer._predictions([data.train[0]])[0]
        assert torch.isfinite(zero.moments).all() and torch.isfinite(zero.jacobian).all()
        first = trainer.step(); checkpoint = trainer.save_checkpoint()
        second = trainer.step(); target_state = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
        trainer.save_checkpoint()
        assert first['committed'] and second['committed'] and trainer.global_step == 2
        assert first['cuda_memory_bytes'] is None
        assert trainer._frozen_digest() == initial_frozen
        trainer.close()
        resumed = Trainer(data, tmp_path/'resumed', config=config, exposure_schedule=schedule,
                          resume=checkpoint['receipt_path'], max_steps=2, max_attempts=2)
        replay = resumed.step()
        assert replay['record_ids'] == second['record_ids'] and replay['loss'] == second['loss']
        assert all(torch.equal(value, resumed.model.state_dict()[key]) for key, value in target_state.items())
        final = resumed.save_checkpoint()
        stats = resumed.model.efficiency_statistics()
        assert stats['frozen_features']['hits'] >= 0
        frozen = load_frozen_checkpoint(final['path'], final['sha256'])
        try:
            assert frozen.config.response_type == 'monotone_warp'
            encoded, _ = frozen.encode(data.train[0]['case'])
            predicted = frozen.predict(encoded, data.train[0]['case'], 'tetra4', 65536)
            assert torch.isfinite(predicted.moments).all()
        finally:
            frozen.close()
    finally:
        trainer.close()
        if resumed is not None: resumed.close()


def test_explicit_head_migration_resets_only_output_and_rejects_full_semantic_load(tmp_path, rapid_data):
    _, data = rapid_data
    source = Trainer(data, tmp_path/'source', max_steps=1, max_attempts=1)
    try:
        checkpoint = source.save_checkpoint()
        state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['model']
        config = TrainConfig(response_type='monotone_warp', response_zero_init=True, training_scope='response')
        with pytest.raises(ValueError, match='response semantics differ'):
            Trainer(data, tmp_path/'invalid', config=config, initial_weights=checkpoint['path'],
                    initial_weights_sha256=checkpoint['sha256'], max_steps=1, max_attempts=1)
        target = Trainer(data, tmp_path/'target', config=config, initial_weights=checkpoint['path'],
                         initial_weights_sha256=checkpoint['sha256'], initial_weights_scope='geometry_and_hidden',
                         max_steps=1, max_attempts=1)
        try:
            for key, value in target.model.state_dict().items():
                if key in ('response.local.2.weight', 'response.local.2.bias'):
                    assert torch.count_nonzero(value) == 0
                else:
                    assert torch.equal(value, state[key])
            assert target.optimizer.state == {}
            assert target.initial_weights['scope'] == 'geometry_and_hidden'
        finally: target.close()
    finally: source.close()


def test_run_reports_finite_schedule_exhaustion_without_fake_updates(tmp_path, rapid_data):
    _, data = rapid_data
    path, digest = _schedule(tmp_path/'schedule.json', [data.train[0]['record_id']], data.manifest_sha256)
    config = TrainConfig(training_scope='response', sampler='fixed_schedule', exposure_schedule_sha256=digest)
    trainer = Trainer(data, tmp_path/'run', config=config, exposure_schedule=path, max_steps=2, max_attempts=2)
    result = trainer.run(checkpoint_every=1)
    assert result['global_step'] == result['attempts'] == 1
    assert result['stop_reason'] == 'EXPOSURE_SCHEDULE_EXHAUSTED'
    assert result['physical_supervision']['minimum_physics_exposure']['satisfied']
