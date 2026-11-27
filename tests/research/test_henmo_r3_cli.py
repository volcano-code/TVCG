"""R3 commands preserve snapshot authority and recover committed work."""
from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.henmo_v1plus import cli
from experiments.henmo_v1plus.dataset import dataset_binding, load_henmo_dataset, model_contract
from experiments.henmo_v1plus.fixtures import synthetic_dataset
from experiments.henmo_v1plus.training import HENMOTrainer, HENMOTrainingConfig
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def fixture_data():
    data = synthetic_dataset()
    original = data.train[0]
    case = dict(original['case'])
    case['meta'] = dict(case['meta'], source_group='held-out-fixture-source',
                        geometry_id='held-out-cube', geometry_sha256='f' * 64)
    development = dict(original, case=case, role='development', record_id='held-out-query', case_sha256='a' * 64)
    data.development = [development]
    data.records = [*data.train, development]
    return data


@pytest.fixture
def source_bundle(tmp_path):
    data = fixture_data()
    config = HENMOConfig.cpu_smoke(response_zero_init=False)
    training = HENMOTrainingConfig(max_steps=3, max_attempts=6)
    with HENMOOperator(config).double() as model:
        trainer = HENMOTrainer(model, training, dataset_binding(data, model_contract(model)), records=data.train)
        assert trainer.step()['committed']
        checkpoint = tmp_path / 'source.pt'
        receipt = trainer.save_checkpoint(checkpoint)
        config_path = tmp_path / 'source_config.json'
        config_path.write_text(json.dumps(asdict(model.config)))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return SimpleNamespace(data=data, config=config, training=training, checkpoint=checkpoint,
                           receipt=receipt, config_path=config_path, state=state)


def native_loader_fixture(monkeypatch, bundle):
    calls = []
    def load(manifest, **kwargs):
        calls.append((manifest, kwargs))
        assert kwargs['allowed_native_resolutions'] == (16,)
        return bundle.data, {'missing': [], 'scope': 'synthetic_software_only'}
    # Only this test substitutes fixtures. The CLI continues calling the native
    # loader and never adds a synthetic/side16 production admission exception.
    monkeypatch.setattr(cli, 'load_henmo_dataset', load)
    return calls


def common_args(bundle, output):
    return ['--manifest', 'synthetic-test-only.json', '--config', str(bundle.config_path),
            '--checkpoint', str(bundle.checkpoint), '--checkpoint-sha256', bundle.receipt['sha256'],
            '--device', 'cpu', '--dtype', 'float64', '--threads', '2', '--output', str(output)]


@pytest.mark.parametrize('interrupt_at', [0, 1])
def test_committed_state_is_saved_before_interrupted_evaluation(tmp_path, monkeypatch, interrupt_at):
    data = fixture_data()
    config = HENMOConfig.cpu_smoke(response_zero_init=False)
    training = HENMOTrainingConfig(max_steps=2, max_attempts=4)
    output = tmp_path / 'interrupted'
    output.mkdir()
    with HENMOOperator(config).double() as model:
        trainer = HENMOTrainer(model, training, dataset_binding(data, model_contract(model)), records=data.train)
        expected_at_interrupt = {}
        def interrupted_evaluation(records):
            if trainer.step_count == interrupt_at:
                expected_at_interrupt.update({key: value.detach().clone() for key, value in model.state_dict().items()})
                assert (output / 'last.pt').is_file()
                saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
                assert saved['step_count'] == interrupt_at
                assert all(torch.equal(saved['model'][key], value) for key, value in expected_at_interrupt.items())
                if interrupt_at:
                    assert (output / f'step_{interrupt_at:06d}.pt').is_file()
                raise KeyboardInterrupt('interrupted full development evaluation')
            return {'complete': False, 'source_macro_raw_moment_rmse': None}
        monkeypatch.setattr(trainer, 'evaluate', interrupted_evaluation)
        args = SimpleNamespace(evaluate_every=1, save_every=1, dtype='float64')
        with pytest.raises(KeyboardInterrupt, match='development evaluation'):
            cli._run_training_session(trainer, data, args, output)
        expected_next = trainer.step()
        next_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with HENMOOperator(config).double() as restored:
            resumed = HENMOTrainer(restored, training, dataset_binding(data, model_contract(restored)), records=data.train)
            resumed.load_checkpoint(output / 'last.pt')
            assert resumed.step_count == interrupt_at
            assert all(torch.equal(restored.state_dict()[key], value) for key, value in expected_at_interrupt.items())
            replay = resumed.step()
            assert replay['record_ids'] == expected_next['record_ids']
            assert replay['coarse_grids'] == expected_next['coarse_grids']
            assert replay['loss'] == expected_next['loss']
            assert all(torch.equal(restored.state_dict()[key], value) for key, value in next_state.items())


def test_selection_uses_existing_identities_and_rejects_cross_split_or_duplicate_ids(tmp_path):
    data = fixture_data()
    path = tmp_path / 'selection.json'
    specification = dict(split='train', snapshot_id=data.snapshot_id, manifest_sha256=data.manifest_sha256,
                         record_ids=[data.train[1]['record_id'], data.train[0]['record_id']])
    path.write_text(json.dumps(specification))
    rows, receipt = cli._diagnostic_records(data, split='train', selection=path, limit=1)
    assert rows == [data.train[1]] and rows[0] is data.train[1]
    assert receipt['selection']['sha256'] == cli._sha(path)
    for field, bad in [('split', 'development'), ('snapshot_id', 'another-snapshot'),
                       ('manifest_sha256', '0' * 64), ('record_ids', ['missing']),
                       ('record_ids', [data.development[0]['record_id']]),
                       ('record_ids', [data.train[0]['record_id']] * 2)]:
        path.write_text(json.dumps(dict(specification, **{field: bad})))
        with pytest.raises(ValueError):
            cli._diagnostic_records(data, split='train', selection=path)
    with pytest.raises(ValueError, match='positive'):
        cli._diagnostic_records(data, split='train', limit=0)
    with pytest.raises(ValueError, match='train/development'):
        cli._diagnostic_records(data, split='test')


def test_diagnose_paths_runs_from_verified_checkpoint_without_rebinding(tmp_path, monkeypatch, source_bundle):
    bundle = source_bundle
    calls = native_loader_fixture(monkeypatch, bundle)
    before_digest = cli._sha(bundle.checkpoint)
    report_path = tmp_path / 'path_report.json'
    cli.main(['diagnose-paths', *common_args(bundle, report_path), '--split', 'train', '--limit', '1',
              '--coarse-grids', '2', '--diagnostic-nodes', '2'])
    report = json.loads(report_path.read_text())
    assert report['complete'] and report['actual_parameter_updates'] == 0
    assert report['selection']['record_ids'] == [bundle.data.train[0]['record_id']]
    assert report['checkpoint']['binding_sha256'] == bundle.state['binding_sha256']
    assert report['checkpoint']['evaluation_scope'] == 'same_admitted_snapshot_binding_only'
    assert report['manifest_sha256'] == bundle.data.manifest_sha256
    assert cli._sha(bundle.checkpoint) == before_digest
    assert calls and report['source_config'] == json.loads(bundle.config_path.read_text())
    assert {'coarse2', 'coarse2_zero_correction', 'adaptive2', 'deterministic_gl2', 'deterministic_gl3'} <= set(report['routes'])
    for route in report['routes'].values():
        assert len(route['rows'][0]['predicted_moments']) == 10
        assert route['rows'][0]['native_moments'] == bundle.data.train[0]['case']['native_moments'].tolist()


def test_diagnose_rejects_foreign_snapshot_and_hash_before_report_creation(tmp_path, monkeypatch, source_bundle):
    bundle = source_bundle
    native_loader_fixture(monkeypatch, bundle)
    output = tmp_path / 'forbidden.json'
    args = ['diagnose-gradients', *common_args(bundle, output), '--coarse-grid', '2', '--limit', '1']
    bundle.data.snapshot_id += '-foreign'
    with pytest.raises(ValueError, match='same admitted snapshot binding'):
        cli.main(args)
    assert not output.exists()
    bundle.data.snapshot_id = bundle.state['binding']['snapshot_id']
    args[args.index('--checkpoint-sha256') + 1] = '0' * 64
    with pytest.raises(ValueError, match='explicitly requested state'):
        cli.main(args)
    assert not output.exists()


@pytest.mark.parametrize('arm, coefficient', [('moment', 0.), ('joint', 1.)])
def test_screen_keeps_source_weights_but_starts_fresh_fixed_route_training(tmp_path, monkeypatch, source_bundle,
                                                                       arm, coefficient):
    bundle = source_bundle
    native_loader_fixture(monkeypatch, bundle)
    output = tmp_path / ('screen-' + arm)
    source_digest = cli._sha(bundle.checkpoint)
    cli.main(['screen', *common_args(bundle, output), '--coarse-grid', '2', '--arm', arm,
              '--steps', '2', '--evaluate-every', '1', '--save-every', '1'])
    initial = torch.load(output / 'initial.pt', map_location='cpu', weights_only=True)
    final = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
    assert initial['step_count'] == initial['attempts'] == initial['committed_exposures'] == 0
    assert initial['optimizer']['state'] == {} and initial['history'] == []
    assert all(torch.equal(value, initial['model'][name]) for name, value in bundle.state['model'].items())
    assert final['step_count'] == final['committed_exposures'] == 2
    assert final['optimizer']['state']
    assert final['training_config']['jacobian_loss_weight'] == coefficient
    assert final['architecture']['config']['execution'] == 'coarse'
    assert final['architecture']['config']['training_execution'] == 'coarse'
    assert final['architecture']['config']['training_coarse_grids'] == [2]
    assert final['binding']['manifest_sha256'] == bundle.state['binding']['manifest_sha256']
    assert cli._sha(bundle.checkpoint) == source_digest
    run = json.loads((output / 'run.json').read_text())
    assert run['update_budget_reached'] and not run['long_training_promotion']
    assert not run['stage']['old_optimizer_loaded'] and not run['stage']['exact_resume']
    evaluation = json.loads((output / 'development_step_000002.json').read_text())
    assert all(row['diagnostics']['execution'] == 'coarse' for row in evaluation['rows'])
    assert all(row['diagnostics']['coarse_grid'] == 2 for row in evaluation['rows'])


def test_calibration_requires_same_train_checkpoint_grid_and_complete_derivatives(tmp_path):
    data = fixture_data()
    checkpoint = {'sha256': 'b' * 64}
    report = dict(schema='henmo.gradient-diagnostics.v1', complete=True,
                  selection={'split': 'train', 'record_ids': [data.train[0]['record_id']]},
                  checkpoint=checkpoint, manifest_sha256=data.manifest_sha256, snapshot_id=data.snapshot_id,
                  execution_override={'coarse_grid': 2}, nonzero_joint_gradient_queries=1,
                  derivative_objective_complete=True, qualified_derivative_objectives=1,
                  actual_parameter_updates=0)
    path = tmp_path / 'calibration.json'
    args = SimpleNamespace(arm='calibrated', jacobian_weight=.2, calibration_report=str(path), coarse_grid=2)
    path.write_text(json.dumps(report))
    coefficient, receipt = cli._calibrated_weight(args, checkpoint, data)
    assert coefficient == .2 and receipt['sha256'] == cli._sha(path)
    for field, bad in [('selection', {'split': 'development'}), ('checkpoint', {'sha256': 'c' * 64}),
                       ('manifest_sha256', 'd' * 64), ('snapshot_id', 'foreign'),
                       ('execution_override', {'coarse_grid': 4}), ('derivative_objective_complete', False),
                       ('qualified_derivative_objectives', 0), ('actual_parameter_updates', 1),
                       ('nonzero_joint_gradient_queries', 0), ('complete', False)]:
        path.write_text(json.dumps(dict(report, **{field: bad})))
        with pytest.raises(ValueError, match='calibration'):
            cli._calibrated_weight(args, checkpoint, data)
    path.write_text(json.dumps(report))
    for bad in (float('nan'), float('inf'), -1., 0.):
        args.jacobian_weight = bad
        with pytest.raises(ValueError):
            cli._calibrated_weight(args, checkpoint, data)


def test_actual_gradient_report_can_bind_an_explicit_calibrated_screen(tmp_path, monkeypatch, source_bundle):
    bundle = source_bundle
    native_loader_fixture(monkeypatch, bundle)
    gradient_path = tmp_path / 'actual_gradients.json'
    cli.main(['diagnose-gradients', *common_args(bundle, gradient_path), '--coarse-grid', '2', '--limit', '1'])
    gradients = json.loads(gradient_path.read_text())
    assert gradients['complete'] and gradients['derivative_objective_complete']
    assert gradients['nonzero_joint_gradient_queries'] == 1
    assert gradients['selection']['split'] == 'train' and gradients['actual_parameter_updates'] == 0
    output = tmp_path / 'calibrated-screen'
    cli.main(['screen', *common_args(bundle, output), '--coarse-grid', '2', '--arm', 'calibrated',
              '--jacobian-weight', '.2', '--calibration-report', str(gradient_path), '--steps', '1',
              '--evaluate-every', '1', '--save-every', '1'])
    initial = torch.load(output / 'initial.pt', map_location='cpu', weights_only=True)
    final = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
    assert initial['optimizer']['state'] == {} and initial['step_count'] == 0
    assert final['step_count'] == 1 and final['training_config']['jacobian_loss_weight'] == .2
    stage = json.loads((output / 'stage.json').read_text())
    assert stage['calibration']['sha256'] == cli._sha(gradient_path)
    assert stage['calibration']['coefficient_origin'] == 'explicit_training_side_choice_not_auto_tuning'


def test_screen_budget_and_nonfinite_learning_rate_rejected_without_training(tmp_path, monkeypatch, source_bundle):
    bundle = source_bundle
    calls = native_loader_fixture(monkeypatch, bundle)
    args = ['screen', *common_args(bundle, tmp_path / 'must-not-exist'), '--coarse-grid', '2', '--arm', 'joint']
    with pytest.raises(ValueError, match='1..256'):
        cli.main([*args, '--steps', '257'])
    assert calls == []
    for value in ('nan', 'inf', '-1'):
        with pytest.raises(ValueError, match='training coefficient'):
            cli.main([*args, '--learning-rate', value, '--steps', '1'])
    assert not (tmp_path / 'must-not-exist').exists()
    with pytest.raises(ValueError, match='supported native material'):
        load_henmo_dataset('must-not-be-read.json', allowed_native_resolutions=(16,))
