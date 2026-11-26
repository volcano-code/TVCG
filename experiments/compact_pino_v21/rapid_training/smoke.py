"""Actual paired Token16 response training and exact interrupted replay.

Uses synthetic, explicitly bound repository-profile targets and the released
32^3/64^3 shared U-FNO. Each response arm executes two optimizer updates, then
replays its second update from a fresh trainer. This is a software/GPU admission
check, not real-data training quality or native spinning-top acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch

from ..p0_diagnostics.baseline_cases import moment_scale, sha256
from ..p0_diagnostics.exposure_audit import read_events
from ..training.losses import physical_loss
from ..training.manifest import load_training_manifest
from ..training.precision_smoke import make_precision_manifest
from ..training.runner import (TrainConfig, Trainer, _gradient_stats, _json_write,
                               _physical_exposure_at_checkpoint)
from ..training.smoke import _exact
from ..training.state import SCHEDULE_SCHEMA
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import evaluate_packed
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface


def _counts(model):
    return dict(total=sum(parameter.numel() for parameter in model.parameters()),
                trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                trainable_names=[name for name, parameter in model.named_parameters() if parameter.requires_grad])


def _initial_predictions(trainer, data):
    trainer.model.eval()
    with torch.no_grad():
        predictions = trainer._predictions(data.train, evaluation=True)
    result = [dict(record_id=row['record_id'], moments=prediction.moments.detach().cpu().clone(),
                   jacobian=prediction.jacobian.detach().cpu().clone())
              for row, prediction in zip(data.train, predictions)]
    trainer.model.train()
    return result


def _baseline(data, rule, chunk_size):
    rows = []
    for record in data.train:
        case = record['case']; operand = case['operand']
        prediction = evaluate_packed(case['blocks'], prepare_surface(operand.vertices, operand.faces),
            case['blocks'].pitch, operand.operation, case['fixed_moments'], operand.vertex_velocity,
            model=None, chunk_size=chunk_size, full_cell_rule=rule)
        rows.append(dict(record_id=record['record_id'], moments=prediction.moments.detach().cpu(),
                         jacobian=prediction.jacobian.detach().cpu()))
    return rows


def _baseline_errors(initial, baseline, data):
    errors = []
    for predicted, reference, record in zip(initial, baseline, data.train):
        case = record['case']; meta = case['meta']
        scale = torch.tensor(moment_scale(meta['editable_mass'], meta['length_scale']), dtype=torch.float64)
        parameters = torch.tensor(case['parameter_scales'], dtype=torch.float64)
        errors.append(dict(record_id=record['record_id'],
            moment_scaled_max=float(((predicted['moments']-reference['moments'])/scale).abs().max()),
            jacobian_scaled_max=float(((predicted['jacobian']-reference['jacobian'])*parameters[None]/scale[:,None]).abs().max())))
    return errors


def _physical_backward(trainer, record):
    record = dict(record, moment_weight=0., jacobian_weight=0., residual_weight=1.)
    prediction = trainer._predictions([record])[0]
    loss, info = physical_loss([prediction], [record], jacobian_loss_weight=0.,
                              residual_loss_weight=.1, constraint_loss_weight=.1)
    loss.backward()
    finite, groups = _gradient_stats(trainer.model)
    parameter_norms = {name: None if parameter.grad is None else float(parameter.grad.detach().norm())
                       for name, parameter in trainer.model.named_parameters() if name.startswith('response.')}
    result = dict(loss=float(loss.detach()), memberships=info, finite=finite,
                  groups=groups, response_parameter_gradient_norms=parameter_norms,
                  note='Zero final-layer initialization intentionally gives zero initial hidden-layer gradient.',
                  optimizer_step_executed=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    return result


def _run_arm(root, data, schedule, schedule_sha, *, response_type, rule, device, baseline):
    config = TrainConfig(response_type=response_type, response_zero_init=True, training_scope='response',
                         full_cell_rule=rule, sampler='fixed_schedule', exposure_schedule_sha256=schedule_sha,
                         residual_loss_weight=.1, constraint_loss_weight=.1)
    trainer = Trainer(data, root/'continuous', config=config, max_steps=2, max_attempts=4,
                      device=device, run_id='synthetic-rapid-'+response_type, exposure_schedule=schedule)
    initial_counts = _counts(trainer.model)
    e0 = _initial_predictions(trainer, data)
    initial_baseline_errors = _baseline_errors(e0, baseline, data)
    physical_only = _physical_backward(trainer, data.train[0])
    initial = trainer.save_checkpoint()
    first = trainer.step()
    middle = trainer.save_checkpoint()
    second = trainer.step()
    continuous = trainer.save_checkpoint()
    continuous_efficiency = trainer.model.efficiency_statistics()
    final_counts = _counts(trainer.model)
    trainer.close(); del trainer; gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    replay = Trainer(data, root/'resumed', config=config, max_steps=2, max_attempts=4,
                     device=device, resume=middle['receipt_path'], exposure_schedule=schedule)
    repeated = replay.step()
    replay_checkpoint = replay.save_checkpoint()
    replay_efficiency = replay.model.efficiency_statistics()
    replay.close(); del replay; gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    initial_state = torch.load(initial['path'], map_location='cpu', weights_only=True)
    expected = torch.load(continuous['path'], map_location='cpu', weights_only=True)
    observed = torch.load(replay_checkpoint['path'], map_location='cpu', weights_only=True)
    checks = {key+'_exact_replay': bool(_exact(expected[key], observed[key])) for key in
              ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'sampler', 'global_step', 'attempts',
               'frozen_state_sha256', 'exposure_schedule_identity')}
    frozen_keys = [name for name in expected['model'] if not name.startswith('response.')]
    response_keys = [name for name in expected['model'] if name.startswith('response.')]
    checks.update(
        frozen_parameters_and_buffers_unchanged=all(torch.equal(initial_state['model'][name], expected['model'][name])
                                                    for name in frozen_keys),
        response_parameters_updated=any(not torch.equal(initial_state['model'][name], expected['model'][name])
                                        for name in response_keys),
        trainable_scope_is_only_response=all(name.startswith('response.') for name in initial_counts['trainable_names']),
        trainable_count_preserved=initial_counts == final_counts,
        same_second_batch=second['record_ids'] == repeated['record_ids'],
        same_second_loss=second['loss'] == repeated['loss'],
        fixed_schedule_order=[first['record_ids'], second['record_ids']] == [['fixture-0'], ['fixture-1']],
        only_one_continuous_geometry_encoding=continuous_efficiency['encoder_calls'] == 1,
        frozen_cache_reused_across_updates=continuous_efficiency['frozen_features']['hits'] >= 2,
        replay_rebuilds_one_geometry=replay_efficiency['encoder_calls'] == 1,
        all_four_objectives_effective=all(step[key] > 0 for step in (first, second, repeated)
            for key in ('moment_denominator', 'jacobian_denominator', 'residual_denominator', 'constraint_denominator')),
        physical_only_reaches_response=physical_only['finite'] and physical_only['groups']['response.']['norm'] > 0,
        physical_only_keeps_encoder_frozen=all(physical_only['groups'][key]['parameters_with_grad'] == 0
            for key in ('shared_encoder.pino_encoder.', 'geometry_attention.')),
        joint_response_gradients_finite_nonzero=all(step['gradients']['response.']['finite']
            and step['gradients']['response.']['norm'] > 0 for step in (first, second, repeated)),
        e0_matches_unlearned_cubic_within_float64_reduction_roundoff=all(
            max(row['moment_scaled_max'], row['jacobian_scaled_max']) <= 1e-11 for row in initial_baseline_errors))
    events = read_events([root/'continuous'/'exposure.jsonl', root/'resumed'/'exposure.jsonl'])
    participation = _physical_exposure_at_checkpoint(events, replay_checkpoint, config)
    checks['selected_checkpoint_has_two_physical_occurrences'] = all(
        item['committed_occurrences'] == 2 for item in participation['objectives'].values())
    report = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        response_type=response_type, config=expected['config'], runtime=expected['runtime'],
        execution_sources=expected['execution_sources'],
        initial_parameter_counts=initial_counts, final_parameter_counts=final_counts,
        e0=[dict(record_id=row['record_id'], moments=row['moments'].tolist(),
                 jacobian=row['jacobian'].tolist()) for row in e0],
        e0_vs_unlearned=initial_baseline_errors, physical_only_backward=physical_only,
        steps=[first, second, repeated], initial_checkpoint=initial, interrupted_checkpoint=middle,
        continuous_checkpoint=continuous, resumed_checkpoint=replay_checkpoint,
        physical_exposure=participation, continuous_efficiency=continuous_efficiency,
        replay_efficiency=replay_efficiency, real_optimizer_calls_executed=3)
    del initial_state, expected, observed
    return report, e0


def run_smoke(output, *, device='cpu', full_cell_rule='gl2'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('smoke supports cpu or cuda devices')
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; do not report a CPU run as a GPU test')
    manifest = make_precision_manifest(output/'assets', cache_directory=output/'cache')
    data = load_training_manifest(manifest, mode='smoke', cache_directory=output/'cache')
    schedule = output/'exposure_schedule.json'
    _json_write(schedule, dict(schema=SCHEDULE_SCHEMA, training_manifest_sha256=data.manifest_sha256,
                              record_ids=['fixture-0', 'fixture-1', 'fixture-0', 'fixture-1']))
    schedule_sha = sha256(schedule)
    baseline = _baseline(data, full_cell_rule, 65536)
    arms, initial = {}, {}
    for label, response_type in (('A', 'bounded_cubic'), ('B', 'monotone_warp')):
        arms[label], initial[label] = _run_arm(output/label, data, schedule, schedule_sha,
            response_type=response_type, rule=full_cell_rule, device=device, baseline=baseline)
    checks = dict(two_arms_pass=all(arm['status'] == 'PASS' for arm in arms.values()),
                  paired_e0_moments_bitwise_equal=all(torch.equal(a['moments'], b['moments'])
                      for a, b in zip(initial['A'], initial['B'])),
                  paired_e0_jacobians_bitwise_equal=all(torch.equal(a['jacobian'], b['jacobian'])
                      for a, b in zip(initial['A'], initial['B'])),
                  paired_identical_query_order=all(a['record_ids'] == b['record_ids']
                      for a, b in zip(arms['A']['steps'], arms['B']['steps'])),
                  paired_identical_execution_sources=arms['A']['execution_sources'] == arms['B']['execution_sources'])
    profiles = [{key: arm['config'][key] for key in
        ('response_type', 'training_scope', 'response_zero_init', 'token_grid', 'full_cell_rule', 'amp')}
        for arm in arms.values()]
    report = dict(schema='compact_pino.rapid-response-software-smoke.v1',
        scope='synthetic_software_only', status='PASS' if all(checks.values()) else 'FAIL',
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        supported_model_profiles=profiles, execution_sources=arms['A']['execution_sources'],
        smoke_source_sha256=sha256(Path(__file__)),
        checks=checks, arms=arms, training_manifest_path=str(manifest),
        training_manifest_sha256=data.manifest_sha256, exposure_schedule_sha256=schedule_sha,
        actual_device=str(torch.device(device)), real_optimizer_calls_executed=6,
        elapsed_seconds=time.perf_counter()-started,
        native_solver_acceptance='NOT_EVALUATED', scientific_promotion='NOT_EVALUATED',
        exactness_scope='Same declared runtime/device interrupted replay; synthetic analytic material and bound repository physical profile only.',
        evidence_boundary='No deployed teacher, private noodles formula, real-data pilot or cross-geometry physical acceptance was evaluated.')
    _json_write(output/'report.json', report)
    if not all(checks.values()):
        failures = {label: [key for key, value in arm['checks'].items() if not value]
                    for label, arm in arms.items()}
        raise AssertionError('rapid smoke failed: ' + str(dict(checks=checks, arms=failures)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu', help='cpu or a CUDA device such as cuda:0')
    parser.add_argument('--full-cell-rule', choices=('tetra4', 'gl2', 'gl3'), default='gl2')
    args = parser.parse_args()
    report = run_smoke(args.output, device=args.device, full_cell_rule=args.full_cell_rule)
    print(json.dumps(dict(status=report['status'], checks=report['checks'],
                          actual_device=report['actual_device'],
                          real_optimizer_calls_executed=report['real_optimizer_calls_executed'],
                          report=str(Path(args.output).resolve()/'report.json')), indent=2))


if __name__ == '__main__':
    main()
