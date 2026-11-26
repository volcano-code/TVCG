"""Actual Token16 updates with GL3 and physical losses on synthetic fixtures.

Uses the released 32^3/64^3 shared U-FNO. Tests a physical-loss-only backward,
two joint optimizer updates and independent replay after the first update.
The objective is the explicitly bound repository profile, NOT private noodles.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .attach_residual_targets import REQUEST_SCHEMA, attach_targets
from .losses import physical_loss
from .manifest import load_training_manifest
from .residual_matching import TARGET_SCHEMA, REPO_PROFILE, P_CONVENTION, MOMENT_ORDER, REPO_FORMULA_SHA256
from .runner import Trainer, TrainConfig, _gradient_stats, _json_write, _physical_exposure_at_checkpoint
from .smoke import make_smoke_manifest, _exact
from ..p0_diagnostics.baseline_cases import moment_scale
from ..p0_diagnostics.exposure_audit import read_events


def fixture_target(case):
    """Nonzero reference physics, used only for explicit analytic software tests."""
    scale = moment_scale(case['meta']['editable_mass'], case['meta']['length_scale'])
    dual = np.asarray(case['coefficients']) * np.r_[np.ones(7), np.full(3, 2.)]
    dual /= np.linalg.norm(dual)
    return dict(schema=TARGET_SCHEMA, valid=True, profile=REPO_PROFILE,
        teacher_role='synthetic_fixture', moment_order=list(MOMENT_ORDER),
        formula_provenance=dict(source_profile=REPO_PROFILE, source_sha256=dict(REPO_FORMULA_SHA256)),
        objective=dict(mode='SPINNING_TOP', weight=.2, mass_scale=1.),
        reference_moments=case['native_moments'].tolist(), p=dual.tolist(), p_convention=P_CONVENTION,
        multipliers=[.1, -.1, .05, -.05], multiplier_mu=.7,
        stationarity_scale=(1./scale).tolist(), constraint_scale=scale[[1, 2, 8, 7]].tolist())


def make_precision_manifest(directory, *, cache_directory=None):
    directory = Path(directory)
    original = make_smoke_manifest(directory)
    data = load_training_manifest(original, mode='smoke', cache_directory=cache_directory)
    targets = dict(schema=REQUEST_SCHEMA, records=[dict(record_id=row['record_id'], target=fixture_target(row['case']))
                                                 for row in data.train])
    request = directory/'physics_targets.json'
    _json_write(request, targets)
    output = directory/'precision_manifest.json'
    attach_targets(original, request, output, mode='smoke', cache_directory=cache_directory)
    return output


def run_precision_smoke(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = make_precision_manifest(output/'assets', cache_directory=output/'cache')
    data = load_training_manifest(manifest, mode='smoke', cache_directory=output/'cache')
    config = TrainConfig(full_cell_rule='gl3', residual_loss_weight=.1, constraint_loss_weight=.1)
    trainer = Trainer(data, output/'continuous', config=config, max_steps=2, max_attempts=4,
                      run_id='synthetic-precision-exact-resume')
    # Prove that the new term alone reaches the encoder and local response.
    row = dict(data.train[0], moment_weight=0., jacobian_weight=0., residual_weight=1.)
    prediction = trainer._predictions([row])[0]
    loss, info = physical_loss([prediction], [row], jacobian_loss_weight=0.,
                              residual_loss_weight=.1, constraint_loss_weight=.1)
    loss.backward()
    physical_finite, physical_gradients = _gradient_stats(trainer.model)
    physical_only = dict(loss=float(loss.detach()), loss_membership=info,
                         gradients=physical_gradients, optimizer_step_executed=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    del loss, prediction
    initial = trainer.save_checkpoint()
    first = trainer.step()
    middle = trainer.save_checkpoint()
    second = trainer.step()
    continuous = trainer.save_checkpoint()
    trainer.close()
    del trainer
    gc.collect()
    replay = Trainer(data, output/'resumed', config=config, max_steps=2, max_attempts=4,
                     resume=middle['receipt_path'])
    repeated = replay.step()
    replay_checkpoint = replay.save_checkpoint()
    replay.close()
    del replay
    gc.collect()
    expected = torch.load(continuous['path'], map_location='cpu', weights_only=True)
    observed = torch.load(replay_checkpoint['path'], map_location='cpu', weights_only=True)
    checks = {key: bool(_exact(expected[key], observed[key])) for key in
              ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'sampler', 'global_step', 'attempts')}
    checks.update(same_second_batch=second['record_ids'] == repeated['record_ids'],
                  same_second_loss=second['loss'] == repeated['loss'],
                  parameters_updated=initial['model_tensor_sha256'] != continuous['model_tensor_sha256'])
    checks['all_four_objectives_effective'] = all(step[key] > 0 for step in (first, second, repeated)
        for key in ('moment_denominator', 'jacobian_denominator', 'residual_denominator', 'constraint_denominator'))
    checks['physical_only_reaches_encoder_attention_response'] = physical_finite and all(
        value['finite'] and value['norm'] is not None and value['norm'] > 0
        for value in physical_gradients.values())
    checks['joint_gradients_finite_nonzero'] = all(
        value['finite'] and value['norm'] is not None and value['norm'] > 0
        for step in (first, second, repeated) for value in step['gradients'].values())
    events = read_events([output/'continuous'/'exposure.jsonl', output/'resumed'/'exposure.jsonl'])
    participation = _physical_exposure_at_checkpoint(events, replay_checkpoint, config)
    checks['physical_exposure_tracks_selected_checkpoint'] = all(
        value['committed_occurrences'] == 2 for value in participation['objectives'].values())
    report = dict(schema='compact_pino.precision-software-smoke.v1', scope='synthetic_software_only',
        status='PASS' if all(checks.values()) else 'FAIL', checks=checks, config=expected['config'],
        execution_sources=expected['execution_sources'], runtime=expected['runtime'],
        training_manifest_path=str(manifest), training_manifest_sha256=data.manifest_sha256,
        physical_only_backward=physical_only, steps=[first, second, repeated],
        initial_checkpoint=initial, interrupted_checkpoint=middle, continuous_checkpoint=continuous,
        resumed_checkpoint=replay_checkpoint, physical_exposure=participation,
        real_optimizer_calls_executed=3, native_solver_acceptance='NOT_EVALUATED',
        scientific_promotion='NOT_EVALUATED',
        exactness_scope='same CPU runtime; actual default encoder; GPU and private deployed formula not validated')
    _json_write(output/'precision_smoke_report.json', report)
    if not all(checks.values()):
        raise AssertionError('precision smoke failed: ' + str(checks))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run_precision_smoke(args.output)
    print(json.dumps(dict(status=report['status'], checks=report['checks'],
                         report=str(Path(args.output)/'precision_smoke_report.json')), indent=2))


if __name__ == '__main__':
    main()
