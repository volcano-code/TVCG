"""Run one frozen response-repair arm without hand-copying hashes or settings."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from .prepare import BUNDLE_SCHEMA, _asset, read_json
from ..training.manifest import load_training_manifest
from ..training.runner import Trainer, TrainConfig


def load_bundle(path, arm):
    path = Path(path).resolve(strict=True)
    bundle = read_json(path)
    if bundle.get('schema') != BUNDLE_SCHEMA or arm not in bundle.get('configs', {}):
        raise ValueError('prepared rapid bundle and a declared arm required')
    checked = {name: _asset(bundle[name], path.parent) for name in
               ('manifest', 'schedule', 'calibration', 'evaluation_protocol', 'registered_protocol')}
    checked['config'] = _asset(bundle['configs'][arm], path.parent)
    calibrated = read_json(checked['calibration']['path'])
    config = TrainConfig(**read_json(checked['config']['path'])).validate()
    expected_response = {'A_old_response': 'bounded_cubic', 'B_monotone_response': 'monotone_warp'}
    if config.response_type != expected_response.get(arm):
        raise ValueError('arm name must bind its exact old/new response type')
    cohort = _asset(bundle['cohort'], path.parent)
    schedule = read_json(checked['schedule']['path'])
    protocol = read_json(checked['evaluation_protocol']['path'])
    expected = dict(training_manifest_sha256=checked['manifest']['sha256'],
                    cohort_sha256=cohort['sha256'], full_cell_rule=config.full_cell_rule,
                    seed=config.seed, training_scope=config.training_scope,
                    response_zero_init=config.response_zero_init,
                    initial_weights_sha256=bundle.get('initial_weights_sha256'),
                    status='CALIBRATED_NOT_ACCEPTANCE', parameter_updates=0)
    if any(calibrated.get(key) != value for key, value in expected.items()):
        raise ValueError('calibration, cohort, and selected arm identity mismatch')
    if (config.training_scope != 'response' or not config.response_zero_init
            or config.sampler != 'fixed_schedule'
            or config.exposure_schedule_sha256 != checked['schedule']['sha256']
            or schedule.get('training_manifest_sha256') != checked['manifest']['sha256']
            or protocol.get('registered_protocol_sha256') != checked['registered_protocol']['sha256']):
        raise ValueError('prepared response trial schedule/protocol identity mismatch')
    for key in ('jacobian_loss_weight', 'residual_loss_weight', 'constraint_loss_weight'):
        if calibrated.get('weights', {}).get(key) != getattr(config, key):
            raise ValueError('calibration and config loss weights differ: '+key)
    if config.jacobian_loss_weight != 1.:
        raise ValueError('gradient calibration requires the declared unit J weight')
    initial = calibrated.get('initial_weights')
    if calibrated.get('initial_weights_sha256'):
        if not isinstance(initial, dict):
            raise ValueError('calibration must retain its canonical initial checkpoint receipt')
        checked['initial_weights'] = _asset(initial, Path(checked['calibration']['path']).parent)
        if checked['initial_weights']['sha256'] != calibrated['initial_weights_sha256']:
            raise ValueError('initial checkpoint differs from calibration')
    return bundle, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--arm', choices=('A_old_response', 'B_monotone_response'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=('trial', 'long'), default='trial')
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int, default=104)
    parser.add_argument('--max-attempts', type=int)
    parser.add_argument('--cache-directory')
    args = parser.parse_args()
    _, assets = load_bundle(args.bundle, args.arm)
    config = TrainConfig(**read_json(assets['config']['path'])).validate()
    data = load_training_manifest(assets['manifest']['path'], mode=args.mode, cache_directory=args.cache_directory)
    initial = assets.get('initial_weights') if not args.resume else None
    trainer = Trainer(data, args.output, config=config, mode=args.mode, device=args.device,
        max_steps=args.max_steps, max_attempts=args.max_attempts or args.max_steps,
        authorization_path=args.authorization, resume=args.resume,
        evaluation_protocol_path=assets['evaluation_protocol']['path'],
        registered_protocol_path=assets['registered_protocol']['path'],
        exposure_schedule=assets['schedule']['path'],
        initial_weights=initial['path'] if initial else None,
        initial_weights_sha256=initial['sha256'] if initial else None,
        initial_weights_scope='geometry_and_hidden' if initial else 'full')
    result = trainer.run(checkpoint_every=26, evaluate_every=0)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
