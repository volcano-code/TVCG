"""Bind an existing correctness receipt and session permission to one run budget.

No correctness PASS is inferred or manufactured. The Work-side receipt must
provide original evidence assets. Long authorization additionally runs the
unchanged admission check on the exact screened resume candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare import _asset, file_receipt, read_json
from .run import load_bundle
from ..training.admission import SCHEMA, authorize, canonical_digest
from ..training.runner import TrainConfig, _execution_sources, _json_write, model_kwargs
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21
from dataclasses import asdict
import torch

GATES = ('teacher_and_operand_identity', 'source_split_isolation', 'material_ten_moments',
         'derivative_qualification_masks', 'finite_joint_update_and_resume')


def _verify_software_smoke(evidence, config):
    """Do not substitute a legacy-head or CPU smoke for this CUDA run."""
    profiles = {key: getattr(config, key) for key in
                ('response_type', 'training_scope', 'response_zero_init', 'token_grid',
                 'full_cell_rule', 'amp')}
    reports = []
    for asset in evidence:
        try:
            value = read_json(asset['path'])
        except (ValueError, UnicodeError):
            continue
        if isinstance(value, dict) and value.get('schema') == 'compact_pino.rapid-response-software-smoke.v1':
            reports.append((asset, value))
    if not reports:
        raise ValueError('current rapid-response CUDA software smoke evidence is required')
    # Preserve the caller RNG: constructing this model is only for the exact
    # transitive source inventory, not initialization or training.
    with torch.random.fork_rng(devices=[]):
        expected_sources = _execution_sources(CompactPINOv21(**model_kwargs(config)))
    tool_sha = file_receipt(Path(__file__).with_name('smoke.py'))['sha256']
    for asset, report in reports:
        checks = report.get('checks')
        if (report.get('status') != 'PASS' or not isinstance(checks, dict) or not checks
                or any(value is not True for value in checks.values())
                or report.get('real_optimizer_calls_executed') != 6):
            continue
        if not str(report.get('actual_device', '')).startswith('cuda'):
            continue
        if profiles not in report.get('supported_model_profiles', []):
            continue
        if (report.get('execution_sources') != expected_sources
                or report.get('smoke_source_sha256') != tool_sha):
            continue
        return asset
    raise ValueError('rapid software smoke must PASS on CUDA with this model profile and current execution sources')


def bind_authorization(bundle_path, arm, correctness_path, output, *, authorized_by,
                       mode='trial', max_steps=104, max_attempts=None, wall_hours=6,
                       screen_report=None, resume=None):
    bundle, assets = load_bundle(bundle_path, arm)
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        raise ValueError('retain the existing user authorization attribution')
    config = TrainConfig(**read_json(assets['config']['path'])).validate()
    correctness_path = Path(correctness_path).resolve(strict=True)
    correctness = read_json(correctness_path)
    if (correctness.get('schema') != 'compact_pino.rapid-correctness.v1'
            or correctness.get('training_manifest_sha256') != assets['manifest']['sha256']):
        raise ValueError('correctness receipt must bind this admitted training manifest')
    evidence = {}
    for key in GATES:
        check = correctness.get('checks', {}).get(key, {})
        if check.get('status') != 'PASS' or not isinstance(check.get('evidence'), list) or not check['evidence']:
            raise ValueError('external PASS and original evidence required: '+key)
        evidence[key] = [_asset(value, correctness_path.parent) for value in check['evidence']]
    verified_smoke = _verify_software_smoke(evidence['finite_joint_update_and_resume'], config)
    max_attempts = max_steps if max_attempts is None else max_attempts
    value = dict(schema=SCHEMA, mode=mode, training_authorized=True, authorized_by=authorized_by,
        training_manifest_sha256=assets['manifest']['sha256'], config_sha256=canonical_digest(asdict(config)),
        max_optimizer_steps=max_steps, max_attempts=max_attempts, max_wall_seconds=wall_hours*3600,
        evaluation_protocol_sha256=assets['evaluation_protocol']['sha256'],
        correctness_receipt=file_receipt(correctness_path), correctness_evidence=evidence,
        verified_software_smoke=verified_smoke,
        protocol=dict(correctness_gates={key: 'PASS' for key in GATES},
                      screening_thresholds=dict(primary_relative_improvement=.05, protection_max_ratio=1.05),
                      prepared_bundle=file_receipt(bundle_path),
                      authority_scope='existing_user_instruction_and_external_evidence_not_automatic_quality_GO'))
    resumed_sha = None
    if mode == 'long':
        if not screen_report or not resume:
            raise ValueError('long needs the actual development screen and selected resume receipt')
        value['screen_report'] = file_receipt(screen_report)
        resume_path = Path(resume).resolve(strict=True)
        resumed = read_json(resume_path)
        checkpoint = _asset(resumed['checkpoint'], resume_path.parent)
        resumed_sha = checkpoint['sha256']
        if resumed['exposure_receipt']['checkpoint_sha256'] != resumed_sha:
            raise ValueError('resume checkpoint/exposure mismatch')
        value['resume_receipt'] = file_receipt(resume_path)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Prevalidate through the unchanged admission entry point using a private
    # temporary file, then publish the requested authorization atomically.
    import tempfile
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        draft = Path(temporary)/'authorization.json'
        _json_write(draft, value)
        authorize(mode=mode, manifest_sha256=assets['manifest']['sha256'],
                  config_sha256=value['config_sha256'], max_steps=max_steps, max_attempts=max_attempts,
                  authorization_path=draft, resume_checkpoint_sha256=resumed_sha)
    _json_write(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'arm', 'correctness-receipt', 'output', 'authorized-by'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('trial', 'long'), default='trial')
    parser.add_argument('--max-steps', type=int, default=104)
    parser.add_argument('--max-attempts', type=int)
    parser.add_argument('--wall-hours', type=float, default=6.)
    parser.add_argument('--screen-report')
    parser.add_argument('--resume')
    args = parser.parse_args()
    result = bind_authorization(args.bundle, args.arm, args.correctness_receipt, args.output,
        authorized_by=args.authorized_by, mode=args.mode, max_steps=args.max_steps,
        max_attempts=args.max_attempts, wall_hours=args.wall_hours,
        screen_report=args.screen_report, resume=args.resume)
    print(json.dumps(dict(mode=result['mode'], output=str(Path(args.output).resolve()),
                         max_steps=result['max_optimizer_steps'], max_wall_seconds=result['max_wall_seconds']), indent=2))


if __name__ == '__main__': main()
