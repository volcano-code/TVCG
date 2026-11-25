"""Explicit bounded trial authorization and separate long-training promotion."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..p0_diagnostics.baseline_cases import sha256

SCHEMA = 'compact_pino.training-authorization.v1'


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def authorize(*, mode, manifest_sha256, config_sha256, max_steps, max_attempts,
              authorization_path=None, resume_checkpoint_sha256=None):
    """No function creates GO. Trial permission is a bounded declared protocol.

    Long runs additionally require an immutable development research-screen GO
    bound to the exact resumed candidate and its evaluation protocol. A screen
    itself has training_authorized=false; this separate authorization is needed.
    """
    if type(max_steps) is not int or max_steps < 1 or type(max_attempts) is not int or max_attempts < max_steps:
        raise ValueError('positive optimizer-step budget and sufficient attempt budget required')
    if mode == 'smoke':
        if max_steps > 4 or max_attempts > 8:
            raise ValueError('synthetic smoke is limited to four total optimizer steps and eight attempts')
        return dict(mode=mode,scope='synthetic_software_only',training_authorized=False,
                    max_optimizer_steps=max_steps,max_attempts=max_attempts)
    if mode not in ('trial','long') or authorization_path is None:
        raise ValueError('real trial/long requires an explicit budget authorization file')
    path = Path(authorization_path).resolve(strict=True)
    authorization = json.loads(path.read_text())
    if (authorization.get('schema') != SCHEMA or authorization.get('mode') != mode
            or authorization.get('training_authorized') is not True
            or not isinstance(authorization.get('authorized_by'),str)
            or not authorization['authorized_by'].strip()):
        raise ValueError('explicit mode-matched training authorization required')
    for key,expected in (('training_manifest_sha256',manifest_sha256),('config_sha256',config_sha256)):
        if authorization.get(key) != expected:
            raise ValueError(f'authorization identity mismatch: {key}')
    for key,requested in (('max_optimizer_steps',max_steps),('max_attempts',max_attempts)):
        allowed = authorization.get(key)
        if type(allowed) is not int or requested > allowed:
            raise ValueError(f'request exceeds authorized {key}')
    wall = authorization.get('max_wall_seconds')
    if isinstance(wall,bool) or not isinstance(wall,(float,int)) or not math.isfinite(wall) or wall <= 0:
        raise ValueError('positive finite authorized wall-clock budget required')
    if mode == 'trial' and wall > 6*3600:
        raise ValueError('trial is a bounded diagnostic of at most six hours; use long admission beyond that')
    protocol = authorization.get('protocol')
    if not isinstance(protocol,dict) or not protocol:
        raise ValueError('a frozen explicit training protocol with gates is required')
    gates = protocol.get('correctness_gates')
    required = ('teacher_and_operand_identity','source_split_isolation','material_ten_moments',
                'derivative_qualification_masks','finite_joint_update_and_resume')
    if not isinstance(gates,dict) or any(gates.get(key) != 'PASS' for key in required):
        raise ValueError('required correctness gates must explicitly PASS')
    if not isinstance(protocol.get('screening_thresholds'),dict) or not protocol['screening_thresholds']:
        raise ValueError('explicit frozen screening thresholds required')
    if mode == 'long':
        if resume_checkpoint_sha256 is None:
            raise ValueError('long training must continue the exact screened candidate checkpoint')
        record = authorization.get('screen_report',{})
        screen_path = Path(record.get('path',''))
        screen_path = screen_path if screen_path.is_absolute() else path.parent/screen_path
        if not screen_path.is_file() or sha256(screen_path) != record.get('sha256'):
            raise ValueError('immutable research-screen report and SHA256 required')
        screen = json.loads(screen_path.read_text())
        if (screen.get('schema') != 'tvcg.v22.research-screen.v1'
                or screen.get('mode') != 'research_screen' or screen.get('decision') != 'GO'
                or screen.get('scope') != 'development'
                or screen.get('eligible_for_long_training_review') is not True
                or screen.get('training_authorized') is not False):
            raise ValueError('development research-screen GO required for long-training review')
        identity = screen.get('identity',{})
        expected = dict(training_manifest_sha256=manifest_sha256,config_sha256=config_sha256,
                        candidate_checkpoint_sha256=resume_checkpoint_sha256,
                        protocol_sha256=authorization.get('evaluation_protocol_sha256'))
        if any(not value or identity.get(key) != value for key,value in expected.items()):
            raise ValueError('long-training screen identity mismatch')
    return dict(authorization,authorization_path=str(path),authorization_sha256=sha256(path))
