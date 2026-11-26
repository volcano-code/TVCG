"""Bind separately supplied native physics targets to an existing frozen manifest.

This does not derive multipliers, certify a private formula, or create labels.
Input: {schema: 'cpino_residual_attachment_v1', records:
        [{record_id: ..., target: <fixed residual target JSON>}, ...]}.
Each provided target is checked against the actual native M and query. Existing
source/split/teacher receipts remain binding; the original manifest is untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

from .manifest import load_training_manifest
from .residual_matching import validate_residual_target
from ..p0_diagnostics.baseline_cases import sha256


REQUEST_SCHEMA = 'cpino_residual_attachment_v1'


def _absolute(entry, root):
    value = copy.deepcopy(entry)
    path = Path(value['path'])
    value['path'] = str((path if path.is_absolute() else root / path).resolve(strict=True))
    return value


def attach_targets(manifest, request, output, *, mode='trial', cache_directory=None):
    manifest, request = Path(manifest).resolve(strict=True), Path(request).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite a frozen manifest: ' + str(output))
    data = load_training_manifest(manifest, mode=mode, cache_directory=cache_directory)
    source = json.loads(manifest.read_text())
    packet = json.loads(request.read_text())
    if packet.get('schema') != REQUEST_SCHEMA or not isinstance(packet.get('records'), list) or not packet['records']:
        raise ValueError('nonempty cpino_residual_attachment_v1 records required')
    by_id = {record['record_id']: record for record in data.records}
    rows = {row['record_id']: row for row in source['records']}
    attached = []
    for item in packet['records']:
        rid = item.get('record_id')
        if rid in attached or rid not in by_id or not by_id[rid]['loader_eligible']:
            raise ValueError('residual target must name one unique loader-eligible record')
        record = by_id[rid]
        case = record['case']
        target = copy.deepcopy(item.get('target'))
        qualification = validate_residual_target(target, case['native_moments'], coefficients=case['coefficients'])
        if qualification['availability'] == 'AVAILABLE':
            if (mode == 'smoke') != (target['teacher_role'] == 'synthetic_fixture'):
                raise ValueError('native and synthetic physics evidence cannot be interchanged')
            expected = dict(case_sha256=record['case_sha256'], record_id=rid,
                            query_id=case['meta']['query_id'], geometry_sha256=case['meta']['geometry_sha256'])
            supplied = target.get('binding', {})
            if not isinstance(supplied, dict) or any(key in supplied and supplied[key] != value
                                                    for key, value in expected.items()):
                raise ValueError('supplied residual target binding contradicts the selected record')
            target['binding'] = expected
        rows[rid].pop('residual_target_asset', None)
        rows[rid]['residual_target'] = target
        attached.append(rid)
    # The result may live elsewhere. Resolve only known, already verified path
    # fields, retaining each original SHA and every original role/query ID.
    for row in source['records']:
        path = Path(row['case_path'])
        row['case_path'] = str((path if path.is_absolute() else manifest.parent/path).resolve(strict=True))
        if 'residual_target_asset' in row:
            row['residual_target_asset'] = _absolute(row['residual_target_asset'], manifest.parent)
    if 'export_receipt_sha256' in source:
        path = Path(source.get('export_receipt_path', 'export_receipt.json'))
        source['export_receipt_path'] = str((path if path.is_absolute() else manifest.parent/path).resolve(strict=True))
    if 'source_authority' in source:
        authority = _absolute(source['source_authority'], manifest.parent)
        authority['supporting_assets'] = {key: _absolute(value, manifest.parent)
                                         for key, value in authority['supporting_assets'].items()}
        source['source_authority'] = authority
    source['residual_attachment'] = dict(schema=REQUEST_SCHEMA, source_manifest_sha256=sha256(manifest),
        request_sha256=sha256(request), attached_record_ids=attached,
        scope='case_bound_supplied_physics_targets_not_independent_native_certification')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, suffix='.json', delete=False) as stream:
        temp = Path(stream.name)
        json.dump(source, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        verified = load_training_manifest(temp, mode=mode, cache_directory=cache_directory)
        # Same filesystem; link is an exclusive publish, not an overwrite race.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return dict(status='BOUND_SUPPLIED_TARGETS', manifest_path=str(output), manifest_sha256=sha256(output),
                attached_count=len(attached), valid_train=sum(row.get('residual_valid', False) for row in verified.train),
                valid_development=sum(row.get('residual_valid', False) for row in verified.development),
                native_solver_acceptance='NOT_EVALUATED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--targets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('smoke', 'trial'), default='trial')
    parser.add_argument('--cache-directory')
    args = parser.parse_args()
    print(json.dumps(attach_targets(args.manifest, args.targets, args.output, mode=args.mode,
                                   cache_directory=args.cache_directory), indent=2))


if __name__ == '__main__':
    main()
