"""Prepare two response-only trials from admitted data and external evidence.

This writes configurations, a shared immutable exposure schedule, and a new
pre-registration. It never generates labels, qualification receipts, or GO.
The full schedule is frozen before the 104-step trial so an admitted long run
can resume the exact candidate without changing scientific inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..p0_diagnostics.baseline_cases import sha256
from ..p0_diagnostics.promotion_gate import _protocol, retention_bin
from ..training.manifest import load_training_manifest
from ..training.runner import TrainConfig, _json_write

COHORT_SCHEMA = 'compact_pino.rapid-cohort.v1'
SCHEDULE_SCHEMA = 'compact_pino.fixed-exposure-schedule.v1'
BUNDLE_SCHEMA = 'compact_pino.rapid-training-bundle.v1'
STAGES = {'terminal': 1, 'near_terminal': 2, 'high': 2, 'ordinary': 1}


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=reject)


def file_receipt(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('asset SHA256 mismatch: '+str(path))
    return dict(path=str(path), sha256=digest)


def _asset(value, parent):
    if not isinstance(value, dict) or not isinstance(value.get('sha256'), str):
        raise ValueError('evidence requires path and sha256')
    path = Path(value['path'])
    return file_receipt(path if path.is_absolute() else parent/path, value['sha256'])


def native_retention(record):
    case = record['case']
    denominator = float(case['native_full_editable_moments'][0])
    if denominator <= 0:
        raise ValueError('editable mass must be positive')
    ratio = float((case['native_moments'][0]-case['fixed_moments'][0])/denominator)
    if not np.isfinite(ratio) or not -1e-12 <= ratio <= 1+1e-12:
        raise ValueError('invalid native editable retention')
    return float(np.clip(ratio, 0, 1))


def verify_cohort(data, path):
    """Verify binding and bytes of Work's declarations, not recompute geometry."""
    path = Path(path).resolve(strict=True)
    raw = read_json(path)
    if raw.get('schema') != COHORT_SCHEMA or raw.get('training_manifest_sha256') != data.manifest_sha256:
        raise ValueError('cohort must bind the admitted training manifest')
    requested = raw.get('records')
    if not isinstance(requested, list) or not requested:
        raise ValueError('nonempty cohort records required')
    train = {row['record_id']: row for row in data.train}
    seen, geometries, sources, receipts = set(), defaultdict(Counter), set(), []
    synthetic = data.scope == 'synthetic'
    for item in requested:
        rid = item.get('record_id')
        if rid not in train or rid in seen:
            raise ValueError('cohort IDs must be unique admitted train records: '+str(rid))
        seen.add(rid)
        row = train[rid]; case = row['case']; meta = case['meta']
        target = row.get('residual_target')
        if (not row.get('residual_valid') or not isinstance(target, dict) or target.get('valid') is not True
                or row.get('residual_weight', row['moment_weight']) <= 0 or not target.get('multipliers')):
            raise ValueError('cohort needs qualified positive-weight stationarity and constraint targets: '+rid)
        stage = item.get('trajectory_stage')
        if stage not in STAGES:
            raise ValueError('cohort stage must be terminal/near_terminal/high/ordinary')
        ratio = native_retention(row)
        if stage == 'high' and not (.7 <= ratio < 1-1e-8):
            raise ValueError('high cohort row requires nontrivial native retention >= .7: '+rid)
        geometries[meta['geometry_id']][stage] += 1
        sources.add(meta['source_group'])
        if not synthetic:
            evidence = _asset(item.get('legality_evidence'), path.parent)
            declaration = read_json(evidence['path'])
            if (declaration.get('schema') != 'compact_pino.task-legality.v1'
                    or declaration.get('record_id') != rid
                    or declaration.get('case_sha256') != row['case_sha256']
                    or declaration.get('geometry_legal') is not True
                    or declaration.get('pivot_retained') is not True):
                raise ValueError('legality declaration does not certify this total-geometry task: '+rid)
            if type(declaration.get('native_terminal_accepted')) is not bool:
                raise ValueError('explicit terminal acceptance Boolean required')
            if stage == 'terminal' and not declaration['native_terminal_accepted']:
                raise ValueError('terminal cohort row requires native terminal acceptance')
            evidence_parent = Path(evidence['path']).parent
            task = _asset(declaration.get('task_asset'), evidence_parent)
            originals = declaration.get('original_checks')
            if not isinstance(originals, list) or not originals:
                raise ValueError('original native legality/trajectory evidence required')
            originals = [_asset(value, evidence_parent) for value in originals]
            receipts.append(dict(record_id=rid, declaration=evidence, task=task, original_checks=originals,
                                 verification='external_declaration_and_asset_binding_not_independent_geometry_recompute'))
    if not synthetic and (len(requested) != 24 or len(geometries) != 4 or
                          any(dict(counts) != STAGES for counts in geometries.values())):
        raise ValueError('first real cohort requires four geometries with the 1/2/2/1 layout, 24 distinct queries')
    if not synthetic and len(sources) < 3:
        raise ValueError('first cohort requires at least three training source groups; four are preferred')
    return dict(path=str(path), sha256=sha256(path), record_ids=[item['record_id'] for item in requested],
                unique_queries=len(seen), geometries={key: dict(value) for key, value in geometries.items()},
                source_groups=sorted(sources), legality_receipts=receipts,
                scope=data.scope, scientific_coverage_sufficient=False)


class _Picker:
    """Round-robin shuffled sources, then no-replacement passes within a source."""
    def __init__(self, records, rng):
        self.rng = rng
        self.sources = defaultdict(list)
        for row in records:
            self.sources[row['case']['meta']['source_group']].append(row['record_id'])
        self.source_order, self.orders = [], {}

    def next(self):
        if not self.sources:
            return None
        if not self.source_order:
            self.source_order = self.rng.permutation(sorted(self.sources)).tolist()
        source = self.source_order.pop()
        if not self.orders.get(source):
            self.orders[source] = self.rng.permutation(sorted(self.sources[source])).tolist()
        return self.orders[source].pop()


def build_schedule(data, cohort_ids, *, total_steps=10000, seed=830251):
    if type(total_steps) is not int or total_steps < 104:
        raise ValueError('freeze at least 104 steps, optionally the complete long-run horizon')
    by_id = {row['record_id']: row for row in data.train}
    if not cohort_ids or len(set(cohort_ids)) != len(cohort_ids) or not set(cohort_ids) <= set(by_id):
        raise ValueError('unique admitted train cohort required')
    if len(cohort_ids) > 26:
        raise ValueError('cohort cannot receive two occurrences by step 52')
    rng = np.random.default_rng(seed)
    buckets = dict(general=list(data.train), physics=[by_id[rid] for rid in cohort_ids], high=[], low=[], jacobian=[])
    for row in data.train:
        bucket = retention_bin(native_retention(row))
        if bucket == 'high': buckets['high'].append(row)
        if bucket in ('low', 'near_zero'): buckets['low'].append(row)
        if row['case']['meta']['derivative_valid'] and row['jacobian_weight'] > 0:
            buckets['jacobian'].append(row)
    pickers = {key: _Picker(rows, rng) for key, rows in buckets.items()}
    if not buckets['high'] and data.scope != 'synthetic':
        raise ValueError('a real schedule needs admitted high-retention queries')
    ids, fallback = [], Counter()
    def pick(bucket):
        rid = pickers[bucket].next()
        if rid is None:
            fallback[bucket] += 1; rid = pickers['general'].next()
        ids.append(rid)
    for _ in range(2):
        ids.extend(rng.permutation(cohort_ids).tolist())
        while len(ids) % 26:
            pick('general')
    pattern = ('physics', 'high', 'jacobian', 'general', 'physics', 'high', 'low', 'general')
    high_count = sum(retention_bin(native_retention(by_id[rid])) == 'high' for rid in ids)
    for step in range(52, total_steps):
        requested = pattern[(step-52) % len(pattern)]
        if step < 104 and high_count < 26 and (104-step <= 26-high_count):
            requested = 'high'
        pick(requested)
        if retention_bin(native_retention(by_id[ids[-1]])) == 'high': high_count += 1
    first52, first104 = Counter(ids[:52]), Counter(ids[:104])
    if any(first52[rid] < 2 for rid in cohort_ids):
        raise AssertionError('cohort exposure prefix construction failed')
    report = dict(planned_queries=len(ids), unique_queries=len(set(ids)),
        first52_cohort_occurrences={rid: first52[rid] for rid in cohort_ids},
        first104_high_occurrences=sum(count for rid, count in first104.items()
            if retention_bin(native_retention(by_id[rid])) == 'high'),
        sources=dict(Counter(by_id[rid]['case']['meta']['source_group'] for rid in ids)),
        retention_bins=dict(Counter(retention_bin(native_retention(by_id[rid])) for rid in ids)),
        requested_strata_fallbacks=dict(fallback),
        count_scope='planned_forward_queries_not_committed_objective_exposure')
    return dict(schema=SCHEDULE_SCHEMA, training_manifest_sha256=data.manifest_sha256,
                record_ids=ids, seed=seed, report=report)


def prepare_pair(manifest, evaluation_protocol, registered_protocol, cohort, calibration, output,
                 *, mode='trial', full_cell_rule='gl2', total_steps=10000, seed=830251, cache_directory=None):
    data = load_training_manifest(manifest, mode=mode, cache_directory=cache_directory)
    cohort_receipt = verify_cohort(data, cohort)
    old_protocol = read_json(evaluation_protocol)
    ids = _protocol(old_protocol)
    if ids != {row['record_id'] for row in data.development}:
        raise ValueError('development set must equal the frozen evaluation protocol')
    registered = file_receipt(registered_protocol, old_protocol['registered_protocol_sha256'])
    calibration_receipt = file_receipt(calibration)
    calibrated = read_json(calibration)
    expected = dict(schema='compact_pino.rapid-calibration.v1', training_manifest_sha256=data.manifest_sha256,
                    cohort_sha256=cohort_receipt['sha256'], full_cell_rule=full_cell_rule,
                    training_scope='response', response_zero_init=True, seed=seed,
                    status='CALIBRATED_NOT_ACCEPTANCE', parameter_updates=0)
    if any(calibrated.get(key) != value for key, value in expected.items()):
        raise ValueError('calibration must match the exact data/cohort/initialization/numerical configuration')
    weights = calibrated.get('weights', {})
    if any(not isinstance(weights.get(key), (int, float)) or isinstance(weights.get(key), bool)
           or not np.isfinite(weights[key]) or weights[key] <= 0 for key in
           ('jacobian_loss_weight', 'residual_loss_weight', 'constraint_loss_weight')):
        raise ValueError('finite positive calibrated M/J/physics coefficients required')
    if weights['jacobian_loss_weight'] != 1.:
        raise ValueError('paired gradient calibration is defined for jacobian_loss_weight=1')
    schedule = build_schedule(data, cohort_receipt['record_ids'], total_steps=total_steps, seed=seed)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    _json_write(output/'schedule.json', schedule)
    schedule_receipt = file_receipt(output/'schedule.json')
    milestones = sorted({26, 52, 78, 104, *range(500, total_steps+1, 500), total_steps})
    declaration = dict(schema='compact_pino.rapid-preregistration.v1',
        inherited_protocol=registered, inherited_evaluation_protocol=file_receipt(evaluation_protocol),
        training_manifest=file_receipt(manifest, data.manifest_sha256), cohort=cohort_receipt,
        calibration=calibration_receipt, exposure_schedule=schedule_receipt,
        evaluation_steps=milestones, selection_steps=[step for step in milestones if step >= 52],
        constraints='same teacher/source roles/5% primary/1.05 guards; no new labels or GO generated',
        initial_weights_sha256=calibrated.get('initial_weights_sha256'),
        initial_weights_scope='geometry_and_hidden', zero_output_layer=True,
        pilot_budget=dict(max_steps=104, batch_size=1, max_wall_seconds=21600),
        proposed_long_budget=dict(max_steps=total_steps, max_wall_seconds=172800,
                                  admission='same screened candidate/config/manifest/protocol; explicit long admission required'))
    _json_write(output/'registered_protocol.json', declaration)
    protocol = dict(old_protocol, protocol_id=old_protocol['protocol_id']+'-response-repair',
        registered_protocol_sha256=sha256(output/'registered_protocol.json'), evaluation_steps=milestones,
        selection_steps=declaration['selection_steps'],
        selection_minimum_physics_exposure=dict(record_ids=cohort_receipt['record_ids'], minimum_per_record=2,
                                               objectives=['residual', 'constraint']),
        early_stop=dict(consecutive_primary_and_high_regressions=2, after_step=52))
    _json_write(output/'protocol.json', protocol)
    arms = {}
    for name, kind in (('A_old_response', 'bounded_cubic'), ('B_monotone_response', 'monotone_warp')):
        config = TrainConfig(seed=seed, response_type=kind, response_zero_init=True, training_scope='response',
            frozen_feature_cache_bytes=256*1024**2, query_cache_bytes=256*1024**2,
            full_cell_rule=full_cell_rule, sampler='fixed_schedule',
            exposure_schedule_sha256=schedule_receipt['sha256'], **weights).validate()
        _json_write(output/(name+'.json'), asdict(config))
        arms[name] = file_receipt(output/(name+'.json'))
    bundle = dict(schema=BUNDLE_SCHEMA, status='PREPARED_NOT_AUTHORIZED_NOT_TRAINED',
        manifest=file_receipt(manifest, data.manifest_sha256), cohort=cohort_receipt,
        calibration=calibration_receipt, schedule=schedule_receipt, configs=arms,
        evaluation_protocol=file_receipt(output/'protocol.json'),
        registered_protocol=file_receipt(output/'registered_protocol.json'),
        initial_weights_sha256=calibrated.get('initial_weights_sha256'),
        initial_weights=calibrated.get('initial_weights'), planned_schedule= schedule['report'],
        scientific_promotion='NOT_EVALUATED')
    _json_write(output/'bundle.json', bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'evaluation-protocol', 'registered-protocol', 'cohort', 'calibration', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('trial', 'smoke'), default='trial')
    parser.add_argument('--full-cell-rule', choices=('tetra4', 'gl2', 'gl3'), default='gl2')
    parser.add_argument('--total-steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=830251)
    parser.add_argument('--cache-directory')
    args = parser.parse_args()
    result = prepare_pair(args.manifest, args.evaluation_protocol, args.registered_protocol,
                          args.cohort, args.calibration, args.output, mode=args.mode,
                          full_cell_rule=args.full_cell_rule, total_steps=args.total_steps,
                          seed=args.seed, cache_directory=args.cache_directory)
    print(json.dumps(dict(status=result['status'], bundle=str(Path(args.output)/'bundle.json'),
                         schedule=result['planned_schedule']), indent=2))


if __name__ == '__main__': main()
