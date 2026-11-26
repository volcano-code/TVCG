"""Train-source-only, zero-update calibration for the paired rapid trial.

No optimizer is created. Both response profiles start from the same zero-output
weights and the same static geometry features. This is a loss-scale receipt,
not training authorization, physical acceptance, or a model promotion result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict
import gc
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21
from ..p0_diagnostics.baseline_cases import moment_scale, sha256
from ..training.losses import physical_loss
from ..training.manifest import load_training_manifest
from ..training.runner import Trainer, TrainConfig, model_kwargs, _tensor_digest
from ..training.state import seed_everything


SCHEMA = 'compact_pino.rapid-calibration.v1'
COHORT_SCHEMA = 'compact_pino.rapid-cohort.v1'
PROFILES = ('bounded_cubic', 'monotone_warp')
TARGET_GRADIENT_FRACTION = .1


def select_training_records(data, cohort, *, max_records=12, seed=830251):
    """Validate cohort membership, then sample eligible sources round-robin.

    Role comes from the strict loader, never from an optional cohort assertion.
    A held-out or unknown ID is an error, even if it would later be filtered out.
    Task legality remains the responsibility of the separately frozen cohort.
    """
    if type(max_records) is not int or max_records < 1 or type(seed) is not int:
        raise ValueError('positive max_records and integer seed required')
    if not isinstance(cohort, dict) or cohort.get('schema') != COHORT_SCHEMA:
        raise ValueError('explicit rapid-cohort.v1 required')
    if cohort.get('training_manifest_sha256') != data.manifest_sha256:
        raise ValueError('cohort training manifest SHA256 mismatch')
    entries = cohort.get('records')
    if not isinstance(entries, list) or not entries:
        raise ValueError('cohort requires nonempty records')
    by_id = {row['record_id']: row for row in data.train}
    seen, groups, excluded = set(), defaultdict(list), []
    for entry in entries:
        rid = entry.get('record_id') if isinstance(entry, dict) else None
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError('cohort record IDs must be nonempty and unique')
        seen.add(rid)
        if rid not in by_id or by_id[rid].get('role') != 'train':
            raise ValueError('calibration permits only admitted train records: '+rid)
        if 'role' in entry and entry['role'] != 'train':
            raise ValueError('cohort role contradicts train-only calibration: '+rid)
        row = by_id[rid]
        weight = float(row.get('residual_weight', row['moment_weight']))
        if row.get('residual_valid') is not True or not math.isfinite(weight) or weight <= 0:
            excluded.append(dict(record_id=rid, reason='NO_VALID_POSITIVE_PHYSICS_MEMBERSHIP'))
            continue
        source = row['case']['meta']['source_group']
        if not isinstance(source, str) or not source:
            raise ValueError('calibration record requires a source group')
        groups[source].append(row)
    if not groups:
        raise ValueError('no positive, residual-qualified train records in cohort')
    if max_records < len(groups):
        raise ValueError('max_records must cover every eligible cohort source group')
    rng = random.Random(seed)
    queues = {}
    for source in sorted(groups):
        rows = sorted(groups[source], key=lambda row: row['record_id'])
        rng.shuffle(rows)
        queues[source] = deque(rows)
    selected = []
    while len(selected) < max_records and any(queues.values()):
        for source in sorted(queues):
            if queues[source] and len(selected) < max_records:
                selected.append(queues[source].popleft())
    receipt = dict(strategy='seeded_within_source_round_robin', seed=seed,
        max_records=max_records, unique_records=len(selected),
        eligible_unique_records=sum(map(len, groups.values())),
        source_groups=sorted(groups),
        source_counts={source: sum(row['case']['meta']['source_group'] == source for row in selected)
                       for source in sorted(groups)}, excluded=excluded)
    return selected, receipt


def _cosine(left, right):
    denominator = torch.linalg.vector_norm(left)*torch.linalg.vector_norm(right)
    if not bool(torch.isfinite(denominator)) or denominator.item() == 0:
        return None
    value = float(torch.dot(left, right)/denominator)
    return min(1., max(-1., value))


def probe_terms(predictions, records, model):
    """Probe M/J/R/C on ONE prediction graph without filling parameter.grad.

    Uses the existing physical_loss for all normalization and qualification.
    Each isolated term retains its own positive-weight denominator. Missing J
    supervision has zero gradient and zero denominator, not a fabricated label.
    The returned JSON-compatible data is also suitable for checkpoint probes.
    """
    if not records or len(records) != len(predictions):
        raise ValueError('nonempty predictions and records must align')
    response = getattr(model, 'response', model)
    named = [(name, parameter) for name, parameter in response.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError('probe requires trainable response parameters')
    parameters = tuple(parameter for _, parameter in named)
    total_size = sum(parameter.numel() for parameter in parameters)
    vectors, terms = {}, {}
    settings = dict(M=(1., 0., 0., 0.), J=(0., 1., 0., 0.),
                    R=(0., 0., 1., 0.), C=(0., 0., 0., 1.))
    names = dict(M='moment', J='jacobian', R='residual', C='constraint')
    # The combined call establishes effective memberships even when one
    # isolated component would have no usable rows and physical_loss refuses it.
    _, membership = physical_loss(predictions, records, jacobian_loss_weight=1.,
                                  residual_loss_weight=1., constraint_loss_weight=1.)
    for key, (mw, jw, rw, cw) in settings.items():
        prefix = names[key]
        denominator = float(membership[prefix+'_denominator'])
        if denominator == 0:
            vector = torch.zeros(total_size, dtype=torch.float64)
            value = 0.
            info = membership
        else:
            rows = [dict(row, moment_weight=row['moment_weight']*mw,
                         jacobian_weight=row['jacobian_weight'] if jw else 0.) for row in records]
            loss, info = physical_loss(predictions, rows, jacobian_loss_weight=jw,
                                      residual_loss_weight=rw, constraint_loss_weight=cw)
            value = float(loss.detach())
            if loss.requires_grad:
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                vector = torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient)
                                    .detach().reshape(-1).to(device='cpu', dtype=torch.float64)
                                    for parameter, gradient in zip(parameters, gradients)])
            else:
                vector = torch.zeros(total_size, dtype=torch.float64)
        if not math.isfinite(value) or not bool(torch.isfinite(vector).all()):
            raise ValueError('nonfinite loss/gradient during calibration: '+key)
        vectors[key] = vector
        terms[key] = dict(loss=value, denominator=denominator,
            effective_queries=int(sum(membership[prefix+'_mask'])),
            gradient_status='NO_EFFECTIVE_DENOMINATOR' if denominator == 0 else
                            'ZERO_GRADIENT' if not bool(vector.abs().sum()) else 'AVAILABLE')
        if key == 'J':
            terms[key]['qualification_reasons'] = membership['jacobian_reasons']
        if key in ('R', 'C'):
            terms[key]['qualification_reasons'] = membership['residual_reasons']
            terms[key]['profiles'] = membership['residual_profiles']
            terms[key]['approximations'] = membership['residual_approximations']
    vectors['MJ'] = vectors['M']+vectors['J']
    norms = {name: float(torch.linalg.vector_norm(value)) for name, value in vectors.items()}
    cosines = {left+'_'+right: _cosine(vectors[left], vectors[right])
               for left, right in itertools.combinations(('M', 'J', 'R', 'C'), 2)}
    cosines.update(MJ_R=_cosine(vectors['MJ'], vectors['R']), MJ_C=_cosine(vectors['MJ'], vectors['C']))
    return dict(terms=terms, gradient_norms=norms, gradient_cosines=cosines,
                parameter_names=[name for name, _ in named], parameter_count=total_size,
                gradient_scope='trainable_response_parameters', jacobian_loss_weight=1.,
                parameter_updates=0)


def _initial_model(config, device, initial_weights, expected_sha256):
    seed_everything(config.seed, config.deterministic)
    model = CompactPINOv21(**model_kwargs(config)).to(device)
    receipt = None
    if initial_weights is not None:
        # Reuse the SAME strict canonical checkpoint migration as actual
        # training, without constructing a Trainer, optimizer or training run.
        context = SimpleNamespace(model=model, config=config, device=device, initial_weights=None)
        Trainer._load_initial_weights(context, initial_weights, expected_sha256, 'geometry_and_hidden')
        receipt = context.initial_weights
    model.train()
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if not name.startswith('response.')):
        raise ValueError('calibration requires a frozen geometry encoder and attention')
    for name in ('shared_encoder', 'geometry_attention'):
        module = getattr(model, name, None)
        if module is not None and any(child.training for child in module.modules()):
            raise ValueError('frozen geometry modules and buffers must remain in eval mode')
    return model, receipt


def _shared_weights(records):
    samples = {'R': [], 'C': []}
    for record in records:
        for arm in record['arms'].values():
            norms = arm['probe']['gradient_norms']
            ratios = {}
            for term in samples:
                ratio = TARGET_GRADIENT_FRACTION*norms['MJ']/norms[term] if norms[term] > 0 else None
                valid = ratio is not None and math.isfinite(ratio) and ratio > 0
                ratios[term] = ratio if valid else None
                if valid:
                    samples[term].append(ratio)
            arm['candidate_weight_ratios'] = ratios
    if any(not sample for sample in samples.values()):
        raise ValueError('cannot calibrate a physical term without finite positive M/J-to-physics gradient ratios')
    weights = {term: statistics.median(values) for term, values in samples.items()}
    for record in records:
        for arm in record['arms'].values():
            norms = arm['probe']['gradient_norms']
            arm['actual_weighted_gradient_fraction_of_MJ'] = {
                term: weights[term]*norms[term]/norms['MJ'] if norms['MJ'] > 0 else None
                for term in weights}
    return dict(jacobian_loss_weight=1., residual_loss_weight=weights['R'], constraint_loss_weight=weights['C']), {
        'method': 'pooled_positive_finite_per_record_per_arm_ratio_median',
        'target_fraction': TARGET_GRADIENT_FRACTION,
        'valid_ratio_counts': {term: len(values) for term, values in samples.items()},
        'note': 'The shared median does not guarantee a 10% share on every record or future update.'}


def run_calibration(manifest, cohort, output, *, mode='trial', full_cell_rule='tetra4',
                    device='cpu', seed=830251, max_records=12, initial_weights=None,
                    initial_weights_sha256=None, cache_directory=None):
    if mode not in ('trial', 'smoke'):
        raise ValueError('calibration mode must be trial or smoke')
    if (initial_weights is None) != (initial_weights_sha256 is None):
        raise ValueError('initial weights and their explicit SHA256 must be supplied together')
    target_device = torch.device(device)
    if target_device.type not in ('cpu', 'cuda'):
        raise ValueError('calibration supports CPU or CUDA devices')
    if target_device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('requested CUDA calibration but CUDA is unavailable')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing calibration receipt')
    data = load_training_manifest(manifest, mode=mode, cache_directory=cache_directory)
    cohort = Path(cohort).resolve(strict=True)
    cohort_hash = sha256(cohort)
    packet = json.loads(cohort.read_text())
    rows, selection = select_training_records(data, packet, max_records=max_records, seed=seed)
    records = [dict(record_id=row['record_id'], role='train', case_sha256=row['case_sha256'],
                    source_group=row['case']['meta']['source_group'],
                    geometry_id=row['case']['meta']['geometry_id'],
                    query_id=row['case']['meta']['query_id'],
                    residual_target_receipt=row.get('residual_target_receipt'), arms={}) for row in rows]
    provenance, baseline_state = {}, None
    for profile in PROFILES:
        config = TrainConfig(seed=seed, full_cell_rule=full_cell_rule, response_type=profile,
                             response_zero_init=True, training_scope='response').validate()
        model, initialization = _initial_model(config, target_device, initial_weights, initial_weights_sha256)
        before = _tensor_digest(model.state_dict())
        if baseline_state is None:
            baseline_state = before
        elif before != baseline_state:
            raise ValueError('paired calibration arms do not share exactly the same initial tensors')
        try:
            for row, result in zip(rows, records):
                case = row['case']
                encoded = model.encode_geometry(case['blocks'], case['fixed_moments'],
                                                density=case['meta']['density'])
                prediction = model.evaluate_operand(encoded, case['operand'],
                    with_jacobian=case['meta']['derivative_valid'],
                    bandwidth=case['blocks'].pitch, chunk_size=config.chunk_size)
                result['arms'][profile] = dict(probe=probe_terms([prediction], [row], model),
                    initial_moments=prediction.moments.detach().cpu().tolist(),
                    initial_jacobian=None if prediction.jacobian is None else prediction.jacobian.detach().cpu().tolist())
                del prediction, encoded
            after = _tensor_digest(model.state_dict())
            if before != after or any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError('calibration changed parameters/buffers or accumulated parameter gradients')
            provenance[profile] = dict(config=asdict(config), initialization=initialization,
                initial_state_sha256=before, final_state_sha256=after,
                state_unchanged=True, parameter_grad_fields_untouched=True,
                efficiency=model.efficiency_statistics())
        finally:
            model.clear_efficiency_caches()
            del model
            gc.collect()
    for row, result in zip(rows, records):
        first, second = [result['arms'][profile] for profile in PROFILES]
        scale = moment_scale(row['case']['meta']['editable_mass'], row['case']['meta']['length_scale'])
        m0, m1 = np.asarray(first['initial_moments']), np.asarray(second['initial_moments'])
        error_m = float(np.max(np.abs(m0-m1)/scale))
        j0, j1 = first['initial_jacobian'], second['initial_jacobian']
        if (j0 is None) != (j1 is None):
            raise ValueError('paired E0 Jacobian availability differs')
        error_j = None if j0 is None else float(np.max(np.abs(np.asarray(j0)-np.asarray(j1))
            *np.asarray(row['case']['parameter_scales'])[None, :]/scale[:, None]))
        if error_m > 1e-12 or error_j is not None and error_j > 1e-12:
            raise ValueError('paired zero-output arms do not reproduce the same initial M/J')
        result['shared_initial_response'] = dict(moment_scaled_max_difference=error_m,
            jacobian_scaled_max_difference=error_j, moments_bitwise_equal=bool(np.array_equal(m0, m1)),
            jacobian_bitwise_equal=None if j0 is None else bool(np.array_equal(j0, j1)),
            status='MATCHED', jacobian_status='LABEL_UNQUALIFIED' if j0 is None else 'AVAILABLE')
    weights, rule = _shared_weights(records)
    shared_initialization = None
    initializations = [provenance[profile]['initialization'] for profile in PROFILES]
    if any(value is not None for value in initializations):
        if any(value is None for value in initializations):
            raise ValueError('paired calibration checkpoint initialization availability differs')
        shared_keys = ('path', 'sha256', 'scope', 'source_response_type', 'reset_output_keys')
        shared_initialization = {key: initializations[0][key] for key in shared_keys}
        if any({key: value[key] for key in shared_keys} != shared_initialization
               for value in initializations[1:]):
            raise ValueError('paired calibration arms must inherit the same exact checkpoint')
        if shared_initialization['sha256'] != initial_weights_sha256:
            raise ValueError('initialization receipt contradicts requested checkpoint SHA256')
    if sha256(data.path) != data.manifest_sha256 or sha256(cohort) != cohort_hash:
        raise ValueError('manifest or cohort changed during calibration')
    report = dict(schema=SCHEMA, status='CALIBRATED_NOT_ACCEPTANCE', mode=mode, scope=data.scope,
        training_manifest_sha256=data.manifest_sha256, cohort_sha256=cohort_hash,
        full_cell_rule=full_cell_rule, training_scope='response', response_zero_init=True, seed=seed,
        initial_weights_sha256=initial_weights_sha256, initial_weights=shared_initialization,
        weights=weights, calibration_rule=rule,
        selection=selection, records=records, provenance=provenance, device=str(target_device),
        parameter_updates=0, native_solver_acceptance='NOT_EVALUATED', scientific_promotion='NOT_EVALUATED')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--mode', choices=('trial', 'smoke'), default='trial')
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--full-cell-rule', choices=('tetra4', 'gl2', 'gl3'), required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    parser.add_argument('--initial-weights')
    parser.add_argument('--initial-weights-sha256')
    parser.add_argument('--seed', type=int, default=830251)
    parser.add_argument('--max-records', type=int, default=12)
    parser.add_argument('--cache-directory')
    args = parser.parse_args()
    report = run_calibration(args.manifest, args.cohort, args.output, mode=args.mode,
        full_cell_rule=args.full_cell_rule, device=args.device, seed=args.seed, max_records=args.max_records,
        initial_weights=args.initial_weights, initial_weights_sha256=args.initial_weights_sha256,
        cache_directory=args.cache_directory)
    print(json.dumps(dict(status=report['status'], weights=report['weights'],
                         parameter_updates=0, report=str(Path(args.output).resolve())), indent=2))


if __name__ == '__main__':
    main()
