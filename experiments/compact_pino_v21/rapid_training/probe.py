"""Probe a trained checkpoint's loss gradients on admitted training sources.

The saved response weights are retained exactly. Only response gradients are
enabled; no optimizer, output-layer reset, training authorization or quality GO
is created. Use best.json/last.json to bind the exact checkpoint being diagnosed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import torch

from .calibrate import probe_terms, select_training_records
from .prepare import _asset, read_json
from ..p0_diagnostics.baseline_cases import sha256
from ..precision.frozen_checkpoint import load_frozen_checkpoint
from ..training.manifest import load_training_manifest
from ..training.runner import _tensor_digest
from ..training.state import seed_everything


SCHEMA = 'compact_pino.rapid-checkpoint-gradient-probe.v1'


def weighted_gradient_report(probe, config):
    """Apply saved coefficients to norms/dot products without summing norms.

    An independent J probe is still reported when lambda_J is zero, but it then
    contributes zero to the actual checkpoint's weighted M/J gradient.
    """
    norms, cosines = probe['gradient_norms'], probe['gradient_cosines']
    coefficients = dict(M=1., J=config.jacobian_loss_weight,
                        R=config.residual_loss_weight, C=config.constraint_loss_weight)
    if any(not math.isfinite(value) or value < 0 for value in coefficients.values()):
        raise ValueError('saved loss coefficients must be finite and nonnegative')
    def dot(left, right):
        if norms[left] == 0 or norms[right] == 0:
            return 0.
        cosine = cosines.get(left+'_'+right, cosines.get(right+'_'+left))
        if cosine is None:
            raise ValueError('nonzero gradients require a measured pairwise cosine')
        return norms[left]*norms[right]*cosine
    def combined(keys):
        squared = sum((coefficients[key]*norms[key])**2 for key in keys)
        squared += 2*sum(coefficients[left]*coefficients[right]*dot(left, right)
                         for left, right in itertools.combinations(keys, 2))
        return math.sqrt(max(0., squared))
    mj, physics = combined(('M', 'J')), combined(('R', 'C'))
    weighted = {key: coefficients[key]*norms[key] for key in coefficients}
    direction_cosines = {}
    for term in ('R', 'C'):
        denominator = mj*norms[term]
        value = ((dot('M', term)+coefficients['J']*dot('J', term))/denominator
                 if denominator else None)
        direction_cosines['weighted_MJ_'+term] = None if value is None else min(1., max(-1., value))
    return dict(coefficients=coefficients, weighted_term_norms=weighted,
        combined_MJ_norm=mj, combined_physics_norm=physics,
        combined_total_norm=combined(('M', 'J', 'R', 'C')),
        fractions_of_combined_MJ={key: value/mj if mj else None for key, value in weighted.items()},
        combined_physics_fraction_of_MJ=physics/mj if mj else None,
        gradient_cosines=direction_cosines,
        note='Norm ratios describe this query at this checkpoint; they are not loss fractions or guaranteed future shares.')


def run_probe(manifest, cohort, checkpoint_receipt, output, *, mode='trial', device='cpu',
              max_records=12, cache_directory=None):
    if mode not in ('trial', 'smoke'):
        raise ValueError('checkpoint probe mode must be trial or smoke')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing checkpoint diagnostic')
    data = load_training_manifest(manifest, mode=mode, cache_directory=cache_directory)
    cohort_path = Path(cohort).resolve(strict=True)
    cohort_hash = sha256(cohort_path)
    packet = read_json(cohort_path)
    receipt_path = Path(checkpoint_receipt).resolve(strict=True)
    receipt_hash = sha256(receipt_path)
    checkpoint = _asset(read_json(receipt_path), receipt_path.parent)
    frozen = load_frozen_checkpoint(checkpoint['path'], checkpoint['sha256'], device=device)
    model, config = frozen.model, frozen.config
    source_receipt = dict(frozen.receipt)
    # Discard inference-only caches BEFORE enabling response gradients. From
    # here onward no FrozenCheckpoint inference method is called or reused.
    frozen.close()
    if source_receipt.get('training_manifest_sha256') != data.manifest_sha256:
        raise ValueError('checkpoint probe must use its exact training manifest')
    if torch.device(device).type == 'cuda' and config.deterministic:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')
    rows, selection = select_training_records(data, packet, max_records=max_records, seed=config.seed)
    before = _tensor_digest(model.state_dict())
    seed_everything(config.seed, config.deterministic)
    # This changes autograd eligibility and module modes only. It does not
    # construct/reinitialize the response, even if the original config records
    # response_zero_init=True for the historical start of this trained run.
    model.set_training_scope('response')
    model.train()
    if _tensor_digest(model.state_dict()) != before:
        raise ValueError('enabling diagnostic gradients changed saved tensors')
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if not name.startswith('response.')):
        raise ValueError('checkpoint probe requires frozen geometry parameters')
    records = []
    try:
        for row in rows:
            case = row['case']
            encoded = model.encode_geometry(case['blocks'], case['fixed_moments'],
                                            density=case['meta']['density'])
            prediction = model.evaluate_operand(encoded, case['operand'],
                with_jacobian=case['meta']['derivative_valid'], bandwidth=case['blocks'].pitch,
                chunk_size=config.chunk_size)
            measured = probe_terms([prediction], [row], model)
            records.append(dict(record_id=row['record_id'], role='train', case_sha256=row['case_sha256'],
                source_group=case['meta']['source_group'], geometry_id=case['meta']['geometry_id'],
                query_id=case['meta']['query_id'], probe=measured,
                weighted_gradients=weighted_gradient_report(measured, config),
                predicted_moments=prediction.moments.detach().cpu().tolist(),
                predicted_jacobian=None if prediction.jacobian is None else prediction.jacobian.detach().cpu().tolist()))
            del prediction, encoded
        after = _tensor_digest(model.state_dict())
        if before != after or any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError('checkpoint probe changed state or accumulated parameter gradients')
        if (sha256(data.path) != data.manifest_sha256 or sha256(cohort_path) != cohort_hash
                or sha256(receipt_path) != receipt_hash or sha256(checkpoint['path']) != checkpoint['sha256']):
            raise ValueError('diagnostic input identity changed while probing')
        report = dict(schema=SCHEMA, status='TRAIN_GRADIENT_DIAGNOSTIC_NOT_ACCEPTANCE',
            mode=mode, scope=data.scope, training_manifest_sha256=data.manifest_sha256,
            cohort_sha256=cohort_hash, checkpoint_receipt=dict(path=str(receipt_path), sha256=receipt_hash),
            checkpoint=source_receipt, checkpoint_training_scope=config.training_scope,
            probe_training_scope='response', response_type=config.response_type,
            full_cell_rule=config.full_cell_rule, seed=config.seed,
            weights=dict(jacobian_loss_weight=config.jacobian_loss_weight,
                         residual_loss_weight=config.residual_loss_weight,
                         constraint_loss_weight=config.constraint_loss_weight),
            selection=selection, records=records, initial_state_sha256=before, final_state_sha256=after,
            state_unchanged=True, output_layer_reset=False, parameter_grad_fields_untouched=True,
            efficiency=model.efficiency_statistics(), parameter_updates=0,
            correctness_gate='NOT_EVALUATED', native_solver_acceptance='NOT_EVALUATED',
            scientific_promotion='NOT_EVALUATED', training_authorized=False)
    finally:
        model.clear_efficiency_caches()
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
    parser.add_argument('--checkpoint-receipt', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-records', type=int, default=12)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache-directory')
    args = parser.parse_args()
    report = run_probe(args.manifest, args.cohort, args.checkpoint_receipt, args.output,
        mode=args.mode, device=args.device, max_records=args.max_records, cache_directory=args.cache_directory)
    print(json.dumps(dict(status=report['status'], checkpoint_sha256=report['checkpoint']['sha256'],
                         parameter_updates=0, report=str(Path(args.output).resolve())), indent=2))


if __name__ == '__main__':
    main()
