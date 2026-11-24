"""Paired deterministic evaluation. Teacher files enter scoring only after prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import (
    prepare_surface, prepare_measure, evaluate_response,
)
from .diagnostics import accepted_directions, moment_errors, jacobian_errors


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=1, allow_nan=False)+'\n')
    temp.replace(p)


def eval_partitioned(blocks, surface, operation, fixed, velocity, bandwidth):
    """Compute both readouts on exactly the same local retention and derivative.

    The scalar-token control is given the *computed fine mass and its derivative*
    for each token. Differences thus isolate where within a token mass remains.
    The native teacher is never consulted to calculate these fractions.
    """
    grid = 16
    fixed = np.asarray(fixed, float)
    full, near = blocks.classify(surface, bandwidth, operation)
    ijk = np.floor((blocks.centers-blocks.origin)*grid/(blocks.side*blocks.pitch)).astype(int)
    ids = np.ravel_multi_index(ijk.T, (grid,)*3)
    n = grid**3
    total, part = np.zeros((n,10)), np.zeros((n,10))
    deriv = None if velocity is None else np.zeros((n, 10, velocity.shape[2]))
    np.add.at(total, ids, blocks.moments)
    np.add.at(part, ids[full], blocks.moments[full])
    counts = dict(occupied_blocks=len(ids), classified_full=int(full.sum()), near_blocks=int(near.sum()),
                  near_nodes=0, transition_nodes=0, ambiguous=0)
    for k in np.unique(ids[near]):
        for p, w in blocks.chunks(near & (ids == k)):
            result = evaluate_response(prepare_measure(p, w), surface, bandwidth=bandwidth,
                                       operation=operation, vertex_velocity=velocity)
            part[k] += result.moments.numpy()
            if deriv is not None:
                if result.jacobian is None:
                    deriv = None
                else:
                    deriv[k] += result.jacobian.numpy()
            counts['near_nodes'] += len(w)
            counts['transition_nodes'] += result.diagnostics['active_nodes']
            counts['ambiguous'] += result.diagnostics['ambiguous_nodes']
    def entry(m, j):
        return dict(M=(m+fixed).tolist(), J=None if j is None else j.tolist())
    results = {'local': entry(part.sum(0), None if deriv is None else deriv.sum(0))}
    ti = np.indices((grid,)*3).reshape(3,-1).T
    for g in (8, 16):
        coarse = np.ravel_multi_index((ti//(grid//g)).T, (g,)*3)
        all_m = np.zeros((g**3,10))
        keep_m = np.zeros(g**3)
        np.add.at(all_m, coarse, total)
        np.add.at(keep_m, coarse, part[:,0])
        mask = all_m[:,0] > 0
        mean = all_m[mask] / all_m[mask,:1]
        j = None
        if deriv is not None:
            dj = np.zeros((g**3, deriv.shape[2]))
            np.add.at(dj, coarse, deriv[:,0])
            j = mean.T @ dj[mask]
        results[f'scalar_token{g}'] = entry(keep_m[mask] @ mean, j)
    return results, counts


def score(out, row, meta, direction):
    ref = np.asarray(row['M'])
    mass, length = meta['reference_editable_mesh'][0], meta['length']
    for result in out.values():
        result['moment_error'] = moment_errors(result['M'], ref, mass, length)
        if result['J'] is None:
            continue
        dim = np.asarray(result['J']).shape[1]
        if dim == 10:
            refj = row['J_raw'] if row['full_j_valid'] else None
            basis = None
        else:
            refj = row.get('restricted_J') if row.get('restricted_valid') else None
            basis = row.get('requested_chart', {}).get('basis_raw')
        if refj is not None:
            result['jacobian_error'] = jacobian_errors(result['J'], refj, mass, length,
                    float(np.linalg.norm(row['query'])), direction=direction if dim==10 else None, basis=basis)


def run(args):
    torch.set_num_threads(1)
    trace = read(args.trajectory)['solver']['trace'] if args.trajectory else []
    directions = accepted_directions(trace)
    files = sorted({p for pattern in args.rows for p in (args.native_root/'rows').glob(pattern+'.json')})
    if not files:
        raise ValueError('no native query records matched')
    for p in files:
        row = read(p)
        name, gid = row['row_id'], row['geometry_id']
        final = args.output / (name + '.json')
        if final.exists() and not args.replace:
            raise FileExistsError(final)
        meta = read(args.indexes/gid/'identity.json')
        if meta['side'] != 256:
            raise ValueError('paired controls require native256')
        record = dict(row_id=name, geometry=gid, surface=row['surface'], query=row['query'],
                      native_M=row['M'], native_J=row['J_raw'], ambient_event_valid=row['ambient_event_valid'],
                      native_moment_valid=row['moment_valid'], native_full_J_valid=row['full_j_valid'],
                      operand_valid=row['operand_valid'], operand_sha256=row.get('operand_sha256'),
                      source_error=row.get('source_error'),
                      material_error=row.get('material_error'), native_truth_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                      material_identities={k:v['identity'] for k,v in meta['controls'].items()},
                      source_record=row['source_record'], grid_provenance=meta['grid_provenance'],
                      methods={}, interpreter='CPU float64: stored motion, fixed geometry; timing excludes native operand generation')
        if row.get('restricted_valid'):
            record['native_restricted_J'] = row['restricted_J']
            record['native_chart_basis_raw'] = row['requested_chart']['basis_raw']
            record['restricted_event_valid'] = row['restricted_event_valid']
        if not row['operand_valid']:
            write(final, record)
            continue
        operand_path = args.native_root/'operands'/(name+'.npz')
        if hashlib.sha256(operand_path.read_bytes()).hexdigest()!=row['operand_sha256']:
            raise ValueError('operand hash mismatch')
        operand = FiniteOperand.load(operand_path)
        t = time.perf_counter()
        surface = prepare_surface(operand.vertices, operand.faces)
        record['surface_index_seconds'] = time.perf_counter()-t
        num = int(name.rsplit('_',1)[1]) if gid=='noodles' else None
        v = directions.get(num)
        record['outgoing_native_unit_tangent'] = v.tolist() if v is not None else None
        for method in ('rle', 'conservative'):
            b = BlockMeasure.open(args.indexes/gid/method, identity=meta['controls'][method]['identity'], block_voxels=8)
            t = time.perf_counter()
            results, counts = eval_partitioned(b, surface, operand.operation, meta['fixed_moments'],
                                               operand.vertex_velocity, meta['pitch'])
            elapsed = time.perf_counter()-t
            if row['moment_valid']:
                score(results, row, meta, v)
            record['methods'][method] = dict(results=results, counts=counts, compute_seconds=elapsed)
        write(final, record)
        print(name, row['moment_valid'], round(record['methods']['conservative']['compute_seconds'],3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-root', type=Path, required=True)
    ap.add_argument('--indexes', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--trajectory', type=Path)
    ap.add_argument('--replace', action='store_true')
    ap.add_argument('rows', nargs='+', help='shell-quoted query identifiers or glob patterns')
    run(ap.parse_args())
