"""Audit the published paired tables directly from raw predictions and the separate native archive."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

import numpy as np


def read(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root, data=None, native_dir=None):
    data = data or root/'evidence'
    native_dir = native_dir or root/'native/evidence'
    table = read(data/'comparison.json')
    geometry = read(data/'static_material.json')
    gate = read(data/'gate_status.json')
    native = read(native_dir/'native_replay.json')
    registry = {r['row_id']:r for r in native['rows']}
    observed = {p.stem:read(p) for p in (data/'replay').glob('*.json')}
    check(len(registry) == len(native['rows']) and observed.keys() == registry.keys(), 'query inventory mismatch')
    archive = native_dir/native['numerical_archive']['file']
    check(hashlib.sha256(archive.read_bytes()).hexdigest() == native['numerical_archive']['sha256'], 'target checksum mismatch')
    with np.load(archive, allow_pickle=False) as z:
        mref = dict(zip(z['moment_row_id'], z['M'], strict=True))
        jref = dict(zip(z['full_j_row_id'], z['J_raw'], strict=True))
        q = dict(zip(z['row_id'], z['raw_query'], strict=True))
        typed = {}
        for size in (3, 8):
            if f'R{size}_row_id' in z:
                for k,j,b in zip(z[f'R{size}_row_id'],z[f'R{size}_J'],z[f'R{size}_basis_raw'], strict=True):
                    typed[k] = (j,b)
    choices = {'finite_rle':('rle','local'), 'finite_conservative_token8':('conservative','scalar_token8'),
               'finite_conservative_token16':('conservative','scalar_token16'),
               'finite_conservative_local':('conservative','local')}
    pools = {'noodles':{}, 'other_geometry_probes':{}}
    for k,r in observed.items():
        truth = registry[k]
        check(r['row_id'] == k and r['geometry'] == truth['geometry_id'], k+' mismatched identity')
        np.testing.assert_array_equal(r['query'], q[k])
        for left,right in [('operand_valid','operand_valid'),('operand_sha256','operand_sha256'),
                           ('native_moment_valid','moment_valid'),('native_full_J_valid','full_j_valid'),
                           ('ambient_event_valid','ambient_event_valid')]:
            check(r.get(left) == truth.get(right), k+' '+left)
        geom = geometry[r['geometry']]
        check(r['material_identities'] == {t:v['identity'] for t,v in geom['controls'].items()}, k+' static geometry')
        if k not in mref:
            check(r['native_M'] is None and k not in table['query_strata'], k+' rejected query in scores')
            continue
        np.testing.assert_array_equal(r['native_M'], mref[k])
        if k in jref:
            np.testing.assert_array_equal(r['native_J'], jref[k])
        length = geom['length']
        scale = geom['reference_editable_mesh'][0] * np.array([1, length, length, length]+[length**2]*6)
        predictions = {m:r['methods'][t]['results'][s] for m,(t,s) in choices.items()}
        # A scalar readout oracle and the local readout must keep identical material mass.
        for t in ('rle','conservative'):
            local = r['methods'][t]['results']
            for scalar in ('scalar_token8','scalar_token16'):
                np.testing.assert_allclose(local[scalar]['M'][0],local['local']['M'][0],rtol=2e-12,atol=1e-14)
        ap = data/'analytic'/(k+'.json')
        if ap.exists():
            alt = read(ap)['result']
            if 'M' in alt:
                predictions['analytic_rle'] = alt
        scores = {}
        for name,pred in predictions.items():
            delta = np.asarray(pred['M'])-mref[k]
            metrics = {'scaled_moment':np.max(abs(delta)/scale)}
            refnorm = np.linalg.norm(mref[k])
            if refnorm:
                metrics['relative_moment'] = np.linalg.norm(delta)/refnorm
            if pred.get('J') is not None:
                j = np.asarray(pred['J'])
                chart = j.shape[1] != 10
                pair = typed.get(k) if chart else (jref.get(k), None)
                if pair is not None and pair[0] is not None:
                    ref, basis = pair
                    residual = j-ref
                    unit = residual/scale[:,None]*np.linalg.norm(q[k])
                    if chart:
                        unit = np.linalg.solve(np.linalg.qr(basis,mode='reduced')[1].T,unit.T).T
                    metrics['scaled_jacobian'] = np.linalg.norm(unit, axis=1).max()
                    jnorm = np.linalg.norm(ref)
                    if jnorm:
                        metrics['relative_jacobian'] = np.linalg.norm(residual)/jnorm
                    v = r.get('outgoing_native_unit_tangent')
                    if not chart and v is not None and np.linalg.norm(ref@v):
                        metrics['accepted_direction_relative'] = np.linalg.norm(j@v-ref@v)/np.linalg.norm(ref@v)
            scores[name] = metrics
        group = 'noodles' if r['geometry']=='noodles' else 'other_geometry_probes'
        pools[group][k] = scores
    for group, rows in pools.items():
        methods = set(choices) | {'analytic_rle'}
        mkeys = sorted(k for k,v in rows.items() if methods==set(v))
        jkeys = sorted(k for k in mkeys if all('scaled_jacobian' in s for s in rows[k].values()))
        check(table[group+'_matched_ids']=={'moment_ids':mkeys, 'jacobian_ids':jkeys}, group+' paired IDs')
        for method in methods:
            for metric in ('scaled_moment','relative_moment','scaled_jacobian','relative_jacobian','accepted_direction_relative'):
                eligible = mkeys if metric.endswith('moment') else jkeys
                values = [rows[k][method][metric] for k in eligible if metric in rows[k][method]]
                report = table[group][method][metric]
                check(len(values)==report['n'], group+' metric count')
                if values:
                    np.testing.assert_allclose([report['median'],report['p95'],report['max']],
                                               np.percentile(values,[50,95,100]),rtol=1e-11,atol=1e-15)
                else:
                    check(report['p95'] is None, group+' missing data reported numerically')
    check(gate['actual_geometry_count']==len(geometry) and
          gate['largest_scaled_static_moment_error'] <= read(root/'protocol.json')['gate_a_scaled_full_max'],
          'geometry conservation discrepancy')
    blocked = not gate['complete_training_overlay'] or not gate['certified_dev_R3'] or not gate['certified_dev_R8']
    if blocked:
        check(gate['conditional_training']=='NO-GO' and gate['real_pilot_updates']==0 and
              gate['selected_checkpoint'] is None and gate['matched_time_to_qualified_solution'] is None,
              'an unqualified result was admitted')
    return {'registered':len(observed), 'moments':len(mref), 'raw_jacobians':len(jref), 'restricted':len(typed),
            'paired_moments':{c:len(table[c+'_matched_ids']['moment_ids']) for c in pools},
            'physics':gate['phase_a'], 'training':gate['conditional_training']}


def audit_solver_evidence(root):
    calibration = root/'calibration/evidence'
    summary = read(calibration/'threshold_sensitivity.json')
    path = calibration/summary['receipt_file']
    check(hashlib.sha256(path.read_bytes()).hexdigest() == summary['receipt_sha256'], 'native solver receipts checksum')
    with tarfile.open(path,'r:gz') as archive:
        hashes = summary['member_sha256']
        check(sorted(archive.getnames()) == sorted(hashes), 'solver receipt inventory')
        for name,expected in hashes.items():
            check(hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected, name+' checksum')
        rank = json.load(archive.extractfile('rank.json'))
    measured_ranks = [int(np.linalg.matrix_rank(rank[k])) for k in ('original_bordered_system','perturbed_bordered_system')]
    witness = summary['rank_counterexample']
    check(measured_ranks == [witness['original_ranks'][0],witness['counterfactual_ranks'][0]], 'Newton rank counterexample')
    np.testing.assert_allclose(np.linalg.norm(rank['raw_row_perturbation'])/rank['scale'],
                               witness['global_J_direction_bound_on_unit_queries'])
    profile = read(root/'profile_inference/profile.json')
    times = {}
    for mode,entry in profile['modes'].items():
        identified = []
        for phase,rows in entry['passes'].items():
            check([r['call'] for r in rows] == list(range(1,26)), 'profile frozen query order')
            check([r['call'] for r in rows if not r['defined']]==[3], 'profile rejection')
            total = sum(r['callback_wall_s'] for r in rows)
            np.testing.assert_allclose(total,entry['summary'][phase]['all_callback_wall_sum_s'])
            np.testing.assert_allclose(sum(r['archived_native_joint_s'] for r in rows),profile['archive']['callback_seconds_sum'])
            identified.append([(r['query'], r.get('M_sha256'),r.get('J_sha256')) for r in rows])
            if phase=='warm':
                times[mode]=total
        check(len(identified)==2 and identified[0]==identified[1], 'cold and warm query results changed')
    return {'checked_receipts':len(hashes),'Newton_ranks':measured_ranks,'warm_25_callbacks_seconds':times}


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--evidence',type=Path)
    ap.add_argument('--native-evidence',type=Path)
    ap.add_argument('--include-solver-audit',action='store_true',help='also verify the published calibration and CPU profile')
    args = ap.parse_args()
    result = audit(args.root,args.evidence,args.native_evidence)
    if args.include_solver_audit:
        result['solver_audit'] = audit_solver_evidence(args.root)
    print(json.dumps(result,indent=2))
