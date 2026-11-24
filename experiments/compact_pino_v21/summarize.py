"""Recompute the research tables, admission facts and stratified physics diagnostics."""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .diagnostics import moment_errors
from .replay import read, write

METHODS = ("analytic_rle", "finite_rle", "finite_conservative_token8",
           "finite_conservative_token16", "finite_conservative_local")


def stats(values):
    pairs = [(name, v) for name, v in values if v is not None]
    if not pairs:
        return dict(n=0, median=None, p95=None, max=None, worst=None)
    nums = [v for _, v in pairs]
    worst = max(pairs, key=lambda x: x[1])
    return dict(n=len(nums), median=float(np.median(nums)), p95=float(np.percentile(nums,95)),
                max=float(worst[1]), worst=worst[0])


def summarize_rows(rows, method):
    records = [(k, v[method]) for k,v in rows.items() if method in v]
    def metric(f):
        return stats([(k,f(o)) for k,o in records])
    return dict(rows=len(records),
                scaled_moment=metric(lambda o:o['moment_error']['scaled_max']),
                relative_moment=metric(lambda o:o['moment_error']['relative_l2']),
                scaled_jacobian=metric(lambda o:o.get('jacobian_error',{}).get('unit_direction_scaled_max')),
                relative_jacobian=metric(lambda o:o.get('jacobian_error',{}).get('relative_frobenius')),
                accepted_direction_relative=metric(lambda o:o.get('jacobian_error',{}).get('Jv',{}).get('relative')))


def matched_comparison(rows):
    common = {k:v for k,v in rows.items() if all(m in v for m in METHODS)}
    common_j = {k:v for k,v in common.items() if all('jacobian_error' in v[m] for m in METHODS)}
    out = {}
    for method in METHODS:
        m = summarize_rows(common,method)
        j = summarize_rows(common_j,method)
        for key in ('scaled_jacobian','relative_jacobian','accepted_direction_relative'):
            m[key] = j[key]
        out[method] = m
    return out, dict(moment_ids=sorted(common), jacobian_ids=sorted(common_j))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower_band(x):
    if x < .01:
        return "near_zero"
    if x < .10:
        return "low"
    return "middle" if x < .70 else "high"


def publish(evidence, native, indexes, release, historical):
    protocol = read(Path(__file__).with_name('protocol.json'))
    if sha(release) != protocol['four_split_manifest_sha256']:
        raise ValueError("official release differs from preregistration")
    cohorts = {x['geometry_id']:x['source_group_id'] for x in read(historical)['records']}
    cohorts['noodles'] = cohorts['recon-2024-noodles-v8-v1']
    frozen_queries = {}
    for split in ('train','development'):
        ref = read(release)['views'][split]
        view = Path(ref['path'])
        if sha(view) != ref['sha256']:
            raise ValueError('modified official split view')
        for r in read(view)['records']:
            gid = r['geometry_id']
            cohorts[gid] = r['source_group_id']
            bank = Path(r['labels_file'])
            if sha(bank) != r['labels_sha256']:
                raise ValueError('modified publisher query bank')
            with np.load(bank, allow_pickle=False) as label:
                for query_id, q in zip(label['query_id'],label['queries'],strict=True):
                    key = (gid, str(query_id))
                    if key in frozen_queries:
                        raise ValueError('duplicate official registered query')
                    frozen_queries[key] = (split, np.asarray(q,dtype='<f8').tobytes())
    meta = {p.parent.name:read(p) for p in indexes.glob('*/identity.json')}
    docs = {p.stem:read(p) for p in (evidence/'replay').glob('*.json')}
    registry = read(native/'native_replay.json')
    if set(docs) != {r['row_id'] for r in registry['rows']} or set(meta) != set(registry['sources']):
        raise ValueError("replay coverage differs from native registration")

    static = {}
    for g, m in meta.items():
        independent = registry['sources'][g]['mesh_full_editable']
        info = {k:v for k,v in m.items() if k != 'controls'}
        info['controls'] = {kind:dict(data, versus_native_editable=moment_errors(data['full_editable'],independent,
                                    independent[0],m['length'])) for kind,data in m['controls'].items()}
        static[g] = info
    write(evidence/'static_material.json',static)

    results, attributes = {}, {}
    for row_id, r in docs.items():
        if not r['native_moment_valid']:
            continue
        scores = {}
        for name, (material, local) in zip(METHODS[1:], (
            ('rle','local'), ('conservative','scalar_token8'), ('conservative','scalar_token16'), ('conservative','local'))):
            scores[name] = r['methods'][material]['results'][local]
        a = evidence/'analytic'/(row_id+'.json')
        if a.is_file():
            analytic = read(a)['result']
            if 'moment_error' in analytic:
                scores['analytic_rle'] = analytic
        g = r['geometry']; s = static[g]
        retained = (r['native_M'][0]-s['fixed_moments'][0])/s['reference_editable_mesh'][0]
        results[row_id] = scores
        attributes[row_id] = dict(source=cohorts[g], surface=r['surface'], geometry=g, retention=lower_band(retained),
                                  retained_mass_fraction=retained)

    tables = {}
    for cohort, ids in [('noodles',[k for k,v in attributes.items() if v['geometry']=='noodles']),
                        ('other_geometry_probes',[k for k,v in attributes.items() if v['geometry']!='noodles'])]:
        table = {i:results[i] for i in ids}
        tables[cohort], tables[cohort+'_matched_ids'] = matched_comparison(table)
        tables[cohort+'_all_available'] = {m:summarize_rows(table,m) for m in METHODS}
    tables['other_scope'] = 'Descriptive registered probes including signed/scale and frozen-source controls; no independent generalization inference'
    groups = {}
    for axis in ('geometry','source','surface','retention'):
        splits = defaultdict(dict)
        for row_id, attr in attributes.items():
            if attr['geometry'] != 'noodles':
                splits[attr[axis]][row_id] = results[row_id]
        groups[axis] = {k:summarize_rows(v,'finite_conservative_local') for k,v in splits.items()}
    tables['strata_other_geometry_probes'] = groups
    tables['query_strata'] = attributes
    tables['units'] = dict(scaled_moment='max_i abs(delta M_i)/(editable mass L^degree)',
        scaled_jacobian='max scaled component error over a unit tangent at unit coefficient norm (row 2-norm)',
        relative_jacobian='Frobenius error / native Frobenius; null for zero reference',
        accepted_direction_relative='L2 error in Jv / native ||Jv|| for an actual outgoing accepted geodesic direction',
        percentiles='query-level linear interpolation, same query IDs per paired method')
    write(evidence/'comparison.json',tables)

    fixtures = read(evidence/'manufactured.json')
    worst_static = max([v['controls']['conservative']['versus_native_editable']['scaled_max'] for v in static.values()]
                       +[s['static_error']['scaled_max'] for s in fixtures])
    trace = registry['trajectory']
    archive_ok = (trace['registered']==25 and trace['accepted']==24 and trace['identical_retained_hashes']==24
                  and trace['rejections'][0]['row']=='noodles_03' and trace['max_abs_frozen_M']<1e-11
                  and trace['max_abs_frozen_J']<1e-11)
    identities_ok = all(d['material_identities']=={k:v['identity'] for k,v in static[d['geometry']]['controls'].items()}
                        for d in docs.values())
    manufactured_ok = all(s['opposite_max_abs']<1e-12 and s['opposite_J_max_abs']<1e-12 and
                          s['results']['local']['center_z_abs']<s['results']['scalar_token16']['center_z_abs'] for s in fixtures)
    same_face = registry['controls']
    pairs = [(prefix+'_plus', prefix+'_plus_opposite') for prefix in same_face]
    numerical_pairs = []
    for plus, opposite in pairs:
        r, s = docs[plus], docs[opposite]
        g = static[r['geometry']]
        expected = 2*np.array(g['fixed_moments']) + g['controls']['conservative']['full_editable']
        predicted = np.array(r['methods']['conservative']['results']['local']['M']) + s['methods']['conservative']['results']['local']['M']
        numerical_pairs.append(moment_errors(predicted,expected,g['reference_editable_mesh'][0],g['length'])['scaled_max'])
    same_closed_rule = max(numerical_pairs)<1e-8
    endpoints = {}
    for name, row in docs.items():
        if not name.startswith('T2_endpoint_'):
            continue
        local = row['methods'].get('conservative',{}).get('results',{}).get('local',{})
        endpoints[name] = dict(operand=row['operand_valid'], native_moment=row['native_moment_valid'],
                               native_error=row['material_error'], local_mass=local['M'][0] if local else None,
                               scaled_moment_error=local.get('moment_error',{}).get('scaled_max'))
    endpoint_ok = (endpoints['T2_endpoint_plus']['scaled_moment_error'] < protocol['gate_a_scaled_full_max'] and
                   endpoints['T2_endpoint_minus']['local_mass'] == static['T2']['fixed_moments'][0] and
                   not endpoints['T2_endpoint_minus']['native_moment'] and
                   all(not endpoints['T2_endpoint_'+suffix]['operand']
                       for suffix in ('constant_positive','constant_negative','zero')))
    a_pass = bool(archive_ok and identities_ok and manufactured_ok and same_closed_rule and endpoint_ok and
                  worst_static <= protocol['gate_a_scaled_full_max'] and
                  tables['noodles'][METHODS[-1]]['scaled_jacobian']['n'] == trace['accepted'])
    scoped = set()
    for row in docs.values():
        key = (row['geometry'], str(row['source_record'].get('query_id','')))
        if (key in frozen_queries and not row['row_id'].endswith('_opposite') and
                np.asarray(row['query'],dtype='<f8').tobytes() == frozen_queries[key][1]):
            scoped.add(key)
    complete_coverage = len(scoped)==len(frozen_queries)
    registered_coverage = {split:dict(registered=sum(v[0]==split for v in frozen_queries.values()),
                                      probed=sum(frozen_queries[k][0]==split for k in scoped))
                           for split in ('train','development')}
    r3, r8 = registry['strict_development_R3'], registry['strict_development_R8']
    certified = {d:sum(r.get('restricted_event_valid') == 1 and r['restricted_valid']
                       for r in registry['rows'] if r.get('requested_chart_dimension') == d
                       and r['geometry_id'] in table['counts_by_geometry'])
                 for d, table in ((3,r3),(8,r8))}
    admissions = []
    if not complete_coverage:
        admissions.append('The registered train/development deployment overlay is incomplete.')
    for d, table in ((3,r3),(8,r8)):
        if not table['restricted_defined']:
            admissions.append(f'No registered strict-development R{d} numerical target from the deployed finite map.')
        elif not certified[d]:
            admissions.append(f'Certified strict-development R{d} events are missing.')
    # This is an audit of a probe panel. The separate training service must verify a signed receipt.
    write(evidence/'gate_status.json',dict(
        phase_a='PASS' if a_pass else 'NO-GO', actual_geometry_count=len(static),
        largest_scaled_static_moment_error=worst_static, native_trajectory_matches=archive_ok,
        packed_material_identities_match=identities_ok, manufactured_checks=manufactured_ok,
        endpoint_checks=endpoint_ok, endpoints=endpoints,
        same_operand_model_scaled_max=max(numerical_pairs), native_fixed_operand_controls=same_face,
        conditional_training='NO-GO' if admissions or not a_pass else 'requires independent receipt review',
        strict_R3=r3, strict_R8=r8, reason=' '.join(admissions),
        registered_query_coverage=registered_coverage, complete_training_overlay=complete_coverage,
        certified_dev_R3=certified[3], certified_dev_R8=certified[8],
        signed_admission_receipt=None,
        formal_preflight_updates=0, real_pilot_updates=0, selected_checkpoint=None,
        blind_case_opened_for_scoring=False, candidate_native_solutions=None,
        matched_time_to_qualified_solution=None, long_training='NO-GO',
        limits_scope='proposed project screening limits; measured diagnostic errors are distinct from solver certification'))


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','native','indexes','release','historical'):
        ap.add_argument('--'+name,type=Path,required=True)
    publish(**vars(ap.parse_args()))
