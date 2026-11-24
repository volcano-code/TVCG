"""Offline Chat review: reproduce derivative masks, prediction scores and the Pilot decision from the packet."""
import argparse
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from experiments.compact_pino_v21.stratum import prototype, restricted_targets
from experiments.compact_pino_v21.stratum.event_digest import compare as compare_native_events
from experiments.compact_pino_v21.stratum.physical_replay import archive
from experiments.compact_pino_v21.stratum.qualification import report


def load(path):
    return json.loads(path.read_bytes())


def require(ok,message):
    if not ok:
        raise ValueError(message)


def verify_controls():
    comparison=load(HERE/'evidence/noodles_event_comparison.json')
    require(comparison['passed'] and comparison['all_numerical_events_match'],'native event regression failed')
    runs=[load(HERE/'evidence'/name) for name in ('noodles_legacy_direct.json','noodles_full_rank.json')]
    require(compare_native_events(*runs)==comparison,'native events or their comparison receipt differ')
    for run in runs:
        require(len(run['rows'])==25 and all(r['same_query'] for r in run['rows'])
                and run['accepted'] and run['same_trace_hash'] and run['all_rows_match'],
                'native 25-call regression failed')
        good=[r for r in run['rows'] if r['defined']]
        rejected=[r for r in run['rows'] if not r['defined']]
        require(len(good)==24 and len(rejected)==1 and rejected[0]['call']==3 and rejected[0]['same_rejection']
                and all(r['same_geometry'] and r['M_max_abs']<1e-12 and r['J_raw_max_abs']<1e-12 for r in good),
                'native trajectory values or historical refusal differ')
    require(runs[0]['physical_check']['mesh_geometry_sha256']==runs[1]['physical_check']['mesh_geometry_sha256']
            and runs[0]['trace_hash']==runs[1]['trace_hash'],'physical regression differs between entries')
    return dict(physical_acceptance=runs[1]['accepted'],native_calls=len(runs[1]['rows']),numeric_calls=24,
                trace_hash=runs[1]['trace_hash'],raw_event_hash_matches_archive=runs[1]['same_archived_event_hash'],
                paired_scientific_events_match=comparison['all_numerical_events_match'],native_learned_weights=False)


def verify_supported_bridge():
    """Recompute the two native directional pullbacks using independent physical references."""
    folder=HERE/'solver_bridge'
    receipt=load(folder/'receipt.json')
    frozen=load(HERE/'native_backend/build_receipt.json')
    require(receipt['native_module_sha256']==frozen['upstream_module_sha256']
            and receipt['finite_source_module_sha256']==frozen['module_sha256']
            and receipt['public_facade_sha256']==restricted_targets.sha(prototype.ADAPTER/'v21_stratum.py')
            and receipt['replay_sha256']==restricted_targets.sha(folder/'replay.py'),
            'solver bridge source or compiled binaries differ from the captured evaluation')
    plane=load(HERE/'physical_replay/evidence/manufactured_plane.json')
    with ZipFile(HERE/'qualification/native_admission/evidence/replay_inputs.zip') as packet:
        center=json.loads(packet.read('centers/R02_R8_032.json'))
    material=center['material']
    references=(
        ('plane_tetra',3,plane['q'],plane['basis'],plane['analytic_moments'],plane['analytic_jacobian'],plane['identity']),
        ('registered_R8',8,center['query'],center['raw_basis'],material['M_independent_tetra'],
         material['J_independent_tetra'],center['context']))
    result={}
    scale=np.array([1.]*7+[2.]*3)
    for name,d,raw_query,raw_basis,moments,jacobian,map_id in references:
        run=receipt[name]
        event,=run['supported_space']['evaluations']
        field=event['evaluation']
        lin=event['linearization']
        require(run['function_evaluations']==run['settings']['maximum_function_evaluations_per_restart']==1
                and run['mode']=='STATIC_BALANCE' and run['static_objective']=='MAXIMUM_MASS'
                and np.array_equal(run['initial_query'],raw_query)
                and np.array_equal(event['quadric_coefficients'],run['current_chart']['query'])
                and not event['error'] and field['derivative_defined'] and 'moment_jacobian' not in field
                and field['evaluation_map_identity']==map_id and np.shape(lin['matrix'])==(d+3,d+3),
                f'{name}: unexpected native query, physics, derivative or equation assembly')
        unit_basis=np.array(field['native_tangent_basis'])
        unit_j=np.array(field['native_directional_jacobian'])
        require(unit_basis.shape==unit_j.shape==(10,d),f'{name}: wrong native directional layout')
        q=np.array(raw_query)
        u=q/np.linalg.norm(q)
        proj=np.eye(10)-np.outer(u,u)
        np.testing.assert_allclose(event['quadric_coefficients'],u,rtol=0,atol=5e-15)
        np.testing.assert_allclose(unit_basis.T@unit_basis,np.eye(d),rtol=0,atol=1e-12)
        reference_motion=proj@raw_basis/np.linalg.norm(q)
        np.testing.assert_allclose(unit_basis@(unit_basis.T@reference_motion),reference_motion,rtol=0,atol=1e-12)
        raw_j=unit_j@(unit_basis.T@reference_motion)
        np.testing.assert_allclose(field['raw_moments'],moments,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(raw_j,jacobian,rtol=1e-9,atol=1e-12)
        # The strict solver uses dual monomials for its unknowns. Compare the C++ matrix after its chain rule.
        unit=np.array(event['quadric_coefficients'])
        dual=scale*unit/np.linalg.norm(scale*unit)
        tangent=(np.eye(10)-np.outer(unit,unit))@(np.asarray(lin['solver_tangent'])/scale[:,None])
        tangent/=np.linalg.norm(dual/scale)
        newton_j=unit_j@(unit_basis.T@tangent)
        np.testing.assert_allclose(newton_j,lin['moment_tangent_jacobian'],rtol=1e-10,atol=1e-12)
        require(all(np.isfinite(lin[key]).all() for key in ('matrix','right_hand_side','raw_residual','full_residual')),
                f'{name}: nonfinite Newton system')
        if name=='registered_R8':
            require(np.array_equal(unit,raw_query) and field['geometry_hash']==material['material_geometry_hash']
                    and field['topology_hash']==material['material_topology_hash'],
                    'R8 callback differs from the exact registered center')
        result[name]=dict(dimension=d,eq14_dimension=d+3,independent_raw_M_and_J_agree=True,
                          compiled_newton_directional_matrix_agrees=True)
    refusal=receipt['incompatible_fidget_objective']
    rejected,=refusal['supported_space']['evaluations']
    original,=receipt['plane_tetra']['supported_space']['evaluations']
    same_fields=('geometry_hash','topology_hash','evaluation_map_identity','derivative_identity',
                 'raw_moments','native_tangent_basis','native_directional_jacobian')
    require(refusal['function_evaluations']==1 and refusal['mode']=='FIDGET_SPINNER'
            and np.array_equal(rejected['quadric_coefficients'],original['quadric_coefficients'])
            and rejected['evaluation']['derivative_defined'] and rejected['linearization'] is None
            and all(rejected['evaluation'][key]==original['evaluation'][key] for key in same_fields)
            and 'support excludes objective or constraint gradient' in rejected['error'],
            'expected the same physical derivative and a native objective-support rejection')
    return dict(initial_static_assembly=result,incompatible_fidget_rejected=True,learned_optimization_measured=False)


def verify_step_example(physical,ledger):
    example=load(HERE/'evidence/event_scale_example.json')
    key=example['row_id']
    with gzip.open(physical,'rt') as stream:
        row=next(r for r in json.load(stream)['rows'] if r['row_id']==key)
    cert=next(r for r in ledger['rows'] if r['row_id']==key)
    k=example['column']
    mat=row['material']
    require(key=='combB_R3_021' and k==0 and row['context']==cert['evaluation_map_identity']==example['context']
            and cert['restricted_event_valid']==1 and example['certified_raw_linf']==cert['certified_raw_linf_radius']
            and np.array_equal(row['query'],example['center_query'])
            and np.array_equal(np.asarray(row['raw_basis'])[:,k],example['direction'])
            and np.array_equal(np.asarray(mat['raw_restricted_J'])[:,k],example['center_material_column']),
            'post-hoc example differs from the registered query, column, material or certificate')
    center=example['center_topology']
    require(all(mat['center_surface_components'][k]==v for k,v in center.items()),
            'post-hoc center differs from original retained surface')
    check=example['certified_raw_linf']
    tests=[]
    registered=load(HERE/'physical_replay/protocol.json')
    for trial in example['trials']:
        endpoints=trial['endpoints']
        require([e['sign'] for e in endpoints]==[-1,1],'paired endpoint order changed')
        euler=[]
        for e in endpoints:
            topology=e['topology']
            chi=topology['vertices']-topology['edges']+topology['faces']
            require(topology['closed_edge_valence_two'] and chi==topology['euler_characteristic'],
                    'endpoint retained surface Euler counts disagree')
            distance=float(np.max(np.abs(np.array(e['raw_query'])-example['center_query'])))
            require(distance==e['actual_raw_linf'] and (distance<=check)==e['inside_certificate'],
                    'raw endpoint does not match reported coefficient enclosure')
            for digest,flag in (('construction_sha256','same_vertex_constructions'),
                                ('oriented_incidence_sha256','same_oriented_incidence')):
                require((e['retained_signature'][digest]==mat['retained_signature'][digest])==e[flag],
                        'retained construction or incidence comparison differs from the boundary signatures')
            euler.append(chi)
        fd=(np.array(endpoints[1]['moments'])-endpoints[0]['moments'])/(2*trial['raw_step'])
        require(np.array_equal(fd,trial['physical_fd']),'signed moment difference differs')
        ref=example['center_material_column']
        metric=trial['registered_material_error']
        np.testing.assert_allclose(np.linalg.norm(fd-ref)/max(np.linalg.norm(fd),np.linalg.norm(ref)),
                                   metric['relative_l2'],rtol=1e-12,atol=1e-16)
        require(metric['passed']==archive.metrics(fd,ref,'material',registered),'different material FD decision')
        if trial['raw_step']==row['steps'][0]['h']:
            reference=row['steps'][0]['columns'][k]['endpoints']
            for e,stored in zip(endpoints,reference,strict=True):
                require(np.array_equal(e['raw_query'],stored['query']) and np.array_equal(e['moments'],stored['material']['moments'])
                        and all(e[part]==stored['material'][part] for part in
                                ('retained_vertices_sha256','retained_faces_sha256','retained_signature')),
                        'exploratory repeat differs from registered perturbation')
        tests.append(dict(both_inside=all(e['inside_certificate'] for e in endpoints),euler=euler,
                          same_oriented_faces=all(e['same_oriented_incidence'] for e in endpoints)))
    require(tests[0]['euler'][0]!=tests[0]['euler'][1] and not tests[0]['both_inside']
            and tests[1]['both_inside'] and tests[1]['same_oriented_faces']
            and tests[1]['euler']==[center['euler_characteristic']]*2,
            'post-hoc topology example differs from the recorded observations')
    return dict(center=center['euler_characteristic'],raw_linf_certificate=check,trials=tests)


def decision():
    assets=HERE/'physical_replay/evidence'
    physical=assets/'registered_physical_replay.json.gz'
    cert_file=HERE/'qualification/evidence/neighbourhoods.json'
    inventory_file=HERE/'qualification/evidence/published_inventory.json'
    prediction_file=HERE/'evidence/prototype.json'
    with redirect_stdout(io.StringIO()):
        archive.verify(argparse.Namespace(evidence=assets))
    ledger=load(cert_file)
    report.verify(ledger)
    bank,overlay=restricted_targets.build(physical,cert_file,inventory_file)
    restricted_targets.output(bank,overlay,HERE/'evidence',verify=True)
    prototype.verify_scores(physical,prediction_file)
    native=verify_controls()
    bridge=verify_supported_bridge()
    event_scale=verify_step_example(physical,ledger)
    study=load(HERE.parent/'protocol.json')
    approval=study['pilot']
    ceiling=study['screening_limits']
    by_stratum=load(prediction_file)['by_stratum']
    def screen(measure,count,limit):
        return measure['p95']<=limit if count and measure['n']==count else None
    error={k:dict(rows=v['denominator'],M_scored=v['M']['n'],J_scored=v['J']['n'],
                  M_p95=v['M']['p95'],directional_J_p95=v['J']['p95'],
                  moment_screen_pass=screen(v['M'],v['denominator'],ceiling['moment_p95']),
                  jacobian_screen_pass=screen(v['J'],v['denominator'],ceiling['directional_jacobian_p95_per_full_r3_r8']))
           for k,v in by_stratum.items()}
    splits=load(inventory_file)['splits']
    coverage={}
    for name,split in splits.items():
        old=split['historical_native']
        old_ids={r['row_id'] for r in old['attempted_rows']}
        new=[r for r in overlay['rows'] if name=='development']
        new_ids={r['row_id'] for r in new}
        # The legacy typed calls included in this audit all refused. No accepted M or J is double-counted.
        require(not any(r['moment'] and r['row_id'] in new_ids for r in old['attempted_rows']),
                'overlapping numerical target between physics versions')
        checked=len(old_ids|new_ids)
        coverage[name]=dict(registered=split['queries'],geometries=split['parents'],source_groups=split['groups'],
                            checked_rows=checked,unchecked_rows=split['queries']-checked,
                            numerical_moments=old['moments']+sum(r['moment_valid'] for r in new),
                            matching_full_numerical=old['full_numerical'],matching_full_certified=old['full_certified'],
                            typed_numerical=sum(r['restricted_numerical_valid'] for r in new),
                            published_scope=split['published_scope'],original_source_scope=split['preserved_source_scope'])
    complete=all(c['checked_rows']==c['registered'] for c in coverage.values())
    reasons=[]
    if not complete:
        reasons.append('Same-map numerical values, scoped masks or explicit refusals are missing for published train and development queries.')
    if not overlay['by_dimension']['8']['restricted_geometry_event_certified']:
        reasons.append('No registered development R8 has a certified native coefficient neighbourhood.')
    return dict(schema='tvcg.compact-v21.typed-interface.decision.1',registered_release=overlay['release_manifest_sha256'],
                queried_rows=len(overlay['rows']),source_version=ledger['rows'][0]['source_map'],by_scope=overlay['by_dimension'],
                prototype_screening=error,source_coverage=coverage,whole_train_dev_overlay=complete,
                full_rank_regression=native,supported_native_initial_equations=bridge,example_topology=event_scale,
                pilot='NO-GO' if reasons else 'PREFLIGHT_REVIEW',long_training='NO-GO',learned_candidate_updates=0,
                candidate_native_accepted_tops=None,candidate_time_to_qualified_solution=None,blind_case_scored=False,
                outstanding_evidence=reasons,
                r8_passing_every_sampled_check_and_geometry_proof=overlay['by_dimension']['8']['qualified_with_all_sampled_checks'],
                pilot_primary=approval['primary'],pilot_protections=approval['protections'],
                input_sha256={p.name:restricted_targets.sha(p) for p in (physical,cert_file,inventory_file,prediction_file,
                              HERE/'evidence/restricted_targets.npz',HERE/'evidence/noodles_event_comparison.json',
                              HERE/'solver_bridge/receipt.json')})


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--write',action='store_true',help='write the decision after running all portable checks')
    args=cli.parse_args()
    path=HERE/'evidence/decision.json'
    computed=decision()
    if args.write:
        path.write_text(json.dumps(computed,indent=2,allow_nan=False)+'\n')
    else:
        require(computed==load(path),'decision differs from the raw packet')
    print(json.dumps({k:computed[k] for k in ('queried_rows','source_coverage','pilot','long_training')},indent=1))
