"""Exercise the original compiled Newton solver through the new service's ambient callback."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from experiments.compact_pino_v21.stratum.shared_operator import SharedStratumOperator
from experiments.compact_pino_v21.stratum.event_digest import compare
sys.path.insert(0,str(HERE.parent/'native'))
from registry import Registry,read,save_json,sha
from replay import DEFAULT_ALIGNMENT,DEFAULT_CORPUS,DEFAULT_PHYSICS
from v21_native import unit_to_raw_jacobian


def run(out, entry):
    registry = Registry(DEFAULT_ALIGNMENT,DEFAULT_CORPUS,DEFAULT_PHYSICS)
    geom = registry.noodles()
    req = replace(geom.request,settings=replace(geom.request.settings,deterministic_seed=0))
    service = SharedStratumOperator.from_request(req,geom.normalization,np.asarray(geom.physical).tolist())
    from solve import verify
    exact = registry.alignment/'reference/trajectory_M_J.json'
    archived = registry.alignment/'reference/seed-00.json'
    old = read(archived)
    expected = read(exact)['rows']
    starts = read(registry.solver/'protocol.json')['guesses_raw']['0']
    rows = []
    events = []
    evaluate = service.solver if entry=='versioned' else service.backend
    def callback(q):
        i = len(rows)
        record = dict(call=i+1,query=list(q))
        rows.append(record)
        reference = expected[i] if i<len(expected) else None
        record['same_query'] = reference is not None and np.array_equal(reference['query'],q)
        try:
            result = evaluate(q)
            record.update(defined=True,geometry_hash=result['geometry_hash'],context=result['evaluation_map_identity'])
            if record['same_query'] and reference['native']['defined']:
                truth = reference['native']
                delta_j = result['moment_jacobian']@unit_to_raw_jacobian(q)-np.asarray(truth['J'])
                record.update(same_geometry=truth['geometry_hash']==result['geometry_hash'],
                              M_max_abs=float(np.max(abs(result['raw_moments']-np.asarray(truth['M'])))),
                              J_raw_max_abs=float(np.max(abs(delta_j))))
            return result
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            record.update(defined=False,error=str(exc),
                          same_rejection=bool(record['same_query'] and reference is not None and not reference['native']['defined'] and
                                              reference['native']['diagnostic']==str(exc)))
            raise
        finally:
            # The original solver journal also discarded the temporary retained material after each callback.
            service.backend._materials.clear()
            print(record['call'],record.get('defined'), record['same_query'],flush=True)
    t = perf_counter()
    result = registry.panel.spin.solve_spin_it_faster_with_external_moment_evaluator(
        req,callback,initial_quadrics=starts,event_sink=events.append)
    solver_s = perf_counter()-t
    oracle = registry.panel.direct(req,geom.normalization,np.asarray(geom.physical).tolist())
    t = perf_counter()
    verified = verify(oracle,result['quadric_coefficients'],req,result['status'])
    same_trace = result['trace_hash']==old['solver']['trace_hash']
    match = (len(rows)==len(expected)==25 and all(r['same_query'] and
             ((r.get('same_geometry') and r['M_max_abs']<1e-12 and r['J_raw_max_abs']<1e-12)
              or r.get('same_rejection')) for r in rows))
    packet = dict(scope='Full-rank native regression using a real compiled solve, no learned model.', entry=entry,
                  registered_journal_sha256=sha(archived),registered_targets_sha256=sha(exact),
                  accepted=verified['accepted'],same_trace_hash=same_trace,
                  same_archived_event_hash=result['event_hash']==old['solver']['event_hash'],
                  archived_event_hash=old['solver']['event_hash'],all_rows_match=bool(match),
                  solve_seconds=solver_s, independent_check_seconds=perf_counter()-t,
                  native_status=result['status'],trace_hash=result['trace_hash'],event_hash=result['event_hash'],
                  physical_check=verified,rows=rows,events=events)
    save_json(out,packet)
    if not (packet['accepted'] and same_trace and match):
        raise RuntimeError('native full-rank regression differed; see saved outcome')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--entry',choices=['legacy_direct','versioned'],default='versioned')
    parser.add_argument('--compare',nargs=2,type=Path,metavar=('DIRECT','VERSIONED'))
    args=parser.parse_args()
    if args.compare:
        result=compare(*[read(p) for p in args.compare])
        save_json(args.out,result)
        if not result['passed']:
            raise ValueError('direct and versioned native regression differ; see recorded events')
    else:
        run(args.out,args.entry)
