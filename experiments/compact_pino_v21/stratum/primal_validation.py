"""Check the acceleration against the independent, archived moment-only path on real geometry."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE.parent/'native'))
from registry import Registry, save_json
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS
from v21_native import native_operand
from v21_native_physics import evaluate_operand_physics
from experiments.compact_pino_v21.stratum.primal import retained_value


def run(name, out):
    reg = Registry(DEFAULT_ALIGNMENT,DEFAULT_CORPUS,DEFAULT_PHYSICS)
    geom = reg.example(name)
    backend = reg.panel.direct(geom.request,geom.normalization,np.asarray(geom.physical).tolist())
    ids = {'combB':['combB_hyperbolic_plus'], 'T2':['T2_ellipsoid_plus','T2_endpoint_minus']}[name]
    rows = []
    for probe in (p for p in geom.probes if p.id in ids):
        operand = replace(native_operand(backend,probe.query),vertex_velocity=None)
        t0 = perf_counter()
        ref = evaluate_operand_physics(backend,operand,capture_boundary=True)
        t1 = perf_counter()
        row = dict(row_id=probe.id,frozen_context=backend.context_identity,reference_seconds=t1-t0)
        try:
            val = retained_value(backend,operand)
        except ValueError as exc:
            if ref.moment_valid or str(exc)!=ref.moment_diagnostic:
                raise
            row.update(accepted=False,diagnostic=str(exc))
        else:
            if not ref.moment_valid:
                raise ValueError(f'{probe.id}: a reference rejection was accepted')
            np.testing.assert_array_equal(val.moments,ref.moments)
            np.testing.assert_array_equal(val.vertices,ref.boundary_witness.vertices)
            np.testing.assert_array_equal(val.faces,ref.boundary_witness.faces)
            row.update(accepted=True,faces=len(val.faces),vertices=len(val.vertices),moments=val.moments,
                       equal_moments=True,equal_ordered_vertices=True,equal_oriented_faces=True,
                       arithmetic=val.arithmetic,route=val.path)
        row['accelerated_seconds']=perf_counter()-t1
        rows.append(row)
        save_json(out,rows)
        print(name,probe.id,row['accepted'],row['reference_seconds'],row['accelerated_seconds'],flush=True)
    if {r['row_id'] for r in rows}!=set(ids):
        raise RuntimeError('incomplete physics control registration')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('geometry',choices=['T2','combB'])
    ap.add_argument('out',type=Path)
    args=ap.parse_args()
    run(args.geometry,args.out)
