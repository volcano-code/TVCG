"""Post-hoc physical check of the observed Euler change along combB_R3_021 column zero."""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'physical_replay'))
import replay_registered as physical
from topology import measured
from experiments.compact_pino_v21.stratum.qualification.verify import match_center_qualification


def run(scratch,cert_path,out):
    key='combB_R3_021'
    reg=physical.Registry(physical.DEFAULT_ALIGNMENT,physical.DEFAULT_CORPUS,physical.DEFAULT_PHYSICS)
    geom=reg.example('combB')
    probe=next(p for p in geom.probes if p.id==key)
    legacy=reg.panel.direct(geom.request,geom.normalization,np.asarray(geom.physical).tolist())
    operator=physical.SharedStratumOperator.from_evaluator(legacy)
    center=physical.read(scratch/'centers'/(key+'.json'))
    if center['run_fingerprint']!=physical.fingerprint_run(operator,probe,'fast'):
        raise ValueError('center belongs to a different replay')
    source,target,ids=physical.revive(scratch,center,probe.query,probe.chart)
    proof=match_center_qualification(physical.read(cert_path),center)
    radius=proof['certified_raw_linf_radius']
    if proof['restricted_event_valid']!=1:
        raise ValueError('the diagnostic requires a proven coefficient radius')
    index=physical.checks.PhysicalVertexIDs.index_design(legacy.design)
    w=target.boundary_witness
    before=measured(w.vertices,w.faces)
    direction=probe.chart.basis[:,0]
    trials=[]
    for h in (2.e-5, radius/(2*np.max(np.abs(direction)))):
        ends=[]
        for sign in (-1,1):
            q=physical.checks.retracted(probe.query,direction,sign*h,probe.chart.surface)
            value,operand,nodes,witness=physical.endpoint(legacy,operator,q,probe.chart,source,index,'fast')
            if not value.get('material',{}).get('moment_valid') or nodes is None:
                raise ValueError('native endpoint did not yield an auditable physical boundary')
            displacement=float(np.max(np.abs(q-probe.query)))
            ends.append(dict(sign=sign,raw_query=q,actual_raw_linf=displacement,inside_certificate=displacement<=radius,
                             operation=operand.operation,retained_vertices_sha256=value['material']['retained_vertices_sha256'],
                             retained_faces_sha256=value['material']['retained_faces_sha256'],
                             retained_signature=value['material']['retained_signature'],
                             topology=measured(witness.vertices,witness.faces),moments=value['material']['moments'],
                             same_vertex_constructions=nodes.constructions==ids.constructions,
                             same_oriented_incidence=nodes.winding==ids.winding))
        fd=(np.asarray(ends[1]['moments'])-np.asarray(ends[0]['moments']))/(2*h)
        trials.append(dict(raw_step=h,endpoints=ends,physical_fd=fd,
                           registered_material_error=physical.checks.error_metric(fd,target.restricted_jacobian[:,0],'material')))
    physical.save_json(out,dict(scope='Post-hoc diagnostic chosen after observing the finite-step change. '
                           'These extra four native calls do not enter the registered 89-row counts.',
                           row_id=key,dimension=3,column=0,context=operator.identity,center_query=probe.query,
                           direction=direction,certified_raw_linf=radius,center_topology=before,
                           center_material_column=target.restricted_jacobian[:,0],trials=trials))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scratch','certificate','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    run(args.scratch,args.certificate,args.output)
