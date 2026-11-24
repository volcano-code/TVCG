"""Check the raw pullback against independent FD of actual source vertices and cut moments."""
import argparse
from pathlib import Path

import numpy as np

from registry import Registry, save_json
from chart_diagnostic import as_boundary, cut
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS
from v21_native import operand_from_source, unit_to_raw_jacobian


def run(args):
    reg=Registry(args.alignment,args.corpus,args.physics)
    geom=reg.example(reg.panel.cross_ids[0])
    probe=next(p for p in geom.probes if p.stratum=='hyperbolic')
    q=2.75*probe.query
    be=reg.panel.direct(geom.request,geom.normalization,np.asarray(geom.physical).tolist())
    def packet(coeff):
        return be.native.evaluate_spin_it_faster_source_motion(coeff,be.target_design,be.triangulation)
    center=operand_from_source(packet(q),q)
    reference=be(q)
    jac=reference['moment_jacobian']@unit_to_raw_jacobian(q)
    output=dict(scope='Deployed Cherchi raw-coordinate local differential diagnostic; events unclassified',
                geometry=geom.id,registered_probe=probe.origin,query_raw=q,scale=2.75,
                production_context=be.context_identity,operand_vertex_count=len(center.vertices),
                M=reference['raw_moments'],J_raw=jac,steps=[])
    for h in (1e-6,5e-7):
        fds, vertex_errors, source_faces_equal = [],[],[]
        for k,direction in enumerate(np.eye(10)):
            plus,minus=as_boundary(packet(q+h*direction)),as_boundary(packet(q-h*direction))
            same_faces=all(o.operation==center.operation and np.array_equal(o.faces,center.faces) for o in (plus,minus))
            source_faces_equal.append(same_faces)
            if same_faces:
                dv=(plus.vertices-minus.vertices)/(2*h)
                vertex_errors.append(float(np.linalg.norm(dv-center.vertex_velocity[...,k])/max(np.linalg.norm(dv),1e-14)))
            else:
                vertex_errors.append(None)
            hi,_=cut(be,plus)
            lo,_=cut(be,minus)
            fds.append((hi-lo)/(2*h))
        fd=np.stack(fds,axis=1)
        output['steps'].append(dict(h=h,FD_raw=fd,source_faces_equal=source_faces_equal,
                vertex_velocity_relative_error=vertex_errors,raw_J_max_abs=float(np.max(abs(fd-jac))),
                raw_J_relative_fro=float(np.linalg.norm(fd-jac)/np.linalg.norm(jac))))
        save_json(Path(args.out)/'diagnostics/deployed_full_motion.json',output)
        print(h,output['steps'][-1]['raw_J_relative_fro'],sum(source_faces_equal),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment',default=DEFAULT_ALIGNMENT)
    p.add_argument('--corpus',default=DEFAULT_CORPUS)
    p.add_argument('--physics',default=DEFAULT_PHYSICS)
    p.add_argument('--out',required=True)
    run(p.parse_args())
