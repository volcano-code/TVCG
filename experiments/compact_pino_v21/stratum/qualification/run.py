"""Replay the independent coefficient/vertex-neighbourhood checks on registered development inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT),str(HERE),str(HERE.parents[1]/'native')]
from bounds import Interval as I
from charts import (MAP,SIDE,enclose_plane,enclose_checked_paraboloid,finite_guards,
                    physical_box,require,verify_prism_embedding)
from incidence import DesignTriangles,certify_incidence
from inventory import fingerprint
from registry import Registry,save_json
from replay import DEFAULT_ALIGNMENT,DEFAULT_CORPUS,DEFAULT_PHYSICS
from experiments.compact_pino_v21.stratum.shared_operator import SharedStratumOperator

# A confirmed radius is in raw-coefficient infinity norm intersected with the
# specified chart. Trial sizes do not supply a learned or held-out threshold.
RADII=(2.**-32,2.**-40,2.**-44)
SUPPORTED_MODULE='a2bc6ae680b50680930697136561ba9303a51bd8c0a3fc4118dc31617f99d9af'


def source_box(packet,q,radius):
    w=packet['construction_witness']
    r=float(w['initial_radius'])
    if packet['surface_type']=='PLANE':
        near,frame=enclose_plane(q,radius,r)
        require(int(w['frame_reference_axis'])==frame['frame_reference_axis'],'native reference axis mismatch')
    else:
        near,frame=enclose_checked_paraboloid(q,radius,r,w)
    require(frame['pivot']==int(w['coefficient_pivot']) and
            frame['sign']==int(w['coefficient_sign']) and bool(packet['interior_is_positive'])==(frame['sign']>0),
            'native side/pivot mismatch')
    for attr in ('axis','first','second'):
        b=frame[attr]
        seen=np.asarray(w[attr])
        require((b.lo<=seen)&(seen<=b.hi),'center native basis outside enclosure')
    return near,frame


def check_row(backend,design,packet,q,radius):
    near,frame=source_box(packet,q,radius)
    w=packet['construction_witness']
    faces=np.asarray(packet['faces'])
    caps=np.asarray(packet['face_is_cap'],dtype=bool).reshape(-1)
    pts=np.asarray(packet['vertices'])
    num=int(packet['interface_face_count'])
    r=float(w['initial_radius'])
    guards,box=finite_guards(near,frame,r,int(w['cap_depth_doublings']),backend.target_design,faces,num)
    require(np.all((pts>=box.lo)&(pts<=box.hi)),'native source vertex outside analytic enclosure')
    op=physical_box(box,np.asarray(backend.request.target_axes),np.asarray(backend.request.target_origin),
                    backend.length,backend.normalization_origin)
    # Pull the physical binary64 enclosure back into the graph coordinates
    # before bounding its embedding. This includes the physical affine rounding.
    inv=(op-backend.normalization_origin)/backend.length-np.asarray(backend.request.target_origin)
    graph=verify_prism_embedding(inv,faces,caps,frame,pts,float(w['cap_depth']),r)
    incidence=certify_incidence(design,op,faces)
    return dict(closure=guards,embedding=graph,mixed_incidence=incidence,
                frame=dict((k,v) for k,v in frame.items() if k in
                           ('frame_reference_axis','pivot','sign','pivot_gap_lower','axial_lower',
                            'eigenvalue_gap_lower','eigen_check_bound','graph_discriminant_lower')))


def main(args):
    reg=Registry(args.alignment,args.corpus,args.physics)
    geom=reg.example(args.geometry)
    b=reg.panel.direct(geom.request,geom.normalization,np.asarray(geom.physical).tolist())
    d=DesignTriangles(b.design,np.asarray(b.request.design_faces))
    op=SharedStratumOperator.from_evaluator(b,args.build)
    require(op.extension.backend_identity['module_sha256']==SUPPORTED_MODULE,'unknown compiled graph implementation')
    for p in geom.probes:
        if p.chart is None or (args.row and p.id not in args.row):
            continue
        scope='R'+str(p.chart.basis.shape[1])
        if scope not in args.scopes:
            continue
        dest=args.out/(p.id+'.json')
        start=time.perf_counter()
        packet=None
        report=dict(schema='tvcg.finite-stratum.mixed-incidence.v1',row_id=p.id,geometry_id=geom.id,
                    query_id=p.origin['query_id'],raw_query_sha256=fingerprint(p.query),
                    raw_basis_sha256=fingerprint(p.chart.basis),scope=scope,
                    evaluation_map_identity=op.identity,source_map=MAP,
                    native_backend_identity=op.extension.backend_identity,restricted_event_valid=-1,
                    ambient_event_valid=-1,native_output_triangulation_event_valid=-1,
                    material_mesh_vertex_sha256=fingerprint(b.design),parent_bank_sha256=p.origin['sha256'],
                    material_mesh_faces_sha256=hashlib.sha256(np.ascontiguousarray(d.faces,dtype='<i8').tobytes()).hexdigest(),
                    trials=[])
        try:
            packet=op.extension.source_value(p.query,b.target_design,b.triangulation)
            require(packet['defined'] and packet['side_selection_defined'] and packet['source_map_version']==MAP
                    and packet['side_selection_method']==SIDE and packet['surface_type']==p.chart.surface,
                    'matching typed construction unavailable')
            require(packet['surface_type'] in ('PLANE','PARABOLOID'),'unrecognized source family')
            report.update(target_vertices_sha256=fingerprint(packet['vertices']),triangulation_hash=packet['triangulation_hash'],
                          target_faces_sha256=hashlib.sha256(np.ascontiguousarray(packet['faces'],dtype='<i8').tobytes()).hexdigest(),
                          cap_mask_sha256=hashlib.sha256(np.ascontiguousarray(packet['face_is_cap'],dtype='u1').tobytes()).hexdigest(),
                          operation='intersection' if packet['interior_is_positive'] else 'difference')
            # Read native output from the geometry-only source. The interval
            # proof independently checks the branch, cap/side guards and mesh.
            for radius in RADII:
                trial=dict(raw_box_linf_radius=radius)
                report['trials'].append(trial)
                try:
                    trial.update(check_row(b,d,packet,p.query,radius))
                except (ValueError,ArithmeticError) as error:
                    trial['unresolved']=str(error)
                    continue
                if trial['mixed_incidence']['certified']:
                    if scope=='R3':
                        report.update(restricted_event_valid=1,
                           proof_scope='complete registered material and finite source geometric incidence on raw plane chart',
                           certified_raw_linf_radius=radius)
                    else:
                        report.update(checked_source_envelope_incidence=True,
                           checked_source_raw_box_linf_radius=radius,
                           proof_scope='whole material incidence for every accepted source satisfying the bounded eigenframe contract',
                           unproved='uniform native eigensolver convergence, residual acceptance and roundoff-leaf admission on the complete centered chart')
                    break
            if report['restricted_event_valid']!=1 and 'unproved' not in report:
                report['unproved']='no complete coefficient-neighbourhood proof from the interval trials'
        except (ValueError,ArithmeticError,RuntimeError) as error:
            report['unproved']=str(error)
        save_json(dest,report)
        print(p.id,report['restricted_event_valid'],report.get('checked_source_envelope_incidence'),
              report.get('unproved',''),round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',default=DEFAULT_CORPUS)
    parser.add_argument('--alignment',default=DEFAULT_ALIGNMENT)
    parser.add_argument('--physics',default=DEFAULT_PHYSICS)
    parser.add_argument('--geometry',required=True,choices=('maskB','combB','R01','R02'))
    parser.add_argument('--row',action='append')
    parser.add_argument('--scopes',nargs='+',default=['R3','R8'],choices=('R3','R8'))
    parser.add_argument('--build',type=Path,default=None)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
