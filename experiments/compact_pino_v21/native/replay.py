"""Export independent native targets and their pre-Boolean feature inputs on frozen queries."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

from registry import Registry, save_json, save_npz, sha
from v21_native import audit_physical_source, operand_from_source, physical_operand_vertices
from v21_native_physics import evaluate_operand_physics, targets_from_full_packet


DEFAULT_ALIGNMENT = '/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1'
DEFAULT_CORPUS = '/data/xvxinyao/TVCG/datasets/final_voxel_pino_corpus'
DEFAULT_PHYSICS = '/home/lin/Documents/Codex/2026-11-21/tvcg-phase-2-11-full-v1/outputs/v20_acceptance_1124/physics'
NATIVE_EXCEPTIONS = (ValueError, RuntimeError, ArithmeticError)


def geometry_file(out, geom, backend):
    path = out / 'geometries' / (geom.id + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    req = geom.request
    axes = np.asarray(req.target_axes)
    if not np.array_equal(axes,np.eye(3)):
        raise ValueError('RLE origin/axis export requires a Cartesian target-aligned grid')
    with np.load(geom.rle, allow_pickle=False) as grid:
        save_npz(path, vertices=backend.target_design, faces=np.asarray(req.design_faces,dtype=np.int64),
                            fixed_moments=backend.fixed, density=req.material_density,
                            origin=(grid['origin']-req.target_origin)@axes, pitch=grid['h'],
                            side=grid['native_resolution'], columns=grid['columns'], rle_path=str(geom.rle),
                            rle_sha256=sha(geom.rle), target_origin=req.target_origin, target_axes=axes,
                            physical_origin=geom.normalization['origin'], physical_length=geom.normalization['L'])
    return path


def fingerprint(a, dtype):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a,dtype=dtype).tobytes()).hexdigest()


def tetra_motion(vertices, faces, moving, density):
    """Differentiate the signed, oriented tetrahedron volume, first and second moments."""
    t = vertices[faces]
    dt = moving[faces]
    cross = [np.cross(t[:,1],t[:,2]), np.cross(t[:,2],t[:,0]), np.cross(t[:,0],t[:,1])]
    volume = density * np.einsum('ni,ni->n',t[:,0],cross[0]) / 6
    dvol = density * sum(np.einsum('ni,nid->nd',c,dt[:,k]) for k,c in enumerate(cross))/6
    center = t.sum(axis=1)
    dcenter = dt.sum(axis=1)
    first = (center.T @ dvol + np.einsum('n,nid->id',volume,dcenter)) / 4
    second = []
    for i,j in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]:
        c = center[:,i]*center[:,j] + np.einsum('nt,nt->n',t[:,:,i],t[:,:,j])
        dc = (dcenter[:,i]*center[:,j,None] + dcenter[:,j]*center[:,i,None] +
              np.einsum('ntd,nt->nd',dt[:,:,i],t[:,:,j]) + np.einsum('ntd,nt->nd',dt[:,:,j],t[:,:,i]))
        second.append((c @ dvol + volume @ dc)/20)
    return np.vstack([dvol.sum(axis=0), first, second])


def audit_jacobian(backend, retained, operand):
    from make_it_stand.verification.cherchi_motion_differential import recover_construction_vertex_motion
    axes = np.array(backend.request.target_axes)
    def to_physical(vertices):
        return (vertices@axes.T + backend.request.target_origin)*backend.length + backend.normalization_origin
    velocity = np.einsum('vkd,jk->vjd',operand.vertex_velocity,axes)*backend.length
    moving = recover_construction_vertex_motion(retained['resolved_construction_records'],backend.design,
                    to_physical(operand.vertices),velocity,to_physical(retained['vertices']))
    target = np.einsum('vkd,kj->vjd',moving/backend.length,axes)
    return tetra_motion(retained['vertices'],retained['faces'],target,backend.request.material_density)


def record_typed_targets(row, targets, fixed, density, signed_moments):
    """Record a genuine reduced-stratum material evaluation without granting an ambient J."""
    if not targets.moment_valid:
        row['restricted_material_error'] = targets.moment_diagnostic
        return
    chart = row['requested_chart']
    if (targets.chart is None or targets.chart.surface != chart['surface'] or
            not np.array_equal(targets.chart.basis,chart['basis_raw'])):
        raise ValueError('restricted material evaluation used a different raw chart basis')
    witness = targets.boundary_witness
    if witness is None or not targets.geometry_hash:
        raise ValueError('missing retained chart geometry witness')
    if row['moment_valid']:
        if row['geometry_hash'] != targets.geometry_hash:
            raise ValueError('ambient and typed calls returned different retained geometries')
        if (not np.array_equal(row['M'],targets.moments) or
                not np.array_equal(row['M_editable'],targets.editable_moments)):
            raise ValueError('ambient and typed material moments differ on the same retained geometry')
    independent = signed_moments(witness.vertices,witness.faces,density)
    row.update(moment_valid=True,M=targets.moments,M_editable=targets.editable_moments,
               geometry_hash=targets.geometry_hash,M_independent_tetra=independent+fixed,
               tetra_M_max_abs=float(np.max(abs(independent-targets.editable_moments))),
               restricted_material_geometry_hash=targets.geometry_hash,
               restricted_evaluation_map_identity=targets.evaluation_map_identity)
    if not targets.restricted_valid:
        row['restricted_derivative_error'] = targets.derivative_diagnostic
        return
    dim = chart['dimension']
    if (targets.restricted_jacobian.shape != (10,dim) or witness.vertex_velocity is None or
            witness.vertex_velocity.shape != (len(witness.vertices),3,dim)):
        raise ValueError('restricted derivative and boundary velocity require the same chart dimension')
    audited = tetra_motion(witness.vertices,witness.faces,witness.vertex_velocity,density)
    row.update(restricted_valid=True,restricted_J=targets.restricted_jacobian,
               restricted_J_independent_tetra=audited,
               restricted_tetra_J_max_abs=float(np.max(abs(audited-targets.restricted_jacobian))))


def evaluate_typed_route(row, evaluator, operand, chart, signed_moments):
    targets = evaluate_operand_physics(evaluator,operand,chart=chart,capture_boundary=True)
    if targets.evaluation_map_identity != evaluator.context_identity:
        raise ValueError('restricted material evaluator context changed')
    record_typed_targets(row,targets,evaluator.fixed,evaluator.request.material_density,signed_moments)


def attempt(fn):
    try:
        return fn(), None
    except NATIVE_EXCEPTIONS as exc:
        return None, dict(type=type(exc).__name__,diagnostic=str(exc))


def evaluate_material_routes(row, backend, q, operand, chart, signed_moments):
    """Run the ambient entry independently and consume a genuine chart source when supplied."""
    # This is a fresh deployed physics execution. The conditioning path above stops before Boolean.
    result,material_error=attempt(lambda:backend(q))
    row['material_error']=material_error
    if result is not None:
        last = backend.last
        # A full native packet owns its ten-column moving source. A requested chart owns
        # a distinct reduced velocity, even if the two calls produced an identical mesh.
        ambient_operand = operand_from_source(last['source'],q)
        if operand is not None:
            if (not np.array_equal(ambient_operand.vertices,operand.vertices) or
                    not np.array_equal(ambient_operand.faces,operand.faces) or
                    not np.array_equal(ambient_operand.face_is_cap,operand.face_is_cap) or
                    ambient_operand.operation!=operand.operation):
                raise ValueError('full physics and conditioning returned distinct finite source geometry or operation')
        elif chart is None:
            raise ValueError('full physics returned an operand rejected by its own conditioning route')
        targets = targets_from_full_packet(result,q)
        if targets.evaluation_map_identity != backend.context_identity:
            raise ValueError('ambient material evaluator context changed')
        if not targets.moment_valid:
            raise ValueError('production published an invalid Moment payload')
        mv = signed_moments(last['vertices'],last['faces'],backend.request.material_density)
        row.update(M=targets.moments,M_editable=targets.editable_moments,moment_valid=True,
                   geometry_hash=targets.geometry_hash,topology_hash=targets.topology_hash,
                   M_independent_tetra=mv+backend.fixed,
                   tetra_M_max_abs=float(np.max(abs(mv-targets.editable_moments))),
                   derivative_identity=result.get('derivative_identity'),same_source_as_production_boolean=operand is not None)
        if targets.full_j_valid:
            jac = targets.full_jacobian
            audited_j = audit_jacobian(backend,last,ambient_operand)
            row.update(J_raw=jac,full_j_valid=True,J_independent_tetra=audited_j,
                       tetra_J_max_abs=float(np.max(abs(audited_j-jac))))
        elif targets.derivative_diagnostic:
            row['derivative_diagnostic']=targets.derivative_diagnostic

    if chart is not None and operand is not None:
        evaluate_typed_route(row,backend,operand,chart,signed_moments)
    return result,material_error


def run(args):
    registry = Registry(args.alignment,args.corpus,args.physics)
    geom = registry.noodles() if args.asset == 'noodles' else registry.example(args.asset)
    out = Path(args.out).resolve()
    geom_physics = registry.panel.direct(geom.request, geom.normalization, np.asarray(geom.physical).tolist())
    gpath = geometry_file(out,geom,geom_physics)
    design_full = registry.panel.signed_moments(geom_physics.target_design,geom.request.design_faces,
                                                geom.request.material_density)
    manifest = dict(id=geom.id,split=geom.split,files=geom.hashes,geometry_file=str(gpath),geometry_sha256=sha(gpath),
                    geometry_mesh_sha256=registry.panel.meshhash(geom_physics.target_design,geom.request.design_faces),
                    mesh_full_editable=design_full,fixed_moments=geom_physics.fixed,
                    target_frame=dict(origin=geom.request.target_origin,axes=geom.request.target_axes,normalization=geom.normalization),
                    evaluator_identity=geom_physics.context_identity,boolean_identity=geom_physics.backend_identity,
                    integral_profile=geom_physics.integral_profile,arithmetic=geom_physics.arithmetic_identity,
                    profile=registry.panel.profile,profile_sha256=sha(registry.panel.PROFILE_PATH))
    save_json(out/'geometries'/(geom.id+'.json'),manifest)
    for i,probe in enumerate(geom.probes):
        if ((args.only and probe.id not in args.only) or
                (args.prefix and not probe.id.startswith(args.prefix)) or i % args.shards != args.shard):
            continue
        start=time.monotonic()
        q = np.asarray(probe.query)
        row = dict(row_id=probe.id,geometry_id=geom.id,split=geom.split,geometry_file=str(gpath),
                   query=q,surface=probe.stratum,source_record=probe.origin,frame='normalized_solver_target',
                   geometry_manifest=str(out/'geometries'/(geom.id+'.json')),
                   moment_valid=False,full_j_valid=False,restricted_valid=False,ambient_event_valid=-1,
                   restricted_event_valid=-1,event_proof='no independent event-neighborhood certificate',
                   operand_valid=False,M=None,J_raw=None,restricted_J=None,operand_file=None)
        if probe.chart:
            row['requested_chart'] = dict(surface=probe.chart.surface,dimension=probe.chart.basis.shape[1],basis_raw=probe.chart.basis)
        native = geom_physics.native
        source_fn = (native.evaluate_spin_it_faster_source_motion if probe.chart is None
                     else native.evaluate_spin_it_faster_stratum_source_motion)
        packet,source_error = attempt(lambda:source_fn(q,geom_physics.target_design,geom_physics.triangulation))
        operand = None
        if source_error is None:
            operand,source_error = attempt(lambda:operand_from_source(packet,q,chart=probe.chart))
        if operand is not None:
            facts,source_error=attempt(lambda:audit_physical_source(native,operand,geom.request.target_axes,
                                      geom.request.target_origin,geom_physics.length,geom_physics.normalization_origin))
            if source_error is not None:
                operand=None
            else:
                row['preboolean_physical_boundary_digest']=facts['input_digest']
        if operand is not None:
            path = out/'operands'/(probe.id+'.npz')
            path.parent.mkdir(parents=True,exist_ok=True)
            operand.save(path)
            physical_source = physical_operand_vertices(operand,geom.request.target_axes,geom.request.target_origin,
                                                         geom_physics.length,geom_physics.normalization_origin)
            row.update(operand_valid=True,operand_file=str(path),operand_sha256=sha(path),
                       surface=packet['surface_type'],operation=operand.operation,
                       vertex_count=len(operand.vertices),face_count=len(operand.faces),
                       cap_face_count=int(operand.face_is_cap.sum()),truncation_radius=float(packet['truncation_radius']),
                       triangulation_hash=packet.get('triangulation_hash'),side_selection_method=packet['side_selection_method'],
                       vertex_velocity_scope='raw_R10' if probe.chart is None else f'raw_R{probe.chart.basis.shape[1]}',
                       source_vertices_sha256=fingerprint(operand.vertices,'<f8'),source_faces_sha256=fingerprint(operand.faces,'<i8'),
                       physical_source_vertices_sha256=fingerprint(physical_source,'<f8'))
        row['source_error'] = source_error
        result,material_error=evaluate_material_routes(row,geom_physics,q,operand,probe.chart,
                                                     registry.panel.signed_moments)
        if result is not None:
            if probe.opposite:
                saved=geom_physics.material(q)
                side = geom_physics.complement(saved)
                if side['derivative_defined']:
                    raise ValueError('fixed operand complement unexpectedly granted a material derivative')
                op='difference' if operand.operation=='intersection' else 'intersection'
                if side['boolean_operation']!=('subtraction' if op=='difference' else 'intersection'):
                    raise ValueError('incorrect complement operation')
                other_operand=replace(operand,operation=op,vertex_velocity=None)
                name=probe.id+'_opposite'
                path=out/'operands'/(name+'.npz')
                other_operand.save(path)
                variable=np.asarray(side['variable_raw_moments'])
                tetra=registry.panel.signed_moments(side['variable_vertices'],side['variable_faces'],geom.request.material_density)
                fixed_control=dict(row_id=name,geometry_id=geom.id,split=geom.split,geometry_file=str(gpath),
                     geometry_manifest=row['geometry_manifest'],source_record=probe.origin,frame=row['frame'],
                     query=q,surface=row['surface'],operation=op,operand_file=str(path),operand_sha256=sha(path),
                     operand_valid=True,moment_valid=True,full_j_valid=False,restricted_valid=False,ambient_event_valid=-1,
                     restricted_event_valid=-1,same_operand_as=probe.id,source_vertices_sha256=row['source_vertices_sha256'],
                     source_faces_sha256=row['source_faces_sha256'],physics_scope='fixed-operand opposite Boolean with derivative undefined',
                     M=variable+geom_physics.fixed,M_editable=variable,J_raw=None,restricted_J=None,
                     M_independent_tetra=tetra+geom_physics.fixed,
                     complementary_editable_max_abs=float(np.max(abs(result['variable_raw_moments']+variable-design_full))))
                save_json(out/'rows'/(name+'.json'),fixed_control)
            geom_physics._materials.clear()
        if probe.reference is not None:
            ref=probe.reference
            ok=bool(ref['defined'])
            compare=dict(expected_defined=ok,actual_defined=result is not None)
            if ok and result is not None:
                compare['M_max_abs']=float(np.max(abs(row['M']-np.asarray(ref['M']))))
                ref_j=ref.get('J',ref.get('J_raw'))
                if ref_j is not None:
                    compare['expected_jacobian_returned']=row['full_j_valid']
                    if row['full_j_valid']:
                        compare['J_max_abs']=float(np.max(abs(row['J_raw']-np.asarray(ref_j))))
                if 'geometry_hash' in ref:
                    compare['same_geometry_hash']=ref['geometry_hash']==row['geometry_hash']
            elif not ok and result is None:
                compare['expected_diagnostic']=ref['diagnostic']
                compare['same_diagnostic']=material_error['diagnostic']==ref['diagnostic']
            row['frozen_replay']=compare
        row['elapsed_seconds']=time.monotonic()-start
        save_json(out/'rows'/(probe.id+'.json'),row)
        print(probe.id,row['operand_valid'],row['moment_valid'],row.get('tetra_J_max_abs'),row['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment',default=DEFAULT_ALIGNMENT)
    parser.add_argument('--corpus',default=DEFAULT_CORPUS)
    parser.add_argument('--physics',default=DEFAULT_PHYSICS)
    parser.add_argument('--out',required=True)
    parser.add_argument('--asset',required=True)
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--prefix')
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.shard<args.shards:
        parser.error('shard must be in [0,shards)')
    run(args)
