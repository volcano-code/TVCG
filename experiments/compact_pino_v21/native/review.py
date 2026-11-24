"""Verify replay completeness and export compact per-row evidence without large native operands."""
import argparse
from collections import Counter
from pathlib import Path
import platform

import numpy as np

from registry import Registry, read, save_json, save_npz, sha
from v21_native import FiniteOperand, audit_physical_source
from replay import DEFAULT_ALIGNMENT,DEFAULT_CORPUS,DEFAULT_PHYSICS


def num(x):
    return float(np.max(np.abs(x)))


def runtime_provenance(reg):
    import trimesh
    import _cherchi_core_boolean
    import _cherchi_rational_moments
    from make_it_stand import spin
    from make_it_stand.verification import cherchi_inprocess, cherchi_motion_differential
    from phase3_native_teacher import teacher
    expected=read(reg.alignment/'provenance.json')
    by_path={x['path']:x['sha256'] for x in expected['files']}
    provenance=[]
    for module in (spin._native_module(),_cherchi_core_boolean,_cherchi_rational_moments,spin,
                   cherchi_inprocess,cherchi_motion_differential,teacher):
        path=str(Path(module.__file__).resolve())
        actual=sha(path)
        if by_path.get(path)!=actual:
            raise ValueError('deployed library identity mismatch: '+path)
        provenance.append(dict(module=module.__name__,path=path,sha256=actual))
    host_manifest=Path(__file__).with_name('host_inputs.json')
    return dict(host_inputs_file=host_manifest.name,host_inputs_sha256=sha(host_manifest),
                python=platform.python_version(),numpy=np.__version__,trimesh=trimesh.__version__,
                arithmetic=expected['native_arithmetic'],loaded_modules=provenance,
                archive=reg.alignment/'provenance.json',archive_sha256=sha(reg.alignment/'provenance.json'),
                profile_path=reg.panel.PROFILE_PATH,profile_sha256=sha(reg.panel.PROFILE_PATH),
                profile=reg.panel.profile)


def stratum_summary(rows):
    return dict(queried=len(rows), counts_by_geometry=dict(Counter(r['geometry_id'] for r in rows)),
                mask_sources='Matched exact queried raw coefficients and preserved chart query IDs.',
                operand_defined=sum(r['operand_valid'] for r in rows),
                moment_defined=sum(r['moment_valid'] for r in rows),
                restricted_defined=sum(r['restricted_valid'] for r in rows),
                typed_source_errors=dict(Counter(r['source_error']['diagnostic'] for r in rows if r['source_error'])),
                full_production_errors=dict(Counter(r['material_error']['diagnostic'] for r in rows if r['material_error'])))


def run(args):
    out=Path(args.out)
    dest=Path(args.review_out)
    dest.mkdir(parents=True,exist_ok=True)
    reg=Registry(args.alignment,args.corpus,args.physics)
    records={p.stem:read(p) for p in (out/'rows').glob('*.json')}
    official=read(reg.strict_release)
    source_geometry={}
    expected_ids=set()
    assets=['noodles',*reg.strict.keys(),'T2',*reg.train_examples.keys(),*reg.panel.cross_ids[:2],
            *[k for k in reg.panel.train_ids if k!='T2']]
    for key in assets:
        geom=reg.noodles() if key=='noodles' else reg.example(key)
        p=out/'geometries'/(key+'.json')
        gm=read(p)
        geometry_path=Path(gm['geometry_file'])
        if sha(geometry_path)!=gm['geometry_sha256']:
            raise ValueError(f'changed exported geometry: {key}')
        for path,digest in gm['files'].items():
            if sha(path)!=digest:
                raise ValueError('registered source changed: '+path)
        with np.load(geometry_path,allow_pickle=False) as z:
            rle_path=str(z['rle_path'].item())
            source_geometry[key]=dict(split=geom.split,faces=len(z['faces']),vertices=len(z['vertices']),
                 native_grid=int(z['side']),rle_path=rle_path,rle_sha256=sha(rle_path),density=float(z['density']),
                 raw_file_hashes=gm['files'],frame=gm['target_frame'],fixed_moments=gm['fixed_moments'],
                 mesh_full_editable=gm['mesh_full_editable'],geometry_mesh_sha256=gm['geometry_mesh_sha256'],
                 evaluator_identity=gm['evaluator_identity'],boolean_identity=gm['boolean_identity'])
        for probe in geom.probes:
            expected_ids.add(probe.id)
            if not np.array_equal(records[probe.id]['query'],probe.query):
                raise ValueError('raw query mismatch: '+probe.id)
            if probe.opposite:
                expected_ids.add(probe.id+'_opposite')
    if set(records)!=expected_ids:
        raise ValueError(f'row inventory differs from registration: {set(records)^expected_ids}')
    ledger=[]
    moment_ids, full_ids=[],[]
    restricted={3:[],8:[]}
    values, editable, integral_ref, jac, jac_ref=[],[],[],[],[]
    rows=[records[k] for k in sorted(records)]
    for r in rows:
        key=r['row_id']
        fields=['row_id','geometry_id','split','surface','source_record','operand_valid','moment_valid','full_j_valid',
                'restricted_valid','ambient_event_valid','restricted_event_valid','operation','vertex_count','face_count',
                'cap_face_count','truncation_radius','triangulation_hash','side_selection_method','source_vertices_sha256',
                'source_faces_sha256','physical_source_vertices_sha256','operand_sha256','geometry_hash','topology_hash',
                'source_error','material_error','frozen_replay','tetra_M_max_abs','tetra_J_max_abs',
                'restricted_material_error','restricted_derivative_error','restricted_material_geometry_hash',
                'restricted_evaluation_map_identity','restricted_tetra_J_max_abs',
                'complementary_editable_max_abs','same_operand_as','event_proof','physics_scope']
        item={k:r[k] for k in fields if k in r}
        if any(int(r[k])!=-1 for k in ('ambient_event_valid','restricted_event_valid')):
            raise ValueError('event claim without separate proof: '+key)
        if r.get('requested_chart'):
            item['requested_chart']=r['requested_chart']['surface']
            item['requested_chart_dimension']=r['requested_chart']['dimension']
        if r['operand_valid']:
            frame=source_geometry[r['geometry_id']]['frame']
            operand=FiniteOperand.load(r['operand_file'])
            facts=audit_physical_source(reg.panel.spin._native_module(),operand,frame['axes'],frame['origin'],
                                        frame['normalization']['L'],frame['normalization']['origin'])
            item['preboolean_physical_boundary_digest']=facts['input_digest']
            if r.get('preboolean_physical_boundary_digest',facts['input_digest'])!=facts['input_digest']:
                raise ValueError('pre-Boolean physical operand changed')
            if sha(r['operand_file'])!=r['operand_sha256']:
                raise ValueError('changed operand: '+key)
            with np.load(r['operand_file'],allow_pickle=False) as z:
                if z['operation'].item()!=r['operation'] or len(z['face_is_cap'])!=len(z['faces']):
                    raise ValueError('operand side or cap registration changed')
                if r.get('same_operand_as'):
                    source=records[r['same_operand_as']]
                    with np.load(source['operand_file'],allow_pickle=False) as other:
                        for field in ('vertices','faces','face_is_cap'):
                            if not np.array_equal(z[field],other[field]):
                                raise ValueError('fixed-operand control rebuilt its source')
        if r['moment_valid']:
            moment_ids.append(key)
            values.append(r['M']);editable.append(r['M_editable']);integral_ref.append(r['M_independent_tetra'])
            if not np.array_equal(np.asarray(r['M_editable'])+source_geometry[r['geometry_id']]['fixed_moments'],
                                  r['M']):
                raise ValueError('native material decomposition mismatch: '+key)
        elif r['M'] is not None:
            raise ValueError('invalid Moment carries numeric data: '+key)
        if r['full_j_valid']:
            full_ids.append(key)
            jac.append(r['J_raw']);jac_ref.append(r['J_independent_tetra'])
        elif r['J_raw'] is not None:
            raise ValueError('invalid ambient derivative carries numeric data: '+key)
        if r['restricted_valid']:
            dim=r['requested_chart']['dimension']
            typed=np.asarray(r['restricted_J'])
            if typed.shape!=(10,dim) or not np.isfinite(typed).all():
                raise ValueError('invalid raw chart differential: '+key)
            audited=np.asarray(r['restricted_J_independent_tetra'])
            basis=np.asarray(r['requested_chart']['basis_raw'])
            if (audited.shape!=typed.shape or basis.shape!=(10,dim) or not np.isfinite(audited).all()
                    or not np.isfinite(basis).all() or r['restricted_material_geometry_hash']!=r['geometry_hash']):
                raise ValueError('invalid restricted geometry, raw basis or independent derivative: '+key)
            restricted[dim].append((key,typed,audited,basis))
        elif r['restricted_J'] is not None:
            raise ValueError('invalid chart carries numeric data: '+key)
        comp=r.get('frozen_replay')
        if comp is not None:
            if comp['actual_defined']!=comp['expected_defined'] or comp.get('same_diagnostic') is False or comp.get('same_geometry_hash') is False:
                raise ValueError('frozen replay differed: '+key)
        ledger.append(item)
    nums=dict(row_id=np.asarray([r['row_id'] for r in rows]),raw_query=np.array([r['query'] for r in rows]),
              moment_row_id=np.asarray(moment_ids),M=np.asarray(values),M_editable=np.asarray(editable),
              M_independent_tetra=np.asarray(integral_ref),full_j_row_id=np.asarray(full_ids),J_raw=np.asarray(jac),
              J_independent_tetra=np.asarray(jac_ref))
    for dim, entries in restricted.items():
        if entries:
            keys,typed,independent,bases=zip(*entries)
            nums[f'R{dim}_row_id']=np.asarray(keys)
            nums[f'R{dim}_J']=np.stack(typed)
            nums[f'R{dim}_J_independent_tetra']=np.stack(independent)
            nums[f'R{dim}_basis_raw']=np.stack(bases)
    array_path=dest/'native_targets.npz'
    save_npz(array_path,**nums)
    strict_charts={d:[r for r in rows if r['geometry_id'] in reg.strict
                      and r.get('requested_chart',{}).get('dimension')==d] for d in (3,8)}
    noodles=[r for r in rows if r['geometry_id']=='noodles']
    full_noodles=[r for r in noodles if r['full_j_valid']]
    sum_controls={}
    for prefix in ('T2_hyperbolic','combB_hyperbolic','T2_ellipsoid'):
        po,ne,sc,other=[records[prefix+s] for s in ('_plus','_minus','_scale2','_plus_opposite')]
        complete=np.asarray(source_geometry[po['geometry_id']]['mesh_full_editable'])
        sign=np.asarray(po['M_editable'])+ne['M_editable']-complete
        with np.load(po['operand_file']) as a,np.load(sc['operand_file']) as b:
            same_geometry=all(np.array_equal(a[s],b[s]) for s in ('vertices','faces','face_is_cap','operation'))
            scale_motion=num(a['vertex_velocity']-2*b['vertex_velocity'])
        sum_controls[prefix]=dict(plus_mass=po['M_editable'][0],minus_mass=ne['M_editable'][0],editable_full_mass=complete[0],
            standalone_minus_plus_minus_full=sign,relative_mass_sign_residual=sign[0]/complete[0],
            source_hash_same_under_fresh_minus=po['source_vertices_sha256']==ne['source_vertices_sha256'],
            same_operand_opposite_max_abs=other['complementary_editable_max_abs'],coefficient_scale2_same_source=same_geometry,
            scale2_M_max_abs=num(np.array(po['M'])-sc['M']),scale2_J_max_abs=num(np.array(po['J_raw'])-2*np.array(sc['J_raw'])),
            scale2_vertex_motion_max_abs=scale_motion)
    alt=read(out/'diagnostics/alternative_chart.json')
    if {r['chart'] for r in alt['rows']}!={'PLANE','PARABOLOID'}:
        raise ValueError('alternative chart control missing')
    native_fd=read(out/'diagnostics/deployed_full_motion.json')
    grids=read(out/'diagnostic256/manifest.json')
    if len(grids['rows'])!=len(reg.panel.cross_ids[:2])+len(reg.panel.train_ids)-1:
        raise ValueError('separate native256 cross-domain grids incomplete')
    summary=dict(schema='compact-pino.v21.native-replay.1',
        scope='Deployed finite-Cherchi geometry and numerical same-call physical targets; no neighborhood certification',
        coordinate_convention='Normalized solver target. Input cross monomials are doubled. Output mixed moments are undoubled.',
        numerical_masks='Moment, ambient derivative and typed derivative presence are independent of event certificates (unknown=-1).',
        full_j_units='dM/dq_raw. Vertices are normalized target coordinates. Native unit differential pulled back exactly once.',
        release=dict(path=str(reg.strict_release),sha256=sha(reg.strict_release),official_counts=official['counts']),
        numerical_archive=dict(file=array_path.name,sha256=sha(array_path),fields=list(nums)),sources=source_geometry,
        runtime=runtime_provenance(reg),counts=dict(rows=len(rows),operands=sum(r['operand_valid'] for r in rows),
             moments=len(moment_ids),ambient_J=len(full_ids),restricted_J=sum(r['restricted_valid'] for r in rows),
             full_event_certified=sum(r['ambient_event_valid']==1 for r in rows),
             restricted_event_certified=sum(r['restricted_event_valid']==1 for r in rows)),controls=sum_controls,
        trajectory=dict(registered=len(noodles),accepted=len(full_noodles),rejections=[
             dict(row=r['row_id'],source=r['source_error'],material=r['material_error']) for r in noodles if not r['moment_valid']],
             identical_retained_hashes=sum(r['frozen_replay']['same_geometry_hash'] for r in full_noodles),
             max_abs_frozen_M=max(r['frozen_replay']['M_max_abs'] for r in full_noodles),
             max_abs_frozen_J=max(r['frozen_replay']['J_max_abs'] for r in full_noodles),
             max_abs_tetra_M=max(r['tetra_M_max_abs'] for r in full_noodles),
             max_abs_tetra_J=max(r['tetra_J_max_abs'] for r in full_noodles)),
        strict_development_R3=stratum_summary(strict_charts[3]),
        strict_development_R8=stratum_summary(strict_charts[8]),
        full_motion_fd=native_fd,alternative_chart_diagnostic=alt,
        separately_reencoded_256=grids,
        independent_tetra_max_abs=dict(M=num(nums['M']-nums['M_independent_tetra']),J=num(nums['J_raw']-nums['J_independent_tetra'])),
        rows=ledger)
    save_json(dest/'native_replay.json',summary)
    print(summary['counts'],summary['trajectory'],summary['strict_development_R8'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment',default=DEFAULT_ALIGNMENT)
    p.add_argument('--corpus',default=DEFAULT_CORPUS)
    p.add_argument('--physics',default=DEFAULT_PHYSICS)
    p.add_argument('--out',required=True)
    p.add_argument('--review-out',required=True)
    run(p.parse_args())
