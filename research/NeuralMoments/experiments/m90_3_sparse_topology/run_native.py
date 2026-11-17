"""Predeclared native512 diagnostics; fresh outputs, no gate admission."""
import argparse,hashlib,json,sys
from pathlib import Path
from time import monotonic
import numpy as np
from sparse import mesh_runs,components,subtract,Pending,validated
from test_sparse import box_mesh

ROOT=Path('/data/xvxinyao/Spin_test2')
M90=ROOT/'research/NeuralMoments/experiments/m90_2_saved_result_validator'
sys.path.insert(0,str(M90))
from validator import digest,task_module
from geometry import prism_moments

MANIFEST=Path('/data/xvxinyao/Spin_M68_Trellis_Data_Kit/runs/m83/m83-dataset-v2-full-20261113T124505Z/operator-manifest.json')
MANIFEST_SHA='6effecd78cf03cf4ffed0b6d65f4a36f66d1a900fec4b6cbd4c171e99e0da2c4'


def write_rle(p,r):
    with p.open('xb') as f:f.write(np.asarray(r,dtype='<u4').tobytes())
    payload=p.read_bytes()
    if len(payload)%16:raise ValueError('truncated saved RLE')
    result=np.frombuffer(payload,dtype='<u4').reshape(-1,4)
    return result,hashlib.sha256(payload).hexdigest()


def cuboid(lo,hi):
    lo,hi=np.array(lo,float),np.array(hi,float);m=np.prod(hi-lo);c=(lo+hi)/2;s=(lo*lo+lo*hi+hi*hi)/3
    return m*np.array([1,*c,*s,c[1]*c[2],c[0]*c[2],c[0]*c[1]])


def execute(mode,out):
    out.mkdir(parents=True,exist_ok=False);start=monotonic();deadline=start+110
    report={'schema':'m90.3.sparse-topology-diagnostic.v1','status':'pending','mode':mode,'acceptance_status':'pending','gpu_used':False,'optimizer_steps':0,'dense512_allocated':False,'native_compilation':False,'claim':'represented-grid topology diagnostic only; no material/rotation/manufacturing/dynamics acceptance'}
    try:
        if mode=='analytic512':
            n=512;h=1.;origin=np.zeros(3);v,f=box_mesh([8,8,8],[504,504,504]);reference=cuboid([8]*3,[504]*3)
            cut=np.array([[x,y,128,384] for x in range(128,384) for y in range(128,384)],np.int64)
            report['input_provenance']={'kind':'preregistered analytic cube and internal cavity','outer_bounds':[[8]*3,[504]*3],'cavity_bounds':[[128]*3,[384]*3]}
        else:
            if digest(MANIFEST)!=MANIFEST_SHA:raise ValueError('manifest hash mismatch')
            data=json.loads(MANIFEST.read_text());rec=next(r for r in data['records'] if r.get('split')=='train' and r.get('geometry_file'))
            if rec['id']!='gso70-bird_rattle-ac8b085942-54a52d35971a':raise ValueError('preregistered first training case changed')
            cp=MANIFEST.parent/rec['case_file'];gp=MANIFEST.parent/rec['geometry_file']
            if digest(cp)!=rec['case_sha256'] or digest(gp)!=rec['geometry_sha256']:raise ValueError('case/geometry hash mismatch')
            with np.load(cp,allow_pickle=False) as a:v=a['ov'];f=a['of'];bounds=a['bounds'];reference=a['original_ten']*rec['task']['density']
            with np.load(gp,allow_pickle=False) as a:n=int(a['side']);h=float(a['h']);origin=a['origin'];editable=a['columns']
            L=float(np.ptp(bounds,axis=0).max())
            if n!=512 or not np.allclose(origin,bounds.mean(0)-.51*L,rtol=0,atol=1e-12*L) or not np.isclose(h,1.02*L/n,rtol=1e-12):raise ValueError('actual M69 grid binding mismatch')
            task=task_module().TaskSpec.from_dict(rec['task'])
            if task.task_hash!=rec['task_sha256']:raise ValueError('task hash mismatch')
            if task.density!=1.:raise ValueError('this pinned rho=1 diagnostic does not generalize material registration')
            cut=np.empty((0,4),np.int64)
            report['input_provenance']={'record_id':rec['id'],'split':'train','manifest_sha256':MANIFEST_SHA,'case_sha256':digest(cp),'geometry_sha256':digest(gp),'task_hash':task.task_hash,'task':rec['task'],'source_coordinate_units':rec.get('origin',{}).get('source_coordinate_units'),'source_backend':rec['metadata']['oracle_backend'],'historical_outer_center_samples':rec['mesh_import_evidence']['outer_center_samples'],'scope':'unchanged first training record; no labels/held-out/model reads'}
        report['grid']={'side':n,'origin':origin.tolist(),'h':h,'ray_phase':[.5,.5,.5]}
        report['backend_provenance']={'diagnostic_backend':'m90.3.exact-centre-triangle-events.v1','frozen_mesh68_ray_phase':[.500000173,.500000317,.5],'legacy_stream_decode_ray_phase':[.50000017,.50000031,.5],'same_voxel_count_does_not_prove_identical_membership':True,'mesh_precondition':'pinned source mesh registration or analytic closed box; edge incidence alone is not a self-intersection/embedded-manifold certificate'}
        np.savez(out/'outer_mesh.npz',vertices=v,faces=f)
        report['saved_outer_mesh_sha256']=digest(out/'outer_mesh.npz')
        outer,meta=mesh_runs(v,f,origin,h,n,deadline=deadline);report['mesh_reconstruction']=meta
        outer,outer_sha=write_rle(out/'original.rle',outer)
        deleted,delete_sha=write_rle(out/'deleted.rle',cut)
        report['saved_rle']={'original_sha256':outer_sha,'deleted_sha256':delete_sha,'role':'deleted','retained_export_used':False,'semantics':'new diagnostic deletion; no optimizer/model export'}
        original_stats=components(outer,n,deadline=deadline)
        if mode=='m83':
            # Strictly require the frozen editable centre set to be contained in reconstructed original.
            subtract(outer,editable,n,deadline=deadline)
            report['editable_subset_original']=True
        retained=subtract(outer,deleted,n,deadline=deadline)
        retained,keep_sha=write_rle(out/'retained.rle',retained)
        retained_stats=components(retained,n,deadline=deadline)
        cavity_stats=components(deleted,n,deadline=deadline)
        report['topology']={'original':original_stats,'retained':retained_stats,'deleted':cavity_stats,'retained_sha256':keep_sha}
        voxel_original=prism_moments(outer.tolist(),origin,h,1.)
        voxel_retained=prism_moments(retained.tolist(),origin,h,1.)
        voxel_deleted=prism_moments(deleted.tolist(),origin,h,1.)
        L=float(np.ptp(v,axis=0).max());scales=reference[0]*L**np.array([0,1,1,1,2,2,2,2,2,2],float)
        report['moments']={'mesh_or_analytic_original':reference.tolist(),'voxel_original':voxel_original.tolist(),'voxel_retained':voxel_retained.tolist(),'voxel_deleted':voxel_deleted.tolist(),'original_reference_discrepancy_normalized':((voxel_original-reference)/scales).tolist(),'same_representation_partition_residual':((voxel_retained+voxel_deleted-voxel_original)/scales).tolist(),'cross_representation_admission_threshold':None}
        if mode=='analytic512':
            expected_removed=cuboid([128]*3,[384]*3)
            error=max(np.max(abs((voxel_original-reference)/scales)),np.max(abs((voxel_deleted-expected_removed)/scales)),np.max(abs((voxel_retained-(reference-expected_removed))/scales)))
            if error>1e-10 or original_stats['components']!=1 or retained_stats['components']!=1 or cavity_stats['components']!=1:raise ValueError('native512 analytic control failed')
            report['max_analytic_moment_error']=float(error);report['status']='pass'
        else:
            historical=report['input_provenance']['historical_outer_center_samples'];report['historical_outer_voxel_count_matches']=original_stats['total_voxels']==historical
            # Frozen mesh reference, independent sparse whole-solid connectivity; empty deletion diagnostic only.
            report['frozen_task_full_retention']=task_module().evaluate_moments(reference,reference,task,length_scale=L,connectivity=retained_stats['components']==1)
            report['status']='pass' if original_stats['components']==1 and retained_stats['components']==1 and report['historical_outer_voxel_count_matches'] else 'fail'
            report['remaining_pending']=['SI physical scale absent: source expressly uses undeclared OBJ coordinate units/dimensionless thresholds','native mesh/voxel cross-representation error admission has no preregistered tolerance','homogeneous density is a registered TaskSpec material hypothesis; physical source-material identity is not measured','only full-retention topology checked, no nontrivial optimized M83 saved design','this diagnostic is not wired into or promoted by M90.2 acceptance','exact-centre diagnostic differs from frozen epsilon-shifted ray backends; matching voxel totals is not an equivalence certificate','mesh admissibility relies on pinned source registration; new arbitrary mesh self-intersections/manifold validity not independently certified']
    except Pending as e:report['status']='pending';report['reason']=str(e)
    except (ValueError,KeyError,OSError,TypeError) as e:report['status']='fail';report['reason']=str(e)
    report['seconds']=monotonic()-start
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ['mode','status','seconds','reason','topology','max_analytic_moment_error','historical_outer_voxel_count_matches']}))
    return 0 if report['status']=='pass' else 2 if report['status']=='pending' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['analytic512','m83'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.mode,a.out))
