"""One bounded family/backend/resolution; no optimization or model loading."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from engine import *
from decimal import Decimal

def verify_budget(mass_tolerance,inertia_tolerance,registered_budget):
 return Decimal('0.1')*min(Decimal(str(mass_tolerance)),Decimal(str(inertia_tolerance)))==Decimal(str(registered_budget))

def run(family,backend,n,out):
 start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
 prereg=json.loads((REPORT/'preregistration.json').read_text());phash=digest(REPORT/'protocol.json')
 if phash!=prereg['protocol_json_sha256']:raise ValueError('protocol changed after freeze')
 seal=json.loads((REPORT/'execution_seal_v2.json').read_text())
 for row in seal['sources']:
  if digest(row['path'])!=row['sha256']:raise ValueError('execution source changed')
 p=json.loads((REPORT/'protocol.json').read_text());assert n in p['resolutions']
 provenance=dict(family=family,backend=backend,resolution=n,protocol_sha256=phash,execution_seal_sha256=digest(REPORT/'execution_seal_v2.json'),units=p['units'])
 if family=='analytic':
  bounds=np.array(p['geometry']['outer_box']);v,f=box_mesh(*bounds);reference=cuboid(*bounds);rho=1.
  task=None;provenance['source']='predeclared analytic box, internal cavity and disconnecting slab'
 else:
  manifest=Path('/data/xvxinyao/Spin_M68_Trellis_Data_Kit/runs/m83/m83-dataset-v2-full-20261113T124505Z/operator-manifest.json')
  if digest(manifest)!=p['manifest_sha256']:raise ValueError('manifest hash mismatch')
  rec=next(r for r in json.loads(manifest.read_text())['records'] if r['id']==p['record_id'])
  if rec['split']!='train':raise ValueError('held-out case forbidden')
  cp=manifest.parent/rec['case_file'];gp=manifest.parent/rec['geometry_file']
  if digest(cp)!=rec['case_sha256'] or digest(gp)!=rec['geometry_sha256']:raise ValueError('case or geometry hash mismatch')
  with np.load(cp,allow_pickle=False) as data:v=data['ov'];f=data['of'];bounds=data['bounds'];reference=data['original_ten']*rec['task']['density']
  sys.path.insert(0,'/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src')
  from task69 import TaskSpec,evaluate_moments
  task=TaskSpec.from_dict(rec['task']);rho=task.density
  if task.task_hash!=rec['task_sha256'] or task.native_resolution!=512:raise ValueError('frozen TaskSpec mismatch')
  if not verify_budget(task.mass_tolerance,task.inertia_tolerance,p['criteria']['default_accuracy_budget']):raise ValueError('protocol accuracy derivation mismatch')
  provenance.update(record_id=rec['id'],split=rec['split'],case_sha256=rec['case_sha256'],geometry_sha256=rec['geometry_sha256'],task_hash=task.task_hash,task=task.to_dict(),source_units=rec['origin'].get('source_coordinate_units'),historical_outer_center_samples=rec['mesh_import_evidence']['outer_center_samples'])
 L=float(np.ptp(bounds,axis=0).max());o=bounds.mean(0)-.51*L;h=1.02*L/n
 scales=reference[0]*L**np.array(p['moment_degrees'])
 provenance['grid']=dict(origin=o.tolist(),h=h,outer_length=L,oracle_default_length=n*h,encoder_length='not invoked; editable extent is a distinct internal model normalization',native_TaskSpec_resolution=512,diagnostic_resolution=n)
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 phases=[]
 for phase in p['phases']:
  ps=time.monotonic();dest=out/phase['name'];dest.mkdir();row=dict(phase=phase,execution_status='pending',claim='represented-domain diagnostic only')
  try:
   fn=mesh68_runs if backend=='mesh68' else exact_runs
   outer,meta=fn(v,f,o,h,n,phase['xy']);outer,saved=save_rle(dest/'original.rle',outer,n)
   original_m=moments(outer,o,h,rho);original_top=components(outer,n)
   cases={}
   choices=[('full_retention',np.empty((0,4),np.int64),np.zeros(10),1)]
   if family=='analytic':
    for name,bkey,expected in [('internal_cavity','internal_cavity',1),('disconnected_slab','disconnecting_slab',2)]:
     bb=p['geometry'][bkey];choices.append((name,box_runs(*bb,o,h,n),rho*cuboid(*bb),expected))
   for name,deleted,reference_deleted,expected in choices:
    deleted,saved_deleted=save_rle(dest/(name+'.deleted.rle'),deleted,n)
    retained=subtract(outer,deleted,n);retained,saved_retained=save_rle(dest/(name+'.retained.rle'),retained,n)
    mdelete=moments(deleted,o,h,rho);mretain=moments(retained,o,h,rho);top=components(retained,n)
    reference_retained=reference-reference_deleted
    error=(mretain-reference_retained)/scales
    partition=(original_m-mretain-mdelete)/scales
    if np.max(np.abs(partition))>p['criteria']['moment_partition_atol']:raise ValueError('prism partition failed')
    # Independent closed-form integral of the actual chosen cuboid cell bounds.
    analytic_prism_error=None
    if family=='analytic' and len(deleted):
     lo=o+np.min(deleted[:,:3],axis=0)*h
     hi=o+np.array([deleted[:,0].max()+1,deleted[:,1].max()+1,deleted[:,3].max()])*h
     analytic_prism_error=float(np.max(abs((mdelete-rho*cuboid(lo,hi))/scales)))
     if analytic_prism_error>p['criteria']['analytic_prism_atol']:raise ValueError('analytic removed-prism integration mismatch')
    cases[name]=dict(deleted=saved_deleted,retained=saved_retained,retained_moments=mretain.tolist(),removed_moments=mdelete.tolist(),reference_retained=reference_retained.tolist(),reference_removed=reference_deleted.tolist(),signed_normalized_moment_error=error.tolist(),absolute_normalized_moment_error=abs(error).tolist(),max_normalized_moment_error=float(max(abs(error))),retained_volume=float(mretain[0]/rho),reference_volume=float(reference_retained[0]/rho),normalized_volume_error=float(error[0]),partition_residual_normalized=partition.tolist(),analytic_prism_error=analytic_prism_error,retained_topology=top,expected_components=expected,topology_control_pass=top['components']==expected,design_admissibility='negative control deliberately cuts outer shell' if name=='disconnected_slab' else 'topology/moments diagnostic; no material-design admission')
   row.update(execution_status='pass',original=saved,backend_metadata=meta,original_topology=original_top,original_moments=original_m.tolist(),original_reference=reference.tolist(),normalization_scales=scales.tolist(),original_signed_normalized_error=((original_m-reference)/scales).tolist(),cases=cases)
   if task is not None:
    row['frozen_task_full_retention']=evaluate_moments(reference,reference,task,length_scale=L,connectivity=original_top['components']==1)
    row['task_evaluation_path']='task69.evaluate_moments(immutable_mesh_original, immutable_mesh_original, frozen_task, length_scale=outer_L, connectivity=independent_components==1); no Oracle or optimizer called'
  except Pending as exc:row.update(execution_status='pending',reason=str(exc))
  except (ValueError,KeyError,OSError,TypeError,MemoryError) as exc:row.update(execution_status='fail',reason=str(exc))
  row['seconds']=time.monotonic()-ps
  (dest/'phase_report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');phases.append(row)
  print(json.dumps(dict(family=family,backend=backend,n=n,phase=phase['name'],status=row['execution_status'],seconds=row['seconds'],reason=row.get('reason'))),flush=True)
 report=dict(schema='m90.4.bounded-run.v1',provenance=provenance,phases=phases,seconds=time.monotonic()-start)
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 return 1 if any(x['execution_status']=='fail' for x in phases) else 2 if any(x['execution_status']=='pending' for x in phases) else 0

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--family',choices=['analytic','m83'],required=True);a.add_argument('--backend',choices=['mesh68','exact'],required=True);a.add_argument('--n',type=int,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();raise SystemExit(run(args.family,args.backend,args.n,args.out))
