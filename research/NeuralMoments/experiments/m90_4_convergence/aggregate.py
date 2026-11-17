"""Reopen pinned experiment artifacts; immutable prospective criteria only."""
import json,math
from pathlib import Path
import numpy as np
from engine import ROOT,REPORT,digest,validated,membership

def screen(reference_errors,last_changes,complete,budget):
 values=list(reference_errors)+list(last_changes)
 if any(math.isfinite(x) and abs(x)>budget for x in values):return 'fail'
 if not complete or not values or not all(math.isfinite(x) for x in values):return 'pending'
 return 'pass'

def read_optional_run(directory):
 receipt_path=directory/'run.json';report_path=directory/'evidence/report.json'
 receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(status='pending',reason='missing bounded-run receipt')
 report=json.loads(report_path.read_text()) if report_path.exists() else None
 return receipt,report

def task_results(records,resolutions):
 results=[]
 for backend in ['mesh68','exact']:
  for n in resolutions:
   row=records.get(('m83',backend,n,'mesh68' if backend=='mesh68' else 'centre'),{})
   result=row.get('frozen_task_full_retention')
   results.append(dict(backend=backend,resolution=n,status='measured' if result is not None else 'pending',result=result))
 return results

def grid_series(records,family,backend,phase,case,resolutions):
 data=[];previous=None
 for n in resolutions:
  row=records.get((family,backend,n,phase))
  if not row or row['execution_status']!='pass':
   data.append(dict(resolution=n,status='pending'));previous=None;continue
  c=row['cases'][case];m=np.array(c['retained_moments']);sc=np.array(row['normalization_scales'])
  delta=None if previous is None else ((m-previous)/sc).tolist();previous=m
  data.append(dict(resolution=n,status='measured',voxel_count=c['retained_topology']['total_voxels'],volume=c['retained_volume'],reference_volume=c['reference_volume'],signed_normalized_moment_error=c['signed_normalized_moment_error'],max_normalized_moment_error=c['max_normalized_moment_error'],normalized_volume_error=c['normalized_volume_error'],components=c['retained_topology']['components'],expected_components=c['expected_components'],successive_normalized_moment_change=delta,partition_residual_normalized=c['partition_residual_normalized']))
 return data

def main():
 p=json.loads((REPORT/'protocol.json').read_text());pre=json.loads((REPORT/'preregistration.json').read_text())
 assert digest(REPORT/'protocol.json')==pre['protocol_json_sha256']
 records={};runs=[];artifact_checks={}
 for family in ['analytic','m83']:
  for backend in ['mesh68','exact']:
   for n in p['resolutions']:
    suffix='_v2' if family=='m83' and n==128 else ''
    d=REPORT/f'{family}_{backend}_{n}{suffix}'
    receipt,run=read_optional_run(d);runs.append(dict(path=str(d),**receipt))
    if run is None:continue
    assert run['provenance']['protocol_sha256']==pre['protocol_json_sha256']
    for row in run['phases']:
     records[(family,backend,n,row['phase']['name'])]=dict(row,provenance=run['provenance'])
     if row['execution_status']!='pass':continue
     saved=[row['original']]+[c[k] for c in row['cases'].values() for k in ['retained','deleted']]
     for item in saved:
      path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
      assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256']
      artifact_checks[str(path)]=dict(sha256=item['sha256'],bytes=item['bytes'])
 def opened(item,n):
  path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
  if digest(path)!=item['sha256']:raise ValueError('saved RLE changed')
  return validated(np.fromfile(path,dtype='<u4').reshape(-1,4),n)
 def compare(a,b,n,case):
  aa=a['original'] if case=='original' else a['cases'][case]['retained']
  bb=b['original'] if case=='original' else b['cases'][case]['retained']
  ma=a['original_moments'] if case=='original' else a['cases'][case]['retained_moments']
  mb=b['original_moments'] if case=='original' else b['cases'][case]['retained_moments']
  ta=a['original_topology'] if case=='original' else a['cases'][case]['retained_topology']
  tb=b['original_topology'] if case=='original' else b['cases'][case]['retained_topology']
  diff=membership(opened(aa,n),opened(bb,n),n)
  delta=(np.array(mb)-ma)/a['normalization_scales']
  return dict(membership=diff,signed_normalized_moment_difference=delta.tolist(),normalized_volume_difference=float(delta[0]),components_a=ta['components'],components_b=tb['components'],status='pass' if diff['xor_voxels']==0 else 'fail')
 phases=[];backends=[];series=[]
 for family in ['analytic','m83']:
  cases=['full_retention']+(['internal_cavity','disconnected_slab'] if family=='analytic' else [])
  for n in p['resolutions']:
   for backend in ['mesh68','exact']:
    nominal='mesh68' if backend=='mesh68' else 'centre';base=records.get((family,backend,n,nominal))
    for phase in p['phases']:
     if phase['name']==nominal:continue
     row=records.get((family,backend,n,phase['name']))
     for case in ['original',*cases]:
      entry=dict(family=family,backend=backend,resolution=n,case=case,baseline_phase=nominal,probe_phase=phase['name'],status='pending')
      if base and row and base['execution_status']==row['execution_status']=='pass':entry.update(compare(base,row,n,case))
      phases.append(entry)
   # Both nominal-backend comparison and same-phase differential are explicit.
   for pa,pb,label in [('mesh68','centre','nominal'),*[(x['name'],x['name'],'same_phase') for x in p['phases']]]:
    a=records.get((family,'mesh68',n,pa));b=records.get((family,'exact',n,pb))
    for case in ['original',*cases]:
     entry=dict(family=family,resolution=n,case=case,comparison=label,mesh68_phase=pa,exact_phase=pb,status='pending')
     if a and b and a['execution_status']==b['execution_status']=='pass':entry.update(compare(a,b,n,case))
     backends.append(entry)
  for backend in ['mesh68','exact']:
   for phase in p['phases']:
    for case in cases:
     data=grid_series(records,family,backend,phase['name'],case,p['resolutions'])
     complete=all(x['status']=='measured' for x in data)
     last=data[-1];errs=last.get('signed_normalized_moment_error',[]);changes=last.get('successive_normalized_moment_change') or []
     accuracy=screen(errs,changes,complete,p['criteria']['default_accuracy_budget'])
     topology='fail' if any(x.get('components')!=x.get('expected_components') for x in data if x['status']=='measured') else 'pass' if complete else 'pending'
     measured=[x['max_normalized_moment_error'] for x in data if x['status']=='measured']
     series.append(dict(family=family,backend=backend,phase=phase['name'],case=case,grids=data,accuracy_screen=accuracy,topology_control=topology,max_error_nonincreasing=all(b<=a for a,b in zip(measured,measured[1:])) if complete else None,interpretation='negative topology control, never admitted design' if case=='disconnected_slab' else 'representation screen, not M69 feasibility'))
 def combine(values):return 'fail' if 'fail' in values else 'pending' if 'pending' in values or not values else 'pass'
 summary={}
 for family in ['analytic','m83']:
  s=[x for x in series if x['family']==family and x['case']!='disconnected_slab'];ph=[x for x in phases if x['family']==family];be=[x for x in backends if x['family']==family and x['comparison']=='nominal']
  statuses=dict(accuracy=combine([x['accuracy_screen'] for x in s]),topology=combine([x['topology_control'] for x in s]),phase_robustness=combine([x['status'] for x in ph]),nominal_backend_equivalence=combine([x['status'] for x in be]))
  summary[family]=dict(**statuses,representation_screen=combine(list(statuses.values())),M69_design_acceptance='pending: no nontrivial design; full-retention violates unchanged TaskSpec')
 result=dict(schema='m90.4.convergence-comparison.v1',protocol_sha256=pre['protocol_json_sha256'],criteria=p['criteria'],runs=runs,phase_records=len(records),saved_artifacts_verified=len(artifact_checks),artifact_checks=artifact_checks,series=series,phase_comparisons=phases,backend_comparisons=backends,summary=summary,full_retention_task_results=task_results(records,p['resolutions']),claims=['Represented-domain sparse diagnostics only.','No SI or manufacturing/dynamic stability certification.','Same-phase agreement on this panel is not proof of backend equivalence for arbitrary meshes.','Fidelity screen is conservative and does not replace final M69 constraints.'])
 with (REPORT/'convergence_v2.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(dict(summary=summary,phase_records=len(records),artifacts=len(artifact_checks))))
if __name__=='__main__':main()
