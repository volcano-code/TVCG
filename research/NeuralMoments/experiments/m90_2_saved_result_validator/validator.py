"""M90.2 independently reopen and audit a pinned saved deletion result.

Registry/result SHA256 must come from a separate trusted registration, never the
result's self-reported digest. This module imports frozen task69, no model/oracle.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import monotonic
import numpy as np
from geometry import Pending,canonical,bitmap,encode,prism_moments,cell_moments,mesh_occupancy,components,check_budget

TASK_SOURCE=Path('/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src/task69.py')
TASK_SHA256='5463dcfe57f3aea21f577fd77d26867ef9e7ec18098b7ff25e7bde5209fc3389'
CLAIM='represented-domain rotation/material acceptance only; no manufacturability or dynamic stability certification'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def task_module():
    if digest(TASK_SOURCE)!=TASK_SHA256:raise ValueError('pinned task69 source hash mismatch')
    name='_m90_2_pinned_task69'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,TASK_SOURCE);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]

def complement_retained(editable,retained,side):
    """Explicit editable minus retained; rejects support errors and never treats retained as deleted.

    Interval-only implementation: no dense full512 allocation or hidden repair.
    """
    canonical(editable,side);canonical(retained,side)
    domains={}
    for x,y,a,b in editable:domains.setdefault((x,y),[]).append((a,b))
    keeps={}
    for x,y,a,b in retained:
        if not any(lo<=a<b<=hi for lo,hi in domains.get((x,y),[])):raise ValueError('retained outside editable support')
        keeps.setdefault((x,y),[]).append((a,b))
    out=[]
    for x,y,a,b in editable:
        cursor=a
        for lo,hi in keeps.get((x,y),[]):
            if hi<=a or lo>=b:continue
            if cursor<lo:out.append([x,y,cursor,lo])
            cursor=hi
        if cursor<b:out.append([x,y,cursor,b])
    return out

def _json(path):
    if Path(path).stat().st_size>16*1024**2:raise Pending('JSON input exceeds bounded adapter limit')
    def pairs(rows):
        d={}
        for k,v in rows:
            if k in d:raise ValueError('duplicate JSON key: '+k)
            d[k]=v
        return d
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def _required(d,keys):
    missing=[k for k in keys if k not in d or d[k] is None]
    if missing:raise Pending('missing metadata: '+','.join(missing))

def _bound(base,entry):
    _required(entry,['path','sha256']);p=(base/entry['path']).resolve()
    if not p.is_file():raise Pending('registered file unavailable: '+str(p))
    if digest(p)!=entry['sha256']:raise ValueError('registered file hash mismatch: '+str(p))
    return p

def _arrays(path):
    import zipfile
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist())>128*1024**2:raise Pending('expanded NPZ exceeds bounded input budget')
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def validate(registry_path,result_path,*,registry_sha256=None,result_sha256=None):
    report={'schema':'m90.2.saved-result-audit.v1','status':'pending','gates':{f'A{i}':{'status':'pending','reason':'not evaluated'} for i in range(7)},'claim':None,'scope':CLAIM,'gpu_used':False,'optimizer_steps':0}
    stage='A0';start=monotonic();deadline=start+110
    def gate(k,status,reason):report['gates'][k]={'status':status,'reason':reason}
    try:
        if not registry_sha256 or not result_sha256:raise Pending('independently pinned registry and result hashes required')
        rp,sp=Path(registry_path),Path(result_path)
        for p,expected in [(rp,registry_sha256),(sp,result_sha256)]:
            if not p.is_file():raise Pending('saved file unavailable: '+str(p))
            if digest(p)!=expected:raise ValueError('pinned file hash mismatch: '+str(p))
        g,s=_json(rp),_json(sp)
        report['provenance']={'registry_sha256':registry_sha256,'result_sha256':result_sha256,'task_source_sha256':TASK_SHA256}
        _required(g,['schema','id','grid','frame','units','backend','length_scale','density','task','task_hash','task_source_sha256','original','generator','moment_tolerance','coordinate_space'])
        _required(s,['schema','id','grid','frame','units','backend','task_hash','original_sha256','registry_sha256','density'])
        if g['schema']!='m90.2.registry.v1' or s['schema']!='m90.2.saved-result.v1':raise ValueError('schema mismatch')
        if s['registry_sha256']!=registry_sha256:raise ValueError('result registry binding mismatch')
        for k in ['id','grid','frame','units','backend','task_hash']:
            if s[k]!=g[k]:raise ValueError('result '+k+' mismatch')
        if s['original_sha256']!=g['original']['sha256']:raise ValueError('result original hash mismatch')
        if g['backend']!='exact-voxel-prisms.v1':raise Pending('cross-representation backend requires registered discretization/error adapter')
        if g['units']!={'length':'m','mass':'kg','density':'kg/m^3'}:raise Pending('unit conversion not registered for this adapter')
        if g['coordinate_space']!='registered-task':raise Pending('all mesh/grid/moments/TaskSpec inputs must already be expressed in the registered task frame; no implicit transform')
        _required(g['frame'],['name','world_to_task_rotation','translation'])
        R=np.asarray(g['frame']['world_to_task_rotation'],float);t=np.asarray(g['frame']['translation'],float)
        if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all() or not np.allclose(R.T@R,np.eye(3),rtol=0,atol=1e-12) or np.linalg.det(R)<0:raise ValueError('invalid rigid frame')
        _required(g['generator'],['kind'])
        if 'model_checkpoint' not in g['generator']:raise Pending('explicit model/checkpoint provenance required, including null when unused')
        if g['generator']['model_checkpoint'] is not None:_bound(rp.parent,g['generator']['model_checkpoint'])
        grid=g['grid'];_required(grid,['side','origin','h']);n=grid['side'];o=np.asarray(grid['origin'],float);h=grid['h']
        if type(n) is not int or not 4<=n<=2048 or o.shape!=(3,) or not np.isfinite(o).all() or type(h) not in (float,int) or not np.isfinite(h) or h<=0:raise ValueError('invalid registered grid')
        mod=task_module()
        if g['task_source_sha256']!=TASK_SHA256:raise ValueError('wrong task evaluator provenance')
        # TaskSpec.from_dict supplies defaults; this wrapper deliberately forbids omissions.
        required=set(mod.TaskSpec().to_dict())
        if not required<=g['task'].keys():raise Pending('incomplete registered TaskSpec')
        task=mod.TaskSpec.from_dict(g['task'])
        if task.task_hash!=g['task_hash'] or task.native_resolution!=n:raise ValueError('task hash/resolution mismatch')
        if type(g['length_scale']) not in (int,float) or not np.isfinite(g['length_scale']) or g['length_scale']<=0:raise ValueError('invalid length scale')
        if g['moment_tolerance']!=1e-10:raise ValueError('unregistered moment residual tolerance')
        original_path=_bound(rp.parent,g['original']);check_budget(n)
        d=_arrays(original_path);_required(d,['occupied','editable','protected','density','original_moments','fixed_moments'])
        occ_for_extent=d['occupied']
        if occ_for_extent.shape!=(n,)*3 or occ_for_extent.dtype!=bool or not occ_for_extent.any():raise ValueError('invalid occupancy for length reference')
        extent=(np.ptp(np.argwhere(occ_for_extent),axis=0)+1)*h
        measured_length=float(extent.max())
        if not np.isfinite(measured_length) or not np.isclose(g['length_scale'],measured_length,rtol=1e-12,atol=0):raise ValueError('length_scale differs from immutable occupied-prism extent')
        report['provenance']['measured_original_length']=measured_length
        gate('A0','pass','external hashes, identity, complete TaskSpec, rigid frame, SI units and backend bound')
        stage='A1';occ,edit,protected=(d[k] for k in ['occupied','editable','protected'])
        for mask in [occ,edit,protected]:
            if mask.shape!=(n,)*3 or mask.dtype!=bool:raise ValueError('original occupancy/support must be binary boolean arrays')
        if not occ.any() or np.any(edit&~occ) or np.any(protected&~occ):raise ValueError('invalid original support')
        rho=g['density'];dens=d['density']
        if type(rho) not in (int,float) or not np.isfinite(rho) or rho<=0 or rho!=task.density:raise ValueError('positive scalar TaskSpec density required')
        if dens.shape!=occ.shape or not np.isfinite(dens).all() or np.any(dens[occ]!=rho) or np.any(dens[~occ]!=0):raise ValueError('original is not one homogeneous positive density on occupied cells')
        if type(s['density']) not in (int,float) or s['density']!=rho:raise ValueError('saved density increase/change forbidden')
        gate('A1','pass','every original occupied cell has unchanged rho0; outside density zero')
        stage='A2';_required(s,['role','encoding','runs','export_provenance'])
        if s['role']!='deleted' or s['encoding']!='canonical-integer-rle.v1':raise ValueError('saved result must explicitly encode deleted integer RLE')
        deleted=bitmap(s['runs'],n)
        if np.any(deleted&~edit):raise ValueError('deletion outside editable domain, including fixed shell/outside original')
        if np.any(deleted&protected):raise ValueError('protected removal forbidden')
        provenance=s['export_provenance'];_required(provenance,['source_role','operation'])
        if provenance['source_role']=='retained_editable':
            _required(provenance,['retained_file'])
            kept_file=_bound(sp.parent,provenance['retained_file']);kept_json=_json(kept_file)
            if kept_json.get('role')!='retained_editable':raise ValueError('retained source role mismatch')
            if provenance['operation']!='editable-minus-retained' or complement_retained(encode(edit),kept_json['runs'],n)!=s['runs']:raise ValueError('retained/deleted complement mismatch')
        elif provenance!= {'source_role':'deleted','operation':'direct-deletion'}:raise ValueError('unsupported export provenance')
        retained=occ&~deleted;fixed=occ&~edit
        if 'final_density' in s:
            fd=np.asarray(s['final_density'])
            if fd.shape!=occ.shape or not np.array_equal(fd,rho*retained):raise ValueError('fractional/nonbinary/added final material')
        gate('A2','pass','reopened integer canonical deletion; editable subset; zero fixed/protected/outside removal')
        stage='A3';_required(s,['removed_moments','retained_moments','fixed_moments'])
        original=np.asarray(d['original_moments'],np.float64);fixed_ref=np.asarray(d['fixed_moments'],np.float64)
        if original.shape!=(10,) or not np.isfinite(original).all() or original[0]<=0:raise ValueError('invalid immutable original moments')
        removed=prism_moments(s['runs'],o,h,rho);remaining=original-removed
        direct=cell_moments(retained,o,h,rho);fix=cell_moments(fixed,o,h,rho)
        scales=original[0]*g['length_scale']**np.array([0,1,1,1,2,2,2,2,2,2],float)
        if not np.isfinite(scales).all() or (scales<=0).any():raise ValueError('invalid moment normalization scales')
        residuals={}
        def cross(name,a,b):
            a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
            if a.shape!=(10,) or b.shape!=(10,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('ten finite moments required: '+name)
            residual=(a-b)/scales
            if not np.isfinite(residual).all():raise ValueError('nonfinite moment residual: '+name)
            residuals[name]=residual.tolist()
        cross('original_vs_cells',original,cell_moments(occ,o,h,rho));cross('removed_prisms_vs_cells',removed,cell_moments(deleted,o,h,rho))
        cross('retained_subtraction_vs_cells',remaining,direct);cross('fixed_reference_vs_cells',fixed_ref,fix)
        cross('fixed_once_plus_editable',remaining,fix+cell_moments(edit&~deleted,o,h,rho))
        cross('saved_removed',s['removed_moments'],removed);cross('saved_retained',s['retained_moments'],remaining);cross('saved_fixed',s['fixed_moments'],fix)
        report['moments']={'original':original.tolist(),'removed':removed.tolist(),'retained':remaining.tolist(),'fixed':fix.tolist(),'normalized_residuals':residuals,'tolerance':1e-10,'order':['mass','x','y','z','xx','yy','zz','yz','xz','xy']}
        if any(max(map(abs,r))>1e-10 for r in residuals.values()):raise ValueError('independent ten-moment residual exceeded')
        gate('A3','pass','float64 prism antiderivatives agree with independent cell integrals; immutable original minus removed; fixed shell exactly once')
        stage='A4'
        report['task']=mod.evaluate_moments(remaining,original,task,length_scale=g['length_scale'],connectivity=None)
        gate('A4','pass' if report['task']['moment_pass'] else 'fail','all ten frozen TaskSpec conditions and analytic feasibility evaluated without caller connectivity')
        stage='A5'
        if not g.get('outer_mesh'):raise Pending('missing original outer mesh: whole-solid topology unavailable')
        mesh_path=_bound(rp.parent,g['outer_mesh']);mesh=_arrays(mesh_path);_required(mesh,['vertices','faces'])
        whole=mesh_occupancy(mesh['vertices'],mesh['faces'],grid,deadline)
        if not np.array_equal(whole,occ):raise ValueError('independently reconstructed outer mesh differs from pinned original occupancy')
        before,_=components(whole,deadline);after,largest=components(whole&~deleted,deadline)
        report['topology']={'method':'independent mesh-centre parity rays + explicit six-face scipy label','original_components':before,'retained_components':after,'largest_retained_voxels':largest,'whole_original_voxels':int(whole.sum()),'retained_voxels':int((whole&~deleted).sum()),'outer_mesh_sha256':digest(mesh_path)}
        gate('A5','pass' if after==1 else 'fail','whole original including fixed shell independently reconstructed; retained components='+str(after))
        # Only the independently computed topology value may enter TaskSpec.
        report['task']=mod.evaluate_moments(remaining,original,task,length_scale=g['length_scale'],connectivity=(after==1))
    except Pending as e:gate(stage,'pending',str(e))
    except (ValueError,TypeError,KeyError,IndexError,OSError,OverflowError) as e:gate(stage,'fail',str(e))
    statuses=[report['gates'][f'A{i}']['status'] for i in range(6)]
    report['status']='fail' if 'fail' in statuses else 'pending' if 'pending' in statuses else 'pass'
    gate('A6','pass','claims restricted to represented domain; missing evidence never promoted')
    if report['status']=='pass':report['claim']=CLAIM
    report['elapsed_seconds']=monotonic()-start
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('registry',type=Path);p.add_argument('result',type=Path);p.add_argument('--registry-sha256',required=True);p.add_argument('--result-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=validate(a.registry,a.result,registry_sha256=a.registry_sha256,result_sha256=a.result_sha256)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(report['status']);sys.exit(0 if report['status']=='pass' else 2 if report['status']=='pending' else 1)
