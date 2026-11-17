"""Read-only actual M83 training record preflight; never builds native topology."""
import argparse,ast,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from validator import digest,task_module
from geometry import prism_moments,Pending,check_budget

ROOT=Path('/data/xvxinyao/Spin_M68_Trellis_Data_Kit')
MANIFEST=ROOT/'runs/m83/m83-dataset-v2-full-20261113T124505Z/operator-manifest.json'
MANIFEST_SHA='6effecd78cf03cf4ffed0b6d65f4a36f66d1a900fec4b6cbd4c171e99e0da2c4'

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    if digest(MANIFEST)!=MANIFEST_SHA:raise ValueError('M83 manifest pin changed')
    data=json.loads(MANIFEST.read_text());rec=next(r for r in data['records'] if r.get('split')=='train' and r.get('geometry_file'))
    files={}
    for role in ['case','geometry']:
        p=MANIFEST.parent/rec[role+'_file']
        if digest(p)!=rec[role+'_sha256']:raise ValueError(role+' file hash changed')
        files[role]=p
    # Load geometry + outer mesh/reference only; labels and held-out remain unread.
    with np.load(files['geometry'],allow_pickle=False) as a:g={k:a[k] for k in a.files}
    with np.load(files['case'],allow_pickle=False) as a:c={k:a[k] for k in ['bounds','original_ten','ov','of'] if k in a.files}
    mod=task_module();task=mod.TaskSpec.from_dict(rec['task'])
    if task.task_hash!=rec['task_sha256']:raise ValueError('actual TaskSpec hash changed')
    bounds=c['bounds'];L=float(np.ptp(bounds,axis=0).max());h=1.02*L/task.native_resolution;origin=bounds.mean(axis=0)-.51*L
    if not np.allclose(g['origin'],origin,rtol=0,atol=1e-12*L) or not np.isclose(g['h'],h,rtol=1e-12,atol=0):raise ValueError('M69 padded grid binding failed')
    original=c['original_ten']*task.density
    scales=original[0]*L**np.array([0,1,1,1,2,2,2,2,2,2])
    residual=(g['original_moments']-original)/scales
    if np.max(abs(residual))>1e-10:raise ValueError('case/geometry immutable reference mismatch')
    # Synthetic full-retention diagnostic only. Reopen empty deletion bytes.
    saved=out/'full_retention_deleted.rle'
    with saved.open('xb') as f:f.write(b'')
    payload=saved.read_bytes();runs=np.frombuffer(payload,dtype='<u4').reshape(-1,4)
    # Run just the inspected frozen pure support_check function, no torch/solver imports.
    source=ROOT/'baseline/src/eval65.py';tree=ast.parse(source.read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='support_check')
    namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
    geom=SimpleNamespace(side=int(g['side']),columns=g['columns'])
    count=namespace['support_check'](geom,saved)
    removed=prism_moments(runs.tolist(),g['origin'],float(g['h']),task.density)
    task_report=mod.evaluate_moments(original-removed,original,task,length_scale=L,connectivity=None)
    try:check_budget(int(g['side']));budget_reason=None
    except Pending as e:budget_reason=str(e)
    result={'schema':'m90.2.m83-readonly-preflight.v1','status':'pending','record_id':rec['id'],'split':rec['split'],'manifest_sha256':MANIFEST_SHA,'internal_manifest_fingerprint':data['fingerprint'],'case_sha256':digest(files['case']),'geometry_sha256':digest(files['geometry']),'task_hash':task.task_hash,'task':rec['task'],'source_task_hash':rec['source_task_sha256'],'backend':rec['metadata']['oracle_backend'],'frame_provenance':rec['mesh_import_evidence']['frame'],'density_scalar':float(g['density']),'native_resolution':int(g['side']),'outer_mesh_present':all(k in c for k in ['ov','of']),'grid_binding_pass':True,'original_reference_normalized_residuals':residual.tolist(),'saved_control':{'kind':'new empty-deletion full-retention diagnostic, not an optimized design','role':'deleted','path':str(saved),'sha256':digest(saved),'removed_voxels':count,'removed_moments':removed.tolist(),'task_evaluation':task_report},'source_support_check_sha256':digest(source),'source_support_check_execution':'AST-isolated exact function body; no eval65 module/model imports','topology_started':False,'topology_compilation_started':False,'historical_grid_workspace_estimate_bytes':rec['mesh_import_evidence']['estimated_grid_workspace_bytes'],'budget_reason':budget_reason,'pending_reasons':['native whole-original occupancy/density field not provided in saved geometry API; only scalar density declaration','explicit SI-unit registration not present in selected record','mixed original-mesh-minus-deleted-voxel moments need preregistered discretization budget and independent retained-solid reference','full512 independent whole-solid reconstruction exceeds bounded adapter preflight'],'gates':{'A0':'pending units and representation registration','A1':'pending original occupied-cell density evidence','A2':'empty deletion support pass only','A3':'pending independent whole-retained integration; immutable reference and zero removal checked','A4':'fail full-retention diagnostic using all actual TaskSpec conditions','A5':'pending budget; outer mesh exists but not voxelized','A6':'no acceptance claim'},'gpu_used':False,'optimizer_steps':0,'labels_loaded':False,'held_out_evaluated':False}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'record':rec['id'],'all_task_checks':len(task_report['checks']),'full_retention_failed_checks':[k for k,v in task_report['checks'].items() if not v],'native_topology':result['budget_reason']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
