"""Safe M90.2 entry point with conservative M90.3 mesh boundary guard.

No original files are modified. This adds a required guard to any would-be pass;
it does not promote legacy pending inputs or admit native512/mixed references.
"""
from pathlib import Path
import sys
import numpy as np
from sparse import mesh_runs,Pending
M90=Path(__file__).resolve().parents[1]/'m90_2_saved_result_validator'
sys.path.insert(0,str(M90))
import validator as legacy
from geometry import encode

def validate(registry_path,result_path,*,registry_sha256=None,result_sha256=None):
    report=legacy.validate(registry_path,result_path,registry_sha256=registry_sha256,result_sha256=result_sha256)
    report['schema']='m90.3.guarded-saved-result-audit.v1'
    report['boundary_guard']={'status':'not_reached','legacy_final_status':report['status']}
    if report['status']!='pass':return report
    try:
        p=Path(registry_path);g=legacy._json(p)
        # Recheck pins before the extra read, never trust changed registration.
        if legacy.digest(p)!=registry_sha256 or legacy.digest(result_path)!=result_sha256:raise ValueError('files changed between legacy and boundary checks')
        mesh=legacy._arrays(legacy._bound(p.parent,g['outer_mesh']))
        grid=g['grid'];rows,meta=mesh_runs(mesh['vertices'],mesh['faces'],grid['origin'],grid['h'],grid['side'])
        original=legacy._arrays(legacy._bound(p.parent,g['original']))
        expected=np.asarray(encode(original['occupied']),np.int64).reshape(-1,4)
        if not np.array_equal(rows,expected):raise ValueError('sparse boundary guard original occupancy mismatch')
        report['boundary_guard']={'status':'pass','legacy_final_status':'pass','mesh_reconstruction':meta}
    except (Pending,ValueError,KeyError,OSError,TypeError) as e:
        state='pending' if isinstance(e,Pending) else 'fail'
        report['boundary_guard']={'status':state,'legacy_final_status':'pass','reason':str(e)}
        report['gates']['A5']={'status':state,'reason':'M90.3 boundary guard: '+str(e)}
        report['status']=state;report['claim']=None
        # Preserve historical physical metrics; do not leave a stale pass certificate.
        if 'task' in report:
            report['task']['complete_pass']=False;report['task']['passes']=False
            report['task']['connectivity_checked']=False;report['task']['connectivity_pass']=None
            report['task']['status']='material_audit_pending' if state=='pending' else 'material_audit_failed'
    return report

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('registry',type=Path);p.add_argument('result',type=Path);p.add_argument('--registry-sha256',required=True);p.add_argument('--result-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=validate(a.registry,a.result,registry_sha256=a.registry_sha256,result_sha256=a.result_sha256)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(report['status']);raise SystemExit(0 if report['status']=='pass' else 2 if report['status']=='pending' else 1)
