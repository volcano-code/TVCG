"""Replay pinned M90.2 fixtures through new guard, retaining all prior evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from guarded_validator import validate,legacy
from test_validator import fixture,save,sha

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    root=Path('/data/xvxinyao/Spin_test2')
    prior=root/'reports/m90_2_saved_result_validator_20261116/verified_controls/cases/summary.json'
    rows=[]
    for i,item in enumerate(json.loads(prior.read_text())['saved_audits']):
        case=root/item['directory']
        report=validate(case/'registry.json',case/'result.json',registry_sha256=item['registry_sha256'],result_sha256=item['result_sha256'])
        (out/f'prior_{i:03d}.json').write_text(json.dumps(report,indent=2)+'\n')
        rows.append({'control':item['control'],'prior_status':item['status'],'new_status':report['status'],'matches':report['status']==item['status']})
    boundary=out/'vertical_boundary';g,r=fixture(boundary)
    with np.load(boundary/'outer.npz') as a:v=a['vertices'];f=a['faces']
    v[:,0]=np.where(v[:,0]>0,2.5,-2.5);np.savez(boundary/'outer.npz',vertices=v,faces=f);g['outer_mesh']['sha256']=sha(boundary/'outer.npz')
    a,b=save(boundary,g,r);old=legacy.validate(boundary/'registry.json',boundary/'result.json',registry_sha256=a,result_sha256=b)
    new=validate(boundary/'registry.json',boundary/'result.json',registry_sha256=a,result_sha256=b)
    for label,report in [('legacy',old),('guarded',new)]:
        (boundary/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    summary={'schema':'m90.3.guarded-replay.v1','source_summary_sha256':sha(prior),'original_fixtures_replayed':len(rows),'all_prior_expected_statuses_preserved':all(r['matches'] for r in rows),'rows':rows,'vertical_boundary':{'registry_sha256':a,'result_sha256':b,'legacy_status':old['status'],'guarded_status':new['status'],'claim_cleared':new['claim'] is None,'task_complete_pass_cleared':not new['task']['complete_pass']}}
    summary['passed']=summary['all_prior_expected_statuses_preserved'] and old['status']=='pass' and new['status']=='pending' and new['claim'] is None and not new['task']['complete_pass']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'}));return 0 if summary['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.out))
