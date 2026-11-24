"""Pack verified row receipts, or read the portable evidence and check its scopes and identities."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from .admission import MODULE, PROOF_DIR, combine, verify_joined
    from .verify import match_center_qualification
except ImportError:
    from admission import MODULE, PROOF_DIR, combine, verify_joined
    from verify import match_center_qualification

HERE=Path(__file__).resolve().parent
EVIDENCE=HERE/'evidence'
PROTOCOL=json.loads((HERE.parent/'protocol.json').read_bytes())
CERT_FILE=EVIDENCE/'neighbourhoods.json'
INVENTORY_FILE=EVIDENCE/'published_inventory.json'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def demand(c,message):
    if not c:
        raise ValueError(message)


def counts(rows):
    def countgroup(group):
        return dict(registered=len(group),
                    restricted_event_positive=sum(x['restricted_event_valid']==1 for x in group),
                    restricted_event_unknown=sum(x['restricted_event_valid']==-1 for x in group),
                    accepted_source_envelope_incidence=sum(x.get('checked_source_envelope_incidence',False) for x in group),
                    exhaustive_native_value_admission=sum('uniform_native_admission' in x for x in group),
                    ambient_event_positive=sum(x['ambient_event_valid']==1 for x in group),
                    output_triangulation_positive=sum(x['native_output_triangulation_event_valid']==1 for x in group))
    return {scope:dict(**countgroup(pop),geometries={g:countgroup([r for r in pop if r['geometry_id']==g])
                 for g in sorted({r['geometry_id'] for r in pop})}) for scope in ('R3','R8')
                 if (pop:=[r for r in rows if r['scope']==scope])}


def verify(data, proof_dir=PROOF_DIR, inventory_file=INVENTORY_FILE):
    demand(data['schema']=='tvcg.finite-stratum.qualification-ledger.v1','wrong ledger schema')
    demand(data['release_manifest_sha256']==PROTOCOL['release_manifest_sha256'],'different release')
    demand(data['inventory_sha256']==sha(inventory_file),'different publication inventory')
    backend=data['native_backend_identity']
    deploy=read(HERE.parent/'native_backend'/'build_receipt.json')
    demand(backend['module_sha256']==MODULE and all(deploy[k]==v for k,v in backend.items()
               if k!='source_map_version') and backend['source_map_version']=='ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1',
           'different native build or finite source map')
    rows=data['rows']
    demand(len({r['row_id'] for r in rows})==len({(r['geometry_id'],r['query_id']) for r in rows})==len(rows),
           'duplicate registered rows')
    publisher=read(inventory_file)['splits']['development']['records']
    pubs={r['geometry']:r for r in publisher}
    registered={c['row_id']:c for p in publisher for c in p['requested_native_charts']}
    demand(set(registered)=={r['row_id'] for r in rows},'missing or additional published chart query')
    computed=counts(rows)
    for scope,table in PROTOCOL['population'].items():
        demand({g:r['registered'] for g,r in computed[scope]['geometries'].items()}==table,'incorrect registered population')
    demand(data['counts']==computed,'incorrect qualification counts')
    for r in rows:
        demand(r['parent_bank_sha256']==pubs[r['geometry_id']]['labels']['sha256'],'unregistered published bank')
        demand(all(r[k]==registered[r['row_id']][k] for k in ('scope','query_id','raw_query_sha256','raw_basis_sha256')),
               'published raw coefficient or chart basis fingerprint mismatch')
        demand(r['native_module_sha256']==data['native_backend_identity']['module_sha256'],'native module mismatch')
        demand(r['ambient_event_valid']==r['native_output_triangulation_event_valid']==-1,
               'a typed material certificate was promoted to a different scope')
        flag=r['restricted_event_valid']
        demand(flag in (1,-1),'invalid proof disposition')
        for trial in r['trials']:
            if trial.get('mixed_incidence',{}).get('certified') is not True:
                continue
            closure=trial['closure']
            margin=closure['contact_clearance_threshold']
            growth=closure['growth']
            demand(len(growth)==closure['cap_doublings']+1 and
                   all(s['minimum_far_gap_upper']<margin for s in growth[:-1]) and
                   growth[-1]['minimum_far_gap_lower']>margin,'unproved cap-growth selection')
            demand(min(closure[k] for k in ('projected_rim_clearance_lower','lateral_cap_clearance_lower'))>margin and
                   closure['slope_lower']>1.e-12 and closure['front_winding_abs_lower']>0 and
                   0<=trial['embedding']['embedding_contraction_upper']<1,'unproved material/graph guard')
            demand(trial['mixed_incidence']['fixed_faces']>0 and trial['mixed_incidence']['moving_faces']>0 and
                   0<=trial['mixed_incidence']['narrow_phase_face_pairs']<=trial['mixed_incidence']['aabb_candidates'],
                   'invalid mixed feature counts')
        if flag==1 or r.get('checked_source_envelope_incidence'):
            t=r['trials'][-1]
            demand(t.get('mixed_incidence',{}).get('certified') is True,'missing mixed-incidence result')
            radius=t['raw_box_linf_radius']
            geometry_radius=r.get('checked_source_raw_box_linf_radius',r.get('certified_raw_linf_radius'))
            demand(0<radius==geometry_radius,'different coefficient enclosure')
        if flag==1 and r['scope']=='R8':
            verify_joined(r, proof_dir)
        elif flag==-1 and r.get('checked_source_envelope_incidence'):
            demand(r['scope']=='R8' and 'unproved' in r,'missing conditional R8 scope')
    demand(data['row_lists']==row_lists(rows),'incorrect explicit row lists')
    return computed


def row_lists(rows):
    return dict(restricted_positive=[r['row_id'] for r in rows if r['restricted_event_valid']==1],
                restricted_unknown=[r['row_id'] for r in rows if r['restricted_event_valid']==-1],
                exhaustive_native_admission=[r['row_id'] for r in rows if 'uniform_native_admission' in r],
                accepted_source_envelope_incidence=[r['row_id'] for r in rows if r.get('checked_source_envelope_incidence')])


def pack(folder, proof_dir=PROOF_DIR):
    rows=[read(p) for p in sorted(folder.glob('*.json'))]
    demand(len(rows)>0,'missing raw row receipts')
    backend=rows[0]['native_backend_identity']
    for r in rows:
        supplement=proof_dir/(r['row_id']+'.json')
        if supplement.is_file():
            link=combine(r.copy(), supplement)
            r['uniform_native_admission']=link
            r['certified_raw_linf_radius']=link['certified_raw_linf_radius']
            r['restricted_event_valid']=1
            r['proof_scope']='every represented raw input in the stated box: uniform native value admission and whole-material finite-operand geometric incidence, including the centered R8 chart in that box'
            r.pop('unproved')
        demand(r.pop('native_backend_identity')==backend,'different native builds in row receipts')
        r['native_module_sha256']=backend['module_sha256']
    data=dict(schema='tvcg.finite-stratum.qualification-ledger.v1',release_manifest_sha256=PROTOCOL['release_manifest_sha256'],
              inventory_sha256=sha(INVENTORY_FILE),native_backend_identity=backend,
              scope='Local geometric incidence of the whole physical design and finite operand including caps. Ambient status and final-output triangulation have independent masks.',
              coefficient_metric='Raw coefficient infinity-norm enclosure intersected with the declared chart. R3 permits all four linear coefficients to vary, including positive gauge.',
              counts=counts(rows),rows=rows,row_lists=row_lists(rows))
    verify(data,proof_dir)
    return data


def join(data,folder,proof_dir=PROOF_DIR):
    out=Counter()
    for r in data['rows']:
        path=folder/(r['row_id']+'.json')
        demand(path.is_file(),f'missing physical center {r["row_id"]}')
        c=read(path)
        event=match_center_qualification(r,c,proof_dir=proof_dir)
        m=c['material']
        key=(r['scope'],bool(m['moment_valid']),bool(m.get('restricted_valid')),event['restricted_event_valid']==1)
        out[key]+=1
    return [dict(scope=s,moment=m,chart_j=d,geometric_event=e,count=v) for (s,m,d,e),v in sorted(out.items())]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,help='assemble output from the 89 freshly recomputed individual records')
    p.add_argument('--output',type=Path,default=CERT_FILE)
    p.add_argument('--admissions',type=Path,default=PROOF_DIR,help='archived exhaustive native admission receipts')
    p.add_argument('--centers',type=Path,help='verify identities against physical experiment center rows')
    args=p.parse_args()
    if args.rows:
        data=pack(args.rows,args.admissions)
        args.output.write_text(json.dumps(data,indent=1,allow_nan=False)+'\n')
    else:
        data=read(args.output)
    print(json.dumps(verify(data,args.admissions),indent=1))
    if args.centers:
        print(json.dumps(join(data,args.centers,args.admissions),indent=1))
