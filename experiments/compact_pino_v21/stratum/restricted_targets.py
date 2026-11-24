"""Assemble and independently re-read the strict-development research overlay.

Rows are indexed by parent and publisher query ID. Numerical targets are stored
only for observed rows. A raw 10×3 or 10×8 matrix is never an ambient Jacobian.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from experiments.compact_pino_v21.stratum.physical_replay.archive import check_row, historic
from experiments.compact_pino_v21.stratum.qualification import report as qualification
from experiments.compact_pino_v21.stratum.qualification.verify import match_center_qualification

HERE=Path(__file__).resolve().parent
METRICS = ('moment_valid','restricted_numerical_valid','both_tetra_checks','all_source_fd','all_material_fd',
           'all_retained_correspondences','all_sampled_checks','restricted_geometry_event_certified',
           'ambient_event_certified','native_output_triangulation_event_certified','qualified_material_j',
           'qualified_with_all_sampled_checks')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def summarize(records):
    return dict(rows=len(records), **{name:sum(r[name] for r in records) for name in METRICS},
                sampled_output_incidence_changed=sum(r['sampled_output_incidence_stable']==0 for r in records))


def masks(row, proof, expected_steps):
    mat=row['material']
    steps=row['steps']
    dim=row['dimension']
    fd_ok=[s['h'] for s in steps]==list(expected_steps) and all(len(s['columns'])==dim and {c['raw_col'] for c in s['columns']}==set(range(dim))
                                for s in steps)
    columns=[c for s in steps for c in s['columns']]
    def all_columns(field,flag):
        return bool(fd_ok and all(c.get(field,{}).get(flag,False) for c in columns))
    m=bool(mat.get('moment_valid'))
    j=bool(mat.get('restricted_valid'))
    tetra=bool(mat.get('moment_tetra',{}).get('passed') and mat.get('jacobian_tetra',{}).get('passed'))
    src=all_columns('source_velocity','passed')
    material_fd=all_columns('material_jacobian','passed')
    cert=proof['restricted_event_valid']==1
    def step_passed(step):
        cols=step['columns']
        return len(cols)==dim and all(
            c.get('source_combinatorics_stable') and c.get('source_velocity',{}).get('passed')
            and c.get('material_jacobian',{}).get('passed') and c.get('retained',{}).get('complete')
            and c.get('retained',{}).get('velocity',{}).get('passed') for c in cols)
    step_results=[bool(step_passed(step)) for step in steps]
    strict=bool(fd_ok and tetra and row.get('same_primal_motion_source') and all(step_results))
    if fd_ok and (step_results!=[s['all_columns_passed'] for s in steps] or strict!=row['sampled_row_passed']):
        raise ValueError('archived step or row acceptance disagrees with the physical columns')
    verified_j=bool(j and m and tetra and src and material_fd and cert and row.get('same_primal_motion_source'))
    sampled=[c.get('retained',{}).get('same_oriented_incidence') for c in columns]
    stencil=0 if False in sampled else (1 if fd_ok and all(s is True for s in sampled) else -1)
    return dict(moment_valid=m,restricted_numerical_valid=j,both_tetra_checks=tetra,all_source_fd=src,
                all_material_fd=material_fd,all_retained_correspondences=all_columns('retained','complete'),
                all_sampled_checks=strict,sampled_output_incidence_stable=stencil,restricted_geometry_event_certified=cert,
                ambient_event_certified=proof['ambient_event_valid']==1,
                native_output_triangulation_event_certified=proof['native_output_triangulation_event_valid']==1,
                qualified_material_j=verified_j,qualified_with_all_sampled_checks=bool(verified_j and strict))


def build(replay_file, certificate_file, inventory_file, admission_dir=None):
    with gzip.open(replay_file,'rt') as stream:
        replay=json.load(stream)
    roots,raw=historic()
    ledger=load(certificate_file)
    qualification.verify(ledger,admission_dir or qualification.PROOF_DIR,inventory_file)
    certs=ledger['rows']
    certificates={r['row_id']:r for r in certs}
    rows=replay['rows']
    inventory=load(inventory_file)
    parents={r['geometry']:r for r in inventory['splits']['development']['records']}
    hashes=load(HERE/'protocol.json')
    physical_protocol=load(HERE/'physical_replay/protocol.json')
    if replay['protocol']!=physical_protocol:
        raise ValueError('physical replay used different thresholds or cohort')
    expected_steps=physical_protocol['central_difference']['raw_steps']
    if (len(rows)!=len(certs) or set(certificates)!=set(roots) or
            len(rows)!=len(roots) or {r['row_id'] for r in rows}!=set(roots)
            or inventory['release_sha256']!=hashes['release_manifest_sha256']):
        raise ValueError('audit, certificate and historical rosters differ')
    groups={}
    receipt=[]
    keys=set()
    for dim in (3,8):
        selected=sorted((r for r in rows if r['dimension']==dim),key=lambda r:r['row_id'])
        if dict(Counter(r['geometry_id'] for r in selected))!=hashes['population'][f'R{dim}']:
            raise ValueError(f'R{dim} registered geometry counts differ')
        records=[]
        for row in selected:
            key,gid=row['row_id'],row['geometry_id']
            check_row(row,roots[key],raw[key],physical_protocol)
            meta,cert=parents[gid],certificates[key]
            source=row['source_record']
            qid=source['query_id']
            index=(meta['parent'],qid)
            if (index in keys or source['sha256']!=meta['labels']['sha256'] or
                    row['split']!='development' or row['full_j_valid']):
                raise ValueError(f'{key} identity or derivative scope mismatch')
            keys.add(index)
            proof=match_center_qualification(cert,row,proof_dir=admission_dir)
            record=dict(row_id=key,query_id=qid,parent=meta['parent'],geometry=gid,source_group=meta['source_group'],
                        dimension=dim,context=row['context'],operation=(row.get('primal') or {}).get('operation',''),
                        operand_sha256=row.get('operand_sha256') or '',
                        restricted_event_valid=proof['restricted_event_valid'],ambient_event_valid=proof['ambient_event_valid'],
                        native_output_triangulation_event_valid=proof['native_output_triangulation_event_valid'],
                        certified_radius=proof.get('certified_raw_linf_radius'),
                        certificate_domain=proof.get('proof_scope','unknown'),**masks(row,proof,expected_steps))
            records.append(record)
        receipt.extend(records)
        arrays={k:np.asarray([r[k] for r in records]) for k in
                ('row_id','query_id','parent','geometry','source_group','context','operation','operand_sha256',
                 'restricted_event_valid','ambient_event_valid','native_output_triangulation_event_valid',
                 'sampled_output_incidence_stable',*METRICS)}
        arrays['raw_query']=np.asarray([r['query'] for r in selected],dtype='<f8')
        arrays['raw_basis']=np.asarray([r['raw_basis'] for r in selected],dtype='<f8')
        cert_ids=np.flatnonzero(arrays['restricted_geometry_event_certified'])
        arrays['certificate_row_index']=cert_ids
        arrays['certificate_raw_linf_radius']=np.asarray([records[i]['certified_radius'] for i in cert_ids],dtype='<f8')
        arrays['certificate_domain']=np.asarray([records[i]['certificate_domain'] for i in cert_ids],dtype=str)
        for mask,field,name,tail in [('moment_valid','moments','moment',(10,)),
                                     ('restricted_numerical_valid','raw_restricted_J','restricted_j',(10,dim))]:
            ids=np.flatnonzero(arrays[mask])
            values=np.asarray([selected[i]['material'][field] for i in ids],dtype='<f8').reshape((len(ids),)+tail)
            if not np.isfinite(values).all():
                raise ValueError('nonfinite physical targets')
            arrays[name+'_row_index']=ids
            arrays[name]=values
        groups.update({f'R{dim}_'+k:v for k,v in arrays.items()})
    def by(field):
        return {str(key):summarize([r for r in receipt if r[field]==key]) for key in sorted({r[field] for r in receipt})}
    report=dict(schema='tvcg.compact-v21.restricted-targets.review.1',
                scope='89 registered development chart queries on four geometries in three sources; partial research overlay',
                released_for_training=False, train_rows_in_this_overlay=0, ambient_jacobian_rows=0,
                physical_replay_sha256=sha(replay_file),certificates_sha256=sha(certificate_file),
                published_inventory_sha256=sha(inventory_file),study_protocol_sha256=sha(HERE/'protocol.json'),
                release_manifest_sha256=inventory['release_sha256'],moment_order=load(HERE.parent/'protocol.json')['moment_order'],
                totals=summarize(receipt),by_dimension=by('dimension'),by_geometry=by('geometry'),
                by_source_group=by('source_group'),rows=receipt)
    return groups,report


def output(arrays,report,destination,verify=False):
    bank=destination/'restricted_targets.npz'
    receipt=destination/'restricted_targets.json'
    if verify:
        with np.load(bank,allow_pickle=False) as data:
            if set(data.files)!=set(arrays) or not all(np.array_equal(data[k],v) and data[k].dtype==v.dtype
                                                      for k,v in arrays.items()):
                raise ValueError('numerical vectors, ordered bases, identity or eligibility masks disagree')
        if load(receipt)!=dict(**report,npz_sha256=sha(bank)):
            raise ValueError('overlay receipt differs from source evidence')
        return
    destination.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=destination,delete=False) as stage:
        np.savez_compressed(stage,**arrays)
    Path(stage.name).replace(bank)
    with NamedTemporaryFile(mode='w',dir=destination,delete=False) as stage:
        json.dump(dict(**report,npz_sha256=sha(bank)),stage,indent=1,allow_nan=False)
        stage.write('\n')
    Path(stage.name).replace(receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,default=HERE/'physical_replay/evidence/registered_physical_replay.json.gz')
    parser.add_argument('--certificates',type=Path,default=HERE/'qualification/evidence/neighbourhoods.json')
    parser.add_argument('--inventory',type=Path,default=HERE/'qualification/evidence/published_inventory.json')
    parser.add_argument('--admissions',type=Path,help='optional directory of independently rebuilt R8 certificates')
    parser.add_argument('--output',type=Path,default=HERE/'evidence')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    bank,summary=build(args.replay,args.certificates,args.inventory,args.admissions)
    output(bank,summary,args.output,args.verify)
    print(json.dumps(summary['by_dimension'],indent=2))
