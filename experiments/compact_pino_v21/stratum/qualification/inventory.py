"""Re-read release 392 train and development only and match the frozen native replay by exact bits."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE.parent/'protocol.json').read_bytes())
DEFAULT_CORPUS = Path('/data/xvxinyao/TVCG/datasets/final_voxel_pino_corpus')
SPLITS = ('train','development')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(pin):
    path = Path(pin['path'])
    if sha(path) != pin['sha256']:
        raise ValueError(f'publisher file hash mismatch at {path}')
    return path


def fingerprint(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype='<f8').tobytes()).hexdigest()


def exact_determinant(q):
    a,b,c,d,e,f = map(Fraction,q[4:])
    return a*b*c+2*d*e*f-a*d*d-b*e*e-c*f*f


def _masked_scope(full, typed, dim):
    if full and typed:
        raise ValueError('overlapping full and restricted rows')
    if typed and dim not in (3,8):
        raise ValueError('unexpected declared chart dimension')
    return 'full' if full else f'R{int(dim)}' if typed else 'moment_only'


def _totals(records, field):
    result = Counter()
    for row in records:
        result.update(row[field])
    return dict(sorted(result.items()))


def audit(corpus):
    release = read(dict(path=corpus/'releases/four_split_delivery_392/manifest.json',
                        sha256=PROTOCOL['release_manifest_sha256']))
    views = json.loads(release.read_bytes())['views']
    geom_rows, keys, lookup, by_gid = {}, set(), {}, {}
    for split in SPLITS:
        manifest = read(views[split])
        records = json.loads(manifest.read_bytes())['records']
        geom_rows[split] = []
        for g in records:
            if g['split'] != split or g['geometry_id'] in by_gid:
                raise ValueError('unexpected split or repeated geometry')
            gid,parent = g['geometry_id'],g['parent_geometry_id']
            if parent in by_gid.values():
                raise ValueError('repeated parent geometry')
            by_gid[gid] = parent
            bank = read(dict(path=g['labels_file'],sha256=g['labels_sha256']))
            if g['labels'] != dict(path=str(bank),sha256=sha(bank)):
                raise ValueError('inconsistent publisher bank aliases')
            row = dict(geometry=gid,parent=parent,source_group=g['source_group_id'],raw_sha256=g['raw_sha256'],
                       labels=g['labels'],geometry_sha256=g['geometry']['sha256'],teacher_version=g['teacher_version'])
            published,source,events,replay = Counter(),Counter(),Counter(),Counter()
            idq = hashlib.sha256()
            determinants=[]
            requested_charts=[]
            with np.load(bank,allow_pickle=False) as z:
                n=len(z['query_id'])
                if n!=g['queries'] or z['queries'].shape!=(n,10) or not np.isfinite(z['queries']).all():
                    raise ValueError('invalid published query inventory')
                # Preload arrays once. NPZ field access inflates the entire member.
                names = ('query_id','queries','derivative_valid','restricted_valid','restricted_dimension',
                         'moment_valid','source_restricted_valid','source_restricted_dimension',
                         'ambient_event_valid','restricted_event_valid','native_output_triangulation_event_valid')
                c = {k:z[k] for k in names if k in z}
                for i,q in enumerate(c['queries']):
                    qid = str(c['query_id'][i])
                    key = (parent,qid)
                    if key in keys:
                        raise ValueError('repeated parent/query ID')
                    keys.add(key)
                    f = bool(c['derivative_valid'][i])
                    view_scope = _masked_scope(f,c['restricted_valid'][i],c['restricted_dimension'][i])
                    scope = _masked_scope(f,c.get('source_restricted_valid',c['restricted_valid'])[i],
                                          c.get('source_restricted_dimension',c['restricted_dimension'])[i])
                    published[view_scope]+=1
                    source[scope]+=1
                    if c['moment_valid'][i]:
                        replay['publisher_moments']+=1
                    qsha=fingerprint(q)
                    # Length framing avoids ambiguity in concatenated IDs.
                    encoded=qid.encode()
                    idq.update(len(encoded).to_bytes(4,'big')+encoded+bytes.fromhex(qsha))
                    lookup[(gid,qid,qsha)] = (split,scope)
                    for k in ('ambient_event_valid','restricted_event_valid','native_output_triangulation_event_valid'):
                        flag = int(c[k][i]) if k in c else -1
                        if flag not in (-1,0,1):
                            raise ValueError(f'invalid publisher mask in {k}')
                        events[f'{scope}/{k}/{flag}']+=1
                    if split=='development' and scope in ('R3','R8'):
                        requested_charts.append(dict(row_id=f'{gid}_{scope}_{i:03d}',row=i,scope=scope,query_id=qid,raw_query_sha256=qsha))
                    if split=='development' and scope=='R8':
                        determinants.append(abs(exact_determinant(q)))
            row.update(queries=n,ordered_query_id_raw_coefficient_sha256=idq.hexdigest(),
                       published_scope=dict(published),preserved_source_scope=dict(source),
                       historical_scoped_events=dict(sorted(events.items())),moment_counts=dict(replay))
            if requested_charts:
                chart_bank=read(g['original_labels'])
                with np.load(chart_bank,allow_pickle=False) as original:
                    ids={str(q):i for i,q in enumerate(original['query_id'])}
                    oq,ob,od,ov=(original[k] for k in ('queries','restricted_basis','restricted_dimension','restricted_valid'))
                    for chart in requested_charts:
                        oi=ids[chart['query_id']]
                        dim=int(chart['scope'][1:])
                        basis=ob[oi,:,:dim]
                        if (not ov[oi] or od[oi]!=dim or fingerprint(oq[oi])!=chart['raw_query_sha256']
                                or basis.shape!=(10,dim) or not np.isfinite(basis).all()):
                            raise ValueError('original typed bank does not match registered chart')
                        chart.update(original_row=oi,raw_basis_sha256=fingerprint(basis))
                        if dim==3:
                            if np.any(basis[4:]!=0.):
                                raise ValueError('published R3 tangent leaves exact plane subspace')
                            bf=[[Fraction(x) for x in c] for c in basis[:4].T]
                            gram=[[sum(a*b for a,b in zip(u,v)) for v in bf] for u in bf]
                            floor=min(gram[j][j]-sum(abs(x) for k,x in enumerate(gram[j]) if j!=k) for j in range(3))
                            if floor<=0:
                                raise ValueError('published R3 basis has no positive exact Gram lower bound')
                            chart.update(r3_curvatures_exactly_zero=True,
                                         r3_exact_gram_eigenvalue_lower_bound=float(np.nextafter(float(floor),-np.inf)))
                row.update(preserved_original_labels=g['original_labels'],requested_native_charts=requested_charts)
            if determinants:
                row['r8_exact_dyadic_determinants'] = dict(zero=sum(x==0 for x in determinants),
                       nonzero=sum(x!=0 for x in determinants),min_abs=float(min(determinants)),max_abs=float(max(determinants)))
            geom_rows[split].append(row)
    archive=HERE.parents[1]/'native/evidence'
    report_file=read(dict(path=archive/'native_replay.json',sha256=PROTOCOL['historical_native_replay_sha256']))
    report=json.loads(report_file.read_bytes())
    pin=report['numerical_archive']
    targets=read(dict(path=archive/pin['file'],sha256=pin['sha256']))
    published_geometries={r['geometry']:r for rows in geom_rows.values() for r in rows}
    matches={s:{} for s in SPLITS}
    with np.load(targets,allow_pickle=False) as z:
        qs=dict(zip(map(str,z['row_id']),z['raw_query'],strict=True))
        for r in report['rows']:
            gid=r['geometry_id']
            qid=r['source_record'].get('query_id')
            if gid not in by_gid or not qid or r['row_id'].endswith('_opposite'):
                continue
            key=(gid,qid,fingerprint(qs[r['row_id']]))
            if key not in lookup:
                continue
            s,scope=lookup[key]
            pub=published_geometries[gid]
            geometry=report['sources'][gid]
            if (geometry['split']!=s or pub['raw_sha256'] not in geometry['raw_file_hashes'].values()
                    or geometry['rle_sha256']!=pub['geometry_sha256']):
                raise ValueError('historical native replay differs in mesh, conditioning grid or split')
            if key in matches[s]:
                raise ValueError('multiple historical base operators on same published query')
            matches[s][key]=dict(row_id=r['row_id'],source_scope=scope,
                moment=bool(r['moment_valid']),full=bool(r['full_j_valid']),
                typed=bool(r['restricted_valid']),ambient_event=int(r['ambient_event_valid']),
                restricted_event=int(r['restricted_event_valid']))
    evidence_sources={
        'event_contract': corpus.parent/'phase2_12_dataset/event_label_contract.json',
        'vertex_margin_audit': corpus/'increments/development_vertex_margins_85/audit.py',
        'r8_whole_material_receipts': corpus/'increments/r8_event_trace_113/report.json',
        'r8_handoff': corpus/'increments/r8_event_handoff_86/CHAT_HANDOFF.md',
        'four_split_builder': corpus/'releases/four_split_delivery_392/build.py',
        'development_withholding_builder': Path(views['development']['path']).with_name('build.py'),
        'publisher_chat_review': corpus/'releases/chat_dataset_review_460/CHAT_HANDOFF.md'
    }
    source_hashes={k:dict(path=str(p),sha256=sha(p)) for k,p in evidence_sources.items()}
    result = dict(schema='tvcg.finite-stratum.publication-inventory.v1',evidence_sources=source_hashes,
        release_sha256=sha(release), historical_native_replay_sha256=sha(report_file), splits={})
    for s, rows in geom_rows.items():
        matched = list(matches[s].values())
        result['splits'][s] = dict(view=views[s],parents=len(rows),queries=sum(r['queries'] for r in rows),
            groups=len({r['source_group'] for r in rows}),published_scope=_totals(rows,'published_scope'),
            preserved_source_scope=_totals(rows,'preserved_source_scope'),
            publisher_moments=_totals(rows,'moment_counts')['publisher_moments'],
            historical_native=dict(attempted=len(matched), moments=sum(r['moment'] for r in matched),
                full_numerical=sum(r['full'] for r in matched), typed_numerical=sum(r['typed'] for r in matched),
                full_certified=sum(r['ambient_event']==1 for r in matched),
                restricted_certified=sum(r['restricted_event']==1 for r in matched),
                attempted_rows=matched), historical_scoped_events=_totals(rows,'historical_scoped_events'),records=rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=DEFAULT_CORPUS)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=audit(args.corpus)
    if args.output:
        args.output.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    for split,rows in result['splits'].items():
        print(split,rows['parents'],rows['queries'],rows['preserved_source_scope'],
              {k:v for k,v in rows['historical_native'].items() if k!='attempted_rows'})
