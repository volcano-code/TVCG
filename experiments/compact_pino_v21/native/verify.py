"""Portable numerical checks on the small exported archive. Only NumPy is required."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def check(condition, label):
    if not condition:
        raise ValueError(label)


def run(root):
    report=json.loads((root/'native_replay.json').read_text())
    source=root/report['numerical_archive']['file']
    check(hashlib.sha256(source.read_bytes()).hexdigest()==report['numerical_archive']['sha256'],'numeric archive digest')
    with np.load(source,allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    rows={r['row_id']:r for r in report['rows']}
    ids=list(arrays['row_id'])
    moments=list(arrays['moment_row_id'])
    derivatives=list(arrays['full_j_row_id'])
    check(len(rows)==len(ids)==report['counts']['rows'] and set(ids)==set(rows),'row inventory')
    check(set(moments)=={k for k,r in rows.items() if r['moment_valid']},'Moment availability')
    check(set(derivatives)=={k for k,r in rows.items() if r['full_j_valid']},'ambient J availability')
    data={k:i for i,k in enumerate(moments)}
    jac={k:i for i,k in enumerate(derivatives)}
    check(arrays['M'].shape==(len(moments),10) and arrays['J_raw'].shape==(len(derivatives),10,10),'physical target shapes')
    check(all(np.isfinite(arrays[k]).all() for k in ['raw_query','M','M_editable','M_independent_tetra','J_raw','J_independent_tetra']),
          'nonfinite numerical targets')
    for key in moments:
        idx=data[key]
        fix=np.asarray(report['sources'][rows[key]['geometry_id']]['fixed_moments'])
        np.testing.assert_array_equal(arrays['M'][idx], arrays['M_editable'][idx]+fix)
    np.testing.assert_allclose(arrays['M'],arrays['M_independent_tetra'],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(arrays['J_raw'],arrays['J_independent_tetra'],rtol=1e-9,atol=1e-11)
    for prefix,control in report['controls'].items():
        pos,neg,other,twice=[prefix+s for s in ['_plus','_minus','_plus_opposite','_scale2']]
        full=np.asarray(report['sources'][rows[pos]['geometry_id']]['mesh_full_editable'])
        np.testing.assert_allclose(arrays['M_editable'][data[pos]]+arrays['M_editable'][data[other]],
                                   full,rtol=1e-9,atol=1e-11)
        np.testing.assert_array_equal(arrays['M_editable'][data[pos]]+arrays['M_editable'][data[neg]]-full,
                                      control['standalone_minus_plus_minus_full'])
        np.testing.assert_array_equal(arrays['M'][data[pos]],arrays['M'][data[twice]])
        np.testing.assert_allclose(arrays['J_raw'][jac[pos]],2*arrays['J_raw'][jac[twice]],rtol=1e-9,atol=1e-11)
        check(rows[pos]['preboolean_physical_boundary_digest']==rows[other]['preboolean_physical_boundary_digest'],
              'fixed operand boundary digest')
        check(rows[pos]['operation']!=rows[other]['operation'],'fixed operand opposite side')
    for d in (3,8):
        typed={k for k,r in rows.items() if r['restricted_valid'] and r['requested_chart_dimension']==d}
        keys=f'R{d}_row_id'
        check(typed==set(arrays.get(keys,[])),'typed numerical inventory')
        if typed:
            shape=(len(typed),10,d)
            check(arrays[f'R{d}_J'].shape==arrays[f'R{d}_basis_raw'].shape==shape,'raw chart dimensions')
            np.testing.assert_allclose(arrays[f'R{d}_J'],arrays[f'R{d}_J_independent_tetra'],rtol=1e-9,atol=1e-11)
        table=report['strict_development_R'+str(d)]
        scoped=[r for r in rows.values() if r.get('requested_chart_dimension')==d and
                    r['geometry_id'] in table['counts_by_geometry']]
        check(len(scoped)==table['queried'] and Counter(r['geometry_id'] for r in scoped)==table['counts_by_geometry'],
              'registered chart inventory')
        check(sum(r['restricted_valid'] for r in scoped)==table['restricted_defined'],'registered chart numerical coverage')
    trajectory=[r for r in rows.values() if r['geometry_id']=='noodles']
    check(len(trajectory)==25,'frozen trajectory query count')
    check(all(r['frozen_replay']['actual_defined']==r['frozen_replay']['expected_defined'] for r in trajectory),'frozen eligibility')
    for d in report['full_motion_fd']['steps']:
        check(d['raw_J_relative_fro']<1.e-5 and all(d['source_faces_equal']),'raw native motion FD')
    for row in report['alternative_chart_diagnostic']['rows']:
        check(not row['deployment_admissible'],'alternate profile promotion')
        check(row['deployed_same_query_status']['moment_valid'] is False,'alternate chart claimed from deployment')
    print(json.dumps(dict(native_counts=report['counts'],trajectory=report['trajectory'],
                          strict_R3=report['strict_development_R3'],strict_R8=report['strict_development_R8']),
                     ensure_ascii=False,indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence',type=Path,nargs='?',default=Path(__file__).resolve().parent/'evidence')
    run(p.parse_args().evidence)
