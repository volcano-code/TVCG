"""Record why a direct uncorrelated enclosure of the pinned QR code was insufficient."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import numpy as np
from qr_trace import Interval, Unresolved, trace

def run(directory):
    rows=[]
    for f in sorted(directory.glob('*R8*.json')):
        e=json.loads(f.read_text());w=e['native_value']['construction_witness']
        u=np.asarray(w['unit_coefficients'])*w['coefficient_sign']
        center=trace(u[4:]);lam=[x.lo for x in center['values']]
        cols=[None]*3;k=w['least_eigenvalue_index'];cols[k]=w['axis']
        others=[j for j in range(3) if j!=k];cols[others[0]]=w['first'];cols[others[1]]=w['second']
        got=np.array([[x.lo for x in row] for row in center['vectors']])
        if not np.array_equal(lam,w['canonical_quadratic_eigenvalues']) or not np.array_equal(
                np.abs(got),np.abs(np.asarray(cols).T)):
            raise ValueError('point trace disagrees with native eigenpairs: '+f.stem)
        try:
            check=trace([Interval(math.nextafter(x-2.**-50,-math.inf),math.nextafter(x+2.**-50,math.inf))
                         for x in u[4:]])
        except Unresolved as exc: blocker=exc.record
        else: blocker=None
        rows.append(dict(row_id=f.stem,center_qr_steps=len(center['steps']),center_spectrum_and_unsigned_axes_bitwise=True,
                         normalized_component_radius=2.**-50,first_unresolved_branch=blocker))
    return dict(schema='tvcg.rank-chart.interval-qr.diagnostic.1',rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('centers',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(run(args.centers),indent=2)+'\n')
