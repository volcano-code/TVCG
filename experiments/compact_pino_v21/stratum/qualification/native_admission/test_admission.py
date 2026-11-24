"""The source cover rejects wrong rank, missing transverse support, and vacuous charts."""
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

import certify
from qr_trace import Interval, Unresolved, trace

ROOT=Path(os.environ.get('TVCG_ADMISSION_PROBE','/tmp/tvcg-stratum-build/admission'))
sys.path.insert(0,str(ROOT))
import _rank_admission_probe as probe


def sample():
    q=np.array([.1,.2,.3,.4,.7,.5,0.,0.,0.,0.]);q=probe.normalize(q)
    axes=[[[float(x)] for x in q[4:]]]
    b=np.column_stack([q[1:4],q[1:4]])
    return q,axes,b


def test_known_failures():
    q,p,b=sample()
    assert probe.cover(p,q,b,0)['distinct_matrices']==1
    with pytest.raises(RuntimeError,match='transverse linear'):
        probe.cover(p,q,np.zeros((3,2)),0)
    with pytest.raises(RuntimeError,match='rank leaf'):
        near=[[list(a) for a in p[0]]]; near[0][2]=[1.e-13]
        probe.cover(near,q,b,0)
    with pytest.raises(RuntimeError,match='rejected|spectral'):
        near=[[list(a) for a in p[0]]]; near[0][2]=[1.e-8]
        probe.cover(near,q,b,0)
    with pytest.raises(RuntimeError,match='spectral'):
        probe.cover(p,q,b,2)
    # Unguarded overlap counting would duplicate every matrix.
    n=probe.cover(p+p,q,b,0)
    assert (n['pair_count'],n['distinct_matrices'],n['overlap_duplicates'])==(2,1,1)


def test_native_normalization_encloses_entire_small_lattice():
    q,p,b=sample()
    lo=q.copy(); hi=q.copy()
    for i in range(3):
        lo[i]=math.nextafter(q[i],-math.inf);hi[i]=math.nextafter(q[i],math.inf)
    ns,_=certify.possible_norms(lo,hi,probe)
    for record in itertools.product(*(certify.floats(l,h) for l,h in zip(lo,hi))):
        unit=probe.normalize(np.asarray(record))
        assert any(np.array_equal(unit,np.asarray(record)/n) for n in ns)


def test_chart_rank_and_outward_endpoints():
    lo=certify.ceil_machine(certify.F(1)+certify.F(1,2**54))
    hi=certify.floor_machine(certify.F(1)+certify.F(1,2**54))
    assert hi==1. and lo==math.nextafter(1.,math.inf)
    assert certify.full_rank_exact([[1,0],[0,1]])==2
    assert certify.full_rank_exact([[1,0],[2,0]])==1
    frame=np.array([[-.03575127960434256,-.20857653852241154,.025316891999215274,.19569599602437499,
                      .21398122132397487,.8195894973481445,.3239237454458185,.17002609021296713,
                      .25346730092758674,.026148835638205588]])[0,4:]
    with pytest.raises(Unresolved,match='QR deflation'):
        trace([Interval(x-2.**-50,x+2.**-50) for x in frame])
