from fractions import Fraction

import numpy as np
import pytest

from experiments.compact_pino_v21.stratum.primal import load_transform


def test_affine_rational_rows_match_fraction_and_round_to_even():
    n = load_transform()
    vertices = [['0/1','0/1','0/1'],['0b1/0b11','0/1','0/1'],['0/1','1/1','0/1'],['0/1','0/1','1/1']]
    points = np.array([[0.,0.,0.],[1/3,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces = np.array([[0,2,1],[0,1,3],[1,2,3],[0,3,2]],np.int64)
    axes = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    length, origin, aim = 1.9, [.16,-.21,.1], [.01,.25,-.5]
    display,tokens = n.transform(vertices,points,faces,origin,length,aim,axes)
    for i, xyz in enumerate(points):
        exact = [Fraction(1,3) if i==1 and j==0 else Fraction(float(v)) for j,v in enumerate(xyz)]
        local = [(v-Fraction(o))/Fraction(length)-Fraction(a) for v,o,a in zip(exact,origin,aim)]
        result = [sum(a*Fraction(x) for a,x in zip(local,column)) for column in axes.T]
        assert [Fraction(t) for t in tokens[i]] == result
        np.testing.assert_array_equal(display[i],list(map(float,result)))
    # Adjacent shells may need opposite winding. Component orientations stay separate.
    reverse = faces[:,[0,2,1]].copy()
    assert n.transform(vertices,points,reverse,origin,length,aim,axes)[1] == tokens
    wrong = faces.copy()
    wrong[0] = wrong[0,::-1]
    with pytest.raises(n.ReferenceOrientation):
        n.transform(vertices,points,wrong,origin,length,aim,axes)
    points[1,0] = 1.
    with pytest.raises(ValueError,match='display'):
        n.transform(vertices,points,faces,origin,length,aim,axes)
