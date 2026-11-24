import numpy as np
from experiments.compact_pino_v21.analytic_control import point_response


def test_plane_and_sphere_same_euclidean_band_and_shape_motion():
    # f > 0 outside the unit sphere, matching a signed Euclidean distance of |x|-1.
    q = np.array([-1.,0,0,0,1,1,1,0,0,0])
    points = np.array([[.98,0.,0.], [1.03,0.,0.], [.01,.01,.01]])
    p, j = point_response(points,q,.05)
    s = np.array([-.4,.6,-1.])
    t = (s+1)/2
    np.testing.assert_allclose(p, t*t*(3-2*t))
    eps = 1e-6
    for i in (0,1,4,7):
        v = np.eye(10)[i]
        d = (point_response(points, q+eps*v,.05)[0]-point_response(points, q-eps*v,.05)[0])/(2*eps)
        np.testing.assert_allclose(d, j[:,i], atol=3e-8)
    plane = np.array([0., 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p,j = point_response(np.array([[.01, 2., 4.]]),plane,.05)
    assert j[0,7] == np.float64(16 * .75*(1-.2**2)/.05)
