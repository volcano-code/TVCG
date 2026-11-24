import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from margins import kkt_score, residual_interval
from perturb_trials import smooth_field


def test_static_moment_box_extrema_are_attained():
    q = np.arange(1.0, 11)
    q /= np.linalg.norm(q)
    m = np.zeros(10)
    m[1:3] = [0.005, -0.03]
    b = 0.01
    score, mu, s2 = kkt_score(q, m)
    lo, hi = residual_interval(m, s2, b)
    extremal = m.copy()
    extremal[1:3] = [0.0, -0.02]
    assert np.isclose(lo, kkt_score(q, extremal)[0])
    extremal[1:3] = [0.015, -0.04]
    assert np.isclose(hi, kkt_score(q, extremal)[0])
    assert lo < score < hi and mu > 0


def test_bounded_raw_tangent_field_matches_central_difference():
    q0 = np.arange(1.0, 11)
    q0 /= np.linalg.norm(q0)
    u = np.eye(10)[3] - q0[3] * q0
    u /= np.linalg.norm(u)
    q = q0 + 0.2 * np.eye(10)[2]
    _, J = smooth_field(q, q0, u, [1, -1], 1.5, 0.01, 0.05)
    v = np.linspace(0, 1, 10)
    v /= np.linalg.norm(v)
    h = 1e-6
    fd = (
        smooth_field(q + h * v, q0, u, [1, -1], 1.5, 0.01, 0.05)[0]
        - smooth_field(q - h * v, q0, u, [1, -1], 1.5, 0.01, 0.05)[0]
    ) / (2 * h)
    np.testing.assert_allclose(fd, J @ v, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(J @ q, 0.0, atol=1e-14)
    assert np.max(np.linalg.norm(J, axis=1)) * np.linalg.norm(q) / 1.5 <= 0.05
