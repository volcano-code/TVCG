"""Versioned Phase-1 mathematical contracts and analytic fixtures.

This module is intentionally independent of Torch, M80, and native teacher code.
It tests the geometry/material/moment/KKT conventions without claiming any
Cherchi/M69 supervision or physical-object admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

CONTRACT_VERSION = "tvcg.geometry-conditioned-quadric-pino.v1"
MOMENT_ORDER = ("mass", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
COEFFICIENT_ORDER = ("constant", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
EXPONENTS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
             (2, 0, 0), (0, 2, 0), (0, 0, 2), (0, 1, 1),
             (1, 0, 1), (1, 1, 0))


def _finite_array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a


def validate_target_frame(origin, axes, *, atol=1e-10):
    """Validate SpinTargetFrame: columns of axes are target axes in model space."""
    o = _finite_array(origin, (3,), "origin")
    R = _finite_array(axes, (3, 3), "axes")
    if not np.allclose(R.T @ R, np.eye(3), rtol=0.0, atol=atol):
        raise ValueError("target axes must be orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, rtol=0.0, atol=atol):
        raise ValueError("target axes must be right-handed")
    return o, R


def raw_monomials(points):
    """Return basis [1,x,y,z,x²,y²,z²,yz,xz,xy] for points [...,3]."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError("points must be finite with final dimension 3")
    px, py, pz = np.moveaxis(x, -1, 0)
    return np.stack((np.ones_like(px), px, py, pz, px*px, py*py,
                     pz*pz, py*pz, px*pz, px*py), axis=-1)


def moment_transform_matrix(linear, translation):
    """T such that m(x')=T m(x), x'=linear@x+translation."""
    A = _finite_array(linear, (3, 3), "linear")
    t = _finite_array(translation, (3,), "translation")
    if abs(np.linalg.det(A)) < 1e-14:
        raise ValueError("linear transform must be invertible")

    zero = (0, 0, 0)
    unit = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    coords = []
    for i in range(3):
        p = {zero: float(t[i])}
        for j in range(3):
            if A[i, j] != 0.0:
                p = p | {unit[j]: float(A[i, j])}
        coords.append(p)

    def multiply(p, q):
        out = {}
        for ep, cp in p.items():
            for eq, cq in q.items():
                e = tuple(ep[k] + eq[k] for k in range(3))
                if sum(e) <= 2:
                    out[e] = out.get(e, 0.0) + cp*cq
        return out

    polys = [{zero: 1.0}, coords[0], coords[1], coords[2],
             multiply(coords[0], coords[0]), multiply(coords[1], coords[1]),
             multiply(coords[2], coords[2]), multiply(coords[1], coords[2]),
             multiply(coords[0], coords[2]), multiply(coords[0], coords[1])]
    T = np.zeros((10, 10), dtype=np.float64)
    for i, poly in enumerate(polys):
        for j, exponent in enumerate(EXPONENTS):
            T[i, j] = poly.get(exponent, 0.0)
    if not np.isfinite(T).all() or abs(np.linalg.det(T)) < 1e-14:
        raise ValueError("moment transform is singular or nonfinite")
    return T


def target_frame_transform(origin, axes):
    """Return (R,t) for x_target=axes.T@(x_model-origin)."""
    o, axes = validate_target_frame(origin, axes)
    R = axes.T
    return R, -R @ o


def transform_raw_moments(moments, linear, translation):
    m = _finite_array(moments, (10,), "moments")
    return moment_transform_matrix(linear, translation) @ m


def transform_quadric_coefficients(coefficients, linear, translation):
    """Dual transform for beta'(x')=beta(linear^{-1}(x'-translation))."""
    c = _finite_array(coefficients, (10,), "coefficients")
    T = moment_transform_matrix(linear, translation)
    # c'^T m(x') = c^T m(x), hence c' = T^{-T} c.
    return np.linalg.solve(T.T, c)


def integrate_weighted_points(points, weights):
    """Numerically integrate ten moments from weighted point samples."""
    p = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or w.shape != (len(p),):
        raise ValueError("expected points[N,3] and weights[N]")
    if not np.isfinite(p).all() or not np.isfinite(w).all():
        raise ValueError("points and weights must be finite")
    return raw_monomials(p).T @ w


def add_disjoint_material_moments(fixed, protected, editable):
    """Add already-integrated moments for pairwise-disjoint masks exactly once."""
    f = _finite_array(fixed, (10,), "fixed moments")
    p = _finite_array(protected, (10,), "protected moments")
    e = _finite_array(editable, (10,), "editable moments")
    return f + p + e


def _power_integral(lo, hi, power):
    return (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)


@dataclass(frozen=True)
class PlaneCutBoxResult:
    moments: np.ndarray
    derivative_da: np.ndarray
    derivative_valid: bool


def plane_cut_box_moments(a, bounds, density=1.0):
    """Exact moments for a box clipped by x<=a; analytic dM/da on regular cuts.

    bounds=((xmin,xmax),(ymin,ymax),(zmin,zmax)).
    The derivative is explicitly invalid at either x-boundary event.
    """
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape != (3, 2) or not np.isfinite(b).all():
        raise ValueError("bounds must be finite [3,2]")
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("each box upper bound must exceed its lower bound")
    if not np.isfinite(a) or not np.isfinite(density) or density <= 0:
        raise ValueError("cut and density must be finite; density must be positive")
    x0, x1 = b[0]
    y0, y1 = b[1]
    z0, z1 = b[2]
    cut = float(np.clip(a, x0, x1))
    moments = np.zeros(10, dtype=np.float64)
    derivative = np.zeros(10, dtype=np.float64)
    for k, (i, j, kpow) in enumerate(EXPONENTS):
        ix = _power_integral(x0, cut, i) if cut > x0 else 0.0
        iy = _power_integral(y0, y1, j)
        iz = _power_integral(z0, z1, kpow)
        moments[k] = density * ix * iy * iz
        if x0 < a < x1:
            derivative[k] = density * (a ** i) * iy * iz
    return PlaneCutBoxResult(moments, derivative, bool(x0 < a < x1))


@dataclass(frozen=True)
class ActiveSetKKT:
    residual: np.ndarray
    stationarity: np.ndarray
    equality: np.ndarray
    active_inequality: np.ndarray
    inactive_primal_violation: float
    active_dual_violation: float


def active_set_kkt_residual(
    objective_gradient, equality_values, equality_jacobian,
    inequality_values, inequality_jacobian, active_mask,
    equality_multipliers, active_inequality_multipliers, *,
    beta=None,
):
    """Fixed-active-set KKT residual for min f, h=0, g<=0.

    L=f+nu*h+lambda*g, lambda>=0. The returned residual is
    [projected stationarity, h, g_active]. beta, if provided, is the first
    ten components of alpha and is projected onto its S^9 tangent space.
    """
    grad = np.asarray(objective_gradient, dtype=np.float64)
    if grad.ndim != 1 or not len(grad) or not np.isfinite(grad).all():
        raise ValueError("objective_gradient must be finite vector")
    h = np.asarray(equality_values, dtype=np.float64)
    Jh = np.asarray(equality_jacobian, dtype=np.float64)
    g = np.asarray(inequality_values, dtype=np.float64)
    Jg = np.asarray(inequality_jacobian, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    nu = np.asarray(equality_multipliers, dtype=np.float64)
    lam = np.asarray(active_inequality_multipliers, dtype=np.float64)
    n = len(grad)
    if h.ndim != 1 or Jh.shape != (len(h), n) or nu.shape != h.shape:
        raise ValueError("equality values/Jacobian/multipliers have inconsistent shapes")
    if g.ndim != 1 or Jg.shape != (len(g), n) or active.shape != g.shape:
        raise ValueError("inequality values/Jacobian/active mask have inconsistent shapes")
    if lam.shape != (int(active.sum()),):
        raise ValueError("active inequality multipliers have wrong shape")
    if not all(np.isfinite(x).all() for x in (h, Jh, g, Jg, nu, lam)):
        raise ValueError("KKT inputs must be finite")
    stationarity = grad + Jh.T @ nu
    if active.any():
        stationarity = stationarity + Jg[active].T @ lam
    if beta is not None:
        beta = _finite_array(beta, (10,), "beta")
        if n < 10 or not np.isclose(np.linalg.norm(beta), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("beta must be unit norm and alpha must start with ten beta coordinates")
        stationarity = stationarity.copy()
        stationarity[:10] -= beta * np.dot(beta, stationarity[:10])
    residual = np.concatenate((stationarity, h, g[active]))
    inactive_violation = float(max(0.0, np.max(g[~active]))) if (~active).any() else 0.0
    dual_violation = float(max(0.0, -np.min(lam))) if len(lam) else 0.0
    return ActiveSetKKT(residual, stationarity, h.copy(), g[active].copy(),
                        inactive_violation, dual_violation)


def validate_contract_record(record):
    """Small fail-closed validator for the versioned Phase-1 contract envelope."""
    if not isinstance(record, dict) or record.get("schema_version") != CONTRACT_VERSION:
        raise ValueError("unsupported or missing contract schema_version")
    required = ("geometry", "task", "material", "moments", "objective", "derivatives", "kkt")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"contract missing required sections: {missing}")
    material = record["material"]
    if material.get("single_material") is not True:
        raise ValueError("Phase-1 strict track requires single_material=true")
    density = material.get("density")
    if not isinstance(density, dict) or density.get("unit") not in ("kg/m^3", "registered_scalar", "dimensionless"):
        raise ValueError("density must declare an explicit supported unit")
    if not np.isfinite(density.get("value", np.nan)) or density["value"] <= 0:
        raise ValueError("density value must be finite and positive")
    if record["moments"].get("order") != list(MOMENT_ORDER):
        raise ValueError("moment order mismatch")
    if record["objective"].get("primary_profile_selected") is not False:
        raise ValueError("primary objective must remain unselected until decision gate")
    regions = material.get("regions", {})
    if set(regions) != {"fixed", "protected", "editable"}:
        raise ValueError("fixed/protected/editable region identities must all be explicit")
    if len(set(regions.values())) != 3:
        raise ValueError("fixed/protected/editable region identities must be disjoint")
    return True
