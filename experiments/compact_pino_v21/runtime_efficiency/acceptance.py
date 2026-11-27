"""Cached form of Work's deployed, independent spinning-top acceptance.

The formula is the one in Work ``solve.py`` (SHA256 below), not the separate
``p0_diagnostics.native_solver_adapter`` profile. Only its SymPy construction
is cached. A production ``verify`` always evaluates the native original-mesh
backend at the supplied q; no moments or acceptance decisions are cached.

``evaluate_moments`` is used only when a backend advertises the actual native
capability. Older evaluators take the explicit joint fallback, which still
pays for their cut Jacobian. Neither route reads that Jacobian for acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from time import perf_counter

import numpy as np


WORK_VERIFY_SHA256 = "ad5093d39d4487daf0c805b79b8747b811595c8f2f773a9341ef06a0d1a2e37f"
ACCEPTANCE_PROFILE = "WORK_SPINNING_TOP_DUAL_MONOMIAL_RAW_V1"
INTEGRAL_PROFILE = "CERTIFIED_MPFR256_GMP_FALLBACK_V1"
MOMENT_ORDER = ("m", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
EQUATION_TOLERANCE = 1e-12
PHYSICAL_RELATIVE_TOLERANCE = 1e-6


def _weight(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("center_of_mass_weight must be finite")
    return value


@lru_cache(maxsize=16)
def _compiled_formula(center_of_mass_weight):
    """Compile the original expression once per physical objective weight."""
    import sympy as sp

    m, x, y, z, xx, yy, zz, yz, xz, xy = r = sp.symbols("m x y z xx yy zz yz xz xy")
    a = yy + zz - (y*y + z*z)/m
    b = xx + zz - (x*x + z*z)/m
    c = xx + yy - (x*x + y*y)/m
    d = -xy + x*y/m
    f = (a*a + b*b + 2*d*d)/c**2 + center_of_mass_weight*z/m
    g = sp.Matrix([x, y, -xz + x*z/m, -yz + y*z/m])
    return sp.lambdify(r, (f, g, sp.Matrix([f.diff(v) for v in r]), g.jacobian(r)), "numpy", cse=True)


def prepare_formula(center_of_mass_weight):
    """Optional worker warmup; charge this once to cold preparation timing."""
    _compiled_formula(_weight(center_of_mass_weight))


def formula_cache_info():
    """Return compilation cache counters, never query/result cache counters."""
    return _compiled_formula.cache_info()._asdict()


def _query(query):
    q = np.asarray(query, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (10,) or not np.isfinite(q).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("acceptance needs ten finite, nonzero raw coefficients")
    return q.copy()


def evaluate_moment_formula(moments, query, center_of_mass_weight, status="CONVERGED"):
    """Numerical attribution only; this function does not evaluate a real mesh.

    Passing predicted moments here can diagnose false convergence but cannot
    turn that prediction into independent native acceptance. Use ``verify``
    for the latter. The returned scope explicitly records this distinction.
    """
    q = _query(query)
    v = np.asarray(moments, dtype=np.float64)
    if v.shape != (10,) or not np.isfinite(v).all() or v[0] <= 0:
        raise ValueError("acceptance needs ten finite moments with positive mass")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        obj, cons, grad, G = _compiled_formula(_weight(center_of_mass_weight))(*v)
    obj = float(obj)
    cons = np.asarray(cons, dtype=np.float64).ravel()
    grad = np.asarray(grad, dtype=np.float64).ravel()
    G = np.asarray(G, dtype=np.float64)
    if not np.isfinite(np.r_[obj, cons, grad, G.ravel()]).all():
        raise ValueError("nonfinite native moment objective or moment gradients")
    # Exact deployed dual convention: mixed quadratic coefficients are doubled.
    dual = q.copy()
    dual[7:] *= 2
    dual /= np.linalg.norm(dual)
    proj = np.column_stack((-G.T, dual))
    coeff = np.linalg.lstsq(proj, -grad, rcond=None)[0]
    stationarity_vector = grad + proj @ coeff
    stationarity = float(np.linalg.norm(stationarity_vector))
    residual = float(np.linalg.norm(np.r_[stationarity_vector, cons]))
    phys = float(np.linalg.norm(cons)/max(1, np.linalg.norm(v)))
    accepted = bool(status == "CONVERGED" and np.isfinite([obj, phys, residual, *coeff]).all()
                    and coeff[-1] > 0 and residual <= EQUATION_TOLERANCE
                    and phys <= PHYSICAL_RELATIVE_TOLERANCE)
    return dict(accepted=accepted, scope="moment_formula_diagnostic", independent_mesh_evaluated=False,
                acceptance_profile=ACCEPTANCE_PROFILE, reference_verify_sha256=WORK_VERIFY_SHA256,
                objective=obj, constraints=cons.tolist(), max_constraint=float(np.abs(cons).max()),
                raw_residual=residual, physical_relative_residual=phys,
                stationarity_residual=stationarity, mu=float(coeff[-1]), mass=float(v[0]),
                center_of_mass=(v[1:4]/v[0]).tolist(), I_xz=float(cons[2]), I_yz=float(cons[3]),
                max_transverse_com=float(np.abs(v[1:3]/v[0]).max()), exact_ten_moments=v.tolist())


def _check_native_request(exact, request):
    """Prevent a cached worker context from verifying a different physical task."""
    actual = getattr(exact, "request", None)
    if actual is None:
        raise ValueError("native acceptance requires a backend bound to the original request")
    for bound in (request, actual):
        settings = getattr(bound, "settings", None)
        if (getattr(bound, "mode", None) != "SPINNING_TOP" or
                getattr(bound, "spinning_top_objective", None) != "INERTIA_AND_LOWERING" or
                getattr(settings, "equation_profile", None) != "DUAL_MONOMIAL_RAW_V1" or
                getattr(settings, "equation_tolerance", None) != EQUATION_TOLERANCE or
                getattr(settings, "physical_relative_tolerance", None) != PHYSICAL_RELATIVE_TOLERANCE):
            raise ValueError("request does not match the frozen Work spinning-top acceptance profile")
    if actual is not request:
        for field in ("design_vertices", "design_faces", "target_axes", "target_origin",
                      "fixed_shell_moments", "material_density", "center_of_mass_weight"):
            if not np.array_equal(getattr(actual, field), getattr(request, field)):
                raise ValueError("native acceptance request mismatch: " + field)
    if (getattr(exact, "integral_profile", None) != INTEGRAL_PROFILE or
            getattr(exact, "rational_arithmetic_backend", None) != "NATIVE_GMP_SURFACE_V1"):
        raise ValueError("native acceptance requires the frozen certified original-mesh arithmetic")


def verify(exact, q, request, status, *, prefer_moment_only=True, require_moment_only=False):
    """Drop-in Work verifier using fresh native moments and cached symbolics.

    Reusing the evaluator's static mesh/context is permitted; reusing a previous
    query's cut material or supplying learned moments is not. A lack of a real
    value-only backend is reported as ``joint_fallback``. Failure of a supported
    native value call propagates; it is not silently retried through another map.
    """
    if type(prefer_moment_only) is not bool or type(require_moment_only) is not bool:
        raise ValueError("moment-only options must be boolean")
    if require_moment_only and not prefer_moment_only:
        raise ValueError("require_moment_only conflicts with prefer_moment_only=False")
    q = _query(q)
    _check_native_request(exact, request)
    weight = _weight(request.center_of_mass_weight)
    preparation_start = perf_counter()
    prepare_formula(weight)
    formula_preparation_s = perf_counter() - preparation_start
    capability = getattr(exact, "moment_only_capabilities", None)
    supported = (isinstance(capability, dict) and capability.get("schema") == "cherchi-native-moment-only-v1"
                 and capability.get("independent_mesh_evaluation") is True
                 and capability.get("material_jacobian_avoided") is True
                 and capability.get("material_result_cache_used") is False
                 and capability.get("integral_profile") == INTEGRAL_PROFILE
                 and callable(getattr(exact, "evaluate_moments", None)))
    if require_moment_only and not supported:
        raise RuntimeError("native Moment-only capability unavailable; joint fallback forbidden")
    moment_only = prefer_moment_only and supported
    # A certificate from an earlier query cannot authenticate this evaluation.
    exact.last_integral_evidence = None
    start = perf_counter()
    out = exact.evaluate_moments(q) if moment_only else exact(q)
    mesh_seconds = perf_counter() - start
    if not str(out.get("boolean_backend_identity", "")).startswith("cherchi-reference-"):
        raise ValueError("independent native Cherchi material required, not predicted moments")
    if not all(isinstance(out.get(k), str) and out[k] for k in ("geometry_hash", "topology_hash")):
        raise ValueError("native mesh identity missing from acceptance evaluation")
    evidence = deepcopy(getattr(exact, "last_integral_evidence", None))
    if (not isinstance(evidence, dict) or evidence.get("enclosure_certified") is not True or
            evidence.get("precision_bits", 0) < 256):
        raise ValueError("fresh native certified integral evidence is missing")
    if moment_only and (out.get("derivative_defined") or out.get("moment_jacobian") is not None):
        raise ValueError("Moment-only native backend returned a derivative claim")
    result = evaluate_moment_formula(out["raw_moments"], q, weight, status)
    result.update(scope="returned_quadric", independent_mesh_evaluated=True,
                  mesh_geometry_sha256=out["geometry_hash"], mesh_topology_sha256=out["topology_hash"],
                  mesh_evaluate_s=mesh_seconds, certified_integral_evidence=evidence,
                  formula_preparation_s=formula_preparation_s,
                  verification_backend_mode="native_moment_only" if moment_only else "joint_fallback",
                  moment_only_supported=bool(supported),
                  moment_only_capabilities=deepcopy(capability) if moment_only else None,
                  cut_jacobian_requested=not moment_only)
    return result
