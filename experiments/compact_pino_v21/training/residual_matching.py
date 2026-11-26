"""Fixed-reference residual matching; never impose zero on arbitrary queries.

``REPO_PROFILE`` mirrors the explicitly bound repository objective/constraints,
not the private noodles ``solve.py``. ``LINEARIZED_PROFILE`` consumes a native
teacher's frozen H_L and Dc at M*: its output is a first-order sensitivity error,
NOT an exact deployed residual or an acceptance decision.

An admitted ``record['residual_target']`` has the following JSON fields::

    schema: 'cpino_fixed_residual_target_v1'
    valid: true
    profile: REPO_PROFILE or LINEARIZED_PROFILE
    teacher_role: native_teacher | frozen_deployment_replay | synthetic_fixture
    formula_provenance:
        source_profile: explicit native formula/solver identity
        source_sha256: {source_name: lowercase_sha256, ...}
    moment_order: ['1', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'yz', 'xz', 'xy']
    reference_moments: [10 raw moments, INCLUDING the fixed shell]
    p_convention: 'unit_dual_monomial_v1'
    p: [10 unit dual monomial coordinates; raw cross coefficients doubled]
    multipliers: [one frozen lambda per constraint]
    multiplier_mu: frozen scalar
    stationarity_scale: [10 strictly positive fixed scales]
    constraint_scale: [one strictly positive fixed scale per constraint]

Exact repository targets additionally declare ``objective={mode, weight,
mass_scale}``, using native_solver_adapter.MODES. Their formula provenance binds
the two source digests below. Linearized targets instead provide
``lagrangian_hessian[10,10]``, ``constraint_jacobian[n,10]``,
``reference_stationarity[10]`` and ``reference_constraints[n]`` from that SAME
native reference state. H_L is already d2(f-lambda.c)/dM2, not an unweighted
objective Hessian. No multiplier fit or Hessian recomputation occurs here.

``binding={'case_sha256': ...}`` and query/geometry identity are checked by the
manifest/export layer. Pure numerical calls do not assert artifact provenance.
An unavailable target uses ``valid: false`` and a nonempty ``reason``; it must
not provide invented zero labels. Every tensor in a target, including its
scales and multipliers, is frozen. Only predicted moments carry gradients.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re

import torch


TARGET_SCHEMA = "cpino_fixed_residual_target_v1"
REPO_PROFILE = "TVCG_REPO_DUAL_MONOMIAL_RAW_V1_CABI_V1"
LINEARIZED_PROFILE = "NATIVE_FROZEN_MOMENT_LINEARIZATION_V1"
P_CONVENTION = "unit_dual_monomial_v1"
MOMENT_ORDER = ("1", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
REPO_FORMULA_SHA256 = {
    "src/spin/SpinMomentProblem.cpp": "5f02921f38ba8ba6b8f03445b8db1dcac23ff79039f558d4ceb9322a6b04bbab",
    "include/spin/MomentInertiaAlgebra.h": "6ed2e361b07e6fff881ffd85743d6bb7fa67c3b5b477a6465648864902d0ecc1",
}
_COUNTS = {"STATIC_MAXIMUM_MASS": 2, "STATIC_LOWER_CENTER": 2,
           "STATIC_PARETO": 2, "FIDGET_SPINNER": 2, "YOYO": 5,
           "SPINNING_TOP": 4, "SPINNING_TOP_LOWER_CENTER": 4}
_TEACHER_ROLES = {"native_teacher", "frozen_deployment_replay", "synthetic_fixture"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class PhysicalDomainError(ValueError):
    """Predicted moments are outside the bound formula's numerical domain.

    Training propagates this error. Read-only metrics may report it only after
    independently validating the reference target and its query/profile.
    """


def _constant(value, *, like, shape, name):
    """Copy no graph from a teacher/reference tensor, even if it has one."""
    try:
        out = torch.as_tensor(value, dtype=torch.float64, device=like.device).detach()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite numeric tensor with shape {shape}") from exc
    if tuple(out.shape) != tuple(shape) or not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out


def _scalar(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64).detach()
        if tensor.ndim != 0:
            raise ValueError
        number = float(tensor)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite scalar")
    return number


@lru_cache(maxsize=1)
def _check_repo_sources():
    root = Path(__file__).resolve().parents[3]
    for name, expected in REPO_FORMULA_SHA256.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("repository formula source changed; revalidate the differentiable profile: " + name)


def _objective_settings(objective):
    if not isinstance(objective, dict) or objective.get("mode") not in _COUNTS:
        raise ValueError("explicit supported repository objective mode required")
    mode = objective["mode"]
    if "weight" not in objective or "mass_scale" not in objective:
        raise ValueError("objective weight and mass_scale must be explicit")
    weight = _scalar(objective["weight"], "objective weight")
    mass_scale = _scalar(objective["mass_scale"], "objective mass_scale")
    if mass_scale <= 0:
        raise ValueError("positive objective mass_scale required")
    if mode == "STATIC_PARETO" and not 0 <= weight <= 1:
        raise ValueError("static Pareto weight must lie in [0,1]")
    if mode.startswith("SPINNING_TOP") and weight < 0:
        raise ValueError("spinning-top weight must be nonnegative")
    if mode == "SPINNING_TOP_LOWER_CENTER" and weight != 0:
        raise ValueError("pure-height endpoint requires zero extra weight")
    return {"mode": mode, "weight": weight, "mass_scale": mass_scale}


def _availability(target):
    if not isinstance(target, dict) or type(target.get("valid")) is not bool:
        raise ValueError("residual target valid must be an explicit boolean")
    if not target["valid"]:
        if not isinstance(target.get("reason"), str) or not target["reason"].strip():
            raise ValueError("unavailable residual target requires its reason")
        return False
    if target.get("schema") != TARGET_SCHEMA:
        raise ValueError("explicit fixed residual target schema required")
    return True


def _prepare(target, native_moments, *, coefficients=None, device=None):
    if not _availability(target):
        raise ValueError("residual target is explicitly unavailable: " + target["reason"])
    profile = target.get("profile")
    if profile not in (REPO_PROFILE, LINEARIZED_PROFILE):
        raise ValueError("unknown residual profile; no implicit deployed formula substitution")
    if tuple(target.get("moment_order", ())) != MOMENT_ORDER:
        raise ValueError("raw undoubled moment_order mismatch")
    if target.get("p_convention") != P_CONVENTION:
        raise ValueError("unsupported p_convention; explicit unit dual monomial coordinates required")
    if target.get("teacher_role") not in _TEACHER_ROLES:
        raise ValueError("explicit residual teacher_role required")
    provenance = target.get("formula_provenance")
    if (not isinstance(provenance, dict) or not isinstance(provenance.get("source_profile"), str)
            or not provenance["source_profile"].strip()):
        raise ValueError("formula_provenance requires an explicit source_profile")
    hashes = provenance.get("source_sha256")
    if (not isinstance(hashes, dict) or not hashes
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not _HASH.fullmatch(v)
                   for k, v in hashes.items())):
        raise ValueError("formula_provenance requires source SHA-256 digests")
    like = torch.empty((), dtype=torch.float64, device=device)
    reference = _constant(native_moments, like=like, shape=(10,), name="native moments")
    if float(reference[0]) <= 0:
        raise ValueError("native reference moments require positive mass")
    bound = _constant(target.get("reference_moments"), like=like, shape=(10,), name="reference_moments")
    if not torch.equal(reference, bound):
        raise ValueError("residual reference_moments differ from the case native teacher")
    p = _constant(target.get("p"), like=like, shape=(10,), name="fixed unit dual p")
    if abs(float(torch.linalg.vector_norm(p)) - 1.) > 1e-12:
        raise ValueError("fixed p must already be unit dual coordinates; no silent normalization")
    if coefficients is not None:
        raw = _constant(coefficients, like=like, shape=(10,), name="query coefficients")
        dual = raw * raw.new_tensor([1.] * 7 + [2.] * 3)
        norm = torch.linalg.vector_norm(dual)
        if float(norm) == 0. or not torch.allclose(p, dual / norm, rtol=0., atol=1e-12):
            raise ValueError("fixed p is not the supplied query's doubled-cross-term unit dual coordinate")
    settings = None
    if profile == REPO_PROFILE:
        if provenance["source_profile"] != REPO_PROFILE:
            raise ValueError("exact repository formula cannot be relabelled as another deployment profile")
        if any(hashes.get(name) != digest for name, digest in REPO_FORMULA_SHA256.items()):
            raise ValueError("exact repository objective/constraint source digest mismatch")
        _check_repo_sources()
        settings = _objective_settings(target.get("objective"))
        count = _COUNTS[settings["mode"]]
        # Reject undefined teacher formulas during admission, before a batch
        # attempts to train against them. The deployed denominator is not
        # changed with a clamp or an added epsilon.
        try:
            reference_objective, reference_constraints = repository_objective_constraints(reference, settings)
        except PhysicalDomainError as exc:
            raise ValueError("invalid native reference physical domain: " + str(exc)) from exc
        if not bool(torch.isfinite(reference_objective)) or not bool(torch.isfinite(reference_constraints).all()):
            raise ValueError("nonfinite repository objective/constraints at the native reference")
    else:
        try:
            values = torch.as_tensor(target.get("constraint_jacobian"), dtype=torch.float64, device=like.device).detach()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("native constraint_jacobian must have shape (n_constraints,10)") from exc
        if values.ndim != 2 or values.shape[1] != 10 or values.shape[0] < 1:
            raise ValueError("native constraint_jacobian must have shape (n_constraints,10)")
        count = int(values.shape[0])
    lambdas = _constant(target.get("multipliers"), like=like, shape=(count,), name="fixed multipliers")
    mu = _constant(target.get("multiplier_mu"), like=like, shape=(), name="fixed multiplier_mu")
    rscale = _constant(target.get("stationarity_scale"), like=like, shape=(10,), name="stationarity_scale")
    cscale = _constant(target.get("constraint_scale"), like=like, shape=(count,), name="constraint_scale")
    if bool((rscale <= 0).any()) or bool((cscale <= 0).any()):
        raise ValueError("fixed residual scales must be strictly positive; zero reference residuals are not scales")
    result = dict(profile=profile, reference=reference, p=p, lambdas=lambdas, mu=mu,
                  stationarity_scale=rscale, constraint_scale=cscale, objective=settings,
                  constraint_count=count)
    if profile == LINEARIZED_PROFILE:
        hessian = _constant(target.get("lagrangian_hessian"), like=like, shape=(10,10), name="native lagrangian_hessian")
        if not torch.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12):
            raise ValueError("native lagrangian_hessian must be symmetric; no silent symmetrization")
        result.update(hessian=hessian,
            dc=_constant(target.get("constraint_jacobian"), like=like, shape=(count,10), name="native constraint_jacobian"),
            reference_stationarity=_constant(target.get("reference_stationarity"), like=like, shape=(10,), name="native reference_stationarity"),
            reference_constraints=_constant(target.get("reference_constraints"), like=like, shape=(count,), name="native reference_constraints"))
    return result


def validate_residual_target(target, native_moments=None, *, coefficients=None):
    """Validate numerical semantics; the caller separately binds file/query bytes."""
    if not _availability(target):
        return {"availability": "UNAVAILABLE_TARGET", "reason": target["reason"]}
    native = target.get("reference_moments") if native_moments is None else native_moments
    context = _prepare(target, native, coefficients=coefficients)
    return dict(availability="AVAILABLE", profile=context["profile"],
                constraint_count=context["constraint_count"], teacher_role=target["teacher_role"],
                approximation=("exact_fixed_multiplier_residual_difference" if context["profile"] == REPO_PROFILE
                               else "first_order_at_native_moments"))


def repository_objective_constraints(moments, objective):
    """Differentiable source-bound SpinMomentProblem; input includes the shell."""
    settings = _objective_settings(objective)
    if moments.shape != (10,) or not bool(torch.isfinite(moments).all()):
        raise ValueError("repository moments require ten finite values")
    if float(moments[0].detach()) <= 0:
        raise PhysicalDomainError("repository moments require positive mass")
    mode, weight, mass_scale = settings["mode"], settings["weight"], settings["mass_scale"]
    m, x, y, z, xx, yy, zz, yz, xz, xy = moments.unbind()
    horizontal = torch.stack((x, y))
    if mode == "STATIC_MAXIMUM_MASS":
        return -mass_scale * m, horizontal
    if mode == "STATIC_LOWER_CENTER":
        return z / m, horizontal
    if mode == "STATIC_PARETO":
        return -weight * mass_scale * m + (1. - weight) * z / m, horizontal
    if mode == "FIDGET_SPINNER":
        return -(xx + yy), horizontal
    i11 = yy + zz - (y*y + z*z) / m
    i22 = xx + zz - (x*x + z*z) / m
    i33 = xx + yy - (x*x + y*y) / m
    i12 = -xy + x*y / m
    i13 = -xz + x*z / m
    i23 = -yz + y*z / m
    constraints = torch.stack((x,y,z,i13,i23) if mode == "YOYO" else (x,y,i13,i23))
    if mode == "SPINNING_TOP_LOWER_CENTER":
        return z / m, constraints
    if abs(float(i33.detach())) <= 1e-14:
        raise PhysicalDomainError("repository axial central inertia denominator is zero")
    value = (i11*i11 + i22*i22 + 2.*i12*i12) / (i33*i33)
    return value if mode == "YOYO" else value + weight*z/m, constraints


def _exact_gradient(moments, context, *, differentiable):
    # enable_grad is needed even for a no_grad diagnostic because Eq.14 itself
    # contains a derivative. Do not create a fake parameter dependency for a
    # saturated prediction that did not depend on any model parameter.
    retain = bool(differentiable and torch.is_grad_enabled() and moments.requires_grad)
    with torch.enable_grad():
        value = moments if moments.requires_grad else moments.detach().requires_grad_(True)
        objective, constraints = repository_objective_constraints(value, context["objective"])
        if not bool(torch.isfinite(objective)) or not bool(torch.isfinite(constraints).all()):
            raise PhysicalDomainError("nonfinite repository objective/constraints at the evaluated moments")
        lagrangian = objective - torch.dot(constraints, context["lambdas"])
        gradient, = torch.autograd.grad(lagrangian, value, create_graph=retain)
    if not bool(torch.isfinite(gradient).all()):
        raise PhysicalDomainError("nonfinite repository Lagrangian gradient at the evaluated moments")
    if not retain:
        return gradient.detach(), constraints.detach()
    return gradient, constraints


def residual_match_terms(predicted_moments, native_moments, target, *, coefficients=None):
    """Return differentiable vector differences with fixed, nonzero references.

    FP64 conversion of a prediction preserves its model gradient. Teacher
    tensors are detached. The common mu*p term cancels analytically in the
    difference, avoiding subtractive loss of precision for large fixed mu.
    """
    prediction = torch.as_tensor(predicted_moments).to(dtype=torch.float64)
    if prediction.shape != (10,) or not bool(torch.isfinite(prediction).all()):
        raise ValueError("predicted moments must contain ten finite values")
    context = _prepare(target, native_moments, coefficients=coefficients, device=prediction.device)
    if context["profile"] == REPO_PROFILE:
        # Reference failures are invalid labels, not prediction diagnostics.
        # Keep that distinction even if a finite teacher objective has an
        # overflowing derivative at the supplied precision.
        try:
            gr, cr = _exact_gradient(context["reference"], context, differentiable=False)
        except PhysicalDomainError as exc:
            raise ValueError("invalid native reference residual: " + str(exc)) from exc
        gp, cp = _exact_gradient(prediction, context, differentiable=True)
        dr, dc = gp - gr, cp - cr
        rp, rr = gp + context["mu"]*context["p"], gr + context["mu"]*context["p"]
        approximation = "exact_fixed_multiplier_residual_difference"
    else:
        delta = prediction - context["reference"]
        dr, dc = context["hessian"] @ delta, context["dc"] @ delta
        rr, cr = context["reference_stationarity"], context["reference_constraints"]
        rp, cp = rr + dr, cr + dc
        approximation = "first_order_at_native_moments"
    result = dict(profile=context["profile"], approximation=approximation,
                stationarity_delta=dr, constraint_delta=dc,
                stationarity_scaled_delta=dr / context["stationarity_scale"],
                constraint_scaled_delta=dc / context["constraint_scale"],
                predicted_stationarity=rp, reference_stationarity=rr,
                predicted_constraints=cp, reference_constraints=cr)
    if any(not bool(torch.isfinite(value).all()) for value in result.values() if isinstance(value, torch.Tensor)):
        raise ValueError("nonfinite fixed residual matching result; native formulas/scales were not modified")
    return result


def residual_metrics(predicted_moments, native_moments, target, *, coefficients=None):
    """JSON diagnostics; optional physics cannot silently change M/J selection.

    A valid target with a prediction outside its formula domain is reported as
    unavailable, with no invented zero metric. Profile, binding, shape, and
    teacher-reference errors still raise. Training uses residual_match_terms
    directly and never swallows a prediction-domain failure.
    """
    if target is None:
        return {"availability": "UNAVAILABLE_PROFILE", "reason": "NO_FIXED_RESIDUAL_TARGET"}
    if not _availability(target):
        return {"availability": "UNAVAILABLE_TARGET", "reason": target["reason"]}
    validated = validate_residual_target(target, native_moments, coefficients=coefficients)
    try:
        with torch.no_grad():
            terms = residual_match_terms(predicted_moments, native_moments, target, coefficients=coefficients)
    except PhysicalDomainError as exc:
        return dict(availability="UNAVAILABLE_PREDICTION_DOMAIN", reason=str(exc),
                    profile=validated["profile"], approximation=validated["approximation"],
                    teacher_role=target["teacher_role"],
                    claim_scope="physical_diagnostic_unavailable_no_native_acceptance_decision")
    result = {"availability": "AVAILABLE", "profile": terms["profile"],
              "approximation": terms["approximation"], "teacher_role": target["teacher_role"],
              "claim_scope": "fixed_reference_error_only_no_native_acceptance_decision"}
    for name in ("stationarity_delta", "constraint_delta", "predicted_stationarity",
                 "reference_stationarity", "predicted_constraints", "reference_constraints"):
        result[name] = terms[name].detach().cpu().tolist()
    for output, term in (("stationarity_error_l2", "stationarity_delta"),
                         ("scaled_stationarity_error_l2", "stationarity_scaled_delta"),
                         ("constraint_error_l2", "constraint_delta"),
                         ("scaled_constraint_error_l2", "constraint_scaled_delta")):
        result[output] = float(torch.linalg.vector_norm(terms[term]).detach())
    return result
