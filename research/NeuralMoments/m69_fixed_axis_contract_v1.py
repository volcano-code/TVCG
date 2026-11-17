"""Faithful M69 fixed-axis TaskSpec constraint adapter (v1)."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

TASK_SCHEMA = "m69.fixed-axis-task.v1"
TASK_SOURCE_SHA256 = "5463dcfe57f3aea21f577fd77d26867ef9e7ec18098b7ff25e7bde5209fc3389"
TASK_HASH = "54a52d35971aacf83e5ab3dc52dcf11e686724ad0199201340009a2bd342b38d"
MOMENT_ORDER = ("mass", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")

def _vec3(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return result

def _moments(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (10,) or not np.isfinite(result).all() or result[0] <= 0:
        raise ValueError(f"{name} must contain ten finite moments with positive mass")
    return result

def second_matrix(moments):
    m = _moments(moments, "moments")
    return np.array([[m[4],m[9],m[8]],[m[9],m[5],m[7]],[m[8],m[7],m[6]]],dtype=np.float64)

@dataclass(frozen=True)
class M69PhysicalMetrics:
    mass: float
    polar: float
    centroid: tuple
    transverse_centroid: float
    centroid_height: float
    alignment_error: float
    principal_gap: float
    covariance_min_eigenvalue: float

@dataclass(frozen=True)
class M69ConstraintReport:
    task_hash: str
    metrics: M69PhysicalMetrics
    constraint_values: dict
    checks: dict
    moment_pass: bool
    connectivity_checked: bool
    connectivity_pass: object
    complete_pass: bool

def physical_metrics(moments, task, length_scale, reference_polar=None):
    """Port of task69._physical; input coordinates remain in the supplied frame."""
    m = _moments(moments, "moments")
    L = float(length_scale)
    if not math.isfinite(L) or L <= 0:
        raise ValueError("length_scale must be finite and positive")
    axis = _vec3(task["axis"], "axis")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-14:
        raise ValueError("axis cannot be zero")
    axis = axis / axis_norm
    origin = _vec3(task["axis_origin"], "axis_origin")
    contact = _vec3(task["contact"], "contact")
    delta_contact = contact - origin
    off_axis = delta_contact - float(np.dot(delta_contact, axis)) * axis
    if np.linalg.norm(off_axis) > 1e-10 * max(1.0, np.linalg.norm(delta_contact)):
        raise ValueError("contact must lie on the prescribed fixed axis")
    center = m[1:4] / m[0]
    S = second_matrix(m)
    covariance = S - m[0] * np.outer(center, center)
    centered_inertia = np.trace(covariance) * np.eye(3) - covariance
    shifted = S - np.outer(origin,m[1:4]) - np.outer(m[1:4],origin) + m[0]*np.outer(origin,origin)
    polar = float(np.trace(shifted) - axis @ shifted @ axis)
    if reference_polar is None:
        reference_polar = polar
    reference_polar = float(reference_polar)
    if not math.isfinite(reference_polar) or reference_polar <= 0:
        raise ValueError("original fixed-axis inertia must be positive")
    displacement = center - origin
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    plane = np.stack((u,v),axis=1)
    axial_centered = float(axis @ centered_inertia @ axis)
    transverse_max = float(np.linalg.eigvalsh(plane.T @ centered_inertia @ plane)[-1])
    return M69PhysicalMetrics(
        mass=float(m[0]), polar=polar, centroid=tuple(float(x) for x in center),
        transverse_centroid=float(np.linalg.norm(displacement-np.dot(displacement,axis)*axis)),
        centroid_height=float(np.dot(center-contact,axis)),
        alignment_error=float(np.linalg.norm(centered_inertia@axis-axial_centered*axis)/reference_polar),
        principal_gap=float((axial_centered-transverse_max)/reference_polar),
        covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance)[0]),
    )

def evaluate_m69_moments(moments, original_moments, task, *, length_scale, connectivity=None):
    """Evaluate all ten M69 moment constraints; connectivity is an independent audit."""
    if task.get("schema") != TASK_SCHEMA:
        raise ValueError("unsupported M69 TaskSpec schema")
    if task.get("objective") != "minimum_retained_mass":
        raise ValueError("unsupported M69 objective")
    if task.get("material_policy") != "single_density_removal_only":
        raise ValueError("unsupported M69 material policy")
    if task.get("inertia_reference") != "original_fixed_axis":
        raise ValueError("unsupported M69 inertia reference")
    density = float(task["density"])
    if not math.isfinite(density) or density <= 0:
        raise ValueError("TaskSpec density must be positive and finite")
    if connectivity is not None and not isinstance(connectivity,(bool,np.bool_)):
        raise ValueError("connectivity must be an independent boolean audit result")
    original = _moments(original_moments,"original_moments")
    candidate = _moments(moments,"moments")
    L = float(length_scale)
    reference = physical_metrics(original,task,L)
    current = physical_metrics(candidate,task,L,reference_polar=reference.polar)
    mass_fraction = current.mass/reference.mass
    polar_retention = current.polar/reference.polar
    values = {
        "positive_mass": mass_fraction,
        "no_added_mass": 1.0+float(task["mass_tolerance"])-mass_fraction,
        "minimum_removal": 1.0-mass_fraction-float(task["min_removal_fraction"])+float(task["mass_tolerance"]),
        "polar_retention": polar_retention-float(task["polar_retention"])+float(task["inertia_tolerance"]),
        "no_added_polar_inertia": 1.0+float(task["inertia_tolerance"])-polar_retention,
        "axis_alignment_com": float(task["com_tolerance"])-current.transverse_centroid/L,
        "axis_alignment_inertia": float(task["alignment_tolerance"])-current.alignment_error,
        "principal_dominance": current.principal_gap-float(task["principal_gap_min"]),
        "centroid_height": float(task["max_centroid_rise"])-(current.centroid_height-reference.centroid_height)/L,
        "physical_second_moments": current.covariance_min_eigenvalue/(reference.mass*L*L)+float(task["moment_psd_tolerance"]),
    }
    checks = {key: bool(value >= 0.0) for key,value in values.items()}
    analytic_infeasible = float(task["polar_retention"]) == 1.0 and float(task["min_removal_fraction"]) > 0.0
    moment_pass = all(checks.values()) and not analytic_infeasible
    connected = connectivity is not None and bool(connectivity)
    complete = moment_pass and (connected or not bool(task["require_connected"]))
    return M69ConstraintReport(
        task_hash=TASK_HASH,metrics=current,constraint_values={k:float(v) for k,v in values.items()},
        checks=checks,moment_pass=bool(moment_pass),connectivity_checked=connectivity is not None,
        connectivity_pass=None if connectivity is None else bool(connectivity),complete_pass=bool(complete)
    )
