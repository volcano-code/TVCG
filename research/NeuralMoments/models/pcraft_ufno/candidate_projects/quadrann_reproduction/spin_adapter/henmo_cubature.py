"""Static positive cubature compression on the *original material support*.

The complete ten moments are preserved within a checked numerical tolerance.
This is not a certificate for partially cut moments or shape derivatives.  A
query-independent spatial partition keeps more information than a single
centroid/covariance fit, but difficult cuts still require refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def moment_basis(points):
    x, y, z = np.asarray(points, dtype=np.float64).T
    return np.column_stack((np.ones(len(x)), x, y, z, x*x, y*y, z*z, y*z, x*z, x*y))


@dataclass(frozen=True)
class CompressedCubature:
    nodes: np.ndarray
    weights: np.ndarray
    source_indices: np.ndarray
    total_moments: np.ndarray
    source_count: int
    normalized_moment_error: float
    compressed: bool
    reason: str
    source_rule: str = 'provided_positive_atoms'


def _freeze(*arrays):
    for array in arrays:
        array.flags.writeable = False


def compress_positive_measure(nodes, weights, max_atoms=64, *, tolerance=2e-11):
    """Return positive source atoms, preserving degree-two moments.

Each deterministic spatial group is reduced by a nonnegative linear program.
Its selected support is polished by a tiny least-squares solve. Any failed
positivity/moment check returns the original positive measure, never an
unverified rule. ``max_atoms`` is a requested budget, not a reason to discard
material when numerical compression is unsuccessful.
"""
    xyz = np.array(nodes, dtype=np.float64, order='C', copy=True)
    weight = np.array(weights, dtype=np.float64, copy=True)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or weight.shape != (len(xyz),)
            or not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(weight).all()
            or np.any(weight <= 0)):
        raise ValueError('compression requires finite material atoms with strictly positive weights')
    if type(max_atoms) is not int or max_atoms < 10:
        raise ValueError('max_atoms must be an integer at least ten')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive compression tolerance required')
    total = (moment_basis(xyz) * weight[:, None]).sum(0)
    source = np.arange(len(xyz), dtype=np.int64)

    def fallback(reason):
        _freeze(xyz, weight, source, total)
        return CompressedCubature(xyz, weight, source, total, len(xyz), 0., False, reason)

    if len(xyz) <= max_atoms:
        return fallback('ALREADY_WITHIN_BUDGET')
    # Local coordinates avoid conditioning the compression on the world origin.
    center = (xyz.min(0) + xyz.max(0)) * .5
    extent = max(float(np.ptp(xyz, axis=0).max()), np.finfo(float).tiny)
    local = (xyz - center) / extent
    local_basis = moment_basis(local)
    normalized_weight = weight / total[0]
    target = (local_basis * normalized_weight[:, None]).sum(0)
    groups = [source]
    desired_groups = min(max_atoms // 10, len(xyz))
    while len(groups) < desired_groups:
        choice = max(range(len(groups)), key=lambda i: len(groups[i]))
        ids = groups.pop(choice)
        if len(ids) < 2:
            groups.append(ids)
            break
        axis = int(np.argmax(np.ptp(local[ids], axis=0)))
        ids = ids[np.argsort(local[ids, axis], kind='stable')]
        half = len(ids) // 2
        groups.extend((ids[:half], ids[half:]))
    from scipy.optimize import linprog
    selected, values = [], []
    for ids in groups:
        if len(ids) <= 10:
            selected.append(ids)
            values.append(normalized_weight[ids])
            continue
        basis = local_basis[ids].T
        rhs = basis @ normalized_weight[ids]
        # A bounded, deterministic linear objective chooses a basic feasible
        # rule with at most rank(basis) support entries. It is not a learned fit.
        objective = np.linspace(0., 1., len(ids), dtype=np.float64)
        fit = linprog(objective, A_eq=basis, b_eq=rhs, bounds=(0, None), method='highs',
                      options={'primal_feasibility_tolerance': 1e-9,
                               'dual_feasibility_tolerance': 1e-9})
        if not fit.success or not np.isfinite(fit.x).all():
            return fallback('POSITIVE_REDUCTION_FAILED')
        active = np.flatnonzero(fit.x > 0)
        if not len(active) or len(active) > 10:
            return fallback('REDUCTION_SUPPORT_EXCEEDED')
        chosen = basis[:, active]
        corrected = fit.x[active] + np.linalg.lstsq(chosen, rhs - chosen @ fit.x[active], rcond=None)[0]
        if np.any(corrected <= 0) or not np.isfinite(corrected).all():
            return fallback('POSITIVE_POLISH_FAILED')
        selected.append(ids[active])
        values.append(corrected)
    index = np.concatenate(selected)
    reduced = np.concatenate(values)
    residual = float(np.max(np.abs(local_basis[index].T @ reduced - target)))
    if len(index) > max_atoms or residual > tolerance:
        return fallback('MOMENT_CHECK_FAILED')
    result_nodes = xyz[index].copy()
    result_weights = reduced * total[0]
    result_total = (moment_basis(result_nodes) * result_weights[:, None]).sum(0)
    # Report the original total, and audit the computed rule in well-scaled
    # coordinates. World-coordinate cancellation is not a compression metric.
    if not np.isfinite(result_total).all():
        return fallback('NONFINITE_WORLD_MOMENTS')
    _freeze(result_nodes, result_weights, index, total)
    return CompressedCubature(result_nodes, result_weights, index, total, len(xyz),
                              residual, True, 'POSITIVE_SOURCE_SUPPORT_MOMENT_MATCH')
