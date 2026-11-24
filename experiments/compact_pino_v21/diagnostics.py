"""Scaling and directional diagnostics shared by the V2.1 physics audit."""
from __future__ import annotations

import numpy as np


def _scale(mass, length):
    if not np.isfinite([mass, length]).all() or min(mass, length) <= 0:
        raise ValueError("positive finite editable mass and length required")
    return mass * np.power(length, [0, 1, 1, 1, 2, 2, 2, 2, 2, 2])


def _relative(delta, reference):
    norm = float(np.linalg.norm(reference))
    return float(np.linalg.norm(delta) / norm) if norm else None


def moment_errors(predicted, reference, mass, length):
    a, b = np.asarray(predicted), np.asarray(reference)
    if a.shape != (10,) or a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("expected finite undoubled ten moments")
    delta = a - b
    absolute = np.abs(delta)
    scaled = absolute / _scale(mass, length)
    return dict(absolute=absolute.tolist(), scaled=scaled.tolist(),
                scaled_max=float(scaled.max()), relative_l2=_relative(delta, b))


def jacobian_errors(predicted, reference, mass, length, coefficient_norm, *, basis=None, direction=None):
    """Supremum over unit input tangents of per-component, dimensionless error.

    For charts, basis maps a local coordinate increment into raw coefficient
    space. QR removes the coordinate metric before taking the row norms.
    """
    a, b = np.asarray(predicted), np.asarray(reference)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != 10 or a.shape[1] not in (10, 3, 8):
        raise ValueError("expected matching [10,D] native derivative matrices")
    if not np.isfinite([a, b]).all() or not np.isfinite(coefficient_norm) or coefficient_norm <= 0:
        raise ValueError("nonfinite derivatives or invalid coefficient norm")
    delta = a - b
    normalized = delta / _scale(mass, length)[:, None] * coefficient_norm
    if a.shape[1] != 10 or basis is not None:
        t = np.asarray(basis)
        if t.shape != (10, a.shape[1]) or not np.isfinite(t).all() or np.linalg.matrix_rank(t) != t.shape[1]:
            raise ValueError("native chart must supply a full-rank raw basis")
        _, r = np.linalg.qr(t, mode="reduced")
        normalized = np.linalg.solve(r.T, normalized.T).T
    out = dict(unit_direction_scaled_max=float(np.linalg.norm(normalized, axis=1).max()),
               relative_frobenius=_relative(delta, b),
               rmse=float(np.sqrt(np.mean(delta**2))), mae=float(np.mean(abs(delta))),
               column_absolute_max=np.max(np.abs(delta), axis=0).tolist())
    if direction is not None:
        v = np.asarray(direction)
        if v.shape != (a.shape[1],) or not np.isfinite(v).all():
            raise ValueError("incompatible direction")
        exact, estimate = b @ v, a @ v
        out['Jv'] = dict(predicted=estimate.tolist(), reference=exact.tolist(),
                         relative=_relative(estimate - exact, exact))
    return out


def accepted_directions(trace):
    """Raw unit tangent of the great-circle arc to the next accepted iterate.

    Normalizing an invertible diagonal coordinate transform preserves great
    circles and the orientation of the local arc. The result is the direction
    in raw coefficient coordinates, not the norm of a trial Newton step.
    """
    directions = {}
    for old, new in zip(trace, trace[1:]):
        if old['restart'] != new['restart']:
            continue
        q, p = (np.asarray(row['quadric_coefficients'], float) for row in (old, new))
        q = q / np.linalg.norm(q)
        tangent = p - q.dot(p) * q
        norm = np.linalg.norm(tangent)
        if norm > 64 * np.finfo(float).eps:
            directions[int(old['function_evaluations'])] = tangent / norm
    return directions
