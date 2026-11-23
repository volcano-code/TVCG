"""Independent Jacobian diagnostics for the Full V1 raw-coefficient contract.

This module deliberately does not change model outputs or training behavior.
It separates (1) model self-consistency, comparing a supplied Jacobian with
finite differences of the same model's Moment function, from (2) physical
accuracy, comparing that Jacobian with an independently supplied teacher.

All Jacobians use [Q, moment_dim, coefficient_dim] and raw input coefficients.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


def _as_batch_coefficients(coefficients: Tensor) -> tuple[Tensor, bool]:
    if not isinstance(coefficients, Tensor) or not coefficients.is_floating_point():
        raise TypeError("coefficients must be a floating-point Tensor")
    single = coefficients.ndim == 1
    values = coefficients.unsqueeze(0) if single else coefficients
    if values.ndim != 2 or values.shape[1] != 10 or values.shape[0] == 0:
        raise ValueError("coefficients must have shape [10] or [Q,10]")
    if not torch.isfinite(values).all():
        raise ValueError("coefficients must be finite")
    if (torch.linalg.vector_norm(values, dim=-1) <= 1e-12).any():
        raise ValueError("coefficient vectors must be nonzero")
    return values, single


def finite_difference_jacobian(
    moment_fn: Callable[[Tensor], Tensor],
    coefficients: Tensor,
    *,
    step: float,
) -> Tensor:
    """Central finite-difference Jacobian of Moment wrt raw coefficients.

    moment_fn must map [Q,10] to [Q,M]. The returned shape is [Q,M,10].
    This routine is diagnostic only; it does not preserve an autograd graph.
    """
    if not (float(step) > 0):
        raise ValueError("step must be positive")
    c, _ = _as_batch_coefficients(coefficients)
    columns = []
    with torch.no_grad():
        for axis in range(c.shape[1]):
            delta = torch.zeros_like(c)
            delta[:, axis] = float(step)
            plus = moment_fn(c + delta)
            minus = moment_fn(c - delta)
            if plus.ndim != 2 or minus.shape != plus.shape or plus.shape[0] != len(c):
                raise ValueError("moment_fn must return [Q,M] for each input batch")
            if not torch.isfinite(plus).all() or not torch.isfinite(minus).all():
                raise ValueError("moment_fn returned nonfinite values during finite differences")
            columns.append((plus - minus) / (2.0 * float(step)))
    return torch.stack(columns, dim=-1)


def compare_jacobians(
    predicted: Tensor,
    reference: Tensor,
    *,
    component_scales: Tensor | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compare [Q,M,D] Jacobians without hiding near-zero reference columns."""
    if predicted.shape != reference.shape or predicted.ndim != 3:
        raise ValueError("predicted and reference must share [Q,M,D] shape")
    if not torch.isfinite(predicted).all() or not torch.isfinite(reference).all():
        raise ValueError("Jacobian inputs must be finite")
    p = predicted.detach().to(torch.float64)
    r = reference.detach().to(device=p.device, dtype=torch.float64)
    diff = p - r
    ref_norm = torch.linalg.vector_norm(r)
    pred_norm = torch.linalg.vector_norm(p)
    relative = torch.linalg.vector_norm(diff) / ref_norm.clamp_min(eps)
    per_column_error = torch.linalg.vector_norm(diff, dim=(0, 1))
    per_column_reference = torch.linalg.vector_norm(r, dim=(0, 1))
    per_column_relative = per_column_error / per_column_reference.clamp_min(eps)
    cosine = (p.flatten() @ r.flatten()) / (
        torch.linalg.vector_norm(p.flatten()) * torch.linalg.vector_norm(r.flatten())
    ).clamp_min(eps)
    result: dict[str, Any] = {
        "queries": int(p.shape[0]),
        "moment_dimension": int(p.shape[1]),
        "coefficient_dimension": int(p.shape[2]),
        "relative_frobenius": float(relative),
        "absolute_frobenius": float(torch.linalg.vector_norm(diff)),
        "rmse": float(diff.square().mean().sqrt()),
        "mae": float(diff.abs().mean()),
        "max_absolute": float(diff.abs().max()),
        "reference_frobenius": float(ref_norm),
        "predicted_frobenius": float(pred_norm),
        "cosine_similarity": float(cosine),
        "per_coefficient_column_relative": [float(x) for x in per_column_relative],
        "per_coefficient_column_absolute": [float(x) for x in per_column_error],
        "near_zero_reference_columns": [
            int(i) for i, value in enumerate(per_column_reference) if float(value) <= eps
        ],
    }
    if component_scales is not None:
        scales = torch.as_tensor(component_scales, device=p.device, dtype=p.dtype)
        if scales.shape != (p.shape[1],) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("component_scales must be positive finite [M]")
        normalized = diff / scales[None, :, None]
        result["component_scaled_rmse"] = float(normalized.square().mean().sqrt())
        result["component_scaled_mae"] = float(normalized.abs().mean())
    return result


def directional_jacobian_error(
    predicted: Tensor,
    reference: Tensor,
    directions: Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compare Jv for matched [Q,M,D] Jacobians and [Q,D] directions."""
    if predicted.shape != reference.shape or predicted.ndim != 3:
        raise ValueError("Jacobians must share [Q,M,D] shape")
    if directions.shape != (predicted.shape[0], predicted.shape[2]):
        raise ValueError("directions must have shape [Q,D]")
    p = torch.bmm(predicted, directions.unsqueeze(-1)).squeeze(-1).to(torch.float64)
    r = torch.bmm(reference, directions.unsqueeze(-1)).squeeze(-1).to(torch.float64)
    delta = p - r
    return {
        "absolute_l2": float(torch.linalg.vector_norm(delta)),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(r).clamp_min(eps)),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": float((p.flatten() @ r.flatten()) / (
            torch.linalg.vector_norm(p.flatten()) * torch.linalg.vector_norm(r.flatten())
        ).clamp_min(eps)),
    }


def audit_jacobian_paths(
    moment_fn: Callable[[Tensor], Tensor],
    model_jacobian_fn: Callable[[Tensor], Tensor],
    coefficients: Tensor,
    *,
    step_sizes: Sequence[float] = (1e-3, 3e-4, 1e-4),
    native_jacobian: Tensor | None = None,
    directions: Tensor | None = None,
    component_scales: Tensor | None = None,
) -> dict[str, Any]:
    """Run separate self-consistency and native-physics comparisons.

    The finite-difference sweep is intentionally reported per step size. A
    single h can be misleading near cut events or under floating-point noise.
    """
    c, _ = _as_batch_coefficients(coefficients)
    model_j = model_jacobian_fn(c)
    if model_j.ndim != 3 or model_j.shape[0] != len(c) or model_j.shape[2] != 10:
        raise ValueError("model_jacobian_fn must return [Q,M,10]")
    result: dict[str, Any] = {"model_self_consistency": {}, "native_physical_accuracy": None}
    for h in step_sizes:
        fd = finite_difference_jacobian(moment_fn, c, step=float(h))
        result["model_self_consistency"][str(float(h))] = compare_jacobians(
            model_j, fd, component_scales=component_scales
        )
    if native_jacobian is not None:
        result["native_physical_accuracy"] = compare_jacobians(
            model_j, native_jacobian, component_scales=component_scales
        )
        if directions is not None:
            result["native_physical_accuracy"]["newton_direction"] = directional_jacobian_error(
                model_j, native_jacobian, directions
            )
    elif directions is not None:
        raise ValueError("directions require native_jacobian")
    return result
