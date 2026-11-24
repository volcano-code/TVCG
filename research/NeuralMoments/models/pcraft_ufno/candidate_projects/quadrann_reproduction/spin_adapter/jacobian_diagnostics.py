"""Diagnostics of raw-coefficient Jacobians, independent of training behavior.

Model self-consistency compares a supplied Jacobian with finite differences of
the same model's Moment function. Physical accuracy requires an independent
teacher Jacobian; passing self-consistency alone does not establish it.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import torch
from torch import Tensor


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _floating_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach()


def _as_batch_coefficients(coefficients: Tensor) -> tuple[Tensor, bool]:
    coefficients = _floating_tensor(coefficients, "coefficients")
    single = coefficients.ndim == 1
    values = coefficients.unsqueeze(0) if single else coefficients
    if values.ndim != 2 or values.shape[1] != 10 or values.shape[0] == 0:
        raise ValueError("coefficients must have shape [10] or nonempty [Q,10]")
    if (values.abs().amax(dim=-1) == 0).any():
        raise ValueError("coefficient vectors must be nonzero")
    return values, single


def _jacobian_pair(predicted: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
    p = _floating_tensor(predicted, "predicted Jacobian")
    r = _floating_tensor(reference, "reference Jacobian")
    if p.shape != r.shape or p.ndim != 3 or any(size == 0 for size in p.shape):
        raise ValueError("Jacobians must share nonempty [Q,M,D] shape")
    p = p.to(dtype=torch.float64)
    return p, r.to(device=p.device, dtype=torch.float64)


def _query_labels(query_ids: Sequence[str] | None, count: int) -> list[str] | None:
    if query_ids is None:
        return None
    labels = list(query_ids)
    if (isinstance(query_ids, str) or len(labels) != count
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("query_ids must contain one unique nonempty string per query")
    return labels


def finite_difference_jacobian(
    moment_fn: Callable[[Tensor], Tensor],
    coefficients: Tensor,
    *,
    step: float,
    assume_query_independence: bool = False,
) -> Tensor:
    """Central differences of d Moment[q] / d coefficients[q], shape [Q,M,10].

    The default perturbs one query at a time, holding other queries fixed. It
    extracts diagonal blocks of a batch mapping; it does not return cross-query
    derivatives. Set ``assume_query_independence=True`` only when each output
    row depends solely on its matching input row. That fast path perturbs all
    rows together and needs 20 rather than 20*Q Moment evaluations.

    ``moment_fn`` must deterministically map [Q,10] to nonempty [Q,M] with a
    consistent dtype and device. Differences accumulate in float64, without
    increasing the precision of model forward passes. The result is detached.
    """
    step = _positive_finite(step, "step")
    if not isinstance(assume_query_independence, bool):
        raise TypeError("assume_query_independence must be bool")
    c, _ = _as_batch_coefficients(coefficients)
    signature = None

    def evaluate(values: Tensor) -> Tensor:
        nonlocal signature
        output = _floating_tensor(moment_fn(values), "moment_fn output")
        if output.ndim != 2 or output.shape[0] != len(c) or output.shape[1] == 0:
            raise ValueError("moment_fn must return nonempty [Q,M] for each input batch")
        current = (output.shape, output.dtype, output.device)
        if signature is not None and current != signature:
            raise ValueError("moment_fn output shape, dtype and device must remain consistent")
        signature = current
        return output.to(dtype=torch.float64)

    query_rows = [None] if assume_query_independence else range(len(c))
    batches = []
    with torch.no_grad():
        for query in query_rows:
            columns = []
            for axis in range(c.shape[1]):
                delta = torch.zeros_like(c)
                selected = slice(None) if query is None else slice(query, query + 1)
                delta[selected, axis] = step
                plus_input, minus_input = c + delta, c - delta
                if (not torch.isfinite(plus_input).all() or not torch.isfinite(minus_input).all()
                        or (plus_input[selected, axis] == c[selected, axis]).any()
                        or (minus_input[selected, axis] == c[selected, axis]).any()):
                    raise ValueError("step must produce finite, representable coefficient perturbations")
                plus, minus = evaluate(plus_input), evaluate(minus_input)
                column = (plus[selected] - minus[selected]) / (2.0 * step)
                if not torch.isfinite(column).all():
                    raise ValueError("finite differences produced nonfinite derivatives")
                columns.append(column)
            batches.append(torch.stack(columns, dim=-1))
    return torch.cat(batches, dim=0).detach()


def _error_metrics(p: Tensor, r: Tensor, eps: float, norm_name: str) -> dict[str, Any]:
    diff = p - r
    reference_norm = torch.linalg.vector_norm(r)
    predicted_norm = torch.linalg.vector_norm(p)
    absolute = torch.linalg.vector_norm(diff)
    if not torch.isfinite(torch.stack((reference_norm, predicted_norm, absolute))).all():
        raise ValueError("Jacobian error norms overflowed; rescale inputs before comparison")
    cosine_defined = bool(reference_norm > eps and predicted_norm > eps)
    # Normalize each operand separately: clamping the product of norms makes
    # two identical small, nonzero vectors appear uncorrelated.
    cosine = ((p.flatten() / predicted_norm) @ (r.flatten() / reference_norm)).clamp(-1, 1) \
        if cosine_defined else None
    return {
        f"relative_{norm_name}": float(absolute / reference_norm.clamp_min(eps)),
        f"absolute_{norm_name}": float(absolute),
        f"reference_{norm_name}": float(reference_norm),
        f"predicted_{norm_name}": float(predicted_norm),
        "rmse": float(diff.square().mean().sqrt()),
        "mae": float(diff.abs().mean()),
        "max_absolute": float(diff.abs().max()),
        "relative_denominator_floored": bool(reference_norm < eps),
        "near_zero_reference": bool(reference_norm <= eps),
        "near_zero_prediction": bool(predicted_norm <= eps),
        "cosine_similarity": float(cosine) if cosine_defined else None,
        "cosine_defined": cosine_defined,
    }


def compare_jacobians(
    predicted: Tensor,
    reference: Tensor,
    *,
    component_scales: Tensor | None = None,
    eps: float = 1e-12,
    query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare [Q,M,D] Jacobians globally, by column, and by query.

    Relative errors use an explicitly flagged denominator floor. Cosine is
    None when either norm is <= eps, so an empty signal is never reported as
    directional agreement. Query indices always preserve input order.
    """
    eps = _positive_finite(eps, "eps")
    p, r = _jacobian_pair(predicted, reference)
    labels = _query_labels(query_ids, len(p))
    scales = None
    if component_scales is not None:
        if torch.as_tensor(component_scales).is_complex():
            raise TypeError("component_scales must be real")
        scales = torch.as_tensor(component_scales, device=p.device, dtype=torch.float64).detach()
        if scales.shape != (p.shape[1],) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("component_scales must be positive finite [M]")

    def metrics(a: Tensor, b: Tensor) -> dict[str, Any]:
        record = _error_metrics(a, b, eps, "frobenius")
        dims = tuple(range(a.ndim - 1))
        errors = torch.linalg.vector_norm(a - b, dim=dims)
        references = torch.linalg.vector_norm(b, dim=dims)
        record.update({
            "per_coefficient_column_relative": (errors / references.clamp_min(eps)).tolist(),
            "per_coefficient_column_absolute": errors.tolist(),
            "near_zero_reference_columns": torch.nonzero(references <= eps).flatten().tolist(),
        })
        if scales is not None:
            normalized = (a - b) / scales[:, None]
            record["component_scaled_rmse"] = float(normalized.square().mean().sqrt())
            record["component_scaled_mae"] = float(normalized.abs().mean())
        return record

    result = metrics(p, r)
    result.update({"queries": len(p), "moment_dimension": p.shape[1],
                   "coefficient_dimension": p.shape[2], "per_query": []})
    for index, (a, b) in enumerate(zip(p, r)):
        record = {"query_index": index, **metrics(a, b)}
        if labels is not None:
            record["query_id"] = labels[index]
        result["per_query"].append(record)
    return result


def directional_jacobian_error(
    predicted: Tensor,
    reference: Tensor,
    directions: Tensor,
    *,
    eps: float = 1e-12,
    query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare Jv for [Q,M,D] Jacobians and matched, unnormalized [Q,D] v.

    All operands detach and align to predicted.device/float64 before bmm.
    Zero directions are allowed but explicitly marked as uninformative.
    """
    eps = _positive_finite(eps, "eps")
    p, r = _jacobian_pair(predicted, reference)
    labels = _query_labels(query_ids, len(p))
    v = _floating_tensor(directions, "directions")
    if v.shape != (p.shape[0], p.shape[2]):
        raise ValueError("directions must have shape [Q,D]")
    v = v.to(device=p.device, dtype=torch.float64)
    a = torch.bmm(p, v.unsqueeze(-1)).squeeze(-1)
    b = torch.bmm(r, v.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Jacobian direction products must be finite")
    norms = torch.linalg.vector_norm(v, dim=-1)
    result = _error_metrics(a, b, eps, "l2")
    result["near_zero_direction_queries"] = torch.nonzero(norms <= eps).flatten().tolist()
    result["per_query"] = []
    for index, (ap, bp) in enumerate(zip(a, b)):
        record = {"query_index": index, "direction_norm": float(norms[index]),
                  "near_zero_direction": bool(norms[index] <= eps),
                  **_error_metrics(ap, bp, eps, "l2")}
        if labels is not None:
            record["query_id"] = labels[index]
        result["per_query"].append(record)
    return result


def audit_jacobian_paths(
    moment_fn: Callable[[Tensor], Tensor],
    model_jacobian_fn: Callable[[Tensor], Tensor],
    coefficients: Tensor,
    *,
    step_sizes: Sequence[float] = (1e-3, 3e-4, 1e-4),
    native_jacobian: Tensor | None = None,
    directions: Tensor | None = None,
    component_scales: Tensor | None = None,
    query_ids: Sequence[str] | None = None,
    assume_query_independence: bool = False,
) -> dict[str, Any]:
    """Report separate self-consistency sweeps and native-physics comparisons.

    Each step and each query is reported; no automatic pass/fail or physical
    accuracy claim is inferred from self-consistency. See the FD routine for
    the explicit query-independence optimization.
    """
    c, _ = _as_batch_coefficients(coefficients)
    labels = _query_labels(query_ids, len(c))
    steps = [_positive_finite(h, "step") for h in step_sizes]
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("step_sizes must be nonempty and unique")
    if directions is not None and native_jacobian is None:
        raise ValueError("directions require native_jacobian")
    model_j = _floating_tensor(model_jacobian_fn(c), "model_jacobian_fn output")
    if model_j.ndim != 3 or model_j.shape[0] != len(c) or model_j.shape[1] == 0 or model_j.shape[2] != 10:
        raise ValueError("model_jacobian_fn must return nonempty [Q,M,10]")
    result: dict[str, Any] = {
        "model_self_consistency": {}, "native_physical_accuracy": None,
        "self_consistency_comparison": "reported_jacobian_vs_moment_finite_difference",
        "finite_difference_semantics": (
            "independent_queries_assumed" if assume_query_independence else "batch_diagonal_blocks"),
    }
    for h in steps:
        fd = finite_difference_jacobian(moment_fn, c, step=h,
            assume_query_independence=assume_query_independence)
        result["model_self_consistency"][str(h)] = compare_jacobians(
            model_j, fd, component_scales=component_scales, query_ids=labels)
    if native_jacobian is not None:
        result["native_physical_accuracy"] = compare_jacobians(
            model_j, native_jacobian, component_scales=component_scales, query_ids=labels)
        if directions is not None:
            result["native_physical_accuracy"]["newton_direction"] = directional_jacobian_error(
                model_j, native_jacobian, directions, query_ids=labels)
            result["native_physical_accuracy"]["direction_semantics"] = (
                "caller_supplied_directions_not_verified_newton_steps")
    return result
