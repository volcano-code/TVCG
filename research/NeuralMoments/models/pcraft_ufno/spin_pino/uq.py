"""Gaussian residual UQ primitives and calibration metrics for Moment/Jacobian outputs.

This module does not fabricate uncertainty labels or train a UQ head. A model
must provide predictive means and positive predictive standard deviations.
Calibration must be measured on held-out geometry/source-family groups.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor


_NOMINAL_Z = {
    0.50: 0.6744897501960817,
    0.68: 0.994457883209753,
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def gaussian_nll(mean: Tensor, target: Tensor, log_std: Tensor, *,
                 min_log_std: float = -7.0, max_log_std: float = 4.0,
                 reduction: str = "mean") -> Tensor:
    """Elementwise Gaussian NLL with bounded log standard deviation."""
    if mean.shape != target.shape or mean.shape != log_std.shape:
        raise ValueError("mean, target and log_std must have identical shapes")
    if not (math.isfinite(min_log_std) and math.isfinite(max_log_std)
            and min_log_std < max_log_std):
        raise ValueError("invalid log_std bounds")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean or sum")
    if not torch.isfinite(mean).all() or not torch.isfinite(target).all() or not torch.isfinite(log_std).all():
        raise FloatingPointError("Gaussian NLL inputs must be finite")
    bounded = log_std.clamp(min_log_std, max_log_std)
    inv_var = torch.exp(-2.0 * bounded)
    values = 0.5 * ((target - mean).square() * inv_var + 2.0 * bounded + math.log(2.0 * math.pi))
    if reduction == "none":
        return values
    return values.mean() if reduction == "mean" else values.sum()


def gaussian_calibration_metrics(mean: Any, target: Any, std: Any, *,
                                 mask: Any | None = None,
                                 group_ids: Any | None = None,
                                 nominal_levels: tuple[float, ...] = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99),
                                 min_std: float = 1e-8) -> dict[str, Any]:
    """Evaluate Gaussian NLL, coverage, interval width and standardized residuals.

    The first axis is the query/sample axis; remaining axes are output dimensions.
    Optional group_ids enables source/geometry-group macro summaries.
    """
    mu = np.asarray(mean, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    sigma = np.asarray(std, dtype=np.float64)
    if mu.shape != y.shape or mu.shape != sigma.shape or mu.ndim < 1:
        raise ValueError("mean, target and std must have identical non-scalar shapes")
    if not (np.isfinite(mu).all() and np.isfinite(y).all() and np.isfinite(sigma).all()):
        raise FloatingPointError("calibration arrays must be finite")
    if min_std <= 0 or not math.isfinite(min_std):
        raise ValueError("min_std must be finite and positive")
    if (sigma <= 0).any():
        raise ValueError("std must be strictly positive")
    keep = np.ones(mu.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if keep.shape != (mu.shape[0],):
        raise ValueError("mask must have shape [N]")
    if not keep.any():
        raise ValueError("calibration mask selects no samples")
    mu, y, sigma = mu[keep], y[keep], np.maximum(sigma[keep], min_std)
    residual = y - mu
    z = residual / sigma
    nll = 0.5 * (np.square(z) + 2.0 * np.log(sigma) + math.log(2.0 * math.pi))
    levels = tuple(float(v) for v in nominal_levels)
    if not levels or any((not 0 < v < 1) for v in levels):
        raise ValueError("nominal_levels must lie strictly between zero and one")
    coverage = {}
    widths = {}
    errors = []
    for level in levels:
        if level not in _NOMINAL_Z:
            raise ValueError(f"unsupported nominal level {level}; supported={tuple(_NOMINAL_Z)}")
        zcrit = _NOMINAL_Z[level]
        observed = float((np.abs(z) <= zcrit).mean())
        coverage[str(level)] = observed
        widths[str(level)] = float((2.0 * zcrit * sigma).mean())
        errors.append(abs(observed - level))
    result = {
        "status": "PASS",
        "count": int(mu.shape[0]),
        "scalar_count": int(mu.size),
        "gaussian_nll": float(nll.mean()),
        "coverage": coverage,
        "mean_interval_width": widths,
        "calibration_error": float(np.mean(errors)),
        "standardized_residual": {
            "mean": float(z.mean()),
            "std": float(z.std()),
            "rmse": float(np.sqrt(np.mean(np.square(z)))),
            "abs_p95": float(np.quantile(np.abs(z), 0.95)),
        },
    }
    if group_ids is not None:
        gids = np.asarray(group_ids)
        if gids.shape != (keep.shape[0],):
            raise ValueError("group_ids must have shape [N]")
        selected = gids[keep]
        group_reports = []
        for gid in sorted(set(selected.tolist()), key=str):
            idx = selected == gid
            group_reports.append({
                "group_id": str(gid),
                "count": int(idx.sum()),
                "gaussian_nll": float(nll[idx].mean()),
                "coverage": {
                    str(level): float((np.abs(z[idx]) <= _NOMINAL_Z[level]).mean())
                    for level in levels
                },
            })
        result["groups"] = group_reports
        result["source_group_count"] = len(group_reports)
        result["group_macro_gaussian_nll"] = float(np.mean([g["gaussian_nll"] for g in group_reports]))
        result["group_macro_coverage"] = {
            str(level): float(np.mean([g["coverage"][str(level)] for g in group_reports]))
            for level in levels
        }
    return result


def joint_uq_report(*, moment: dict[str, Any] | None,
                    jacobian: dict[str, Any] | None,
                    jacobian_status: str = "PENDING") -> dict[str, Any]:
    """Keep Moment and Jacobian uncertainty distinct; never infer missing UQ."""
    if jacobian_status not in {"PASS", "PENDING", "BLOCKED", "NOT_APPLICABLE"}:
        raise ValueError("invalid jacobian_status")
    return {
        "moment_uq": moment if moment is not None else {"status": "PENDING", "reason": "Moment UQ predictions not supplied"},
        "jacobian_uq": jacobian if jacobian is not None else {"status": jacobian_status, "reason": "Jacobian UQ predictions/labels not supplied"},
        "joint_calibration_claim": False,
    }
