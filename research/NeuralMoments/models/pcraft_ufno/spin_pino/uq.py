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


def uncertainty_risk_coverage(mean: Any, target: Any, std: Any, *,
                              mask: Any | None = None,
                              group_ids: Any | None = None,
                              retained_fractions: tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)) -> dict[str, Any]:
    """Risk-coverage curve using per-query mean predictive std as abstention score.

    Lower predicted uncertainty is retained first. Risk is RMSE of normalized
    residual components over retained queries. Group macro results prevent
    large geometry families dominating the summary.
    """
    mu=np.asarray(mean,dtype=np.float64); y=np.asarray(target,dtype=np.float64)
    sigma=np.asarray(std,dtype=np.float64)
    if mu.shape!=y.shape or mu.shape!=sigma.shape or mu.ndim<2:
        raise ValueError("mean, target and std must share [N,...] shape")
    if not (np.isfinite(mu).all() and np.isfinite(y).all() and np.isfinite(sigma).all()):
        raise FloatingPointError("risk-coverage inputs must be finite")
    if (sigma<=0).any(): raise ValueError("std must be strictly positive")
    keep=np.ones(mu.shape[0],dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if keep.shape!=(mu.shape[0],) or not keep.any():
        raise ValueError("mask must select at least one sample and have shape [N]")
    fractions=tuple(float(v) for v in retained_fractions)
    if not fractions or any(not 0<v<=1 for v in fractions) or tuple(sorted(fractions))!=fractions:
        raise ValueError("retained_fractions must be sorted values in (0,1]")
    mu,y,sigma=mu[keep],y[keep],sigma[keep]
    residual=y-mu
    score=sigma.reshape(len(sigma),-1).mean(-1)
    per_query_mse=np.square(residual).reshape(len(residual),-1).mean(-1)
    order=np.argsort(score,kind="stable")
    curve=[]
    for fraction in fractions:
        count=max(1,int(math.ceil(fraction*len(order))))
        chosen=order[:count]
        curve.append({"retained_fraction":fraction,"retained_count":count,
            "mean_uncertainty":float(score[chosen].mean()),
            "rmse_risk":float(np.sqrt(per_query_mse[chosen].mean()))})
    result={"status":"PASS","count":len(order),"curve":curve,
        "risk_at_full_coverage":curve[-1]["rmse_risk"]}
    if group_ids is not None:
        groups=np.asarray(group_ids)
        if groups.shape!=(len(keep),): raise ValueError("group_ids must have shape [N]")
        selected=groups[keep]
        group_curves=[]
        for gid in sorted(set(selected.tolist()),key=str):
            idx=selected==gid
            if idx.sum()<1: continue
            sub=uncertainty_risk_coverage(mu[idx],y[idx],sigma[idx],
                retained_fractions=fractions)
            group_curves.append({"group_id":str(gid),"count":int(idx.sum()),"curve":sub["curve"]})
        result["groups"]=group_curves
        result["group_macro_risk_curve"]=[
            {"retained_fraction":fraction,
             "rmse_risk":float(np.mean([g["curve"][i]["rmse_risk"] for g in group_curves]))}
            for i,fraction in enumerate(fractions)]
    return result


def calibration_reliability_bins(mean: Any, target: Any, std: Any, *,
                                 mask: Any | None = None, bins: int = 10) -> dict[str, Any]:
    """Quantile-bin predictive std against empirical absolute residual scale.

    Reports predicted RMS std and observed RMS residual per bin. This is a
    diagnostic reliability table, not a replacement for interval coverage.
    """
    mu=np.asarray(mean,dtype=np.float64); y=np.asarray(target,dtype=np.float64)
    sigma=np.asarray(std,dtype=np.float64)
    if mu.shape!=y.shape or mu.shape!=sigma.shape or mu.ndim<2:
        raise ValueError("mean, target and std must share [N,...] shape")
    if not (np.isfinite(mu).all() and np.isfinite(y).all() and np.isfinite(sigma).all()):
        raise FloatingPointError("reliability inputs must be finite")
    if (sigma<=0).any(): raise ValueError("std must be strictly positive")
    if isinstance(bins,bool) or not isinstance(bins,int) or bins<1: raise ValueError("bins must be positive")
    keep=np.ones(mu.shape[0],dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if keep.shape!=(len(mu),) or not keep.any(): raise ValueError("mask must select samples")
    score=sigma[keep].reshape(int(keep.sum()),-1).mean(-1)
    error=np.square(y[keep]-mu[keep]).reshape(int(keep.sum()),-1).mean(-1)
    order=np.argsort(score,kind="stable")
    chunks=np.array_split(order,min(bins,len(order)))
    table=[]
    for chunk in chunks:
        table.append({"count":int(len(chunk)),"predicted_rms_std":float(np.sqrt(np.mean(np.square(score[chunk])))),
                      "observed_rms_error":float(np.sqrt(np.mean(error[chunk])))})
    return {"status":"PASS","count":int(len(order)),"bins":table}


def event_classification_metrics(logits: Any, targets: Any, *,
                                 mask: Any | None = None,
                                 group_ids: Any | None = None,
                                 bins: int = 10) -> dict[str, Any]:
    """Event-validity binary metrics; target is teacher event_valid, not derivative_valid."""
    x=np.asarray(logits,dtype=np.float64).reshape(-1)
    y=np.asarray(targets,dtype=bool).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all():
        raise ValueError("event logits/targets must be finite matching vectors")
    keep=np.ones(len(y),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if keep.shape!=y.shape or not keep.any(): raise ValueError("event mask must select samples")
    x=x[keep]; y=y[keep]
    p=1.0/(1.0+np.exp(-np.clip(x,-60,60)))
    pred=p>=.5
    tp=int(np.sum(pred & y)); fp=int(np.sum(pred & ~y)); fn=int(np.sum(~pred & y)); tn=int(np.sum(~pred & ~y))
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    result={"status":"PASS","count":len(y),"positive_count":int(y.sum()),"negative_count":int((~y).sum()),
        "brier":float(np.mean(np.square(p-y.astype(float)))),
        "precision":float(precision),"recall":float(recall),
        "f1":float(2*precision*recall/max(precision+recall,1e-15)),
        "confusion":{"tp":tp,"fp":fp,"fn":fn,"tn":tn},
        "auroc":None,"auprc":None,"ece":None}
    if y.any() and (~y).any():
        order=np.argsort(p,kind="stable"); ranks=np.empty(len(p),dtype=float);ranks[order]=np.arange(1,len(p)+1)
        pos_ranks=ranks[y]
        auc=float((pos_ranks.sum()-len(pos_ranks)*(len(pos_ranks)+1)/2)/(len(pos_ranks)*(len(y)-len(pos_ranks))))
        result["auroc"]=auc
        thresholds=np.unique(p)[::-1]
        precisions=[];recalls=[]
        for threshold in thresholds:
            selected=p>=threshold
            precisions.append(np.sum(selected & y)/max(np.sum(selected),1))
            recalls.append(np.sum(selected & y)/np.sum(y))
        result["auprc"]=float(np.sum(np.diff(np.r_[0.,recalls])*np.asarray(precisions)))
    chunks=np.array_split(np.argsort(p,kind="stable"),min(max(int(bins),1),len(p)))
    ece=0.
    for chunk in chunks:
        if len(chunk):
            ece+=len(chunk)/len(p)*abs(float(p[chunk].mean())-float(y[chunk].mean()))
    result["ece"]=float(ece)
    if group_ids is not None:
        gids=np.asarray(group_ids)
        if gids.shape!=(len(keep),):raise ValueError("group_ids must have shape [N]")
        selected=gids[keep]
        result["groups"]=[]
        for gid in sorted(set(selected.tolist()),key=str):
            idx=selected==gid
            sub=event_classification_metrics(x[idx],y[idx],bins=bins)
            result["groups"].append({"group_id":str(gid),**sub})
    return result
