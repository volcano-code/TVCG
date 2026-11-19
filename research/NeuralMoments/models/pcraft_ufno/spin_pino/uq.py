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



def _binary_auc_ap(probability: np.ndarray, target: np.ndarray) -> tuple[float | None,float | None]:
    """Tie-correct AUROC and step-wise average precision."""
    positives=int(target.sum()); negatives=int((~target).sum())
    if positives==0 or negatives==0:return None,None
    order=np.argsort(probability,kind="mergesort")
    sorted_p=probability[order]
    # Average ranks for ties (Mann-Whitney U).
    ranks=np.empty(len(target),dtype=np.float64)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and sorted_p[j]==sorted_p[i]:j+=1
        ranks[order[i:j]]=(i+1+j)/2.0
        i=j
    auc=float((ranks[target].sum()-positives*(positives+1)/2)/(positives*negatives))
    descending=np.argsort(-probability,kind="mergesort")
    y=target[descending]
    tp=np.cumsum(y); fp=np.cumsum(~y)
    # Evaluate only at distinct score boundaries.
    boundary=np.r_[np.where(np.diff(probability[descending])!=0)[0],len(y)-1]
    recall=tp[boundary]/positives
    precision=tp[boundary]/(tp[boundary]+fp[boundary])
    previous=np.r_[0.0,recall[:-1]]
    ap=float(np.sum((recall-previous)*precision))
    return auc,ap


def _event_metrics_from_probability(probability: np.ndarray,target: np.ndarray,*,threshold:float,bins:int)->dict[str,Any]:
    pred=probability>=threshold
    tp=int(np.sum(pred&target));fp=int(np.sum(pred&~target));fn=int(np.sum(~pred&target));tn=int(np.sum(~pred&~target))
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    auc,ap=_binary_auc_ap(probability,target)
    chunks=np.array_split(np.argsort(probability,kind="stable"),min(max(int(bins),1),len(target)))
    ece=sum(len(ix)/len(target)*abs(float(probability[ix].mean())-float(target[ix].mean())) for ix in chunks if len(ix))
    return {"status":"PASS","count":len(target),"positive_count":int(target.sum()),"negative_count":int((~target).sum()),
        "threshold":float(threshold),"brier":float(np.mean(np.square(probability-target.astype(float)))),
        "precision":float(precision),"recall":float(recall),"f1":float(2*precision*recall/max(precision+recall,1e-15)),
        "confusion":{"tp":tp,"fp":fp,"fn":fn,"tn":tn},"auroc":auc,"auprc":ap,"ece":float(ece)}


def event_classification_metrics(logits: Any, targets: Any, *,
                                 mask: Any | None = None,
                                 group_ids: Any | None = None,
                                 bins: int = 10,
                                 threshold: float = 0.5) -> dict[str, Any]:
    """Event metrics for teacher event_valid labels; never use derivative_valid as target."""
    x=np.asarray(logits,dtype=np.float64).reshape(-1)
    y=np.asarray(targets,dtype=bool).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all():raise ValueError("event logits/targets must be finite matching vectors")
    if not 0<threshold<1 or not math.isfinite(threshold):raise ValueError("threshold must lie strictly between zero and one")
    if isinstance(bins,bool) or not isinstance(bins,int) or bins<1:raise ValueError("bins must be a positive integer")
    keep=np.ones(len(y),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if keep.shape!=y.shape or not keep.any():raise ValueError("event mask must select samples")
    x=x[keep];y=y[keep];p=1/(1+np.exp(-np.clip(x,-60,60)))
    result=_event_metrics_from_probability(p,y,threshold=threshold,bins=bins)
    if group_ids is not None:
        gids=np.asarray(group_ids)
        if gids.shape!=(len(keep),):raise ValueError("group_ids must have shape [N]")
        selected=gids[keep];groups=[]
        for gid in sorted(set(selected.tolist()),key=str):
            idx=selected==gid
            groups.append({"group_id":str(gid),**_event_metrics_from_probability(p[idx],y[idx],threshold=threshold,bins=bins)})
        result["groups"]=groups
        valid_auc=[g["auroc"] for g in groups if g["auroc"] is not None]
        result["group_macro_auroc"]=float(np.mean(valid_auc)) if valid_auc else None
        result["group_macro_brier"]=float(np.mean([g["brier"] for g in groups]))
        result["group_macro_ece"]=float(np.mean([g["ece"] for g in groups]))
    return result


def select_event_threshold(logits: Any,targets: Any,*,mask:Any|None=None,
                           objective:str="f1",thresholds:tuple[float,...]|None=None)->dict[str,Any]:
    """Select a threshold on calibration data only; caller must keep test data separate."""
    x=np.asarray(logits,dtype=np.float64).reshape(-1);y=np.asarray(targets,dtype=bool).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all():raise ValueError("logits/targets must be finite matching vectors")
    keep=np.ones(len(y),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if keep.shape!=y.shape or not keep.any():raise ValueError("mask must select calibration samples")
    if objective not in {"f1","balanced_accuracy","youden"}:raise ValueError("unsupported threshold objective")
    candidates=np.asarray(thresholds if thresholds is not None else np.linspace(.01,.99,99),dtype=np.float64)
    if candidates.ndim!=1 or not len(candidates) or not np.isfinite(candidates).all() or ((candidates<=0)|(candidates>=1)).any():
        raise ValueError("threshold candidates must lie in (0,1)")
    p=1/(1+np.exp(-np.clip(x[keep],-60,60)));yt=y[keep]
    if not yt.any() or yt.all():raise ValueError("threshold selection requires both event classes in calibration data")
    reports=[]
    for th in candidates:
        pred=p>=th;tp=np.sum(pred&yt);fp=np.sum(pred&~yt);fn=np.sum(~pred&yt);tn=np.sum(~pred&~yt)
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1)
        f1=2*precision*recall/max(precision+recall,1e-15)
        score=f1 if objective=="f1" else .5*(recall+specificity)
        if objective=="youden":score=recall+specificity-1
        reports.append((float(score),float(th),float(f1),float(recall),float(specificity)))
    # Deterministic tie break: best score, then threshold closest to 0.5, then lower threshold.
    best=sorted(reports,key=lambda z:(-z[0],abs(z[1]-.5),z[1]))[0]
    return {"status":"PASS","objective":objective,"threshold":best[1],"objective_value":best[0],
            "f1":best[2],"recall":best[3],"specificity":best[4],"calibration_count":int(len(yt)),
            "warning":"threshold selected on calibration split only; do not retune on locked test"}


def grouped_event_bootstrap(logits:Any,targets:Any,group_ids:Any,*,mask:Any|None=None,
                            n_bootstrap:int=1000,seed:int=0,threshold:float=.5,
                            confidence:float=.95)->dict[str,Any]:
    """Cluster bootstrap by geometry/source group, never by individual query."""
    x=np.asarray(logits,dtype=np.float64).reshape(-1);y=np.asarray(targets,dtype=bool).reshape(-1)
    groups=np.asarray(group_ids)
    if x.shape!=y.shape or groups.shape!=y.shape or not np.isfinite(x).all():raise ValueError("logits, targets and group_ids must be matching finite vectors")
    if isinstance(n_bootstrap,bool) or n_bootstrap<100:raise ValueError("n_bootstrap must be >=100")
    if not 0<confidence<1 or not 0<threshold<1:raise ValueError("invalid confidence or threshold")
    keep=np.ones(len(y),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if keep.shape!=y.shape or not keep.any():raise ValueError("mask must select samples")
    x=x[keep];y=y[keep];groups=groups[keep]
    unique=np.asarray(sorted(set(groups.tolist()),key=str),dtype=object)
    if len(unique)<2:raise ValueError("grouped bootstrap requires at least two geometry/source groups")
    rng=np.random.default_rng(seed);samples={"brier":[],"f1":[],"ece":[]}
    for _ in range(n_bootstrap):
        draw=rng.integers(0,len(unique),size=len(unique))
        # Keep duplicated clusters as duplicated observations.
        chunks=[np.where(groups==unique[k])[0] for k in draw]
        idx=np.concatenate(chunks)
        p=1/(1+np.exp(-np.clip(x[idx],-60,60)))
        m=_event_metrics_from_probability(p,y[idx],threshold=threshold,bins=10)
        for key in samples:samples[key].append(m[key])
    alpha=(1-confidence)/2
    intervals={key:{"lower":float(np.quantile(v,alpha)),"upper":float(np.quantile(v,1-alpha)),
                    "mean":float(np.mean(v))} for key,v in samples.items()}
    return {"status":"PASS","bootstrap_unit":"geometry_or_source_group","group_count":int(len(unique)),
            "query_count":int(len(y)),"n_bootstrap":int(n_bootstrap),"seed":int(seed),
            "confidence":float(confidence),"intervals":intervals}
