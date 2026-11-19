"""Paired, inference-only fusion ablation harness (Phase 2.8.5.5.1).

This module measures already-constructed variants on identical geometries and
queries. It never trains, mutates weights, or claims a winner. Geometry encode
and query evaluation timings are reported separately.
"""
from __future__ import annotations

from time import perf_counter
from typing import Mapping, Any

import torch
from torch import Tensor, nn


def _state_equal(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k].detach().cpu(), sb[k].detach().cpu()) for k in sa)


def _parameter_counts(model: nn.Module) -> dict[str, int]:
    total = list(model.parameters())
    trainable = [p for p in total if p.requires_grad]
    return {
        "total_parameters": sum(p.numel() for p in total),
        "trainable_parameters": sum(p.numel() for p in trainable),
    }


def _relative_l2(pred: Tensor, target: Tensor) -> float:
    denom = torch.linalg.vector_norm(target).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(pred - target) / denom).detach().cpu())


@torch.no_grad()
def run_fusion_ablation(
    variants: Mapping[str, nn.Module],
    geometry: Mapping[str, Any],
    coefficients: Tensor,
    *,
    reference_moments: Tensor | None = None,
    reference_jacobian: Tensor | None = None,
    encode_chunk: int | None = None,
    warmup_runs: int = 1,
    timed_runs: int = 3,
) -> dict[str, Any]:
    """Evaluate variants fairly on the same geometry/query batch.

    All variants must have byte-identical shared-encoder and SQNO state_dicts.
    Each variant encodes the same geometry independently, then evaluates the
    same query tensor. Timings are per variant and separated into encode and
    query stages. CUDA synchronization is used only when CUDA is active.
    """
    if not variants:
        raise ValueError("at least one fusion variant is required")
    if isinstance(warmup_runs,bool) or warmup_runs<0 or isinstance(timed_runs,bool) or timed_runs<1:
        raise ValueError("warmup_runs must be >=0 and timed_runs must be >=1")
    names = list(variants)
    anchor = variants[names[0]]
    for name, model in variants.items():
        if not hasattr(model, "shared_encoder") or not hasattr(model, "sqno"):
            raise TypeError(f"{name} is not a UnifiedM80SQNO-compatible model")
        if not _state_equal(anchor.shared_encoder, model.shared_encoder):
            raise ValueError("shared encoder weights differ across variants")
        if not _state_equal(anchor.sqno, model.sqno):
            raise ValueError("SQNO weights differ across variants")
        model.eval()
    q = torch.as_tensor(coefficients, dtype=torch.float64)
    if q.ndim != 2 or q.shape[-1] != 10 or not torch.isfinite(q).all():
        raise ValueError("coefficients must be finite [Q,10]")
    if reference_moments is not None and reference_moments.shape != (len(q), 10):
        raise ValueError("reference_moments must have shape [Q,10]")
    if reference_jacobian is not None and reference_jacobian.shape != (len(q), 10, 10):
        raise ValueError("reference_jacobian must have shape [Q,10,10]")

    reports: dict[str, Any] = {}
    geometry_ids = set()
    for name, model in variants.items():
        device = next(model.parameters()).device
        def sync():
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device)
        def evaluate_once():
            sync()
            t0 = perf_counter()
            cache = model.encode_geometry(geometry, chunk=encode_chunk, inference=True)
            sync()
            encode_elapsed = perf_counter() - t0
            geometry_ids.add(cache.geometry_id)
            t1 = perf_counter()
            result = model.evaluate_queries(cache, q.to(device), create_graph=False)
            sync()
            query_elapsed = perf_counter() - t1
            return cache, result, encode_elapsed, query_elapsed
        for _ in range(warmup_runs):
            evaluate_once()
        encode_times=[]; query_times=[]; final_cache=None; final_result=None
        if torch.cuda.is_available() and device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(timed_runs):
            final_cache,final_result,et,qt=evaluate_once()
            encode_times.append(et); query_times.append(qt)
        result=final_result
        moments=result["moments"].detach().cpu()
        jacobian=result["jacobian_coefficients"].detach().cpu()
        encode_mean=sum(encode_times)/len(encode_times)
        query_mean=sum(query_times)/len(query_times)
        report={
            **_parameter_counts(model),
            "geometry_encode_seconds_mean":encode_mean,
            "geometry_encode_seconds_min":min(encode_times),
            "query_evaluation_seconds_mean":query_mean,
            "query_evaluation_seconds_min":min(query_times),
            "total_seconds_mean":encode_mean+query_mean,
            "timed_runs":timed_runs,
            "warmup_runs":warmup_runs,
            "moment_relative_l2":None if reference_moments is None else _relative_l2(moments,reference_moments.cpu()),
            "jacobian_relative_l2":None if reference_jacobian is None else _relative_l2(jacobian,reference_jacobian.cpu()),
            "fusion_mode":model.fusion_mode,
            "interaction_variant":model.interaction_variant,
            "gno_enabled":bool(model.shared_encoder.config.gno_enabled),
            "geometry_attention_enabled":model.geometry_physics_attention is not None,
            "cross_scale_attention_enabled":model.cross_scale_attention is not None,
            "fusion_weight_mean":float(result["fusion_weight"].float().mean().cpu()),
            "fusion_weight_std":float(result["fusion_weight"].float().std(unbiased=False).cpu()),
            "fusion_gate_jacobian_norm_mean":float(result["fusion_gate_jacobian_norm"].float().mean().cpu()),
            "parallel_attention_gate":None if "parallel_attention_gate" not in result else float(result["parallel_attention_gate"].float().mean().cpu()),
            "attention_diagnostics":{
                key:float(value.float().mean().cpu()) for key,value in result.items()
                if key.endswith("_entropy") or key.endswith("_effective_support")
            },
            "peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated(device))
                if torch.cuda.is_available() and device.type=="cuda" else None,
        }
        reports[name]=report
    if len(geometry_ids) != 1:
        raise RuntimeError("variants encoded different geometry identities")
    return {
        "protocol": "paired_frozen_weights_same_geometry_same_queries",
        "fairness_contract": {
            "shared_encoder_weights_identical": True,
            "sqno_weights_identical": True,
            "geometry_identity_identical": True,
            "query_tensor_identical": True,
            "geometry_encode_timed_separately": True,
            "query_evaluation_timed_separately": True,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "training_performed": False,
        },
        "geometry_id": next(iter(geometry_ids)),
        "query_count": len(q),
        "variants": reports,
        "claims": {
            "accuracy_comparison": reference_moments is not None,
            "jacobian_comparison": reference_jacobian is not None,
            "speed_comparison": True,
            "training_performed": False,
        },
    }
