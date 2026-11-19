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


@torch.inference_mode()
def run_fusion_ablation(
    variants: Mapping[str, nn.Module],
    geometry: Mapping[str, Any],
    coefficients: Tensor,
    *,
    reference_moments: Tensor | None = None,
    reference_jacobian: Tensor | None = None,
    encode_chunk: int | None = None,
) -> dict[str, Any]:
    """Evaluate variants fairly on the same geometry/query batch.

    All variants must have byte-identical shared-encoder and SQNO state_dicts.
    Each variant encodes the same geometry independently, then evaluates the
    same query tensor. Timings are per variant and separated into encode and
    query stages. CUDA synchronization is used only when CUDA is active.
    """
    if not variants:
        raise ValueError("at least one fusion variant is required")
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
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = perf_counter()
        cache = model.encode_geometry(geometry, chunk=encode_chunk, inference=True)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_seconds = perf_counter() - t0
        geometry_ids.add(cache.geometry_id)

        t1 = perf_counter()
        result = model.evaluate_queries(cache, q.to(device), create_graph=False)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        query_seconds = perf_counter() - t1
        moments = result["moments"].detach().cpu()
        jacobian = result["jacobian_coefficients"].detach().cpu()
        report = {
            **_parameter_counts(model),
            "geometry_encode_seconds": encode_seconds,
            "query_evaluation_seconds": query_seconds,
            "total_seconds": encode_seconds + query_seconds,
            "moment_relative_l2": None if reference_moments is None
                else _relative_l2(moments, reference_moments.cpu()),
            "jacobian_relative_l2": None if reference_jacobian is None
                else _relative_l2(jacobian, reference_jacobian.cpu()),
            "fusion_mode": model.fusion_mode,
            "fusion_weight_mean": float(result["fusion_weight"].float().mean().cpu()),
            "fusion_weight_std": float(result["fusion_weight"].float().std(unbiased=False).cpu()),
            "fusion_gate_jacobian_norm_mean": float(
                result["fusion_gate_jacobian_norm"].float().mean().cpu()),
            "attention_diagnostics": {
                key: float(value.float().mean().cpu())
                for key, value in result.items()
                if key.endswith("_entropy") or key.endswith("_effective_support")
            },
        }
        reports[name] = report
    if len(geometry_ids) != 1:
        raise RuntimeError("variants encoded different geometry identities")
    return {
        "protocol": "paired_frozen_weights_same_geometry_same_queries",
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
