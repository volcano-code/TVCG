"""Preflight and one-batch CUDA smoke test for Unified Training V13."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import (
    architecture_manifest,
    build_model,
)
from .train_unified import MomentNormalizer, _batch_tensors
from .unified_corpus import UnifiedOperatorCorpus


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_preflight(
    manifest: str | Path,
    *,
    config_name: str = "full_v1_candidate",
    device: str = "cuda",
    grid_size: int = 64,
    run_update: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    report: dict[str, Any] = {
        "schema": "tvcg.unified-training-v13.preflight.v1",
        "status": "running",
        "config_name": config_name,
        "device": str(device_obj),
        "checks": [],
    }

    def check(name: str, **details: Any) -> None:
        report["checks"].append({"name": name, "status": "pass", **details})

    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        props = torch.cuda.get_device_properties(device_obj)
        report["gpu"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "torch_cuda": torch.version.cuda,
            "allocated_bytes_before": torch.cuda.memory_allocated(device_obj),
        }
    elif device_obj.type != "cpu":
        raise ValueError("preflight supports cpu or cuda devices")

    corpus = UnifiedOperatorCorpus(manifest, grid_size=grid_size)
    summary = corpus.split_summary()
    if not corpus.splits["train"] or not corpus.splits["val"] or not corpus.splits["test"]:
        raise ValueError("train/val/test splits must all be nonempty")
    check(
        "manifest_and_source_groups",
        manifest_sha256=corpus.manifest_sha256,
        split_summary=summary,
    )

    normalizer = MomentNormalizer.fit(corpus)
    check("train_only_normalizer", query_count=normalizer.count,
          scales=normalizer.scales.tolist())

    manifest_config = architecture_manifest(config_name)
    model = build_model(config_name, device=device_obj)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    check("model_factory", architecture_fingerprint=manifest_config["config_fingerprint"],
          parameter_count=parameter_count, trainable_parameters=trainable_count,
          modules=manifest_config["modules"])

    first_row = corpus.splits["train"][0]
    batch = corpus.to_training_batch(first_row)
    tensors = _batch_tensors(batch, device_obj)
    check(
        "first_training_batch",
        geometry_id=first_row["id"],
        source_group_id=first_row["group_id"],
        queries=int(tensors["coefficients"].shape[0]),
        moment_labels=int(tensors["valid_mask"].sum()),
        full_jacobian_labels=int(tensors.get("jacobian_valid_mask", torch.zeros(0, dtype=torch.bool, device=device_obj)).sum()),
        restricted_jacobian_labels=int(tensors.get("restricted_valid", torch.zeros(0, dtype=torch.bool, device=device_obj)).sum()),
        native_resolution=int(batch["geometry"]["native_resolution"]),
        native_intervals=int(len(batch["geometry"]["columns"])),
    )

    if run_update:
        model.train()
        optimizer = model.build_optimizer()
        scales = torch.as_tensor(normalizer.scales, device=device_obj, dtype=torch.float64)
        training_contract = model.training_contract(moment_scale=scales)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if device_obj.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_obj)
        started = time.perf_counter()
        cache = model.encode_geometry(batch["geometry"], inference=False)
        metrics = model.train_step(
            optimizer,
            cache,
            tensors["coefficients"],
            tensors["target_moments"],
            tensors.get("target_jacobian"),
            moment_scale=scales,
            valid_mask=tensors.get("valid_mask"),
            jacobian_valid_mask=tensors.get("jacobian_valid_mask"),
            restricted_basis=tensors.get("restricted_basis"),
            restricted_jacobian=tensors.get("restricted_jacobian"),
            restricted_valid=tensors.get("restricted_valid"),
            restricted_dimension=tensors.get("restricted_dimension"),
        )
        _sync(device_obj)
        elapsed = time.perf_counter() - started
        delta_sq = 0.0
        for name, parameter in model.named_parameters():
            if name in before:
                delta_sq += float((parameter.detach() - before[name]).to(torch.float64).square().sum())
        if not math.isfinite(delta_sq) or delta_sq <= 0:
            raise RuntimeError("smoke update did not produce a finite nonzero parameter update")
        if not metrics.get("cache_invalidated"):
            raise RuntimeError("train_step did not report geometry-cache invalidation")
        check(
            "one_optimizer_update",
            loss=metrics["loss"],
            moment_loss=metrics["moment"],
            jacobian_loss=metrics["jacobian"],
            gradient_norm=metrics["gradient_norm"],
            parameter_delta_l2=math.sqrt(delta_sq),
            elapsed_seconds=elapsed,
            peak_memory_bytes=(torch.cuda.max_memory_allocated(device_obj)
                               if device_obj.type == "cuda" else None),
        )
        try:
            model.evaluate_queries(cache, tensors["coefficients"], with_jacobian=True)
        except ValueError as exc:
            if "stale" not in str(exc) and "cache" not in str(exc):
                raise
            check("cache_invalidated_after_update", error=str(exc))
        else:
            raise RuntimeError("stale geometry cache was unexpectedly accepted")

        if output is not None:
            checkpoint = Path(output)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            model.save_checkpoint(
                checkpoint,
                optimizer=optimizer,
                scheduler=scheduler,
                data_fingerprint=corpus.manifest_sha256,
                training_contract=training_contract,
                extra={"smoke_update": True, "global_step": 1},
            )
            check("checkpoint_written", path=str(checkpoint), bytes=checkpoint.stat().st_size)

    report["status"] = "pass"
    report["summary"] = {
        "checks_passed": len(report["checks"]),
        "smoke_update_executed": bool(run_update),
        "training_started": False,
        "formal_training_claim": False,
    }
    if output is not None and not run_update:
        report_path = Path(output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified V13 manifest/model/CUDA preflight")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="full_v1_candidate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--smoke-update", action="store_true")
    parser.add_argument("--output", required=True, help="JSON report path; checkpoint path when --smoke-update")
    args = parser.parse_args()
    report = run_preflight(
        args.manifest,
        config_name=args.config,
        device=args.device,
        grid_size=args.grid_size,
        run_update=args.smoke_update,
        output=args.output,
    )
    if args.smoke_update:
        report_path = Path(args.output).with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Smoke checkpoint: {args.output}")
        print(f"Preflight report: {report_path}")
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
