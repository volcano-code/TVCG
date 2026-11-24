"""Preflight and one-batch CUDA smoke test for Unified Training V13."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import (
    architecture_manifest,
    build_model,
)
from ..spin_adapter.memory_profile import MemoryProfile, profile_stage
from .train_unified import MomentNormalizer, _load_training_batch, _seed_all, _train_geometry
from .unified_corpus import UnifiedOperatorCorpus
from .native256_readiness import resolve_readiness


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _checkpoint_predictions(model, geometry, coefficients):
    model.eval()
    with torch.no_grad():
        cache = model.encode_geometry(geometry, inference=True)
        prediction = model.evaluate_queries(cache, coefficients, create_graph=False, with_jacobian=True)
        return {key: prediction[key].detach().cpu() for key in ("moments", "jacobian_coefficients")}


def _compare_predictions(expected, actual):
    differences = {}
    for name, baseline in expected.items():
        observed = actual[name]
        torch.testing.assert_close(observed, baseline, rtol=1e-5, atol=1e-8)
        error = (observed - baseline).abs()
        relative = error / baseline.abs().clamp_min(torch.finfo(baseline.dtype).eps)
        differences[name] = {"shape": list(baseline.shape), "max_abs": error.max().item(),
                             "max_rel": relative.max().item()}
    return differences


def run_preflight(
    manifest: str | Path,
    *,
    config_name: str = "full_v1_candidate",
    device: str = "cuda",
    grid_size: int = 64,
    run_update: bool = False,
    output: str | Path | None = None,
    allow_absolute_assets: bool = False,
    geometry_id: str | None = None,
    memory_profile: str | Path | None = None,
    seed: int | None = None,
    native_cut_contract: str | Path | None = None,
    derivative_normalization: str | None = None,
    branch_aux_weight: float | None = None,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    if seed is not None:
        _seed_all(seed)
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

    corpus = UnifiedOperatorCorpus(
        manifest,
        grid_size=grid_size,
        allow_absolute_paths=allow_absolute_assets,
    )
    summary = corpus.split_summary()
    if not corpus.splits["train"] or not corpus.splits["val"] or not corpus.splits["test"]:
        raise ValueError("train/val/test splits must all be nonempty")
    check(
        "manifest_and_source_groups",
        manifest_sha256=corpus.manifest_sha256,
        split_summary=summary,
    )

    admission, derivative_norm = resolve_readiness(corpus, config_name, native_cut_contract, derivative_normalization)
    if admission is not None:
        check("native_cut_admission", **admission)
    check("derivative_normalization", **derivative_norm)
    normalizer = MomentNormalizer.fit(corpus)
    check("train_only_normalizer", query_count=normalizer.count,
          scales=normalizer.scales.tolist())

    manifest_config = architecture_manifest(config_name)
    model = build_model(config_name, device=device_obj)
    if branch_aux_weight is None:branch_aux_weight=getattr(model,"default_branch_aux_weight",0.25)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    check("model_factory", architecture_fingerprint=manifest_config["config_fingerprint"],
          parameter_count=parameter_count, trainable_parameters=trainable_count,
          modules=manifest_config["modules"])

    if geometry_id is None:
        row = corpus.splits["train"][0]
    else:
        row = next((r for r in corpus.splits["train"] if r["id"] == geometry_id), None)
        if row is None:
            raise ValueError(f"geometry is absent from the training split: {geometry_id}")
    profile = MemoryProfile(device_obj, memory_profile) if memory_profile else None
    with profile if profile is not None else nullcontext():
        if profile is not None:
            profile.metadata.update(geometry_id=row["id"], manifest_sha256=corpus.manifest_sha256,
                                    geometry_sha256=row["geometry_sha256"], labels_sha256=row["labels_sha256"],
                                    config=config_name, seed=seed, model_manifest=manifest_config)
        batch, tensors = _load_training_batch(corpus, row, device_obj)
        moment_count = int(tensors["valid_mask"].sum())
        full_count = int(tensors["jacobian_valid_mask"].sum()) if "jacobian_valid_mask" in tensors else 0
        restricted = tensors.get("restricted_valid")
        dims = tensors.get("restricted_dimension")
        by_dim = ({str(int(d)): int(((dims == d) & restricted).sum()) for d in dims[restricted].unique()}
                  if restricted is not None and bool(restricted.any()) else {})
        query_count = len(tensors["coefficients"])
        support = {"queries": query_count, "moment": moment_count, "full_J": full_count}
        support.update({f"restricted{d}": count for d, count in by_dim.items()})
        advertised = row.get("label_support", {})
        for key in set(advertised) & ({"queries", "moment", "full_J"} |
                                     {k for k in advertised if k.startswith("restricted") and k[10:].isdigit()}):
            if support.get(key, 0) != int(advertised[key]):
                raise ValueError(f"{row['id']}: manifest label support disagrees for {key}")
        batch_check = dict(
            geometry_id=row["id"], source_group_id=row["group_id"],
            geometry_sha256=row["geometry_sha256"], labels_sha256=row["labels_sha256"],
            queries=query_count, moment_labels=moment_count, full_jacobian_labels=full_count,
            restricted_jacobian_labels=sum(by_dim.values()), restricted_by_dimension=by_dim,
            native_resolution=int(batch["geometry"]["native_resolution"]),
            native_intervals=len(batch["geometry"]["columns"]), full_batch=True,
        )
        check("first_training_batch" if geometry_id is None else "selected_training_batch", **batch_check)
        if profile is not None:
            profile.metadata["batch"] = batch_check

        if run_update:
            model.train()
            optimizer = model.build_optimizer()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            scales = torch.as_tensor(normalizer.scales, device=device_obj, dtype=torch.float64)
            derivative_options = {key: torch.as_tensor(derivative_norm[key], device=device_obj, dtype=torch.float64)
                                  for key in ("jacobian_scale", "restricted_scale")} if derivative_norm["mode"] != "legacy" else {}
            training_contract = model.training_contract(moment_scale=scales, branch_aux_weight=branch_aux_weight, **derivative_options)
            before = {name: p.detach().to("cpu", copy=True)
                      for name, p in model.named_parameters() if p.requires_grad}
            if device_obj.type == "cuda" and profile is None:
                torch.cuda.reset_peak_memory_stats(device_obj)
            with profile_stage("geometry_metadata_cache", phase="cache"):
                geometry = model.prepare_geometry(batch["geometry"])
            _sync(device_obj)
            started = time.perf_counter()
            with profile_stage("one_optimizer_update"):
                cache, metrics = _train_geometry(model, optimizer, geometry, tensors, moment_scale=scales, branch_aux_weight=branch_aux_weight, **derivative_options)
            _sync(device_obj)
            elapsed = time.perf_counter() - started
            peak_allocated = (profile.peak_allocated if profile is not None else torch.cuda.max_memory_allocated(device_obj)
                              ) if device_obj.type == "cuda" else None
            peak_reserved = (profile.peak_reserved if profile is not None else torch.cuda.max_memory_reserved(device_obj)
                             ) if device_obj.type == "cuda" else None
            delta_sq = 0.
            changed = 0
            for name, p in model.named_parameters():
                if name in before:
                    delta = (p.detach().cpu() - before[name]).abs().double().square().sum().item()
                    changed += int(delta > 0)
                    delta_sq += delta
            del before
            if not math.isfinite(delta_sq) or delta_sq <= 0:
                raise RuntimeError("smoke update did not produce a finite nonzero parameter update")
            if not metrics.get("cache_invalidated"):
                raise RuntimeError("train_step did not report geometry-cache invalidation")
            consumed = (metrics["valid_queries"], metrics["valid_jacobian_queries"],
                        metrics["valid_restricted_queries"], metrics["restricted_dimensions"])
            expected = (moment_count, full_count, sum(by_dim.values()),
                        sum(int(d) * n for d, n in by_dim.items()))
            if consumed != expected:
                raise RuntimeError(f"train_step consumed {consumed}, expected {expected}")
            if not all(math.isfinite(metrics[k]) for k in ("loss", "moment", "jacobian", "gradient_norm")):
                raise FloatingPointError("nonfinite smoke training metrics")
            update = dict(
                loss=metrics["loss"], moment_loss=metrics["moment"], jacobian_loss=metrics["jacobian"],
                full_jacobian_loss=metrics["full_jacobian"], restricted_jacobian_loss=metrics["restricted_jacobian"],
                gradient_norm=metrics["gradient_norm"], parameter_delta_l2=math.sqrt(delta_sq),
                changed_parameter_tensors=changed, elapsed_seconds=elapsed,
                peak_memory_bytes=peak_allocated, peak_reserved_bytes=peak_reserved, optimizer_steps=1,
                consumed_moment_queries=consumed[0], consumed_full_jacobians=consumed[1],
                consumed_restricted_jacobians=consumed[2], consumed_restricted_directions=consumed[3],
            )
            check("one_optimizer_update", **update)
            if profile is not None:
                profile.metadata["update"] = update
            with profile_stage("cache_invalidation_check", phase="cache"):
                try:
                    model.evaluate_queries(cache, tensors["coefficients"], with_jacobian=True)
                except ValueError as exc:
                    if "stale" not in str(exc):
                        raise
                    check("cache_invalidated_after_update", error=str(exc))
                else:
                    raise RuntimeError("stale geometry cache was unexpectedly accepted")
            del cache
            if profile is not None:
                profile.metadata["cache_invalidated"] = True

            if output is not None:
                checkpoint = Path(output)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                scheduler.step()
                with profile_stage("checkpoint_save", phase="other"):
                    model.save_checkpoint(
                        checkpoint, optimizer=optimizer, scheduler=scheduler,
                        data_fingerprint=corpus.manifest_sha256,
                        training_contract=training_contract,
                        extra={"smoke_update": True, "global_step": 1,
                               "geometry_id": row["id"], "normalizer": normalizer.state_dict(),
                               "native_cut_admission": admission, "derivative_normalization": derivative_norm},
                    )
                check("checkpoint_written", path=str(checkpoint), bytes=checkpoint.stat().st_size)
                with profile_stage("checkpoint_resume", phase="other"):
                    restored = build_model(config_name, device=device_obj)
                    restored_optimizer = restored.build_optimizer()
                    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=2)
                    payload = restored.load_checkpoint(
                        checkpoint, optimizer=restored_optimizer, scheduler=restored_scheduler,
                        expected_data_fingerprint=corpus.manifest_sha256, restore_rng=True,
                        map_location="cpu", expected_training_contract=training_contract,
                    )
                    torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
                    torch.testing.assert_close(optimizer.state_dict(), restored_optimizer.state_dict(), rtol=0, atol=0)
                    torch.testing.assert_close(scheduler.state_dict(), restored_scheduler.state_dict(), rtol=0, atol=0)
                    if payload["extra"]["global_step"] != 1 or payload["extra"]["geometry_id"] != row["id"]:
                        raise RuntimeError("checkpoint smoke step or geometry identity mismatch")
                with profile_stage("checkpoint_prediction_comparison", phase="verification"):
                    predictions = _checkpoint_predictions(model, geometry, tensors["coefficients"])
                    reloaded = _checkpoint_predictions(restored, geometry, tensors["coefficients"])
                    differences = _compare_predictions(predictions, reloaded)
                resumed = dict(predictions=differences, prediction_rtol=1e-5, prediction_atol=1e-8,
                               relative_error_denominator="max(abs(original_output), dtype_epsilon)",
                               restored_parameters=len(model.state_dict()), global_step=1,
                               optimizer_state_entries=len(restored_optimizer.state),
                               scheduler_last_epoch=restored_scheduler.last_epoch, rng_restored=True)
                check("checkpoint_resume_roundtrip", **resumed)
                if profile is not None:
                    profile.metadata["checkpoint_resume_roundtrip"] = resumed

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
    parser.add_argument("--geometry-id", help="select a complete record from the training split")
    parser.add_argument("--memory-profile", help="write stage measurements to JSON and sibling Markdown")
    parser.add_argument("--branch-aux-weight",type=float)
    parser.add_argument("--native-cut-contract")
    parser.add_argument("--derivative-normalization", choices=("legacy", "train_rms"))
    parser.add_argument("--seed", type=int, help="seed model initialization and update")
    parser.add_argument("--allow-absolute-assets", action="store_true",
                        help="allow Work manifests to reference absolute local asset paths")
    parser.add_argument("--output", required=True, help="JSON report path; checkpoint path when --smoke-update")
    args = parser.parse_args()
    report = run_preflight(
        args.manifest,
        config_name=args.config,
        device=args.device,
        grid_size=args.grid_size,
        run_update=args.smoke_update,
        output=args.output,
        allow_absolute_assets=args.allow_absolute_assets, geometry_id=args.geometry_id,
        memory_profile=args.memory_profile, seed=args.seed,
        native_cut_contract=args.native_cut_contract, derivative_normalization=args.derivative_normalization,
        branch_aux_weight=args.branch_aux_weight,
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
