"""Training V13 for the unified P-CRAFT + M80 + SQNO operator.

One optimizer update is performed per geometry record. The native RLE
geometry is encoded with an autograd-enabled cache for that update; the cache
is discarded immediately after train_step invalidates it.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import (
    build_model,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_prediction import (
    predict_full,
)
from ..spin_adapter.ablation_initialization import initialize_from_reference
from ..spin_adapter.memory_profile import MemoryProfile, profile_cache_tensors, profile_count, profile_stage, profile_tensors
from .unified_corpus import UnifiedOperatorCorpus


class MomentNormalizer:
    """Train-split RMS scales; no centering of physical moments."""

    def __init__(self, scales: np.ndarray, count: int):
        self.scales = np.asarray(scales, dtype=np.float64)
        self.count = int(count)
        if self.scales.shape != (10,) or not np.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("normalizer scales must be finite positive [10]")
        if self.count < 1:
            raise ValueError("normalizer requires at least one training query")

    @classmethod
    def fit(cls, corpus: UnifiedOperatorCorpus) -> "MomentNormalizer":
        square = np.zeros(10, dtype=np.float64)
        count = 0
        load_labels = getattr(corpus, "load_labels", None)
        for row in corpus.splits["train"]:
            labels = load_labels(row) if load_labels is not None else corpus.load(row)[1]
            mask = labels["valid_mask"]
            values = labels["moments"][mask].astype(np.float64)
            square += np.square(values).sum(axis=0)
            count += len(values)
        if count == 0:
            raise ValueError("no valid training Moment labels")
        return cls(np.maximum(np.sqrt(square / count), 1e-12), count)

    def state_dict(self) -> dict[str, Any]:
        return {"method": "train-only-rms-no-centering", "scales": self.scales.tolist(), "count": self.count}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    tensor_keys = {
        "coefficients", "target_moments", "target_jacobian", "valid_mask",
        "jacobian_valid_mask", "restricted_basis", "restricted_jacobian",
        "restricted_valid", "restricted_dimension",
    }
    for key, value in batch.items():
        if key in tensor_keys:
            result[key] = torch.as_tensor(np.array(value, copy=True) if isinstance(value, np.ndarray) else value,
                                          device=device)
    return result


def _load_training_batch(corpus, row, device):
    with profile_stage("rle_decode", phase="other"):
        batch = corpus.to_training_batch(row)
        geometry = batch["geometry"]
        profile_count(active_intervals=len(geometry["columns"]))
        profile_tensors(**{key: value for key, value in geometry.items()
                           if isinstance(value, np.ndarray)})
    with profile_stage("training_batch_transfer", phase="other"):
        tensors = _batch_tensors(batch, device)
        profile_count(queries=len(tensors["coefficients"]))
        profile_tensors(**tensors)
    return batch, tensors


def _train_geometry(model, optimizer, geometry, tensors, **options):
    with profile_stage("encode_geometry"):
        cache = model.encode_geometry(geometry, inference=False)
        profile_cache_tensors(cache)
    with profile_stage("train_step"):
        metrics = model.train_step(
            optimizer, cache, tensors["coefficients"], tensors["target_moments"],
            tensors.get("target_jacobian"), **options,
            valid_mask=tensors.get("valid_mask"),
            jacobian_valid_mask=tensors.get("jacobian_valid_mask"),
            restricted_basis=tensors.get("restricted_basis"),
            restricted_jacobian=tensors.get("restricted_jacobian"),
            restricted_valid=tensors.get("restricted_valid"),
            restricted_dimension=tensors.get("restricted_dimension"),
        )
    return cache, metrics


def _evaluate_split(model, corpus, split: str, normalizer: MomentNormalizer,
                    device: torch.device, grid_size: int) -> dict[str, Any]:
    model.eval()
    moment_sq = 0.0
    moment_count = 0
    full_j_sq = 0.0
    full_j_count = 0
    restricted_sq = 0.0
    restricted_count = 0
    geometry_count = 0
    query_count = 0
    started = time.perf_counter()
    scales = torch.as_tensor(normalizer.scales, device=device, dtype=torch.float64)

    for row in corpus.splits[split]:
        batch = corpus.to_training_batch(row)
        geometry = batch["geometry"]
        cache = model.encode_geometry(geometry, inference=True)
        values = _batch_tensors(batch, device)
        # Prediction internally differentiates Moments with respect to raw query coefficients.
        prediction = predict_full(model, cache, values["coefficients"], create_graph=False)
        target = values["target_moments"].to(dtype=prediction.moments.dtype)
        valid = values["valid_mask"].bool()
        if not prediction.prediction_finite.all():
            raise FloatingPointError(f"{row['id']}: nonfinite validation prediction")
        error = (prediction.moments[valid] - target[valid]) / scales
        moment_sq += float(error.square().sum().detach())
        moment_count += error.numel()

        if "target_jacobian" in values:
            mask = values.get("jacobian_valid_mask", valid).bool()
            if mask.any():
                target_j = values["target_jacobian"].to(dtype=prediction.jacobian_coefficients.dtype)
                delta = prediction.jacobian_coefficients[mask] - target_j[mask]
                full_j_sq += float(delta.square().sum().detach())
                full_j_count += delta.numel()
        if "restricted_basis" in values and values.get("restricted_valid") is not None:
            mask = values["restricted_valid"].bool()
            if mask.any():
                basis = values["restricted_basis"].to(dtype=prediction.jacobian_coefficients.dtype)
                target_r = values["restricted_jacobian"].to(dtype=prediction.jacobian_coefficients.dtype)
                dims = values["restricted_dimension"].long()
                for d in torch.unique(dims[mask]).tolist():
                    idx = torch.nonzero(mask & (dims == d), as_tuple=False).flatten()
                    predicted = prediction.jacobian_coefficients[idx] @ basis[idx, :, :d]
                    delta = predicted - target_r[idx, :, :d]
                    restricted_sq += float(delta.square().sum().detach())
                    restricted_count += delta.numel()
        geometry_count += 1
        query_count += len(values["coefficients"])

    return {
        "split": split,
        "geometries": geometry_count,
        "queries": query_count,
        "moment_normalized_rmse": math.sqrt(moment_sq / max(moment_count, 1)),
        "moment_elements": moment_count,
        "full_jacobian_rmse": math.sqrt(full_j_sq / max(full_j_count, 1)) if full_j_count else None,
        "full_jacobian_elements": full_j_count,
        "restricted_jacobian_rmse": math.sqrt(restricted_sq / max(restricted_count, 1)) if restricted_count else None,
        "restricted_jacobian_elements": restricted_count,
        "elapsed_seconds": time.perf_counter() - started,
    }


def train(
    manifest: str | Path,
    output: str | Path,
    *,
    config_name: str = "full_v1_candidate",
    device: str = "cuda",
    grid_size: int = 64,
    epochs: int = 10,
    seed: int = 830251,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    jacobian_weight: float = 0.1,
    branch_aux_weight: float = 0.25,
    gradient_clip: float = 1.0,
    gradient_mode: str = "fixed",
    resume: str | None = None,
    allow_absolute_assets: bool = False,
    memory_profile: str | Path | None = None,
    initialization_reference: str | None = None,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if resume and initialization_reference is not None:
        raise ValueError("initialization_reference is for fresh runs and cannot be combined with resume")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    corpus = UnifiedOperatorCorpus(
        manifest,
        grid_size=grid_size,
        allow_absolute_paths=allow_absolute_assets,
    )
    normalizer = MomentNormalizer.fit(corpus)
    _seed_all(seed)
    model = build_model(config_name, device=device)
    initialization = None
    if initialization_reference is not None:
        initialization = initialize_from_reference(
            model, reference_config_name=initialization_reference, seed=seed,
        )
        # Align stochastic training after construction as well as common
        # weights; optional modules otherwise shift the training RNG stream.
        _seed_all(seed)
        initialization["training_rng_reset_to_seed"] = seed
    device_obj = torch.device(device)
    optimizer = model.build_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scales = torch.as_tensor(normalizer.scales, device=device_obj, dtype=torch.float64)
    training_contract = model.training_contract(
        gradient_mode=gradient_mode,
        jacobian_weight=jacobian_weight,
        branch_aux_weight=branch_aux_weight,
        gradient_clip=gradient_clip,
        moment_scale=scales,
    )
    start_epoch = 0
    global_step = 0
    if resume:
        payload = model.load_checkpoint(
            resume,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_data_fingerprint=corpus.manifest_sha256,
            restore_rng=True,
            map_location=device_obj,
            expected_training_contract=training_contract,
        )
        start_epoch = int(payload.get("extra", {}).get("epoch", 0))
        global_step = int(payload.get("extra", {}).get("global_step", 0))
        initialization = payload.get("extra", {}).get("config", {}).get("initialization")

    config_record = {
        "schema": "tvcg.unified-training-v13.run.v1",
        "config_name": config_name,
        "manifest_sha256": corpus.manifest_sha256,
        "split_summary": corpus.split_summary(),
        "normalizer": normalizer.state_dict(),
        "training_contract": training_contract,
        "device": str(device_obj),
        "seed": seed,
        "epochs": epochs,
        "architecture_manifest": model.architecture_contract(),
        "initialization": initialization,
    }
    (output / "training_config.json").write_text(json.dumps(config_record, indent=2, allow_nan=False))

    history_path = output / "metrics.jsonl"
    best_val = float("inf")
    for epoch in range(start_epoch, epochs):
        model.train()
        rows = list(corpus.splits["train"])
        rng = np.random.default_rng(seed + epoch)
        rows = [rows[i] for i in rng.permutation(len(rows))]
        epoch_loss = 0.0
        epoch_moment = 0.0
        epoch_jacobian = 0.0
        epoch_seconds = 0.0
        for row in rows:
            profile = MemoryProfile(device_obj, memory_profile) if memory_profile else None
            with profile if profile is not None else nullcontext():
                if profile is not None:
                    profile.metadata.update(geometry_id=row["id"], global_step=global_step + 1,
                                            manifest_sha256=corpus.manifest_sha256, config=config_name)
                batch, tensors = _load_training_batch(corpus, row, device_obj)
                started = time.perf_counter()
                cache, metrics = _train_geometry(
                    model, optimizer, batch["geometry"], tensors, moment_scale=scales,
                    jacobian_weight=jacobian_weight, branch_aux_weight=branch_aux_weight,
                    gradient_clip=gradient_clip, gradient_mode=gradient_mode,
                )
                epoch_seconds += time.perf_counter() - started
                if profile is not None:
                    profile.metadata["metrics"] = metrics
                del cache, batch, tensors
            # Record only the first update of this invocation, including resumed runs.
            memory_profile = None
            epoch_loss += metrics["loss"]
            epoch_moment += metrics["moment"]
            epoch_jacobian += metrics["jacobian"]
            global_step += 1

        scheduler.step()
        train_metrics = {
            "split": "train",
            "geometries": len(rows),
            "optimizer_steps": len(rows),
            "loss_mean_per_geometry": epoch_loss / len(rows),
            "moment_loss_mean_per_geometry": epoch_moment / len(rows),
            "jacobian_loss_mean_per_geometry": epoch_jacobian / len(rows),
            "elapsed_seconds": epoch_seconds,
        }
        val_metrics = _evaluate_split(model, corpus, "val", normalizer, device_obj, grid_size)
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

        model.save_checkpoint(
            output / "checkpoints" / "latest.pt",
            optimizer=optimizer,
            scheduler=scheduler,
            data_fingerprint=corpus.manifest_sha256,
            training_contract=training_contract,
            extra={"epoch": epoch + 1, "global_step": global_step,
                   "normalizer": normalizer.state_dict(), "config": config_record},
        )
        if val_metrics["moment_normalized_rmse"] < best_val:
            best_val = val_metrics["moment_normalized_rmse"]
            model.save_checkpoint(
                output / "checkpoints" / "best_val.pt",
                optimizer=optimizer,
                scheduler=scheduler,
                data_fingerprint=corpus.manifest_sha256,
                training_contract=training_contract,
                extra={"epoch": epoch + 1, "global_step": global_step,
                       "normalizer": normalizer.state_dict(), "config": config_record},
            )
        print(json.dumps(record, allow_nan=False), flush=True)

    return {"output": str(output), "global_step": global_step, "best_val_moment_normalized_rmse": best_val}


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Spin Neural Moments V13 training")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="full_v1_candidate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=830251)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--jacobian-weight", type=float, default=0.1)
    parser.add_argument("--branch-aux-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-mode", choices=("fixed", "pcgrad"), default="fixed")
    parser.add_argument("--resume")
    parser.add_argument("--initialization-reference",
                        help="fresh CPU random reference config for common ablation initialization; incompatible with --resume")
    parser.add_argument("--memory-profile", help="write first-update JSON and sibling Markdown profiles")
    parser.add_argument("--allow-absolute-assets", action="store_true",
                        help="allow Work manifests to reference absolute local asset paths")
    args = parser.parse_args()
    result = train(
        args.manifest, args.output, config_name=args.config, device=args.device,
        grid_size=args.grid_size, epochs=args.epochs, seed=args.seed,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        jacobian_weight=args.jacobian_weight, branch_aux_weight=args.branch_aux_weight,
        gradient_clip=args.gradient_clip, gradient_mode=args.gradient_mode, resume=args.resume,
        allow_absolute_assets=args.allow_absolute_assets, memory_profile=args.memory_profile,
        initialization_reference=args.initialization_reference,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
