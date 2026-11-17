import os
import json
import time
import numpy as np
import torch

from pathlib import Path

from training_v12.dataset import M83Corpus
from training_v12.normalization import MomentNormalizer
from training_v12.metrics import evaluate_moments
from training_v12.geometry_cache import prepare_m83_geometry

from spin_adapter.spin_quadrann_v12 import SpinQuadrANNV12


DATASET = Path(os.environ["DATASET"])
RUN = Path(os.environ["RUN"])
DEVICE = torch.device("cuda:0")

manifest = DATASET / "operator-manifest.json"
checkpoint_path = RUN / "checkpoints/best_val.pt"

corpus = M83Corpus(manifest)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

assert checkpoint["manifest_sha256"] == corpus.manifest_sha256

config = checkpoint["model_config"]
grid_size = config["grid_size"]

model_args = {
    k: v for k, v in config.items()
    if k != "grid_size"
}

model = SpinQuadrANNV12(**model_args).to(DEVICE)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

normalizer = MomentNormalizer.from_state_dict(
    checkpoint["normalizer"]
)

scales = normalizer.scales

all_results = {}

for split in ("train", "val"):
    predictions = []
    targets = []
    per_geometry = []

    started = time.perf_counter()

    for index, row in enumerate(corpus.splits[split]):
        raw, labels = corpus.load(row)

        geometry = prepare_m83_geometry(
            raw, grid_size
        )

        with torch.inference_mode():
            cache = model.encode_geometry(geometry)

            queries = torch.tensor(
                labels["queries"].copy(),
                device=DEVICE,
                dtype=next(model.parameters()).dtype,
            )

            pred_chunks = []

            for start in range(0, len(queries), config["query_chunk"]):
                result = model.evaluate_queries(
                    cache,
                    queries[start:start + config["query_chunk"]],
                    with_jacobian=False,
                )

                pred_chunks.append(
                    result["moments_total"].cpu().numpy()
                )

            prediction = np.concatenate(pred_chunks)
            target = labels["moments"].copy()

        predictions.append(prediction)
        targets.append(target)

        gm = evaluate_moments(
            prediction, target, scales
        )

        per_geometry.append({
            "record_id": row["id"],
            "relative_l2_mean": gm["relative_l2_mean"],
            "relative_l2_p95": gm["relative_l2_p95"],
            "normalized_rmse": gm["normalized_rmse"],
            "max_absolute_error": gm["max_absolute_error"],
        })

        print(
            f"[{split}] {index + 1}/{len(corpus.splits[split])} "
            f"{row['id']} "
            f"relative_l2={gm['relative_l2_mean']:.6f}",
            flush=True,
        )

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)

    summary = evaluate_moments(
        prediction, target, scales
    )

    normalized_rmse_by_moment = (
        np.asarray(summary["rmse_by_moment"])
        / scales
    )

    target_norm = np.linalg.norm(target, axis=1)
    threshold = np.percentile(target_norm, 5)
    near_zero = target_norm <= threshold

    near_zero_error = (
        np.linalg.norm(
            prediction[near_zero] - target[near_zero],
            axis=1,
        )
        / np.maximum(target_norm[near_zero], 1e-12)
    )

    per_geometry.sort(
        key=lambda x: x["relative_l2_mean"],
        reverse=True,
    )

    all_results[split] = {
        "summary": summary,
        "normalized_rmse_by_moment": (
            normalized_rmse_by_moment.tolist()
        ),
        "near_zero_query_count": int(near_zero.sum()),
        "near_zero_target_norm_p05": float(threshold),
        "near_zero_relative_l2_mean": float(
            near_zero_error.mean()
        ),
        "worst_geometries": per_geometry[:10],
        "all_geometries": per_geometry,
        "elapsed_seconds": time.perf_counter() - started,
    }

    print(f"\n===== {split.upper()} SUMMARY =====")
    print(json.dumps(
        all_results[split]["summary"],
        indent=2,
    ))

    print(
        "Normalized RMSE by moment:",
        normalized_rmse_by_moment,
    )

    print(
        "Near-zero relative L2:",
        all_results[split]["near_zero_relative_l2_mean"],
    )

output = RUN / "reports/frozen_model_evaluation.json"
output.parent.mkdir(parents=True, exist_ok=True)

output.write_text(json.dumps({
    "checkpoint": str(checkpoint_path),
    "epoch": checkpoint["epoch"],
    "global_step": checkpoint["global_step"],
    "manifest_sha256": corpus.manifest_sha256,
    "model_config": config,
    "results": all_results,
}, indent=2))

print("\nFROZEN MODEL EVALUATION COMPLETE")
print("Saved:", output)
