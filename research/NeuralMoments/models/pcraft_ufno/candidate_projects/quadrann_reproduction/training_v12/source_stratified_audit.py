import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

from training_v12.dataset import M83Corpus
from spin_adapter.quadrann_3d import interval_component_ids

root = Path(os.environ["DATASET"])
run = Path(os.environ["RUN"])

report = json.loads(
    (run / "reports/frozen_model_evaluation.json").read_text()
)

corpus = M83Corpus(root / "operator-manifest.json")

def family(record_id):
    if record_id.startswith("gso70-"):
        return "gso"
    if record_id.startswith("m70-procedural-"):
        return "procedural"
    if record_id.startswith("m70-design-panel-"):
        return "design_panel"
    return "other"

records = {
    row["id"]: row
    for split in ("train", "val", "test")
    for row in corpus.splits[split]
}

rows = []

for split in ("train", "val"):
    metrics = report["results"][split]["all_geometries"]

    for item in metrics:
        rid = item["record_id"]
        row = records[rid]

        geometry, labels = corpus.load(row)

        columns = geometry["columns"]
        side = int(geometry["native_resolution"])

        components = interval_component_ids(columns)

        occupied = int(
            np.sum(columns[:, 3] - columns[:, 2])
        )

        bbox = np.array([
            columns[:, 0].max() - columns[:, 0].min() + 1,
            columns[:, 1].max() - columns[:, 1].min() + 1,
            columns[:, 3].max() - columns[:, 2].min(),
        ])

        rows.append({
            "split": split,
            "id": rid,
            "family": family(rid),
            "rle_intervals": len(columns),
            "components": len(np.unique(components)),
            "occupied_voxels": occupied,
            "occupancy_ratio": occupied / side**3,
            "bbox_aspect_ratio": float(
                bbox.max() / max(1, bbox.min())
            ),
            "relative_l2": item["relative_l2_mean"],
            "relative_l2_p95": item["relative_l2_p95"],
            "normalized_rmse": item["normalized_rmse"],
        })

print("===== SOURCE FAMILY SUMMARY =====")

for split in ("train", "val"):
    groups = defaultdict(list)

    for row in rows:
        if row["split"] == split:
            groups[row["family"]].append(row)

    for name, values in sorted(groups.items()):
        errors = np.array([
            x["relative_l2"] for x in values
        ])

        print({
            "split": split,
            "family": name,
            "count": len(values),
            "mean_relative_l2": float(errors.mean()),
            "median_relative_l2": float(np.median(errors)),
            "p95_relative_l2": float(np.percentile(errors, 95)),
            "max_relative_l2": float(errors.max()),
            "mean_rle_intervals": float(np.mean([
                x["rle_intervals"] for x in values
            ])),
            "mean_components": float(np.mean([
                x["components"] for x in values
            ])),
        })

print("\n===== COMPLEXITY CORRELATION =====")

for split in ("train", "val"):
    selected = [r for r in rows if r["split"] == split]

    for key in (
        "rle_intervals",
        "components",
        "occupancy_ratio",
        "bbox_aspect_ratio",
    ):
        x = np.array([r[key] for r in selected])
        y = np.array([r["relative_l2"] for r in selected])

        result = spearmanr(x, y)

        print({
            "split": split,
            "feature": key,
            "spearman_rho": float(result.statistic),
            "p_value": float(result.pvalue),
        })

out = run / "reports/source_stratified_audit.json"

out.write_text(json.dumps({
    "schema": "sqno.source-stratified-audit.v1",
    "results": rows,
}, indent=2))

print("\nAUDIT SAVED:", out)
