"""Bundle all fixed training-side attempts, numeric evidence and replay identities."""

import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile

from capture import HERE, PROTOCOL, save, sha

CAPTURES = (
    "T2_seed",
    "T2_extended",
    "N002_registered",
    "S001_registered",
    "T2_hyperbolic",
    "T2_high_hyper",
    "T2_high_ell",
    "T2_mid_ell",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("/data/xvxinyao/TVCG/datasets/final_voxel_pino_corpus"),
    )
    args = p.parse_args()
    work = args.work
    args.output.mkdir(exist_ok=True, parents=True)
    files = {
        k: work / f"{k}.json"
        for k in (*CAPTURES, "margins", "probe1", "probe2", "rank")
    }

    def read(name):
        return json.loads(files[name].read_text())

    margins = read("margins")
    rank = read("rank")
    probes = [read("probe1"), read("probe2")]
    assert len(margins["prefixes"]) == len(CAPTURES) and all(
        p["complete"] for p in probes
    )
    prefixes = []
    decisions = []
    geometry_rejections = 0
    for name, summary in zip(CAPTURES, margins["prefixes"]):
        captured = read(name)
        assert (
            summary["capture_sha256"] == sha(files[name])
            and captured["result"]["event_hash"] == summary["native_event_hash"]
        )
        attempts = summary["decisions"]
        scored = [d for d in attempts if not d.get("geometry_rejected")]
        decisions.extend(scored)
        geometry_rejections += len(attempts) - len(scored)
        prefixes.append(
            dict(
                id=name,
                geometry=summary["geometry"],
                query=captured["query_id"],
                initialization=summary["start"],
                native_status=summary["status"],
                native_terminal=summary["termination_reason"],
                true_call_count=summary["function_evaluations"],
                scored_accepts=summary["scored_native_accepts"],
                scored_backtracks=summary["scored_native_backtracks"],
                event_hash=summary["native_event_hash"],
                trace_hash=summary["native_trace_hash"],
            )
        )
    fd = [
        (v["step"], v["max_scaled_directional_error"])
        for v in probes[0]["native_directional_fd"]
    ]
    assert (
        rank["exact_bordered_row_dependence"]
        and rank["counterfactual_ranks"] == [12, 12]
        and rank["original_ranks"] == [13, 13]
    )
    source_domain = args.corpus / "manifests/train_manifest.json"
    records = json.loads(source_domain.read_text())["records"]
    reg = {
        r["geometry_id"]: r
        for r in records
        if r["geometry_id"] in {"T2", "S001", "N002"}
    }
    assert len(reg) == 3
    catalogue = args.corpus.parent / "faster_optimization_trajectories_v1"
    admission = catalogue / "admission_failures.jsonl"
    trajectories = catalogue / "trajectories.jsonl"
    selected_admissions = [
        json.loads(line)
        for line in admission.read_text().splitlines()
        if json.loads(line).get("geometry_id") in reg
    ]
    archive = args.output / "native_training_receipts.tar.gz"
    with (
        archive.open("wb") as dest,
        gzip.GzipFile(fileobj=dest, mode="wb", mtime=0, filename="") as g,
        tarfile.open(fileobj=g, mode="w") as tar,
    ):
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{name}.json")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    outcome = dict(
        schema="compact-v21.native-training-threshold-sensitivity.v1",
        scope="Three expressly registered TRELLIS training geometries, native STATIC_BALANCE / MAXIMUM_MASS, seed 0. No blind geometry, learned weights, predictions or training.",
        limits=dict(moment_p95=0.01, directional_jacobian_p95=0.05),
        decision="PROPOSED_SCREENING_ONLY__NATIVE_NEWTON_STABILITY_NOT_VALIDATED",
        material_source_readback={
            str(k): sha(k) for k in [source_domain, admission, trajectories]
        },
        registered_training_support={
            g: [
                k
                for k in (
                    "contact_pose",
                    "support_pose",
                    "support_axis",
                    "spinning_top_objective",
                    "initial_gyroscope_id",
                )
                if k in rec
            ]
            for g, rec in reg.items()
        },
        published_training_trajectory_catalog_bytes=trajectories.stat().st_size,
        selected_trajectory_admissions=selected_admissions,
        original_native_prefixes=prefixes,
        n_accepted=sum(p["scored_accepts"] for p in prefixes),
        n_scored_backtracks=sum(p["scored_backtracks"] for p in prefixes),
        n_geometry_rejected_trials=geometry_rejections,
        n_scored_margins_stable_for_all_componentwise_1pct_M_errors=sum(
            int(
                (
                    x["outcome"] == "accepted"
                    and x["positive_decrease_certified_under_moment_box"]
                )
                or (
                    x["outcome"] == "rejected"
                    and x["nondecrease_certified_under_moment_box"]
                )
            )
            for x in decisions
        ),
        min_observed_residual_drop=min(x["native_margin"] for x in decisions),
        max_original_residual_reproduction_error=max(
            p["max_native_score_reproduction_error"] for p in margins["prefixes"]
        ),
        max_original_candidate_coordinate_reproduction_error=max(
            x["jacobian_sensitivity"]["baseline_reproduction_max_abs"]
            for x in decisions
        ),
        physical_rechecks=[
            dict(
                iteration=p["native_iteration"],
                original_margin=p["unperturbed_true_margin"],
                minimum_true_counterfactual_margin=min(
                    d["true_margin"] for d in p["proposals"]
                ),
                maximum_true_counterfactual_margin=max(
                    d["true_margin"] for d in p["proposals"]
                ),
                realized_finite_true_calls=len(p["proposals"]),
                true_decreasing_count=sum(
                    d["true_trial_decreases"] for d in p["proposals"]
                ),
            )
            for p in probes
        ],
        native_directional_fd=fd,
        rank_counterexample={
            k: rank[k]
            for k in (
                "geometry",
                "start",
                "seed",
                "global_M_component_bound",
                "global_J_direction_bound_on_unit_queries",
                "base_radial_leak_scaled",
                "base_M_error",
                "original_ranks",
                "counterfactual_ranks",
                "exact_bordered_row_dependence",
                "original_first_trial",
                "native_counterfactual_status",
                "native_diagnostic",
                "modified_function_evaluations",
                "original_event_hash",
                "modified_event_hash",
            )
        },
        conclusions=[
            "All observed fixed-trial native static decisions survived the 1% moment component box. No scored original backtrack was observed, so rejected-step sensitivity is unmeasured.",
            "The post-hoc S001 bounded analytic perturbation changes the Newton system rank and terminates the native routine despite satisfying both pointwise budgets. It tests the implications of the metric alone; it is not a learned model result.",
            "The inspected train registrations lack a spinning contact pose and initial gyroscope. These static traces do not validate the spinning-top gate or a population P95. No converged train solve is claimed.",
        ],
        receipt_file=archive.name,
        receipt_sha256=sha(archive),
        member_sha256={path.name: sha(path) for path in files.values()},
        analysis_source_sha256={
            p.name: sha(p) for p in [PROTOCOL, *sorted(HERE.glob("*.py"))]
        },
    )
    save(args.output / "threshold_sensitivity.json", outcome)


if __name__ == "__main__":
    main()
