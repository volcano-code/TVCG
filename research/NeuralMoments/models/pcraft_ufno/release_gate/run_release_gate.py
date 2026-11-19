"""Phase 2.8.6 deterministic software architecture release gate.

This gate runs repository-owned CPU contract suites and emits an auditable
JSON/Markdown report. Data- and GPU-dependent scientific gates are explicitly
reported as BLOCKED/PENDING; they are never inferred from software tests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[5]
NATIVE = ROOT / "research/NeuralMoments/models/pcraft_ufno/spin_pino"
ADAPTER = ROOT / "research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter"
NATIVE_TESTS = NATIVE / "tests"
ADAPTER_TESTS = ADAPTER / "tests"

ARCHITECTURE_MODES = {
    "global_scalar_baseline": {"fusion_mode": "global_scalar", "geometry_attention": False, "cross_scale_attention": False},
    "query_conditioned_fusion": {"fusion_mode": "query_conditioned", "geometry_attention": False, "cross_scale_attention": False},
    "geometry_attention": {"fusion_mode": "global_scalar", "geometry_attention": True, "cross_scale_attention": False},
    "cross_scale_attention": {"fusion_mode": "global_scalar", "geometry_attention": False, "cross_scale_attention": True},
    "joint_attention": {"fusion_mode": "global_scalar", "geometry_attention": True, "cross_scale_attention": True},
    "joint_attention_query_fusion": {"fusion_mode": "query_conditioned", "geometry_attention": True, "cross_scale_attention": True},
}
SCIENTIFIC_PENDING = [
    ("formal_dataset_admission", "100 TRELLIS candidates and multi-source geometry manifests are still being processed/audited"),
    ("paper_benchmark_validation", "eligible geometries and reproducible labels from Spin-It and Spin-It Faster are not yet admitted"),
    ("locked_test_evaluation", "source-family-isolated locked test set is not yet frozen"),
    ("multi_seed_training", "formal multi-seed training has not been run"),
    ("resolution_64_to_512", "64^3/128^3/256^3/512^3 trained-model evaluation has not been run"),
    ("ood_generalization", "OOD evaluation requires the admitted multi-source split"),
    ("rtx5090d_performance", "matched GPU latency/peak-memory profiling has not been run"),
    ("hybrid_optimizer_acceptance", "Neural Proposal/Newton/Exact Correction requires trained checkpoint and independent physical verification"),
]


def _run(name: str, command: list[str], cwd: Path, env: dict[str, str]) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    return {"name": name, "status": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode, "duration_seconds": round(time.perf_counter()-started, 3),
            "command": command, "output_tail": proc.stdout[-12000:]}


def _environment() -> dict:
    packages = {}
    for package in ("torch", "numpy", "scipy", "pytest", "torch-geometric", "torch-scatter"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        import torch
        cuda = {"available": bool(torch.cuda.is_available()), "version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:
        cuda = {"available": False, "error": repr(exc)}
    return {"python": sys.version, "platform": platform.platform(), "packages": packages, "cuda": cuda}


def _architecture_contract_check() -> dict:
    required = [
        NATIVE / "native_pcraft.py",
        NATIVE / "tests/test_native_pcraft.py",
        ADAPTER / "unified_m80_sqno.py",
        ADAPTER / "geometry_physics_attention.py",
        ADAPTER / "cross_scale_attention.py",
        ADAPTER / "fusion_ablation.py",
        ADAPTER / "tests/test_unified_m80_sqno.py",
        NATIVE / "uq.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    wrapper = (ADAPTER / "unified_m80_sqno.py").read_text() if not missing else ""
    native = (NATIVE / "native_pcraft.py").read_text() if not missing else ""
    markers = {
        "global_scalar_fusion": '"global_scalar"' in wrapper,
        "query_conditioned_fusion": '"query_conditioned"' in wrapper,
        "geometry_cross_attention": "GeometryPhysicsCrossAttention" in wrapper,
        "cross_scale_attention": "QueryConditionedCrossScaleAttention" in wrapper,
        "gno_interaction_variants": "parallel_gno_attention" in wrapper and "gno_then_attention" in wrapper,
        "adaptive_physical_slicing": "transolver_adaptive" in native and "transolver_active_slices" in native,
        "fusion_ablation_harness": (ADAPTER / "fusion_ablation.py").is_file(),
        "gaussian_uq_primitives": (NATIVE / "uq.py").is_file() and "def gaussian_calibration_metrics" in (NATIVE / "uq.py").read_text(),
        "uq_risk_coverage": (NATIVE / "uq.py").is_file() and "def uncertainty_risk_coverage" in (NATIVE / "uq.py").read_text(),
        "uq_reliability_bins": (NATIVE / "uq.py").is_file() and "def calibration_reliability_bins" in (NATIVE / "uq.py").read_text(),
        "event_classification_metrics": (NATIVE / "uq.py").is_file() and "def event_classification_metrics" in (NATIVE / "uq.py").read_text(),
        "event_threshold_selection": "def select_event_threshold" in (NATIVE / "uq.py").read_text(),
        "grouped_event_bootstrap": "def grouped_event_bootstrap" in (NATIVE / "uq.py").read_text(),
        "event_label_mask_contract": "event_label_valid" in (NATIVE / "native_train.py").read_text(),
        "event_conditioned_jacobian_supervision": "event_jacobian_weight" in (NATIVE / "native_train.py").read_text(),
        "learned_moment_jacobian_uq_heads": "def predict_uq" in native and "uq_enabled" in native,
        "uq_training_loss_integration": "uq_weight" in (NATIVE / "native_train.py").read_text(),
    }
    failed = [k for k, ok in markers.items() if not ok]
    return {"status": "PASS" if not missing and not failed else "FAIL",
            "missing_files": missing, "architecture_markers": markers,
            "missing_architecture_markers": failed, "supported_ablation_modes": ARCHITECTURE_MODES}


def _markdown(report: dict) -> str:
    lines = ["# Phase 2.8.6 Architecture Release Gate", "",
             f"- Commit: `{report['commit']}`", f"- Generated: {report['generated_at']}",
             f"- Overall software gate: **{report['software_gate']}**", "",
             "## Executed checks", "", "| Check | Status | Duration (s) |", "|---|---:|---:|"]
    for item in report["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item.get('duration_seconds', 0)} |")
    lines += ["", "## Scientific/data-dependent gates", "",
              "| Gate | Status | Reason |", "|---|---|---|"]
    for item in report["scientific_gates"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['reason']} |")
    lines += ["", "## Interpretation", "",
              "A PASS means the listed software checks executed successfully. It does not establish trained-model accuracy, OOD generalization, GPU speedup, or scientific readiness.",
              "Scientific gates remain blocked/pending until the formal multi-source dataset, trained checkpoints, and matched independent evaluations are available.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--output", type=Path, default=Path("runs/phase2_8_6_release"))
    parser.add_argument("--skip-tests", action="store_true", help="Only emit static contract/environment report")
    args = parser.parse_args(argv)
    root = ROOT.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONPATH": f"{root}:{ADAPTER.parent}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1"})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        commit = "unknown"
    checks = []
    static = _architecture_contract_check()
    checks.append({"name": "architecture_contract_inventory", **static})
    if not args.skip_tests:
        checks.append(_run("compile_native", [sys.executable, "-m", "compileall", "-q", str(NATIVE)], root, env))
        checks.append(_run("compile_adapter", [sys.executable, "-m", "compileall", "-q", str(ADAPTER)], root, env))
        checks.append(_run("native_pcraft_tests", [sys.executable, "-m", "pytest", "-q", str(NATIVE_TESTS)], root, env))
        checks.append(_run("sqno_adapter_tests", [sys.executable, "-m", "pytest", "-q", str(ADAPTER_TESTS)], root, env))
    software_status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else ("PASS" if not args.skip_tests else "PENDING")
    report = {
        "schema": "tvcg.phase2_8_6.release_gate.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": commit, "profile": args.profile, "software_gate": software_status,
        "checks": checks,
        "scientific_gates": [{"name": name, "status": "BLOCKED", "reason": reason} for name, reason in SCIENTIFIC_PENDING],
        "environment": _environment(),
        "data_strategy": {
            "mini7": "pilot_only",
            "additional_trellis_100": "formal_training_candidates_pending_processing_and_audit",
            "validation": "eligible_Spin-It_and_Spin-It_Faster_paper_geometries_plus_additional_TRELLIS_artifacts",
            "split_unit": "geometry_and_source_family",
            "locked_test": "separate_and_unseen_during_model_selection",
        },
        "notes": ["No training was started by this release gate.",
                  "A software PASS is not a scientific/model freeze.",
                  "CUDA profiling and formal-data accuracy remain blocked."],
    }
    (output / "release_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "release_report.md").write_text(_markdown(report))
    print(json.dumps({"software_gate": software_status, "commit": commit,
                      "json": str(output / "release_report.json"),
                      "markdown": str(output / "release_report.md"),
                      "checks": [{"name": c["name"], "status": c["status"]} for c in checks],
                      "scientific_gates_blocked": len(report["scientific_gates"])}, indent=2))
    return 1 if software_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
