"""Create reviewable, pinned preflight/training scripts without starting a run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
from typing import Any

import torch

from ..spin_adapter.ablation_initialization import initialize_from_reference
from ..spin_adapter.unified_factory import architecture_manifest, build_model
from .unified_corpus import UnifiedOperatorCorpus


STAGES = {
    "screen": ("ablation_g1_t1", "ablation_no_cross_scale", "compact_v1_candidate"),
    "interaction": ("ablation_g1_t1", "ablation_g0_t0", "ablation_g1_t0", "ablation_g0_t1"),
    "scales": ("ablation_g1_t1", "ablation_two_scales", "ablation_one_scale"),
    "all": ("ablation_g1_t1", "ablation_g0_t0", "ablation_g1_t0", "ablation_g0_t1",
            "ablation_no_cross_scale", "ablation_two_scales", "ablation_one_scale",
            "compact_v1_candidate"),
}
_MODULE = "research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13"
_SOURCE_PATH = "research/NeuralMoments/models/pcraft_ufno"
_REFERENCE = "full_v1_candidate"


def _git_snapshot(root: str | Path | None) -> tuple[Path, str]:
    location = Path.cwd() if root is None else Path(root)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(location), *args], text=True).strip()
    repository = Path(git("rev-parse", "--show-toplevel")).resolve()
    commit = git("rev-parse", "HEAD")
    changed = subprocess.run(["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--", _SOURCE_PATH])
    if changed.returncode == 1:
        raise ValueError("tracked neural-operator source has uncommitted changes; commit it before planning")
    changed.check_returncode()
    untracked = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard", "--", _SOURCE_PATH],
        text=True,
    ).strip()
    if untracked:
        raise ValueError("untracked neural-operator source prevents a pinned plan: " + untracked)
    return repository, commit


def _script(plan: dict[str, Any], phase: str) -> str:
    q = shlex.quote
    hash_check = (
        "import hashlib,pathlib,sys; "
        "actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); "
        "sys.exit(0 if actual == sys.argv[2] else 'Manifest SHA-256 changed; generate a new plan')"
    )
    lines = [
        "#!/usr/bin/env bash", "set -euo pipefail",
        "# Generated plan: this script runs only when explicitly invoked.",
        "# Preflight uses constructor initialization; train uses the common random reference.",
        'TVCG_PYTHON="${TVCG_PYTHON:-python}"',
        f"TVCG_PLAN_REPOSITORY={q(plan['repository_root'])}",
        f"TVCG_PLAN_COMMIT={q(plan['source_commit'])}",
        'cd "$TVCG_PLAN_REPOSITORY"',
        "check_inputs() {",
        '  if [[ "$(git rev-parse HEAD)" != "$TVCG_PLAN_COMMIT" ]]; then',
        '    echo "Source commit changed; generate a new plan" >&2; exit 1',
        "  fi",
        f"  if ! git diff --quiet HEAD -- {q(_SOURCE_PATH)}; then",
        '    echo "Tracked neural-operator source has uncommitted changes" >&2; exit 1',
        "  fi",
        f'  if [[ -n "$(git ls-files --others --exclude-standard -- {q(_SOURCE_PATH)})" ]]; then',
        '    echo "Untracked neural-operator source prevents a pinned run" >&2; exit 1',
        "  fi",
        f'  "$TVCG_PYTHON" -c {q(hash_check)} {q(plan["manifest_path"])} {q(plan["manifest_sha256"])}',
        "}",
        "check_inputs", "",
    ]
    for arm in plan["arms"]:
        target = Path(arm[f"{phase}_output"])
        lines.extend([
            f"# {arm['config_name']}", "check_inputs",
            f"if [[ -e {q(str(target))} ]]; then",
            f"  echo {q('Refusing to overwrite run directory: ' + str(target))} >&2; exit 1",
            "fi",
            f"mkdir -p -- {q(str(target.parent))}",
            # Atomic creation also rejects a concurrent run of this script.
            f"mkdir -- {q(str(target))}",
            '"$TVCG_PYTHON" ' + shlex.join(arm[f"{phase}_argv"])
                + " 2>&1 | tee -- " + q(str(target / "console.log")),
            "",
        ])
    return "\n".join(lines)


def generate_plan(
    manifest: str | Path,
    output: str | Path,
    *,
    stage: str = "screen",
    epochs: int = 3,
    seed: int = 830251,
    learning_rate: float = 3e-5,
    device: str = "cuda",
    geometry_id: str | None = None,
    allow_absolute_assets: bool = False,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate manifest metadata, count CPU models, and write a new plan folder.

    No geometry arrays are loaded or encoded and no training is performed.
    The supplied model states must all be covered by the fresh Full V1
    reference. The combined compact candidate is not a single-factor ablation.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("plan device must be cpu or cuda")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"plan output must be a new directory: {output}")
    corpus = UnifiedOperatorCorpus(manifest, grid_size=64,
                                   allow_absolute_paths=allow_absolute_assets)
    if geometry_id is not None and geometry_id not in {row["id"] for row in corpus.splits["train"]}:
        raise ValueError("geometry_id must identify a complete admitted training record")
    repository, commit = _git_snapshot(repository_root)
    common = ["--manifest", str(corpus.path), "--device", device,
              "--grid-size", "64", "--seed", str(seed)]
    if allow_absolute_assets:
        common.append("--allow-absolute-assets")
    arms = []
    for name in STAGES[stage]:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            model = build_model(name, device="cpu")
        initialization = initialize_from_reference(model, reference_config_name=_REFERENCE, seed=seed)
        preflight_output = output / "preflight" / name
        training_output = output / "training" / name
        preflight = ["-m", f"{_MODULE}.preflight_unified", *common, "--config", name,
                     "--smoke-update", "--output", str(preflight_output / "checkpoint.pt"),
                     "--memory-profile", str(preflight_output / "memory_profile.json")]
        if geometry_id is not None:
            preflight.extend(["--geometry-id", geometry_id])
        training = ["-m", f"{_MODULE}.train_unified", *common, "--config", name,
                    "--output", str(training_output), "--epochs", str(epochs),
                    "--lr", str(learning_rate), "--initialization-reference", _REFERENCE,
                    "--memory-profile", str(training_output / "memory_profile.json")]
        arms.append({
            "config_name": name,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "architecture_manifest": architecture_manifest(name),
            "initialization": initialization,
            "preflight_output": str(preflight_output),
            "training_output": str(training_output),
            "preflight_argv": preflight,
            "training_argv": training,
        })
        del model
    plan = {
        "schema": "tvcg.simplification-plan.v1",
        "stage": stage,
        "repository_root": str(repository),
        "source_commit": commit,
        "manifest_path": str(corpus.path),
        "manifest_sha256": corpus.manifest_sha256,
        "split_summary": corpus.split_summary(),
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": device,
        "initialization_reference": _REFERENCE,
        "preflight_geometry_id": geometry_id,
        "preflight_initialization": "independent_config_constructor_execution_check_only",
        "training_protocol": "fresh_common_random_reference_equal_updates",
        "scientific_status": "candidate_screening_no_performance_claim",
        "training_started": False,
        "arms": arms,
    }
    # The returned plan and its on-disk representation use the same JSON
    # types, including list-valued native configuration tuples.
    plan = json.loads(json.dumps(plan, allow_nan=False))
    # Everything above is read-only apart from transient CPU model allocation.
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for phase in ("preflight", "training"):
        path = output / ("preflight.sh" if phase == "preflight" else "train.sh")
        path.write_text(_script(plan, phase), encoding="utf-8")
        path.chmod(0o755)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="new plan directory; existing directories are rejected")
    parser.add_argument("--stage", choices=tuple(STAGES), default="screen")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=830251)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry-id")
    parser.add_argument("--allow-absolute-assets", action="store_true")
    args = parser.parse_args()
    plan = generate_plan(args.manifest, args.output, stage=args.stage, epochs=args.epochs,
                         seed=args.seed, learning_rate=args.lr, device=args.device,
                         geometry_id=args.geometry_id, allow_absolute_assets=args.allow_absolute_assets)
    print(json.dumps({"output": str(Path(args.output).resolve()), "stage": plan["stage"],
                      "arms": [arm["config_name"] for arm in plan["arms"]],
                      "source_commit": plan["source_commit"], "training_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
