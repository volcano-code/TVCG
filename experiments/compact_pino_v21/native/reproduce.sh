#!/usr/bin/env bash
set -euo pipefail
out=${1:?Pass the output directory for the new replay}
cpu=${2:-$(python -c 'import os; print(min(os.sched_getaffinity(0)))')}
here=$(cd -- "$(dirname -- "$0")" && pwd)
align=${V21_ALIGNMENT:-/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1}
corpus=${V21_CORPUS:-/data/xvxinyao/TVCG/datasets/final_voxel_pino_corpus}
physics=${V21_PRIOR_PHYSICS:-/home/lin/Documents/Codex/2026-11-21/tvcg-phase-2-11-full-v1/outputs/v20_acceptance_1124/physics}
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
opts=(--alignment "$align" --corpus "$corpus" --physics "$physics" --out "$out")
assets=(noodles T2 combB maskB R01 R02 S001 N002
        oa_0202ec72baf548bb957406ed2d88961d oa_3529b9ef4e2c4a32add55948b5361609
        oa_5da9f52f20844b1ba33393044945ef37 oa_f6d71295090c49f590c91b9e0ee05323
        oa_04d9983ba04242fe99182d4506ea0714 oa_555d36846ba546df8344af8f9b6ebd1f)
for asset in "${assets[@]}"; do
    "$align/run.sh" "$cpu" "$here/replay.py" "${opts[@]}" --asset "$asset"
done
"$align/run.sh" "$cpu" "$here/full_motion_diagnostic.py" "${opts[@]}"
"$align/run.sh" "$cpu" "$here/chart_diagnostic.py" "${opts[@]}"
taskset -c "$cpu" bash "$corpus/../phase2_12_dataset/train_work/env.sh" "$here/diagnostic_grids.py" "${opts[@]}"
"$align/run.sh" "$cpu" "$here/review.py" "${opts[@]}" --review-out "$out/review"
"$align/run.sh" "$cpu" "$here/verify.py" "$out/review"
