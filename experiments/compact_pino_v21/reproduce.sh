#!/usr/bin/env bash
# Rebuild the numerical study on the archive machine. Data inputs are read-only.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../.." && pwd)
out=${1:?Pass a new output directory}
cpu=${2:-$(python -c 'import os; print(min(os.sched_getaffinity(0)))')}
mkdir -p -- "$(dirname -- "$out")"
mkdir -- "$out"
out=$(cd -- "$out" && pwd)
align=${V21_ALIGNMENT:-/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1}
corpus=${V21_CORPUS:-/data/xvxinyao/TVCG/datasets/final_voxel_pino_corpus}
historical=${V21_HISTORICAL:-"$corpus/../phase2_12_dataset/manifest_v2.json"}
py=${V21_PYTHON:-python}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export V21_MATERIAL_THREADS=1
cd -- "$root"
bash "$here/native/reproduce.sh" "$out/native" "$cpu"
mapfile -t geometries < <("$py" -c 'import json,sys; print(*sorted(json.load(open(sys.argv[1]))["sources"]),sep="\n")' "$out/native/review/native_replay.json")
runner=(taskset -c "$cpu" "$py" -m)
study=experiments.compact_pino_v21
"${runner[@]}" "$study.pack_geometry" --native-root "$out/native" --indexes "$out/indexes" "${geometries[@]}"
common=(--native-root "$out/native" --indexes "$out/indexes" --trajectory "$align/reference/seed-00.json")
"${runner[@]}" "$study.replay" "${common[@]}" --output "$out/evidence/replay" '*'
"${runner[@]}" "$study.analytic_control" "${common[@]}" --output "$out/evidence/analytic" '*'
"${runner[@]}" "$study.fixtures" --cache "$out/fixtures" --output "$out/evidence/manufactured.json"
"${runner[@]}" "$study.summarize" --evidence "$out/evidence" --native "$out/native/review" --indexes "$out/indexes" \
    --release "$corpus/releases/four_split_delivery_392/manifest.json" --historical "$historical"
"$py" "$here/verify_results.py" --evidence "$out/evidence" --native-evidence "$out/native/review"
