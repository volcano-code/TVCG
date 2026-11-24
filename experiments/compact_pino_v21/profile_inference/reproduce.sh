#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
out=${1:-"$here"}
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
alignment=${V21_ALIGNMENT:-/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1}
corpus=${V21_CORPUS:-/data/xvxinyao/TVCG/datasets/final_voxel_pino_corpus}
physics=${V21_PRIOR_PHYSICS:-/home/lin/Documents/Codex/2026-11-21/tvcg-phase-2-11-full-v1/outputs/v20_acceptance_1124/physics}
pack=${V21_NOODLES_PACK:-/tmp/tvcg-v21-stage/final/noodles}
py=${V21_PYTHON:-/home/lin/anaconda3/bin/python}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH="$root" PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export NUMBA_NUM_THREADS=2 PYTHON_CPU_COUNT=2
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cd -- "$root"
for mode in M joint; do
    export NUMBA_CACHE_DIR="$work/jit-$mode"
    mkdir "$NUMBA_CACHE_DIR"
    /usr/bin/time -f '%e' -o "$work/$mode.elapsed" nice -n 10 taskset -c 6,7 "$py" -Bu "$here/worker.py" \
        --alignment "$alignment" --corpus "$corpus" --physics "$physics" --pack "$pack" --mode "$mode" --output "$work/$mode.json"
done
"$py" -B "$here/report.py" --work "$work" --output "$out" --alignment "$alignment" --pack "$pack"
