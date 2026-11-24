#!/usr/bin/env bash
# `native-root` is produced by the V2.1 original-mesh native export.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
cal_work=${1:?need a new or reusable output directory}
native_root=${2:-/tmp/tvcg-v21-native}
cal_output=${3:-$cal_work/evidence}
cal_cpus=${CALIBRATION_CPUS:-8-11,24-27}
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
mkdir -p "$cal_work" "$cal_output"
py=(taskset -c "$cal_cpus" /home/lin/anaconda3/bin/python -Bu)
cap() { local name=$1; shift; "${py[@]}" "$here/capture.py" --native-root "$native_root" --output "$cal_work/$name.json" "$@"; }
cap T2_seed
cap T2_extended --start registered --extended
cap N002_registered --geometry N002 --start registered
cap S001_registered --geometry S001 --start registered
cap T2_hyperbolic --start registered --extended --query-id T2_hyperbolic_plus
cap T2_high_hyper --start registered --extended --query-id T2_high_HYPERBOLOID
cap T2_high_ell --start registered --extended --query-id T2_high_ELLIPSOID
cap T2_mid_ell --start registered --extended --query-id T2_middle_ELLIPSOID
"${py[@]}" "$here/margins.py" "$cal_work"/{T2_seed,T2_extended,N002_registered,S001_registered,T2_hyperbolic,T2_high_hyper,T2_high_ell,T2_mid_ell}.json --output "$cal_work/margins.json"
for i in 1 2; do
    "${py[@]}" "$here/perturb_trials.py" --capture "$cal_work/T2_extended.json" --native-root "$native_root" --iteration "$i" --output "$cal_work/probe$i.json"
done
"${py[@]}" "$here/rank_witness.py" --capture "$cal_work/S001_registered.json" --native-root "$native_root" --output "$cal_work/rank.json"
"${py[@]}" "$here/summarize.py" --work "$cal_work" --output "$cal_output"
