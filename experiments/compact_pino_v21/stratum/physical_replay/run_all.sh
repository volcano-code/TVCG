#!/usr/bin/env bash
# One context per geometry, shared read-only via fork. Interrupted runs resume from verified row/column files.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
align=/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1
out=${1:?pass a scratch output directory}
mkdir -p "$out/logs"
run_geometry() {
  {
    "$align/run.sh" "$2" "$here/replay_registered.py" --asset "$1" --workers "$3" --mode centers --integrator fast --out "$out" &&
    "$align/run.sh" "$2" "$here/assist_columns.py" --asset "$1" --workers "$3" --cpus "$4" --pending --out "$out" &&
    "$align/run.sh" "$2" "$here/replay_registered.py" --asset "$1" --mode complete --integrator fast --out "$out"
  } > "$out/logs/$1.log" 2>&1
}
run_geometry maskB 4,5,12,13,20,21,28,29 4 4,5,12,13 &
mask_pid=$!
run_geometry combB 6,7,14,15,22,23,30,31 4 6,7,14,15 &
comb_pid=$!
run_geometry R01 8,9,16,17,24,25 5 8,9,16,17,24 &
r01_pid=$!
run_geometry R02 10,11,18,19,26,27 5 10,11,18,19,26 &
r02_pid=$!
python "$here/affinity.py" "$out" > "$out/logs/worker_affinity.log" &
affinity_pid=$!
failed=0
for pid in "$mask_pid" "$comb_pid" "$r01_pid" "$r02_pid" "$affinity_pid"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
