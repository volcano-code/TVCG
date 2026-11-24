#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
centers=${1:?Pass the physical replay centers directory}
geometric=${2:?Pass the conditional geometric proof directory}
out=${3:?Pass a fresh verification output directory}
python=${TVCG_NATIVE_PYTHON:-/home/lin/anaconda3/bin/python}
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 OPENBLAS_NUM_THREADS=1
"$python" "$here/build_probe.py" "$out/build"
export TVCG_ADMISSION_PROBE="$out/build"
"$python" -m pytest -q "$here/test_admission.py"
for pair in R02_R8_032,2 maskB_R8_035,4; do
    row=${pair%,*}
    ulps=${pair#*,}
    "$python" "$here/certify.py" --probe "$out/build" --center "$centers/$row.json" \
        --conditional "$geometric/$row.json" --ulps "$ulps" --out "$out/$row.json"
done
"$python" "$here/inspect_trace.py" "$centers" "$out/interval_qr.json"
