# Phase 2.7.2 — UG-PINO trainable baseline handoff

Date: 2026-11-18
Repository: volcano-code/TVCG
Branch: research/phase2-6-r3-m69-hybrid-ablation

## Decision

Do not create a duplicate trainer. The existing M80 native training entry point is the current trainable operator core:

`research/NeuralMoments/models/pcraft_ufno/spin_pino/native_train.py`

Model implementation:
`research/NeuralMoments/models/pcraft_ufno/spin_pino/native_pcraft.py`

It trains the P-CRAFT native interval operator with a geometry-conditioned moment response and a Jacobian obtained by differentiating the same predicted moment function. The model supports optional GNO, Transolver, multi-resolution P-CRAFT and convex-potential variants through named presets. These are controlled architectural candidates, not all mandatory components of one serial stack.

The initial UG-PINO trainable baseline should retain M80.1 P-CRAFT as the permanent control. For the first 5090 pilot, use the known single-scale `large` profile unless a fresh GPU preflight validates another profile. Do not default to `xl_ms_gno_transolver`: the M80 status records multiscale/GNO GPU preflight as pending and long training not started.

SQNO/QuadrANN remains a separately implemented geometry-once/query-many candidate. It should only be integrated into the shared UG-PINO feature path after a matched-budget ablation shows benefit; do not duplicate geometry encoders by simply serially stacking two complete models. Ensemble-based uncertainty calibration and the M69 Neural-Newton outer loop remain system-level components, not outputs already jointly trained by the current M80 trainer.

## What the current trainer already does

- Strict frozen M69/M72 corpus manifest and audit receipt validation.
- Manifest fingerprint and per-file geometry/label SHA-256 verification.
- Source-group train/validation/test isolation.
- Query, ten-moment and 10x10 Jacobian shape validation.
- Derivative-valid masking for Jacobian supervision.
- Joint normalized Moment and Jacobian loss.
- AdamW, gradient clipping, deterministic seed and wall-clock/step budgets.
- Per-query validation tail metrics (P50/P95/P99/max), plus Jacobian symmetry/eigenvalue diagnostics.
- Best and last checkpoints, exact-resume contract with source/data/runtime/config hashes.
- Optional one-time test evaluation, separate from model selection.

This is a trainable M80 baseline, not proof of a final-quality UG-PINO. It does not by itself train a calibrated uncertainty model, execute Neural Newton, or demonstrate exact-solver acceleration.

## Local 5090 workflow

Set paths to the actual frozen corpus and audit receipt. Do not guess these paths:

```bash
cd /data/xvxinyao/TVCG

export PYTHONPATH="$PWD:$PWD/research/NeuralMoments"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export M72_MANIFEST="/ABSOLUTE/PATH/TO/FROZEN_M72_FORMAL_MANIFEST.json"
export M72_AUDIT="/ABSOLUTE/PATH/TO/M72_FORMAL_AUDIT.json"
export OUT="$PWD/runs/ug_pino_large_pilot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

python research/NeuralMoments/models/pcraft_ufno/spin_pino/native_train.py preflight \
  --manifest "$M72_MANIFEST" \
  --audit "$M72_AUDIT" \
  --out "$OUT/preflight.json" \
  --preset large \
  --device cuda:0 \
  --amp bf16 \
  --batch-queries 16 \
  --chunk 2048 \
  --seed 270201
```

Only if preflight succeeds and the corpus meets the formal query/group gate, run the bounded pilot:

```bash
python research/NeuralMoments/models/pcraft_ufno/spin_pino/native_train.py train \
  --manifest "$M72_MANIFEST" \
  --audit "$M72_AUDIT" \
  --out "$OUT/train" \
  --preset large \
  --device cuda:0 \
  --amp bf16 \
  --max-steps 300 \
  --max-seconds 7200 \
  --min-queries-per-record 256 \
  --min-train-groups 50 \
  --min-val-groups 7 \
  --min-test-groups 10 \
  --batch-queries 16 \
  --chunk 2048 \
  --jacobian-weight 0.05 \
  --val-every 25 \
  --save-every 25 \
  --val-max-records 7 \
  --val-max-queries 256 \
  --seed 270201
```

If the frozen corpus does not have at least 256 queries for every qualified record, do not lower the formal threshold and call the run a formal training result. A separate wiring smoke may use a reduced query threshold, but it must be labelled smoke-only and cannot support scientific accuracy claims.

## Same-day acceptance definition

A same-day run counts as “trainable baseline verified” only if:
1. preflight completes on the RTX 5090;
2. at least one optimizer step updates finite parameters;
3. Moment and Jacobian losses/gradients are finite;
4. validation executes on the held-out validation groups;
5. best/last checkpoint save and reload succeed;
6. resume accepts the exact same contract;
7. report records data/source/runtime/config hashes and P50/P95/P99/max metrics.

This is an engineering training milestone. It is not a perfect model, 512^3 quality result, trained UQ model, real TRELLIS generalization, Newton speedup or CCF-A result.

## Next architecture work

1. Keep `large` M80.1 as permanent baseline.
2. Run matched-budget pilot for `large_gno`, `large_transolver`, `large_gno_transolver`; only promote if held-out tails/downstream outcomes improve at acceptable memory and wall-clock cost.
3. Run multiscale profiles only after a fresh 5090 GPU preflight; do not infer GPU readiness from CPU CI.
4. Add a shared-cache SQNO adapter only after proving it does not duplicate the P-CRAFT geometry encoding and improves query-many total cost.
5. Train at least three seeds before UQ calibration; calibrate risk-coverage on held-out geometry groups.
6. Integrate the trained operator into M69 Neural Newton with exact admission/fallback; only then measure end-to-end acceleration.

## Current hard blocker

The M80 status snapshot says 67 qualified records across 50/7/10 source groups, but the formal 256-query-per-record corpus is not confirmed. It also says long training has not started, multiscale GPU preflight is pending, calibrated UQ is unavailable and solver end-to-end is false. The local manifest and audit receipt must be checked before claiming the formal training gate is open.
