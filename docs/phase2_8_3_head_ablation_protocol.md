# Phase 2.8.3.1–2.8.3.4 — Matched Moment Head Ablation

## Scope

This change adds a fail-closed experiment plan, multi-seed runner, aggregate
reporter and release gate for three heads on the same XL multi-scale P-CRAFT +
GNO + Transolver trunk:

- `xl_ms_gno_transolver_bounded_odd`
- `xl_ms_gno_transolver_convex` (legacy sigmoid amplitude)
- `xl_ms_gno_transolver_convex_atan`

The runner fingerprints the relevant source files at plan creation and rejects a run if the source changes afterward. The aggregate verifies that all nine runs share the same normalized source/runtime/training contract, differing only in seed and the declared head fields. The plan fingerprints the source files and fixes the native chunk size, minimum native resolution, thread/cache settings, gradient clipping, validation cadence, save cadence and validation sample budget. The run command rejects data/group/resolution arguments that differ from the admitted plan, and passes the frozen settings to every child training process. The runner checks that only head-specific config fields differ, and requires
the same frozen formal M69/M72 manifest, matching audit receipt, split,
source-group isolation, seed list, optimizer, budgets and loss weights. It
does not claim a winning head or fabricate GPU results.

## 2.8.3.1 — Fair protocol

Defaults require >=256 queries per qualified record and >=50/7/10 independent
source groups for train/validation/test. The manifest must be frozen
`m69.corpus.v1`; the audit must be valid, ready for training, blocker-free
and fingerprint-bound. These thresholds may only be changed by a documented
protocol amendment before candidate runs.

## 2.8.3.2 — Diagnostics

The native trainer reports per-object Moment/Jacobian scaled errors, query
p95/p99 distributions, convex gate fraction/logit/bound/effective amplitude,
and Jacobian symmetry/eigenvalue diagnostics. The trainer now also persists the pre-clipping global gradient norm at each step, includes its p50/p95/p99/max distribution in the report, and fails closed on non-finite gradients. Resume checkpoints retain the gradient history. The new aggregate reporter
compares these metrics across the same seeds. Convex-head Jacobian symmetry
and PSD are properties of that parameterization, not a claim about the exact
physical moment Jacobian.

## 2.8.3.3 — Multi-seed run

```bash
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.head_ablation plan \
  --manifest /path/to/operator-manifest.json --audit /path/to/audit.json \
  --output runs/phase2_8_3_head/plan --seeds 830301,830302,830303 \
  --min-queries 256 --max-steps 3000 --max-seconds 86400 --device cuda:0 --amp bf16
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.head_ablation run \
  --plan runs/phase2_8_3_head/plan/plan.json \
  --manifest /path/to/operator-manifest.json --audit /path/to/audit.json \
  --output runs/phase2_8_3_head/results --min-queries 256 --min-native-resolution 512
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.head_ablation aggregate \
  --plan runs/phase2_8_3_head/plan/plan.json \
  --run-index runs/phase2_8_3_head/results/run-index.json \
  --output runs/phase2_8_3_head/aggregate.json
```

The matrix contains 3 heads x 3 seeds. Test split evaluation is deliberately
not enabled during selection. The report retains stop reason and step count;
wall-time-limited runs must not be presented as full-step matched comparisons.

## 2.8.3.4 — Freeze gate

The aggregate status is always `pending`; it cannot automatically freeze a
head. Freeze requires complete matched runs, Moment and Jacobian accuracy/tails,
gradient stability, saturation review, checkpoint parity and independent
exact-solver replay. If evidence is incomplete, retain the legacy sigmoid
checkpoint as compatibility baseline and continue without freezing.

CI validates code/contracts only; it does not run GPU training.
