# Phase 3.2.3 — Native256 readiness and first query-path acceleration

Base cd7dc3097654681535f96d178d49b27f612e236c. This change is a code candidate, not a GPU or native-teacher accuracy result.

## What changed

- Add exact-256 configurations: ablation_g1_t1_256, ablation_no_cross_scale_256, compact_v1_candidate_256, G/T controls with _256, compact_lite_256 (32/64), compact_single_256 (64). The old names and config fingerprints are preserved. The factory sets the native minimum to 256; the unified encoder checks exact input side (not just minimum). Context grids must divide the native side. No resampling or relabeling.
- New native256 entry points require a versioned Work A receipt bound to manifest and each admitted train/dev geometry/label SHA. Verify actual NPZ side, TRELLIS source, explicit valid masks and restricted dimensions. Test labels are never opened for admission or scale fitting. Receipt verification is NOT an independent native cut audit; Work A must actually establish the physical contract before issuing it.
- Default native256 derivative normalization is train-only RMS, separate Full and restricted output scales. Full uses a [10,10] repeated per-output scale; restricted pools valid R3/R8 directional columns into a [10] scale. Separate R3/R8 counts and RMS are reported; this is not per-basis whitening. Legacy mode remains available for a paired normalization ablation. All loss weights stay fixed unless explicitly changed. Both preflight and train use the same options. No direction/physics loss is added yet.
- Validation reports source-group macro Moment and separate R3/R8 RMSE and denominators; native256 training archives each epoch. Existing pooled best_val remains diagnostic, not promotion. Native256 resume checks the cut receipt hash as well as checkpoint contracts.
- Add eval/inference-cache-only model.predict_moments(cache, q), avoiding M80 full AD rows, SQNO Hessian contractions, gate derivative and attention tangents. Existing evaluate_queries remains the M+J path. Solver integration is NOT changed. Validate parity and measure actual residual-only call frequency before claiming speedup.
- Optional prepare_inference_acceleration(cache, max_extra_bytes=...) caches exact geometry K/V and static FP64 quadrature basis/coordinates within an additional persistent-byte budget. Off by default; no training cache, no approximate subsampling, stale/mutated cache rejected. It does not eliminate the per-query full-domain reduction; actual GPU speed/peak-memory require measurement.
- New tests cover explicit configuration, wrong resolution, actual side=256 reduced-context joint update, checkpoint, admission/scale checks, moment-only parity and exact acceleration parity/budget/staleness. These tests have not been executed in the Chat repository environment; run before approval. Historical tests and checkpoint source fingerprint checks remain strict. Use historical revision for historical checkpoints.

## Work A receipt

Publish JSON (not a placeholder approval) with schema tvcg.native-cut-admission.v1, status approved_native_discrete, exact manifest_sha256, native_resolution 256, moment_order ["1","x","y","z","xx","yy","zz","yz","xz","xy"], jacobian_basis raw_coefficients. Also provide coefficient_multipliers [1,1,1,1,1,1,1,2,2,2] and retained_side positive (in model raw-coefficient coordinates; map native signs and derivatives consistently). Required nonempty fields: teacher_version, retained_side, finite_cut_extent, cap_rule, fixed_shell, coordinate_frame, units, coefficient_basis, density_rule, rejection_rule. Each records entry is keyed by normalized geometry ID in train/development and contains status approved_native_discrete, geometry_sha256 and labels_sha256. The receipt must cover exactly train/dev. No fixed-test labels used; retain locked benchmark separately. Document original teacher vs native cut reconciliation, per-query evidence, event exclusions and hash-backed review independently; software cannot establish physical correctness from a JSON string.

## Work B instructions

Use a fresh worktree of the new exact commit; preserve original datasets, reports, checkpoints. Run the Phase 3.2.2 regression suite plus:

```bash
QROOT=research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q "$QROOT/training_v13/tests/test_native256_readiness.py" \
  "$QROOT/spin_adapter/tests/test_phase3_2_3_native256.py"
```

Then, after actual Work A receipt is issued, set TVCG_MANIFEST and TVCG_CUT_CONTRACT to the approved absolute local paths; TVCG_RUN_ROOT must be new.

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.simplification_plan \
  --manifest "$TVCG_MANIFEST" --native-cut-contract "$TVCG_CUT_CONTRACT" \
  --output "$TVCG_RUN_ROOT" --stage screen256 --epochs 3 --seed 830251 \
  --lr 3e-5 --device cuda --allow-absolute-assets
bash "$TVCG_RUN_ROOT/preflight.sh"
# Only after reviewing all three preflights and native physics/data gates:
bash "$TVCG_RUN_ROOT/train.sh"
```

Select a representative native256 geometry via --geometry-id; require all three complete updates, restored predictions, finite gradients, correct masks, physical native side=256 and measured GPU memory. Repeat on thin/cavity cases. Use scales256 in a new directory to compare 64/128 vs 32/64 vs 64 with the compact topology fixed. Compare train_rms vs legacy separately on the same fresh initial state. Do not compare a new normalized compact model to the old 1024 checkpoint as a controlled architecture experiment. Profile cached M-only vs M+J with synchronized timings and verify output equivalence, then decide solver integration. Test acceleration separately with explicit byte budgets (zero, 64 MiB, 256 MiB, 512 MiB): after eval and inference encoding, call fast_cache, report = model.prepare_inference_acceleration(cache, max_extra_bytes=budget). Run both paths on identical queries; report setup, additional bytes, peak allocated/reserved, per-query P50/P95 and numerical parity. Keep it disabled if it fails parity or worsens time-to-solution. Report source macro, low/mid/high retention, Full/R3/R8, actual Newton Jv, native trajectory and feasibility; do not accept relative noninferiority to an inaccurate model as final readiness.

## Remaining priorities

Real RTX5090D production 256 update and checkpoint, physical teacher alignment on real cut queries (including 24 known Noodles trajectory points), directional/retention loss and physical bounded response, solver residual-only dispatch, target-GPU validation of the implemented bounded KV/basis cache, single M80/SQNO response experiments. No long training or native solver replacement is authorized by this code alone.
