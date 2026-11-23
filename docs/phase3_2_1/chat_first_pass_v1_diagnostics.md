# Phase 3.2.1 — Chat First-Pass V1 Diagnostics

## Baseline

- Repository: `volcano-code/TVCG`
- Base commit: `0d4231674a7fe4df048f561165ee67f9247be050`
- Working branch: `research/phase3-2-chat-v1-optimization`
- The local working tree reported by the owner contains uncommitted `WORK_HANDOFF.md` and untracked `datasets/`, `docs/phase2_9_6/`, `reports/`. These paths are intentionally not part of this branch's changes.

## Evidence-led diagnosis

The supplied Work B audit establishes four distinct observations:

1. One hyperboloid query differs by 32.5% between the historical analytic cut and the native discrete cut. Native-cut-then-voxelized mass differs from native discrete mass by about 0.009% for that query. This is a cut-contract discrepancy, not evidence of a global 32.5% label error.
2. On 24 valid points from the native successful noodles trajectory, Epoch 36 has 15.53% aggregated ten-moment relative error and 71.73% aggregated full-Jacobian relative error.
3. Five held-out source groups show a 621.68% equal-source average low-retention relative error; low true values amplify relative error, while the mean predicted retained mass is still high by 17.25 percentage points of editable mass.
4. The low-retention pilot improves the low-retention Moment metric by 5.45% but worsens the high-retention metric by 6.85%, crossing the 5% guard. Its noodles Jacobian error remains 71.72%.

These findings do not establish that the P-CRAFT/GNO geometry trunk must be replaced. They justify first separating cut-contract validity, model self-consistency, native-physics accuracy, and optimizer-trajectory accuracy.

## First-pass code addition

`spin_adapter/jacobian_diagnostics.py` adds a read-only diagnostic utility:

- raw-coefficient central finite differences over a sweep of step sizes;
- full Frobenius, absolute, RMSE/MAE and per-coefficient-column Jacobian errors;
- optional output-component scaling;
- directional `Jv` comparison;
- separate reports for model self-consistency and comparison against an independently supplied native Jacobian.

The utility does not modify the model, teacher, checkpoint, training loop, or solver. This is intentional: first establish whether the 71.73% discrepancy is (a) an internal derivative implementation mismatch, or (b) a model-to-native physics mismatch. The two failure modes require different fixes.

## Required follow-up before training

1. Work A must publish a versioned native-cut contract and corrected teacher records, with retained side, finite cut extent, cap, fixed shell, coordinates, parameterization and reject rules explicit.
2. Work B must run the diagnostic on the same 24 valid native trajectory queries and preserve the per-query records.
3. Compare AD-vs-model-FD separately from model-Jacobian-vs-native-Jacobian.
4. Evaluate the actual Newton directions and low/mid/high retention strata; do not accept a low-retention-only improvement that breaches the high-retention guard.
5. Only after the above pass should V1.1 training be admitted.

## Test command

```bash
python -m pytest -q research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_jacobian_diagnostics.py
```

This branch was authored through repository file operations. It has not been executed on the owner's RTX 5090 D, and no GPU performance or model-accuracy improvement is claimed. Work B's second pass must run the tests and integrate the diagnostic against the actual Epoch 36 model/teacher path.
