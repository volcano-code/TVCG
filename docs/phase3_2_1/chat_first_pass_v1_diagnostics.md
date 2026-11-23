# Phase 3.2.1 — Executable Jacobian Diagnostics and Verified Scope

Historical scope through commit `7a522c5`. The subsequent model simplification,
implemented on 2026-11-24, is documented in
[Phase 3.2.2 Work instructions](../phase3_2_2/WORK_INSTRUCTIONS.md).
Statements below about the diagnostics-only revision do not describe that
subsequent implementation.

## Baseline

- Repository: `volcano-code/TVCG`
- Base commit: `0d4231674a7fe4df048f561165ee67f9247be050`
- Working branch: `research/phase3-2-chat-v1-optimization`
- First-pass head: `b8194f421e0c835bbe941cf0f14a31ea36ce71e0`
- The local working tree reported by the owner contains uncommitted `WORK_HANDOFF.md` and untracked `datasets/`, `docs/phase2_9_6/`, `reports/`. These paths are intentionally not part of this branch's changes.

## Scope correction — 2026-11-23

The first pass added exactly three files (315 additions, no deletions): this
document, `spin_adapter/jacobian_diagnostics.py`, and its unit tests. It did
**not** change the model architecture, training loop, teacher, optimizer, or
checkpoint. Calling that pass an overall framework upgrade would be incorrect.

The baseline already includes shared P-CRAFT features, M80 and SQNO branches,
query-conditioned fusion, geometry-physics attention, cross-scale attention,
and the Full V1 GNO/Transolver configuration. Its factory explicitly records
`scientific_status = "candidate_configuration_not_performance_validated"`.
The Phase 2.11 streaming/chunking/checkpointing work also predates this PR.

The unified model already uses SQNO's `jacobian_input` (raw coefficients),
including coefficient normalization. Its query-conditioned fusion already
includes the product-rule term:

```text
J = alpha * J_m80 + (1 - alpha) * J_sqno
    + (M_m80 - M_sqno) outer d(alpha)/d(raw_coefficients)
```

Neither term is a newly implemented architecture improvement in this PR.

## Historical inputs to the first pass

The first-pass document recorded four observations supplied by Work B. These
are retained as reported inputs, not experiments independently reproduced by
this revision:

1. One hyperboloid query differs by 32.5% between the historical analytic cut and the native discrete cut. Native-cut-then-voxelized mass differs from native discrete mass by about 0.009% for that query. This is a cut-contract discrepancy, not evidence of a global 32.5% label error.
2. On 24 valid points from the native successful noodles trajectory, Epoch 36 has 15.53% aggregated ten-moment relative error and 71.73% aggregated full-Jacobian relative error.
3. Five held-out source groups show a 621.68% equal-source average low-retention relative error; low true values amplify relative error, while the mean predicted retained mass is still high by 17.25 percentage points of editable mass.
4. The low-retention pilot improves the low-retention Moment metric by 5.45% but worsens the high-retention metric by 6.85%, crossing the 5% guard. Its noodles Jacobian error remains 71.72%.

These reports motivate separating cut-contract validity, model derivative
self-consistency, native-physics accuracy, and optimizer-trajectory accuracy.
They do not by themselves identify a derivative bug or justify replacing the
geometry trunk.

## First-pass code addition

`spin_adapter/jacobian_diagnostics.py` adds a read-only diagnostic utility:

- raw-coefficient central finite differences over a sweep of step sizes;
- full Frobenius, absolute, RMSE/MAE and per-coefficient-column Jacobian errors;
- optional output-component scaling;
- directional `Jv` comparison;
- separate reports for model self-consistency and comparison against an independently supplied native Jacobian.

The utility compares the **model-reported Jacobian** with finite differences
of the same model's Moment function. A reported Jacobian may combine AD and
analytic derivatives; this comparison alone is not independent AD of the
final Moment output. Independent direct AD is a separate reference. Comparison
against a native teacher is yet another question and requires matching cut
semantics, coefficient basis, units, Moment order, and validity masks.

## Diagnostic contract corrections

- The default finite-difference calculation perturbs one query at a time and
  returns the batch Jacobian's diagonal blocks, `dM[q]/dc[q]`. It does not
  return cross-query blocks. The optional `assume_query_independence=True`
  path reduces Moment evaluations from `20 * Q` to `20`; the caller must
  establish row independence before using it.
- Step sizes must be finite and positive, produce representable perturbations,
  and form a nonempty, nonduplicated sweep. Callback output shape, dtype and
  device must remain consistent across each set of finite-difference calls.
- Directional products align dtype/device and detach tensors before `Jv`.
  Directions are caller-supplied; they count as Newton directions only when
  obtained from the actual solver trajectory.
- Reports retain each query's index and optional unique ID, along with
  absolute, relative and per-column errors. Near-zero reference norms and
  denominator flooring are flagged. Undefined cosine values are `None`,
  rather than a misleading score of zero or one.

Finite differences accumulate in FP64 but retain the forward function's
precision. A smaller step is not automatically a better reference for FP32
neural components. Compare multiple steps and absolute errors. Native event
validity masks must be applied before sending queries to these helpers; they
do not infer physical label validity.

## Import failure and repository fix

The owner's command failed during collection with
`ModuleNotFoundError: No module named 'spin_adapter'`. `spin_adapter` is inside
`quadrann_reproduction`; adding only the repository root to `PYTHONPATH` is
insufficient for the legacy top-level imports. No Jacobian assertion ran.

`pyproject.toml` now supplies both pytest search paths relative to the
repository root. After synchronizing this revision, run from the repository
root without a `PYTHONPATH` export:

```bash
python -m pytest -q \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_jacobian_input_chain_rule.py
```

For an older checkout, the equivalent temporary command is:

```bash
PYTHONPATH="$PWD:$PWD/research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction" \
python -m pytest -q \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_jacobian_input_chain_rule.py
```

The pytest configuration affects test collection; standalone Python scripts
using the legacy `spin_adapter` import still need the package parent on their
search path. Module entry points under `research.*` have their own existing
invocation contract.

## Required follow-up before training

1. Work A must publish a versioned native-cut contract and corrected teacher records, with retained side, finite cut extent, cap, fixed shell, coordinates, parameterization and reject rules explicit.
2. Work B must run the diagnostic on the same 24 valid native trajectory queries and preserve the per-query records.
3. Compare model-reported J, independent direct AD of the final Moment, and model FD separately from model-Jacobian-vs-native-Jacobian.
4. Evaluate the actual Newton directions and low/mid/high retention strata; do not accept a low-retention-only improvement that breaches the high-retention guard.
5. Only after the above pass should V1.1 training be admitted.

## Validation boundary

The import failure was reproduced locally with `PYTHONPATH` unset. After the
pytest configuration fix, the same chain-rule test command passed all three
tests. The local environment is Python 3.12.14, PyTorch 2.8.0+cpu, pytest 9.2.0;
CUDA is unavailable here. This is not a rerun in the owner's Python 3.13 / RTX
5090 D environment.

The final combined run also explicitly removed `PYTHONPATH` and used one
OpenMP/BLAS/MKL thread. It completed with **35 passed, 3 warnings in 6.82s**:

| Test file | Passed | Scope |
| --- | ---: | --- |
| `test_jacobian_input_chain_rule.py` | 3 | Existing raw-coefficient chain-rule regression |
| `test_jacobian_diagnostics.py` | 31 | Diagnostic metrics, validation, query identity, FD batch semantics |
| `test_jacobian_diagnostics_unified.py` | 1 | Real Unified module composition, independent AD, FD and negative control |

Reproduce from the repository root after synchronizing the branch:

```bash
TVCG_SPIN_TESTS=research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q \
  "$TVCG_SPIN_TESTS/test_jacobian_input_chain_rule.py" \
  "$TVCG_SPIN_TESTS/test_jacobian_diagnostics.py" \
  "$TVCG_SPIN_TESTS/test_jacobian_diagnostics_unified.py"
```

The three warnings were existing environment/plugin notices: anyio's pytest
`CallSpec2` import, an unrecognized `asyncio_default_fixture_loop_scope` option
in this environment, and torch_geometric's `torch-cluster` deprecation. They
did not cause collection or test failures.

The new integration test activates nonzero geometry-attention and cross-scale
outputs and a nonzero query-conditioned gate. It independently differentiates
the final Moment tensor, verifies the off-diagonal query blocks, checks the
raw-coefficient inverse-scale chain rule, and uses two FD steps (`2e-4`,
`1e-4`). The per-query and aggregate FD relative errors must be below `2e-4`,
and the maximum absolute error below `1e-3`. Deliberately removing the gate
product-rule term must produce an error above `1e-3` and more than ten times
the correct result at both steps.

This is a deterministic, randomly initialized tiny Unified fixture: one
synthetic geometry, the same query direction at two coefficient magnitudes,
and one valid scale token. It exercises real model modules, but does not
instantiate the full GNO/Transolver production preset or validate broad query
coverage. The 24-query helper test is also explicitly synthetic. Native
physical accuracy remains `None` in the integration report because no native
teacher was provided.

The actual Epoch 36 checkpoint, native teacher records and 24 trajectory queries
have not been evaluated by this revision. Passing small deterministic tests is
evidence about the implemented contracts, not about trained model accuracy,
training throughput, solver convergence, or readiness for full training.

## What a substantive framework upgrade still requires

1. A versioned native-cut/data contract and a reproducible diagnostic entry
   point for the exact checkpoint, manifest, geometry and query identities.
2. Per-query comparison of reported J / independent AD / FD, followed by a
   semantically matched native-J comparison with valid-event filtering and
   the actual Newton directions.
3. An identified failure mechanism before changing sampling, losses,
   quadrature, fusion or attention. A candidate core change must be visible
   in the model/trainer diff and preserve the existing physical conventions.
4. Paired validation with fixed splits, seed and training budget, covering
   Moment, full Jacobian, directional error, all retention strata, native
   trajectory behavior, memory and runtime. The reported high-retention guard
   must be checked alongside low-retention gains.
5. Only after those results support the candidate should the long-training
   configuration be admitted. This diagnostics PR does not satisfy that gate
   by itself.
