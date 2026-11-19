# Phase 2.9 — Physics-consistent Learning

## Scope

Phase 2.8.6 passed the software Release Gate on commit
`45bb16628b08126fbf119e2460f043ff9d5010ef`. The next work package starts
physics-consistent learning while formal dataset construction continues in
parallel. No long training is started in this phase.

## Existing foundations audited

The current native trainer already supports:
- normalized Moment supervision;
- event-gated Jacobian supervision, with optional finite-difference audit and
  Jacobian error-bound filters;
- optional event-validity BCE, requiring explicit teacher `event_valid`
  labels rather than incorrectly reusing `derivative_valid`;
- validation summaries for Moment/Jacobian errors, Jacobian symmetry and
  minimum eigenvalue.

The native model already exposes an optional event-validity head. A learned
uncertainty head and calibrated uncertainty evaluation are not yet present;
they are not claimed as complete by this increment.

## Phase 2.9.1 — Convex Moment-Jacobian consistency regularization

For a twice-differentiable scalar convex potential, its Hessian (the Moment
Jacobian under the model's potential semantics) should be symmetric positive
semidefinite. The trainer now provides an opt-in structural regularizer:

```
L_structure = mean((J - J^T)^2) + mean(relu(-eigvalsh((J+J^T)/2))^2)
```

The total training objective becomes

```
L = L_moment + w_J L_Jacobian + w_event L_event + w_physics L_structure
```

The term is enabled only when `convex_potential_enabled=True`; attempting
to enable it for a non-convex head fails closed. Its default weight is zero,
so existing training runs and checkpoints retain their previous objective.
The CLI exposes `--physics-weight`, and the exact value is part of the
resume contract to prevent resuming with a different objective.

This regularizer is not a replacement for teacher Jacobian labels or exact
finite-difference audits. It penalizes structural violations; it does not
prove physical correctness or global convexity of the full learned mapping.

## Tests added

- zero penalty for symmetric PSD matrices;
- positive penalty for asymmetric and indefinite matrices;
- invariance under orthogonal basis changes;
- finite gradients and rejection of malformed/nonfinite inputs;
- opt-in trainer integration and rejection for non-convex configurations.

## Next work

Phase 2.9.2 should add a calibrated uncertainty interface only after deciding
the prediction target (per-query Moment residual, Jacobian residual, or both),
the distributional parameterization, and calibration protocol. Candidate
metrics include NLL, coverage and interval width, evaluated on held-out
geometry/source families. Do not use uncertainty as an optimizer acceptance
criterion until calibration is demonstrated.

Phase 2.9.3 should define multi-task loss balancing and event-validity
calibration, with all task weights and masks recorded in the training
contract.

## Data and scientific gates

TRELLIS Mini7 remains pilot-only. The additional 100 TRELLIS models are
formal training candidates pending processing and provenance/quality/leakage
audit. Validation includes eligible geometries from both Spin-It and
Spin-It Faster and additional TRELLIS artifacts. Split by geometry and
source family, preserving a separate locked test set.

Formal training, multi-seed stability, 64^3–512^3 evaluation, OOD, RTX 5090 D
performance and full hybrid optimizer acceptance remain BLOCKED until the
formal dataset and independent physical verification are available.
