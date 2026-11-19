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


## Phase 2.9.2.1 — Gaussian UQ contract and calibration primitives

The first UQ increment defines a distributional contract, not a trained
uncertainty model. It provides:
- bounded-log-standard-deviation Gaussian NLL for a future heteroscedastic
  head;
- calibration metrics for Moment or Jacobian residual tensors;
- empirical coverage at nominal 50/68/80/90/95/99 percent intervals;
- mean interval width, Gaussian NLL, calibration error and standardized
  residual diagnostics;
- optional macro summaries by held-out geometry/source group;
- explicit PENDING/BLOCKED reporting when Jacobian UQ predictions or labels
  are absent.

No teacher variance labels are assumed or fabricated. Gaussian NLL can train
a predictive scale against point targets once a scale head is implemented,
but calibration must be evaluated on held-out geometry/source families.
This utility does not itself create a learned UQ head or establish calibrated
uncertainty. Moment and Jacobian UQ remain separate outputs; missing Jacobian
uncertainty must not be imputed from Moment uncertainty.

Uncertainty is not an optimizer acceptance signal. Event validity, derivative
validity, and exact solver checks remain separate gates.


## Phase 2.9.2.2 — Learned Moment/Jacobian UQ heads

The native P-CRAFT operator now has an opt-in heteroscedastic UQ head.
Conditioning uses the cached global geometry latent, geometry statistics,
and the positive-scale-normalized ten-dimensional quadric query.

Outputs are kept separate:
- `moment_log_std`: `[Q,10]`;
- `jacobian_log_std`: `[Q,10,10]`.

Log standard deviations are clamped to configured finite bounds. The head
does not require teacher variance labels: when enabled, Gaussian NLL is
trained against the existing point-valued Moment labels and only those
Jacobian labels admitted by the existing derivative-supervision mask.
Targets are normalized with the same Moment and Moment×coordinate scales as
the deterministic losses. If a batch contains no valid Jacobian labels, its
Jacobian UQ likelihood is omitted rather than fabricating supervision.

Training integration is opt-in:
- `--uq-weight 0` preserves the previous objective and default architecture;
- positive `--uq-weight` enables the separate UQ head and adds Moment plus
  valid-Jacobian Gaussian NLL;
- the UQ configuration and weight are recorded in the strict resume contract;
- legacy checkpoints without UQ fields load with `uq_enabled=False`.

This implementation establishes learned predictive scales and software
integration only. It does not establish calibration. Moment and Jacobian
coverage/NLL must be evaluated independently on held-out geometry/source
families after formal data admission and training. No uncertainty output may
override event validity, derivative-validity labels, or exact physical
acceptance.


## Phase 2.9.2.3 — Risk/Coverage and reliability diagnostics

The UQ evaluation utilities now include:
- uncertainty-ranked selective risk/coverage curves, retaining low-uncertainty
  queries first and reporting residual RMSE at each retained fraction;
- optional per-geometry/source-group risk curves and macro risk summaries;
- quantile-binned reliability tables comparing predicted RMS standard
  deviation with observed RMS residual scale.

These are evaluation primitives only. They do not perform dataset splitting,
fit a calibration map, or prove that uncertainty is calibrated. Calibration
must be run separately for Moment and Jacobian predictions on a held-out
geometry/source-family split. Hyperparameter/model selection must not inspect
the locked test set. Group macro metrics must accompany pooled metrics so a
large geometry family cannot hide poor calibration on smaller families.

Risk/coverage curves quantify selective prediction behavior; they do not
authorize the optimizer to accept or reject a physical design. The exact
solver and event/derivative validity checks remain authoritative.


## Phase 2.9.3.1 — Event-label semantics and masked event learning

The event-validity target is explicitly teacher-provided and remains distinct
from `derivative_valid`. The former describes whether the query is in the
teacher-defined regular/event regime; the latter describes whether a
Jacobian target is admissible for derivative supervision. Neither label is
inferred from the other.

The native trainer now supports an independent optional
`event_label_valid` mask. Unlabeled event rows are excluded from BCE; an
all-unlabeled batch contributes zero event loss rather than fabricated
negative labels. Positive-class imbalance can be adjusted through
`--event-pos-weight`, separately from the overall `--event-weight`.
Both remain opt-in through the existing event-validity head configuration.

Event classification evaluation reports confusion counts, precision,
recall, F1, Brier score, ECE, AUROC and AUPRC when both classes exist.
Single-class subsets report AUROC/AUPRC as null rather than inventing a
ranking score. Optional geometry/source group summaries are included.

Event probabilities are risk indicators only. They must not change exact
solver event checks, derivative-validity labels, or physical pass/fail
acceptance. Dataset-wide event coverage and calibration remain BLOCKED until
the formal multi-source corpus has audited event labels and source-family
splits.
