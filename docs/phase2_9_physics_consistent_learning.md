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


## Phase 2.9.3.2 — Event-conditioned Jacobian supervision

The first implementation is an opt-in teacher-conditioned Jacobian loss
policy, not a new predicted-event-conditioned physical mapping. This avoids
feeding the Event Head's own probability back into the derivative path and
creating a self-confirming gradient loop.

The policy intersects two independent masks:
- `derivative_valid`: the teacher Jacobian label is admissible;
- `event_label_valid`: the teacher event class is known.

Only rows passing both masks contribute. `event_valid=True` is treated as
the teacher-defined regular regime; `event_valid=False` as the
teacher-defined event/nonsmooth regime. Unknown event labels are excluded.
No event class is inferred from derivative validity or predicted event logits.

The opt-in trainer parameters are:
- `--event-jacobian-weight` (default 0; disabled path is unchanged);
- `--event-regular-jacobian-scale` (default 1);
- `--event-event-jacobian-scale` (default 0).

The two scales are relative sample weights inside the Jacobian loss. This
supports conservative supervision on regular rows while suppressing
classical-Jacobian regression on teacher-labeled nonsmooth rows. If no
admissible labeled rows remain, the conditioned Jacobian term is zero.
These weights must be chosen using training/validation groups only.

This increment does not yet add an event embedding to the forward Moment
operator. Such a feature would change the mathematical function being
differentiated and requires a separate design with a well-defined exogenous
event-state input and complete raw-coefficient chain rule. The current
teacher-conditioned loss policy preserves the exact derivative of the
existing Moment mapping.

Formal event-boundary behavior and any near-event categories remain BLOCKED
until the dataset provides audited event labels with documented semantics.
Event probabilities do not override exact contact/event checks or physical
acceptance.


## Phase 2.9.3.3 — Event calibration and grouped risk evaluation

The event evaluation toolkit now supports:
- tie-correct AUROC and average precision/AUPRC, returning null when only
  one class is present;
- Brier score, ECE, confusion counts, precision, recall and F1 at an
  explicitly supplied threshold;
- per-geometry/source-group metrics and macro Brier/ECE/AUROC;
- deterministic threshold selection on a calibration mask only, with an
  explicit warning not to retune on the locked test;
- percentile bootstrap intervals clustered by geometry/source group, never
  by individual query rows.

The calibration API accepts teacher `event_valid` targets and an independent
`event_label_valid` mask. It does not infer event labels from
`derivative_valid`. Threshold selection requires both classes in the
calibration subset. Cluster bootstrap requires at least two independent
geometry/source groups and resamples whole groups with replacement.

These are evaluation primitives. They do not create a dataset split or load
model predictions automatically. A downstream evaluator must provide
predictions from a frozen checkpoint and use a pre-declared calibration split;
validation and locked-test predictions must remain separate. Event
probabilities are risk indicators only and never replace exact contact/event
checks or physical acceptance.

Formal event calibration, source-family confidence intervals and risk
claims remain BLOCKED until Work's multi-source dataset has audited event
labels and a frozen geometry/source-family split.


## Phase 2.9.3.4 — Event-aware release contracts

A dedicated release contract module and CPU tests now verify:
- `event_valid`, `event_label_valid`, and `derivative_valid` are
  independently supplied boolean vectors with the query axis shape;
- event-label coverage is reported separately from derivative-validity
  coverage, without deriving one mask from another;
- software contract readiness is separated from scientific readiness;
- event label coverage, event-boundary calibration and source-family
  generalization remain BLOCKED until audited formal data and trained
  checkpoints exist;
- learned event probabilities cannot serve as physical solver acceptance;
- Mini7 remains pilot-only, the additional 100 TRELLIS models remain
  candidates pending audit, and validation includes eligible geometries
  from both papers plus additional TRELLIS artifacts;
- split unit is geometry/source family and locked test remains unseen during
  threshold selection.

Run the focused release contract tests with:

```bash
python -m pytest -q research/NeuralMoments/models/pcraft_ufno/release_gate/tests/
```

These tests validate software contracts only; they do not claim that event
labels or calibration are scientifically adequate on the formal dataset.


## Phase 2.9.4.1 — Multi-task balancing primitives

The first increment adds independent, opt-in utilities:
- fixed weighted objective as the baseline;
- per-task gradient extraction over an explicitly supplied shared-parameter set;
- symmetric PCGrad projection with pre-projection cosine matrix, task norms,
  projected norms and projection counts;
- a stateful GradNorm weight controller that updates only tasks present in
  the current batch;
- homoscedastic learned task weighting, kept distinct from predictive
  aleatoric Moment/Jacobian UQ scales.

These are primitives, not yet a claim that the native training loop uses
PCGrad or GradNorm end-to-end. In particular, PCGrad must be applied to
separate task gradients on shared parameters before they are summed; applying
it to an already aggregated scalar loss is invalid. Task-specific head
gradients must still be retained. The default native objective and existing
training CLI remain unchanged in this increment. A later integration step
must define the shared parameter groups, absent-task semantics, per-task
gradient extraction cost and checkpointed balancer state before enabling
adaptive balancing in formal runs.

No long training is started. Task-balance effectiveness, convergence and
generalization remain data-dependent and blocked pending the audited
multi-source corpus and frozen geometry/source-family splits.


## Phase 2.9.4.2 — Per-task shared-gradient integration

The native P-CRAFT trainer now exposes an opt-in `--multitask-mode pcgrad`
path. It computes Moment, Jacobian, Event, UQ and Physics task losses
separately, preserving the fixed weighted objective as the default
`fixed` mode.

For PCGrad mode, shared parameters are explicitly partitioned from the
Event and UQ task-specific heads. Each active weighted task loss is
differentiated separately with `autograd.grad`. PCGrad projection is
performed only on the flattened shared-parameter gradients before they are
combined. Task-specific head gradients are accumulated directly with their
task weights and are not projected against unrelated heads.

The partition is checked for overlap, duplicates and exhaustive parameter
coverage. Per-step history records task losses, task label counts, shared
gradient norms, private-head gradient norms, cosine matrix and projection
counts. The selected mode is part of the strict training resume contract.

This is the native P-CRAFT trainer integration. It does not yet claim that
the separate Unified M80+SQNO wrapper training pipeline uses the same
gradient controller; that needs its own shared-parameter audit and adapter
integration before claiming end-to-end unified training support.

GradNorm and learned task weighting remain for Phase 2.9.4.3. The current
PCGrad path does not use predictive aleatoric UQ scales as task weights.
Default `fixed` mode reconstructs the existing weighted objective exactly.
No formal training is started; task-balance efficacy remains blocked on the
audited multi-source dataset.


## Phase 2.9.4.3 — Adaptive scheduling primitives and safe boundaries

This increment hardens adaptive task-scheduling state without silently
changing the current native training objective.

GradNorm state now has an explicit, contract-checked `state_dict` /
`load_state_dict` interface. It persists task names, hyperparameters,
initial reference losses and current weights; it rejects task-set,
hyperparameter and non-finite-state mismatches. Missing tasks remain
untouched by the existing update method.

An event positive-class weighting helper now computes a bounded
negative/positive ratio using only explicitly labeled rows. It consumes
`event_valid` and the independent `event_label_valid` mask; unknown rows
are excluded, and a training split with no positive examples fails closed.
The report includes class counts, clipping status and provenance. This
helper must be computed from the training partition only; validation or
locked-test labels must never be used to estimate training weights.

Learned homoscedastic task weights remain a separate module whose parameters
and optimizer state can be checkpointed with the normal PyTorch
`state_dict` interface. They are not the predictive aleatoric UQ scales.

Important implementation boundary: this increment does not yet activate
GradNorm or LearnedTaskWeights in the native training loop. Doing so safely
requires a dedicated controller update schedule, a separate optimizer/state
contract for learned weights, and per-task shared gradient norms that are
computed before PCGrad projection. The existing fixed and PCGrad training
modes remain unchanged. The next integration step must be explicit and
opt-in; these primitives alone are not evidence of adaptive training.

No formal training was started. Event class balance, adaptive-weight
stability and downstream scientific benefit remain data-dependent and
blocked pending the audited multi-source dataset and matched experiments.
