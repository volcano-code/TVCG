# Phase 2.9 — Physics-consistent Learning

**Phase 2.9.5 coordinate correction:** native coefficients use doubled cross
terms while reported moments do not. In structural formulas below, the
Hessian is now `H = J_c @ B^-1`, with
`B = diag(1,1,1,1,1,1,1,2,2,2)`; raw `J_c` need not be symmetric.
See [Faster107 pilot readiness](phase2_9_5_faster107_pilot_readiness.md) for
the corrected convex-head semantics, partial derivative supervision, current
reported dataset counts, and the independent development limitation.

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


## Phase 2.9.4.4 — Unified M80 + SQNO task-level PCGrad

The unified adapter now exposes differentiable task decomposition:

- fused Moment loss;
- fused raw-coefficient Jacobian loss;
- branch auxiliary loss (M80 and SQNO branch Moment/Jacobian supervision).

The existing fixed objective is reconstructed exactly as
`moment + jacobian_weight * jacobian + branch_aux_weight * branch_aux`.
The `train_step` API accepts `gradient_mode="fixed"` (default) or
`gradient_mode="pcgrad"`. PCGrad computes the three task gradients
separately over the audited shared parameter group, projects conflicts,
combines and clips the resulting gradient, and reports active task names,
cosine matrix, projection counts and gradient norms. Optimizer construction
and every training step check unique, complete trainable parameter coverage.

`training_parameter_partition()` returns ordered parameter names:

| Group | Members | Reason |
|---|---|---|
| shared | P-CRAFT geometry encoder, M80 query/convex head, SQNO decoder, fusion and enabled attention | Both branch decoders affect multiple objectives; fusion/attention affect the fused Moment and Jacobian objectives. |
| private | Empty for the current three-task objective | Branch ownership alone does not make a decoder task-private. |
| inactive | Optional native Event and UQ heads | Unified losses do not call these heads. They remain in the optimizer's complete parameter inventory but keep `grad=None`. |

The general PCGrad assignment also preserves `grad=None` for shared parameters
unused by every active task, including parameters with existing AdamW state.
Weight decay and historical momentum therefore cannot update an unused head.
Task-private gradients retain their weighted sum.

Unified PCGrad now defaults to `pcgrad_reduction="sum"`. With no gradient
conflicts, this agrees with the fixed weighted-sum objective rather than
introducing a factor of `1 / active_task_count`. Explicit `"mean"` remains
available and is recorded in the training contract. The low-level assignment
and existing native trainer retain their historical `"mean"` default. These
are explicit choices, not interchangeable experiment settings.

Training metrics retain the component names and their `*_loss` aliases.
`valid_queries` and `valid_jacobian_queries` count their respective masks;
`joint_training_loss` returns the actual fusion tensor, including
query-conditioned weights. `train_step` reports its mean as a scalar.
Fixed-mode reports list only active weighted objectives and explicitly mark
task-gradient diagnostics as unavailable; branch diagnostic terms are not
misreported as additional optimization tasks.


### PCGrad complex-gradient correction (Unified M80 + SQNO integration)

Unified P-CRAFT contains spectral/operator parameters whose autograd gradients
may be complex-valued. PCGrad therefore uses the real Euclidean inner product
on complex vectors, equivalently the real part of the Hermitian product
`Re(sum(conj(g_i) * g_j))`. Projection coefficients and cosine diagnostics
are real scalars, while assigned gradients retain each parameter's dtype.
This is mathematically the real-parameter-space projection under the
identification of each complex coordinate with its real and imaginary parts.

Assignment packs complex gradients into adjacent real/imaginary coordinates,
resolves lazy conjugate views, and allocates aligned storage before rebuilding
complex tensors. This handles an odd number of preceding real coordinates
without discarding imaginary gradients or relying on `contiguous()` to reset
a storage offset. Regression tests exercise actual optimizer updates with
mixed real/complex dtypes, not just successful assignment.

### Training checkpoint and resume contract

`training_contract(...)` records the mode, reduction, task weights, clipping,
normalization scales, named partition, parameter shapes and dtypes. Checkpoints
also store ordered optimizer parameter names and state IDs, optimizer settings,
scheduler configuration/state, model state, CPU/NumPy/Python/CUDA RNG state,
data fingerprint and training source hashes (including `multitask.py`).

After a successful `train_step`, `save_checkpoint(..., optimizer=...)`
automatically saves its training contract. A nonempty data fingerprint is
required; it must identify the actual batch/label/mask provenance and split.
An explicit contract can also be saved before the first update. A contract
different from the last successful training step is rejected.

Example API usage within a caller-owned training loop:

```python
settings = dict(gradient_mode="pcgrad", pcgrad_reduction="sum",
                jacobian_weight=.1, branch_aux_weight=.25, gradient_clip=1.,
                moment_scale=moment_scale, jacobian_scale=jacobian_scale)
contract = model.training_contract(**settings)
optimizer = model.build_optimizer(learning_rate=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.7)

cache = model.encode_geometry(geometry)
metrics = model.train_step(optimizer, cache, q, target_m, target_j,
                          valid_mask=moment_mask, jacobian_valid_mask=jacobian_mask,
                          **settings)
scheduler.step()
model.save_checkpoint(path, optimizer=optimizer, scheduler=scheduler,
                      data_fingerprint=data_fingerprint, extra={"completed_steps": 1})

# Construct the same architecture, optimizer and scheduler for `restored`.
restored.load_checkpoint(path, optimizer=restored_optimizer,
                        scheduler=restored_scheduler,
                        expected_training_contract=restored.training_contract(**settings),
                        expected_data_fingerprint=data_fingerprint, restore_rng=True)
# Re-encode geometry before the next update. Never reuse an old training cache.
```

Resume checks the expected contract and actual serialized state before loading
live parameters, optimizer, scheduler or RNG state. Parameter order changes,
missing AdamW moments, altered runtime LR/scheduler settings, and incompatible
mode/weights/normalization fail before mutation. The resumed model also rejects
a subsequent `train_step` that silently changes the contract. Deterministic
RNG restoration requires a compatible CPU/CUDA runtime and CUDA device count;
cross-backend bitwise parity is not claimed.

StepLR is exercised by the end-to-end resume tests. Scheduler contracts include
the configuration and types of nested data-driven schedules. Callable schedules
such as LambdaLR and MultiplicativeLR, including nested uses, are rejected:
their state dictionaries do not identify the callback implementation, so a
different callback could otherwise silently alter the resumed learning rate.

Normal fixed and PCGrad resume tests first create real AdamW state, save, then
compare the next uninterrupted and resumed update exactly, including complex
moment buffers, scheduler and RNG. Separate tests check raw Jacobian finite
differences and positive coefficient-scale invariance after an optimizer step.

Inference-only checkpoint loading does not require a training contract, but
it still enforces the existing architecture and source-fingerprint checks.
The narrowly defined historical global-fusion migration remains supported;
arbitrary checkpoints from changed source versions are not silently accepted.
Legacy trained optimizer state without a training contract is rejected for
resume. The historical empty-optimizer roundtrip remains supported.

### Paired short training diagnostic

`spin_adapter.unified_training_ablation.run_training_ablation` accepts a model
and an ordered sequence of caller-supplied geometry/query/target/mask batches.
It trains two independent clones from the identical complete state. Both arms
use the same batch order, normalization, per-step random seeds, AdamW settings,
clipping threshold and number of optimizer updates. The caller's model, inputs,
existing gradients and RNG streams are preserved, including on failure.

Each update encodes geometry again. Reports contain source hashes, architecture
and training/optimizer contracts, initial/input/final state hashes, normalized
losses before and after training, per-step losses and gradient norms, PCGrad
cosines/projection counts, real/imaginary parameter changes and synchronized
step timing. Fixed-mode task cosines are not collected. Timing includes fresh
encoding and training, excludes evaluation, and has no benchmark warmup.

The built-in CLI is bounded to 1--100 steps per arm. Its synthetic fixture has
five geometry columns and four queries with artificial zero targets; its
nominal `side=512` metadata is not a 512-cubed resolution experiment. Two
Fourier modes exercise nonzero imaginary spectral updates. The fixture model
initialization seed is fixed and recorded; `--seed` selects replayed per-step
RNG seeds, not independent model-initialization replicates.

From the repository root:

```bash
export ADAPTER="$PWD/research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter"
export PYTHONPATH="$PWD:$(dirname "$ADAPTER")${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m pytest -q \
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_multitask.py \
  "$ADAPTER/tests/test_unified_m80_sqno.py" \
  "$ADAPTER/tests/test_unified_training_state.py" \
  "$ADAPTER/tests/test_unified_training_ablation.py"

python -m spin_adapter.unified_training_ablation \
  --steps 3 --device cpu \
  --output runs/phase2_9_4_4/unified_training_smoke.json

python -m research.NeuralMoments.models.pcraft_ufno.release_gate.run_release_gate \
  --profile full --output runs/phase2_9_4_4/release
```

Equal optimizer-step budgets do not imply equal FLOPs, memory or wall time:
PCGrad differentiates each task separately. The report explicitly marks compute
budgets as unmatched, timings as diagnostic, and PCGrad superiority as
unestablished. This software increment provides a repeatable short-run harness,
not a formal unified dataset trainer or scientific fixed-vs-PCGrad ablation.
Formal convergence, accuracy, runtime tradeoffs and downstream hybrid acceptance
remain pending the audited multi-source corpus and matched experiments. Default
training remains fixed; GradNorm and learned task weighting remain unintegrated.
