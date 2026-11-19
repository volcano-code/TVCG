# Phase 2.8.5 — Advanced Geometry–Physics Attention Design

## Purpose and current code audit

The current `native_pcraft.py` already contains mass-aware deterministic
`PhysicsStateSlicing` (Transolver-style slice/pool/attention/deslice),
optional GNO message passing with cached neighborhood graphs, optional
multi-scale feature extraction and scale gates, and a shared interval feature
cache exported to SQNO. Phase 2.8.5 must not add a generic Transformer merely
to duplicate these mechanisms.

## Frozen Phase 2.8.4 interface

Keep one shared `PcraftNativeMomentOperator`, one SQNO branch in external
`pcraft_width` mode, one geometry encoding reused across Query-many calls,
physical interval masses, cache identity/version checks, and raw-input
coefficient Jacobians. Do not change the Faster task, positive-side convention,
ten-moment ordering, normalization, or exact-solver acceptance. Do not freeze
a Moment Head from Mini7.

## 2.8.5.1 — Geometry–Physics Cross Attention

For each quadric query `beta`, construct a token from its normalized
ten-dimensional coefficients and invariant query descriptors. Use it as Q
over cached shared P-CRAFT interval tokens as K,V. Add a numerically safe
log-physical-mass key prior; optional geometric bias may use normalized
coordinates and interval descriptors with an explicit coordinate convention.

Requirements:
- batched `[Q,10]` queries and chunked interval attention;
- mass-aware stable softmax;
- interval permutation invariance and query-row-order invariance;
- reuse cached interval features; never rerun geometry encoding per query;
- finite output/gradients and attention entropy/effective-support diagnostics;
- bounded memory; no unchunked `Q x N x N` tensors;
- selectable identity/no-attention ablation path.

## 2.8.5.2 — Cross-scale Attention

Operate on existing per-scale summaries/features rather than creating a
duplicate voxel pyramid. Every scale needs an explicit ID/resolution.
Missing-scale masks must be deterministic. Permuting scale order together
with IDs must preserve output. Scale weights must be finite and normalized.
Each scale must be independently disable-able, with per-scale memory/time
reported.

## 2.8.5.3 — Adaptive Physical Slicing

Extend existing mass-aware `PhysicsStateSlicing` only if a measurable
limitation is found. Candidate controls include query-conditioned slice
temperature/count or adaptive allocation from physical mass and feature
variation. Preserve deterministic inference. Enforce mass conservation,
stable sparse/low-mass behavior, bounded slice count/memory, and diagnostics
for entropy, effective slice count, empty slices and mass error. Do not make
stochastic Gumbel assignments the default.

## 2.8.5.4 — GNO / Attention Interaction

Compare GNO only, cross-attention only, GNO then attention, attention then GNO,
and residual-gated parallel fusion. Keep the same geometry, queries, width,
optimizer and training budget; report parameter counts. Do not stack operators
without measured benefit.

## 2.8.5.5 — Query-conditioned Fusion and exact Jacobian

Keep Phase 2.8.4 global scalar convex fusion as the mandatory baseline. A
query-conditioned gate may use the attention context:
`alpha(beta,G)=sigmoid(g([query_token,physics_context,global_context]))`.

For `M=alpha*M_M80+(1-alpha)*M_SQNO`, the raw-input Jacobian MUST include
the gate derivative:

`J=alpha*J_M80+(1-alpha)*J_SQNO
 +(M_M80-M_SQNO) outer_product grad_beta(alpha)`.

Blending branch Jacobians alone is wrong when alpha depends on beta. Check the
gate derivative by central finite differences. The gate must be invariant to
positive rescaling of the coefficient vector. Preserve global scalar fusion
as an explicit ablation.

## Fair ablation matrix

Compare global scalar baseline, cross-attention only, cross-scale only,
cross+cross-scale, GNO/attention interaction variants, and query-conditioned
versus global fusion. Report parameter count, peak memory, geometry encode
time, per-query time, total Query-many time, Moment error, raw-input Jacobian
error, mass error, attention entropy/effective support and gradient stability.
Do not exclude geometry encoding for one method but include it for another.

## Data and evaluation policy

### Mini7 pilot training partition

Mini7 contains S001, S019, S023, S024, S078, S082 and S083, 69 queries each
(483 total), all from source prompt family
`ordinary_solid_text_batch_01`. The handoff assigns all seven geometries to
training and zero to internal validation/test. Keep that assignment. The 483
query rows are not independent geometry samples.

Of 483 labels, 258 have full native 10x10 Jacobians, 170 have restricted
planar 10x3 Jacobians and 55 have restricted parabolic 10x8 Jacobians. Use
per-row derivative validity and restricted-basis metadata. Never replace
NaN/NULL derivative entries with zero. For restricted rows compare
`J_pred @ restricted_basis` against the restricted target.

### Formal training candidates: additional TRELLIS collection

Beyond Mini7, an additional collection of 100 TRELLIS-generated models is
currently being processed. These are formal-training candidates, not
automatically accepted training records. After geometry, provenance, topology,
quality and cross-source leakage audits, assign them to train, validation or
locked test at geometry/source-family level. Do not place near-duplicates or
objects from the same prompt/source family across splits. Final split counts
must come from the audited manifest, not a target quota.

Mini7 remains the seven-object pilot set. Its 483 query rows are not 483
independent geometry samples.

### Independent validation: both papers plus TRELLIS

The intended validation collection is broader than two objects. It includes
eligible original benchmark geometries from both Spin-It (2014) and Spin-It
Faster (2024), together with additional TRELLIS-generated artifacts reserved
for validation. Heart spinning top and noodles gyro are important named
benchmark candidates, not the entire validation set.

For each paper benchmark, verify the exact original mesh, provenance,
coordinate normalization, design/material/fixed-region semantics and label
oracle before admission. Do not replace an unavailable original mesh with a
newly generated lookalike. If exact source geometry or reproducible teacher
evidence is unavailable, mark that benchmark BLOCKED.

Generate labels independently from model predictions using the frozen Faster
teacher contract. Include ten moments and derivative labels only where
independently validated. Clearly distinguish the project's local libigl/CGAL
teacher from the authors' official implementation unless independent
value-level equivalence has been established.

### Locked test and split governance

Maintain a separate locked test set of additional source-independent
geometries. Validation geometries may be used for model selection; locked test
geometries may not be used for tuning or architecture decisions.

All splits are geometry/source-family-level, never query-row-level. Publish
the split rationale, source family, prompt provenance, exact mesh hashes,
canonical geometry hashes and near-duplicate report. Quarantine unresolved
provenance or ambiguous topology rather than silently assigning it to a split.

## Release gate

Phase 2.8.5 is not complete until new attention modules pass synthetic
shape/gradient/permutation tests; mass and geometry invariants pass; raw-input
Jacobians pass central finite differences; global-fusion baseline remains
checkpoint-compatible; every mechanism has a disable/ablation path; CPU CI
passes; and data contracts are documented. No real-data accuracy, OOD or
speed claim before corresponding data/GPU evidence exists.


## Phase 2.8.5.1 — Geometry–Physics Cross Attention implementation

The first implementation is in
`spin_adapter/geometry_physics_attention.py` and is opt-in through
`UnifiedM80SQNO(cross_attention_enabled=True)`. It attends normalized
quadric query tokens over cached shared P-CRAFT interval features, with
normalized physical interval mass as a log-prior. Online log-sum-exp processes
bounded interval chunks; attention diagnostics include entropy and effective
support.

The attention correction is added to the fused Moment prediction. Its
Jacobian with respect to the raw ten-dimensional input coefficient vector is
computed by autograd and added to the M80/SQNO raw-input Jacobian blend. This
is required because the correction depends on the query. The final output
projection is zero-initialized so the new branch begins as an exact residual
identity; the existing global scalar fusion remains the default when attention
is disabled.

The opt-in synthetic tests cover query ordering, interval permutation,
chunked finite gradients, raw-input Jacobian finite differences, positive
quadric scale invariance, disabled-path behavior and legacy global-fusion
checkpoint migration. Passing these tests is a software gate only; real-data
accuracy and GPU throughput remain unmeasured until the formal dataset is
approved.


## Phase 2.8.5.2 — Query-conditioned Cross-scale Attention

The M80 encoder already computes per-resolution global summaries
(`global_stack`) and a global scale gate. The new module must consume those
existing summaries; it must not rebuild or duplicate the voxel pyramid.
The geometry cache now retains `scale_features` and their native-resolution
`scale_ids`, including an explicit single-scale token when multiscale mode is
disabled.

`QueryConditionedCrossScaleAttention` uses normalized quadric query tokens
to attend over the cached scale summaries. Scale identity is represented by
a stable logarithmic resolution bucket, while the original native resolution
remains available for masking and diagnostics. It supports missing-scale
masks and explicit disabled resolution IDs, reports normalized attention
weights, entropy and effective support, and fails closed if all scales are
masked.

The optional module is integrated into `UnifiedM80SQNO` as a selectable
ablation. Its query-dependent Moment correction is differentiated with
respect to raw input coefficients and its Jacobian is added to the fused
Jacobian. The global scalar fusion and Phase 2.8.5.1 geometry cross-attention
remain separately selectable. No performance or accuracy gain is claimed
before formal-data and GPU evaluation.

The Mini7/100-TRELLIS/two-paper-benchmark/additional-TRELLIS validation and
locked-test split policy remains as specified above. Mini7 is a pilot only;
the additional 100 TRELLIS geometries are pending processing and audit.


## Phase 2.8.5.3 — Adaptive Physical Slicing

The existing `PhysicsStateSlicing` remains the fixed-slice Transolver-style
baseline. The opt-in adaptive mode does not add a second slicing network or
change the learned parameter shapes. It selects an active prefix of the
existing slice slots using geometry-only statistics:

- effective support of normalized physical interval masses;
- mass-weighted feature variation relative to feature energy.

The deterministic complexity score selects an active slice count bounded by
`transolver_min_slices` and `transolver_slices`. Inactive logits are masked
before the assignment softmax. All valid intervals retain normalized
assignments, and the sum of normalized physical mass over active slice tokens
is checked. Zero-mass intervals are permitted when total physical mass is
positive; all-zero mass is rejected.

Configuration:
- `transolver_adaptive=False` preserves the existing fixed-slice path;
- `transolver_adaptive=True` enables geometry-adaptive slice budgeting;
- `transolver_min_slices` controls the lower bound.

This selector is intentionally geometry-conditioned, not query-conditioned,
so it does not introduce an omitted query derivative into the Moment Jacobian.
It is a discrete, non-differentiable compute-budget decision; gradients still
flow through feature projections, assignment logits, pooling, slice attention
and deslicing for the selected active slices.

Diagnostics include active slice count, feature variation, effective mass
support, complexity score, assignment entropy, temperature range and mass
normalization error. Synthetic tests cover deterministic output, permutation
equivariance, sparse mass, zero-mass intervals, bounded slice count, mass
conservation, finite gradients and fixed-path equivalence.

Older Unified checkpoints are normalized with defaults for the two new
configuration fields when loaded through the explicit legacy global-fusion
migration path. No new parameters are introduced by adaptive slicing.
Real-data quality and throughput remain unmeasured pending formal dataset
acceptance and GPU profiling.


### Adaptive softmax numerical safeguard

The adaptive implementation computes softmax only over the selected active
slice prefix, then pads exact zeros for inactive slots. It does not feed
negative-infinity masked logits through the softmax backward graph. This
keeps inactive slice slots exactly empty while avoiding unstable gradients
through masked softmax branches.


### Phase 2.8.5.3 follow-up — adaptive temperature ablation

The next increment adds an independently selectable adaptive-temperature
policy while retaining fixed temperature as the default. Adaptive temperature
is enabled only together with adaptive slicing. It derives a bounded scalar
factor from the same geometry-only complexity score:
`factor = 1.5 - complexity`. The per-head/per-point learned temperature is
multiplied by this factor and clamped to explicit positive bounds
(`transolver_temperature_min`, `transolver_temperature_max`). Higher
feature/mass complexity therefore sharpens assignments; simpler geometry
produces smoother assignments. This is a hypothesis to be tested, not an
established optimum.

The ablation matrix is now:
1. fixed slice count + fixed learned temperature (legacy baseline);
2. adaptive slice count + fixed learned temperature;
3. adaptive slice count + adaptive bounded temperature.

Diagnostics report the configured bounds, observed temperature range,
active slice budget and effective slice utilization. Effective slice
utilization is measured from normalized pooled mass entropy and must not be
confused with the selected active slice budget.

The selector remains geometry-only and deterministic. No query-dependent
slicing is introduced, so the Moment Jacobian has no additional query path.
The adaptive policy adds no trainable parameters. The new config fields have
explicit defaults in the legacy Unified checkpoint migration path.

Synthetic tests cover bounded temperatures, deterministic repeatability,
sparse-mass permutation equivariance, finite gradients, effective-slice
utilization and configuration validation. Empirical superiority, wall-clock
speedup and memory savings remain unproven until matched GPU profiling and
formal dataset evaluation.
