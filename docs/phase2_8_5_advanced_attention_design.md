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

### Independent benchmark validation

Prepare original Spin-It / Spin-It Faster benchmark geometries including the
heart spinning top and noodles gyro as a separate geometry-level validation
source, not as additional Mini7 training objects. Verify exact source mesh,
source provenance, coordinate normalization, material/fixed-region semantics,
positive-side convention and label-generation oracle before admission.

Generate labels independently from model predictions using the same frozen
Faster teacher contract. Include ten moments and derivative labels only where
independently validated. If the exact source mesh or reproducible teacher
contract is unavailable, mark blocked; do not substitute a newly generated
lookalike or claim equivalence.

Heart and noodles are a small benchmark validation set, not a sufficient final
OOD test set. Maintain a separate locked test set of additional
source-independent geometries. Split at geometry/source-family level, never
at query-row level.

## Release gate

Phase 2.8.5 is not complete until new attention modules pass synthetic
shape/gradient/permutation tests; mass and geometry invariants pass; raw-input
Jacobians pass central finite differences; global-fusion baseline remains
checkpoint-compatible; every mechanism has a disable/ablation path; CPU CI
passes; and data contracts are documented. No real-data accuracy, OOD or
speed claim before corresponding data/GPU evidence exists.
