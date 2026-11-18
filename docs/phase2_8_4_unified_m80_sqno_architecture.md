# Phase 2.8.4 — Unified M80 + SQNO Architecture Audit

## Status

Architecture audit and interface planning only. No training is required for
this work, and no GPU performance or accuracy claim is made here. The selected
Moment Head remains provisional until the Phase 2.8.3 matched multi-seed
experiment and independent exact-solver replay are complete.

## Existing implementation verified in this branch

### M80 / native P-CRAFT path

`spin_pino/native_pcraft.py` already provides:

- `PcraftNativeMomentOperator.encode(..., retain_interval_features=True)`;
- a differentiable encoded geometry cache containing native interval features,
  global context, interval mass, descriptors, normalized coordinates and
  multi-scale weights;
- `export_shared_geometry_features(cache)`, which validates cache ownership,
  parameter versions and feature count before exporting the shared payload;
- native Moment prediction and Moment/Jacobian evaluation from the same cache.

The exporter returns `local_features`, `global_context`,
`global_scale_weights`, `interval_mass`, `descriptor`,
`normalized_xyz`, `geometry_id` and `n_intervals`.

### SQNO path

`spin_quadrann_v12.py` already provides:

- one-time `encode_geometry(..., pcraft_features=...)`;
- a geometry cache reused by `evaluate_queries(cache, coefficients)`;
- batched Moment and Jacobian evaluation;
- mass conservation and fixed-moment restoration checks;
- an explicit external P-CRAFT feature input contract.

Its current external feature contract is shape-checked and binds a supplied
geometry identity when available.

## Identified integration gap

The current SQNO v1.2 constructor always constructs its own
`PcraftFeatureAdapter`. With `share_pcraft_encoder=True`, that internal
adapter is frozen, while an external feature payload can be supplied to
`encode_geometry`. This works as an experimental interface, but a production
unified model should not retain a second dormant P-CRAFT parameter tree merely
to obtain the interval width. That creates redundant checkpoint state and
makes optimizer ownership less explicit.

The next implementation should add a width-only external-encoder mode to SQNO:

- accept `pcraft_width` when no local P-CRAFT adapter is owned;
- require external `pcraft_features` in that mode;
- preserve the existing constructor behavior and checkpoint layout when the
  new argument is omitted;
- do not freeze the externally owned shared encoder as a side effect.

The unified wrapper should own exactly one
`PcraftNativeMomentOperator`. It should encode once with
`retain_interval_features=True`, export the feature payload, and pass it to
SQNO. The wrapper should retain both branch outputs and Jacobians, then expose
a unified output contract. The first integration must use a simple, auditable
fusion parameterization; query-conditioned attention belongs to Phase 2.8.5.

## Phase 2.8.4 implementation sequence

### 2.8.4.1 — Shared encoder ownership

- Add SQNO external-width mode without a duplicate P-CRAFT adapter.
- Validate interval width, feature count, geometry identity, dtype and device.
- Keep old SQNO constructors and checkpoints backward compatible.
- Test that one geometry encode serves both branches and gradients reach the
  single shared encoder.

### 2.8.4.2 — Query-many SQNO contract

- Preserve the existing `[Q,10]` coefficient input and `[Q,10]` Moment
  output contract.
- Validate query chunking and query-order invariance.
- Compare batched queries with concatenated single-query calls.
- Keep fixed moments added exactly once.
- Verify the Jacobian chain rule and finite-difference agreement.

### 2.8.4.3 — Unified wrapper

- Own the shared M80/P-CRAFT encoder and SQNO query branch.
- Encode geometry once and cache both branch-specific derived states.
- Return M80 Moments/Jacobian, SQNO Moments/Jacobian and fused
  Moments/Jacobian in one named result.
- Check that fusion preserves output shape, finite values, mass consistency,
  complement/scale invariants where both branches guarantee them.
- Keep the final Moment Head selectable; do not freeze an unvalidated winner.

### 2.8.4.4 — Optimizer and checkpoint

- Expose one deduplicated parameter iterator.
- Assert that shared encoder parameters occur once in optimizer groups.
- Save architecture schema, both branch configs, fusion config, model state,
  optimizer state, scheduler state, random states and source/data fingerprints.
- Reject incompatible or stale checkpoint contracts before loading weights.
- Retain an explicit migration path for legacy standalone M80 and SQNO
  checkpoints; never silently reinterpret them as a unified checkpoint.

### 2.8.4.5 — Mathematical and software release tests

- shared encoder called exactly once per geometry;
- SQNO query-many reuses the cache;
- shared encoder receives gradients from both branches in joint-training mode;
- optimizer has no duplicated parameter identities;
- checkpoint round-trip preserves branch and fused outputs;
- finite-difference Jacobian agreement;
- SO(3) and normalization contracts remain intact;
- legacy checkpoint loading remains unchanged.

## Data-independent work allowed now

The following can proceed while the formal corpus is being repaired:

- API and module design;
- shared encoder ownership refactor;
- synthetic unit fixtures for shapes, cache lifetime, gradients and checkpoint
  round trips;
- optimizer parameter de-duplication;
- CLI and checkpoint contract tests;
- CPU CI and static checks.

The following must wait for the approved data/audit receipt:

- long training;
- model selection;
- OOD claims on real geometry families;
- final Head Freeze;
- performance claims at 512^3.

## Acceptance boundary

Phase 2.8.4 software integration is complete only when the shared encoder,
SQNO query-many path, unified wrapper, optimizer and checkpoint all pass the
tests above. It does not imply that the selected Moment Head has been
scientifically frozen or that large-scale training has succeeded.
