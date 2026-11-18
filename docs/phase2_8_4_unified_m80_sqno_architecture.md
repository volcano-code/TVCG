# Phase 2.8.4 — Unified M80 + SQNO

## Scope

Phase 2.8.4 creates a single trainable P-CRAFT geometry encoder shared by the
native M80 Moment branch and SQNO v1.2 Query-many branch. This is a
data-independent architecture/software integration. It makes no large-scale
training, real-corpus generalization, or final Moment Head selection claim.

## 2.8.4.1 — Shared encoder ownership

SQNO v1.2 supports explicit `pcraft_width` external-encoder mode. In this
mode it registers no `PcraftFeatureAdapter` and requires externally supplied
P-CRAFT features. Omitting `pcraft_width` preserves the legacy constructor
and checkpoint layout. The old `share_pcraft_encoder=True` mode remains
backward compatible.

The shared-feature contract checks feature count/width, global-context width,
scale-weight rank, geometry identity, device, finite values, positive physical
interval masses and agreement with the native geometry mass total.

**Mass convention:** the native M80 geometry cache stores normalized interval
pooling weights. SQNO quadrature consumes physical interval masses. The shared
feature exporter restores physical masses by multiplying normalized weights
by the editable design mass. Normalized pooling weights are never passed as
physical quadrature weights.

## 2.8.4.2 — Query-many

SQNO keeps the established `[Q,10]` coefficient input, `[Q,10]` Moment
output and `[Q,10,10]` query Jacobian. Geometry is encoded once and reused
across queries. The unified wrapper exposes the same batched output contract
for both branches and their fused prediction.

## 2.8.4.3 — Unified wrapper

`UnifiedM80SQNO` owns one `PcraftNativeMomentOperator` and one external-
feature SQNO branch. The shared geometry is encoded once and exported to SQNO.
The wrapper returns branch-specific Moments/Jacobians and a fused result.

The initial fusion is a trainable global scalar convex combination:
`M = alpha*M80 + (1-alpha)*SQNO`, with the same linear combination applied
to the query Jacobian. This is an auditable baseline. Query-conditioned
fusion and attention remain Phase 2.8.5 work.

## 2.8.4.4 — Optimizer and checkpoint

- Expose deduplicated trainable parameters and build one AdamW optimizer.
- Verify unique parameter identities and no duplicate P-CRAFT parameter tree.
- Save architecture schema/config, source fingerprints, optional data
  fingerprint, model state, optimizer/scheduler state and RNG states.
- Reject architecture, source or requested data-fingerprint mismatches.
- Legacy standalone M80/SQNO checkpoints are not silently reinterpreted as
  unified checkpoints. Load them through their legacy loaders and perform an
  explicit migration.

## Data-independent tests

Synthetic fixtures cover shared encoder ownership, physical mass preservation,
feature/geometry identity, joint gradients, Query-many output shapes, optimizer
uniqueness, stale-cache rejection and checkpoint round-trip. These do not
substitute for formal M69/M72 data validation.

## Remaining gates

Formal-corpus and GPU evidence is still required for joint-training stability,
loss balancing, matched multi-seed Moment Head selection, real-geometry OOD,
256^3/512^3 memory and throughput, and independent exact-solver correction.
The global scalar fusion is an integration baseline, not the final advanced
fusion mechanism.
