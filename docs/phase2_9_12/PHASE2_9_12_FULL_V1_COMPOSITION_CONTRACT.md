# Phase 2.9.12 — Full Spin Neural Operator V1 Composition Contract

Status: implementation specification; not an implementation or training result.
Branch: `research/phase2-6-r3-m69-hybrid-ablation`
Baseline: Phase 2.9.11 Step 1 research contract.

## 1. Goal

Compose existing P-CRAFT, M80, SQNO, geometry-physics cross-attention, cross-scale attention, GNO/Transolver-style slicing, UQ and Event components into one named, reproducible model configuration. Do not duplicate existing modules or change the Spin-It/Faster task.

The V1 candidate is a complete computational graph with explicit module configuration. It is not asserted to be the scientifically best configuration. Optional branches remain independently switchable for controlled later ablations.

## 2. Composition layers

1. Geometry/material encoder: one shared `PcraftNativeMomentOperator`; one geometry cache reused across query batches.
2. Query-many physical readouts: M80 and SQNO, both conditioned on the same geometry and raw ten-coefficient query.
3. Query-dependent interactions: geometry-physics cross-attention over cached interval tokens and cross-scale attention over existing scale summaries. Their query derivatives must be included in the fused raw-input Jacobian.
4. Geometry interactions: existing GNO and Transolver-style PhysicsStateSlicing stay inside the native P-CRAFT configuration. Do not instantiate duplicate geometry pyramids or duplicate GNO/slicing blocks outside their current owner.
5. Fusion: query-conditioned fusion is the Full V1 candidate; global scalar fusion remains the reference configuration. Any query-dependent gate derivative must be included.
6. Auxiliary outputs: Moment UQ, Jacobian UQ and Event prediction are separate outputs with separate labels/masks and calibration status. They never replace exact event/derivative checks.
7. Optimizer boundary: model predictions can form a proposal; exact Faster-compatible code owns trial evaluation, acceptance/rejection, fallback and final verification.

## 3. Named configurations

### `reference_v1`

- Shared P-CRAFT encoder.
- M80 + SQNO.
- Global scalar fusion.
- No query cross-attention corrections.
- Existing UQ/Event heads follow their explicit config flags; absent heads return an explicit unavailable status, not fabricated zeros.

Purpose: stable reference and backward-compatible baseline.

### `full_v1_candidate`

- Shared P-CRAFT encoder with GNO enabled.
- Existing Transolver-style PhysicsStateSlicing enabled through the P-CRAFT config.
- Geometry-physics cross-attention enabled.
- Cross-scale attention enabled.
- M80 + SQNO.
- Query-conditioned fusion.
- Moment UQ, Jacobian UQ and Event heads enabled only where the existing model implementation exposes compatible modules and the output contract can report their status correctly.

This is the intended complete candidate configuration, not a proven optimum. If any listed native capability is absent from a particular P-CRAFT preset, the factory must fail with a clear configuration error rather than silently dropping it.

## 4. Unified prediction record

The stable prediction result must contain:

- `moments`: [Q,10], fused raw ten-moment output;
- `jacobian_coefficients`: [Q,10,10], derivative of that exact fused moment function with respect to the raw input coefficient vector;
- `moment_log_std`: [Q,10] or explicit unavailable status;
- `jacobian_log_std`: [Q,10,10] or explicit unavailable status;
- `event_logits` and `event_probability`: [Q] or explicit unavailable status;
- separate `derivative_validity`, `event_label_availability`, and `prediction_finite`;
- branch outputs, fusion weights, attention/slicing diagnostics, geometry identity, cache version and timing.

Never conflate predictive uncertainty with derivative validity, event classification with event truth, or either with exact solver acceptance.

## 5. Jacobian correctness

For the final moment function (M_	heta(G,eta)), the exposed Jacobian must be:

[
J_	heta(G,eta)=rac{partial M_	heta(G,eta)}{partialeta_{mathrm{raw}}}.
]

It must include:
- M80 raw-coefficient derivative;
- SQNO derivative after its coefficient-normalization chain rule;
- query-conditioned fusion-gate derivative;
- geometry cross-attention correction derivative;
- cross-scale attention correction derivative;
- any additional query-dependent interaction derivative.

Finite-difference tests must compare the public fused `moments` against the public `jacobian_coefficients` in the raw coefficient coordinate system.

## 6. No silent architecture assumptions

- Do not enable all modules globally in the existing constructor defaults.
- The named `full_v1_candidate` explicitly enables its intended modules.
- Keep `reference_v1` available.
- Record every module flag and P-CRAFT preset in the architecture fingerprint.
- No module may be silently skipped because a preset does not support it.
- No scientific gain is claimed from the configuration alone.

## 7. Phase 2.9.12 implementation checklist

- [ ] Inspect P-CRAFT config fields and confirm exact GNO/Transolver ownership.
- [ ] Inspect current UQ/Event implementation and identify how outputs are exposed.
- [ ] Add typed architecture config and deterministic factory.
- [ ] Add unified prediction result type without breaking existing dictionary callers.
- [ ] Integrate UQ/Event outputs or explicitly report unavailable status.
- [ ] Add full_v1_candidate and reference_v1 configuration snapshots.
- [ ] Add config/source fingerprint to checkpoints.
- [ ] Add tests for construction, shape contracts, raw-coordinate finite differences, attention gradients, auxiliary output semantics and backward compatibility.
- [ ] Run targeted tests and the full relevant release gate.

## 8. Explicitly out of scope

No formal data training, long training, model selection, architecture ablation claims, OBJ data mutation, Spin-It reimplementation, or Faster task changes in Phase 2.9.12.
