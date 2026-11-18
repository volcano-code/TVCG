# Phase 2.7.3 — Unified feature and derivative-validity contracts

## Architecture decision

UG-PINO is a shared geometry/task representation with parallel, trainable operator branches and joint physical heads. It is not a serial chain of independently trained full models.

The M80 P-CRAFT native interval encoder remains the primary geometry encoder. It can now retain and export its per-interval features once:

```python
cache = pcraft.encode(geometry, chunk=512, retain_interval_features=True)
shared = pcraft.export_shared_geometry_features(cache)
```

The export includes local interval features, global context, scale weights, interval mass, descriptors, normalized coordinates, and geometry identity. The returned tensors retain the autograd graph, allowing downstream SQNO/QuadrANN layers to train jointly through the same P-CRAFT encoding.

SQNO v1.2 now accepts this external feature contract:

```python
sqno = SpinQuadrANNV12(..., share_pcraft_encoder=True)
sqno_cache = sqno.encode_geometry(geometry, pcraft_features=shared)
```

In shared mode the SQNO-local P-CRAFT adapter is frozen and not invoked; the caller must provide the shared cache. The SQNO topology/GNN and query-many quadrature heads remain independently trainable. This removes duplicate P-CRAFT geometry encoding, but it does not yet mean the full SQNO branch is part of the maximal M80 checkpoint or trainer. A joint wrapper, optimizer ownership, checkpoint schema and matched ablation remain required before claiming full SQNO fusion.

## Event-aware Jacobian supervision

The trainer now uses `derivative_supervision_mask` for training, preflight and validation. A Jacobian label must first pass the corpus `derivative_valid` flag. Additional optional gates are:

- `--min-event-margin`: if event-margin labels exist, exclude queries at or below the threshold;
- `--require-fd-audit`: require `finite_difference_audit=true`;
- `--max-jacobian-error-bound`: exclude labels whose recorded Jacobian error bound exceeds the threshold.

Moment supervision remains active for all queries; only Jacobian supervision is masked. If a requested gate's evidence field is absent, the run fails closed. All gate settings are included in the exact-resume training contract and report. Validation reports derivative counts before/after masking.

These gates do not automatically classify every event type. The teacher's derivative_valid/event-margin/audit labels remain authoritative. The threshold values must be selected from teacher audit evidence, not guessed to improve metrics.

## Mathematical audit status

The bounded-odd moment head and convex-potential candidate both obtain the coefficient Jacobian by differentiating their own moment function. Existing tests cover convex-head finite differences, complement, positive-scale invariance and potential Hessian properties. New tests cover the bounded-odd head finite-difference Jacobian, positive coefficient-scale invariance and q/-q complement.

The convex-potential PSD Jacobian property is a property of that specific convex-potential parameterization in smooth regions; it is not asserted as a universal property of the exact Spin-It Faster moment map. The bounded-odd head is not required to have a symmetric or PSD Jacobian.

## Learned event-validity branch

An opt-in `event_validity_head_enabled` branch is now available through the `xl_ms_gno_transolver_convex_event` preset. It predicts the teacher's `derivative_valid` label from the shared geometry latent, global statistics and normalized quadric query. The training loss uses BCE-with-logits with a separately configurable `--event-weight`; validation reports BCE, accuracy and Brier score. This is a derivative-validity risk classifier, not a complete taxonomy of geometric event types and not a substitute for teacher derivative validity or exact solver event checks. The earlier `xl_ms_gno_transolver_convex` preset remains unchanged for local compatibility.

## Uncertainty status

The repository has ensemble spread and split-conformal calibration utilities. These are calibration tools, not a trained uncertainty head. A calibrated reliability gate still requires independently trained seeds, held-out geometry-group calibration, risk-coverage evaluation, and frozen model/data/calibration hashes.

## Not yet complete

- A single top-level UG-PINO wrapper/checkpoint that jointly owns M80 and SQNO trainable branches.
- The optional event-validity head requires trained calibration and threshold selection; no calibrated event gate is claimed.
- SQNO joint trainer and matched-budget ablation against M80 alone.
- Event classifier trained against explicit event labels (currently event-aware Jacobian label masking).
- Calibrated ensemble with multiple trained seeds.
- RTX 5090 GPU preflight for the maximal multiscale/GNO/Transolver/convex candidate.
- Formal corpus coverage and production-quality 256-query labels.
- 512^3 trained-quality validation and exact-solver acceleration evidence.
