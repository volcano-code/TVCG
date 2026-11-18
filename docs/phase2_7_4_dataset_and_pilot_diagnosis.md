# Phase 2.7.4 — TRELLIS/Faster dataset handoff and first GPU pilot diagnosis

Date: 2026-11-18

## 1. Source-grounded data status

The TRELLIS handoff reports:

- First ordinary-geometry release: 100 original meshes retained with SHA-256; seven selected source meshes have 64 valid native Faster moment queries each (448 accepted query labels total).
- Those 448 labels contain 163 planar and 285 curved queries. Query types are ellipsoid 160, plane 163, hyperbolic 49, mixed 28 and parabolic 48.
- The teacher ran 460 attempts; 448 were accepted, 10 were resource-limited, one had constant response and one returned a native error.
- The Jacobian-valid flag is false for these labels. The handoff reports no newly qualified Jacobian labels, no new protected M69 feasible material states and no training steps.
- The remaining 93 first-release source meshes were triaged: 30 interpretable as-is, 19 with separate local repair copies, 44 unresolved. No repaired copy was admitted to the main supervision set.
- The second release contains 100 single-shell, closed, consistently oriented, positive-volume meshes across 25 families. These are geometry candidates, not 100 additional Faster-labelled tasks.
- Across the two batches there are 200 original geometry candidates, but only seven objects currently have the described accepted Faster moment-query supervision.
- The teacher is the repository's libigl/CGAL implementation of the Faster formula. A value-by-value comparison with the paper authors' program is still pending.

This means the new TRELLIS data is valuable for geometry coverage and Moment supervision, but is not yet a complete Moment+Jacobian+material-state training corpus.

## 2. First 5090D pilot: exact scope

The supplied training report is from the existing M72 formal corpus, not from the newly handed-off 448 Faster labels:

- registered 104, qualified 67;
- train/validation/test rows and source groups: 50/7/10;
- every qualified record has 32 queries;
- minimum/maximum query coverage: 32;
- preset: `xl_ms_gno_transolver_convex`;
- parameters: 3,697,421;
- RTX 5090 D, PyTorch 2.8.0+cu128, CUDA 12.8, BF16;
- 300 optimizer steps, 776.80 seconds, best checkpoint at step 200.

Validation on seven records / 224 queries:

| Metric | Initial | Best checkpoint |
|---|---:|---:|
| Moment scaled MSE | 0.00241344 | 0.00241880 |
| Jacobian scaled MSE | 1449.5507 | 1426.8007 |
| Combined selection score | about 72.48 | 71.34245 |

Moment error did not improve materially. The Jacobian aggregate is dominated by strong per-object heterogeneity: three GSO-like records have Jacobian scaled MSE in the range of roughly 2.85 to 8,745, while four M70 design-panel records are around 0.0065 to 0.012. This is a diagnosis signal, not proof of which side has incorrect labels. The next step is to audit units, coordinate transforms, query normalization, teacher provenance and per-family Jacobian finite-difference agreement before increasing the training budget.

The model reports extremely small Jacobian symmetry error and near-zero minimum eigenvalues. These are consistent with the convex-potential architecture's structural constraints; they do not demonstrate agreement with the exact teacher Jacobian.

The convex-potential amplitude fraction is approximately 0.9999995, indicating near-saturation of its configured amplitude bound. This deserves a targeted ablation/diagnostic before promoting the convex head.

The run completed a training and checkpoint pipeline, but it did not perform an exact solver call (`exact_calls=0`) and did not evaluate the test split. It is not evidence of end-to-end Spin-It Faster acceleration or design acceptance.

## 3. Trainer changes required by the handoff

The training code now permits a preflight batch to fall back to Moment-only queries when no Jacobian labels are valid. This is necessary for a Moment-only teacher dataset. The Jacobian loss remains zero for those samples; Moment supervision remains active.

Event regularity is now a separate label contract: the optional event-validity head requires an explicit boolean `event_valid` teacher label. It must not infer event class from `derivative_valid`, because a missing/unvalidated Jacobian is not equivalent to a geometrically irregular event.

The current M72 corpus reader still expects the M69/M72 manifest and the arrays `queries`, `moments`, `jacobian`, and `derivative_valid`. The TRELLIS Faster release therefore still needs a dedicated, provenance-preserving importer/adapter. Until the actual label archive and manifest are available to that adapter, do not claim the 448 labels are integrated into the trainer.

## 4. Next data and model gates

1. Import the seven-object Faster moment labels without fabricating Jacobian targets. Store an explicit `derivative_valid=false` mask and preserve teacher binary hash, source mesh hash, original face indices, coordinate frames, query family and attempt/failure status.
2. Add a moment-only training mode that can consume the seven-object data, while keeping it clearly separate from formal multi-family training and avoiding leakage across source geometry families.
3. Obtain/derive Jacobian labels only after independent finite-difference audits and event regularity checks. Do not convert invalid flags to valid.
4. Audit the large per-object Jacobian discrepancy in the current M72 pilot before longer training.
5. Run matched controls for bounded-odd versus convex-potential heads; inspect amplitude saturation.
6. Integrate SQNO through the shared P-CRAFT feature cache. The shared SQNO feature width must match the UG-PINO interval width; for the current XL multiscale width 192, instantiate the SQNO adapter with `preset="xl_ms"`.
7. Only after these gates, conduct multi-seed architecture screening and long training. Then evaluate 512-resolution geometry, exact Newton replay and independent final material feasibility.

## 5. Claims that are and are not supported

Supported:
- 200 curated geometry candidates exist across the two TRELLIS batches.
- 448 accepted native Faster Moment labels exist for seven first-release objects.
- The existing M72 maximal architecture completed a 300-step RTX 5090D pilot.
- The maximal model has multi-scale P-CRAFT, GNO, mass-aware Transolver-style slicing and a convex moment-potential head.

Not supported yet:
- 200 Faster-labelled training objects.
- Valid new Jacobian labels or new protected M69 material states.
- Trained generalization to the new TRELLIS release.
- A calibrated event or uncertainty model.
- 512-resolution trained-quality generalization.
- End-to-end Spin-It Faster speedup or final design acceptance.
