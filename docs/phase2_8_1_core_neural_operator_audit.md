# Phase 2.8.1 — Core Neural Operator Architecture & Mathematical Audit

Repository: `volcano-code/TVCG`
Branch: `research/phase2-6-r3-m69-hybrid-ablation`
Audit basis: source at the branch tip before this audit's test commits.
Scope: model/code contracts only. TRELLIS data remediation and GPU training are out of scope.

## 1. Research definition

The learned core is a geometry-conditioned, query-conditioned operator:

```text
G_theta : (rho(x), Omega, beta, psi) -> (M(beta), J_M(beta))
```

where `rho(x)` is the material/geometry field, `Omega` is the represented domain, `beta` is the ten-coefficient quadric query and `psi` collects fixed physical/task conditions. The learned operator must support varying geometry and query inputs without retraining its weights for every query. The outer Event/UQ/Neural-Newton/Exact-Solver layers are reliability and optimization system components; they are not themselves the neural operator.

## 2. Audited implementation inventory

| Component | Current implementation | Audit finding |
|---|---|---|
| Spatial trunk | `PINOFeatureEncoder` in `pino_backbone.py` | U-shaped factorized 3D F-FNO/PINO with x/y/z rFFT spectral mixing, local 1x1 skip, coordinate channels and pooled + spatial-map readout |
| Geometry/query adapter | `PcraftNativeMomentOperator.encode/predict` | Native RLE intervals, moment basis, descriptors and geometry-conditioned cache; query is ten-dimensional |
| Multi-scale | Shared-weight P-CRAFT encoder at configured grids (maximal preset 32/64/128) | Softmax global scale gating and local scale gating; not cross-scale token self-attention |
| GNO | `gno_kernel/value/update` and cached KNN graph | Local graph message passing; optional 6-connected component filtering; not Transformer attention |
| Transolver | `PhysicsStateSlicing` | Learned soft interval-to-slice assignments, mass-weighted pooling, per-head Q/K/V scaled-dot-product attention over slice tokens, then deslicing |
| Moment readout A | Bounded-odd query head | Odd response around the half-design baseline, tanh bounded by absolute moment envelope |
| Moment readout B | `ConvexMomentPotentialHead` | Sum of smooth SPD quadratic norms; gradient gives response, Hessian gives a symmetric PSD Jacobian within this parameterization and smooth nonzero queries |
| Jacobian | `moments_and_jacobian` | Autograd differentiates the same predicted moment function; create_graph supports higher-order training |
| Event risk | Optional `event_validity_head` | Requires explicit teacher `event_valid`; must not infer event regularity from `derivative_valid` |
| SQNO | `SpinQuadrANNV12` | Geometry-once/query-many GNN + quadrature; accepts exported P-CRAFT features, but no single joint M80+SQNO wrapper/trainer/checkpoint yet |
| Outer optimization | Phase 2.6 M69 KKT / safeguarded Newton | Separate solver controller; exact oracle remains authority |

## 3. Frozen tensor and physical contracts

### 3.1 Ten-dimensional ordering

Moment order:
`(mass, x, y, z, xx, yy, zz, yz, xz, xy)`.

Quadric query order:
`(1, x, y, z, xx, yy, zz, 2yz, 2xz, 2xy)`.

The native coefficient basis buffer is `[1,1,1,1,1,1,1,2,2,2]`. The factor two on all three cross terms is intentional and must be preserved by all importers, adapters and derivative transforms.

### 3.2 Fixed/editable decomposition

The geometry encoder computes editable moments from native interval mass and basis, while fixed moments remain separate. It checks:

`fixed_moments + design_moments == original_moments`

within the implementation's scale-aware tolerance. Predicted total moments combine fixed material with the query-conditioned editable contribution. Jacobian supervision is with respect to the ten quadric coefficients and only differentiates the query-dependent response.

### 3.3 Query normalization and complement

The predictor normalizes positive coefficient scale and enforces the complement identity:

`M(q) + M(-q) = 2 M_fixed + M_design`

The full/empty query endpoints are exact. The audit added explicit tests for moment/query order, cross-term factors, positive scale invariance and complement.

### 3.4 Jacobian contract

The authoritative learned Jacobian is the derivative of the same predicted moment function, not an independently unconstrained matrix head:

`J_theta(q) = d M_theta(q) / d q`.

Existing tests already compare AD against central finite differences. This audit adds a multi-step finite-difference sweep for the bounded-odd head. Teacher Jacobians still require their own independent validation; passing model self-consistency tests does not validate teacher labels.

### 3.5 Convex potential interpretation

For the convex-potential parameterization, the potential is a positive sum of smooth norms `sqrt(q^T C_k q)` with SPD `C_k`. Its gradient is the editable response and its Hessian is symmetric PSD in smooth nonzero query directions. This is a mathematical property of that parameterization.

Do not automatically impose this property on every alternative head or assume that an observed symmetric/PSD Jacobian proves agreement with the exact teacher. The bounded-odd head is not constrained to have a symmetric or PSD Jacobian.

The reported pilot's `convex_potential_scale_fraction ~= 0.999999463` is near the configured amplitude ceiling. This is a saturation diagnostic requiring a bounded-odd-vs-convex and amplitude-parameterization ablation; it is not by itself proof that the convex formulation is mathematically invalid.

### 3.6 SO(3) and units

The audited native operator module does not expose a single explicit public SO(3) moment/Jacobian transform API. Rotation augmentation/equivariance utilities exist elsewhere in the Spin-PINO tree, but the end-to-end chain rule from rotated geometry and quadric coefficients to the ten raw moments/Jacobian is not certified by this audit. Treat full SO(3) equivariance as open until a dedicated transform contract and independent property tests are added.

Likewise, the native M72 path carries geometry origin, voxel width, density and moment scales, but a universal external physical-unit conversion contract across all teachers/importers must be frozen separately.

## 4. Resolution-transfer audit

`PINOFeatureEncoder.forward_with_map(..., preserve_resolution=True)` accepts spatial dimensions at least 8 and divisible by 4. The factorized spectral convolution clips the retained modes to the available rFFT modes. The same encoder weights can therefore execute at multiple compatible grid sizes.

This establishes variable-resolution execution, not resolution-invariant accuracy or trained 512^3 generalization. The new synthetic test checks one model instance at 8^3, 16^3 and 32^3 and confirms stable output feature dimensions. A true resolution-transfer claim still requires trained checkpoints and matched physical evaluations at 64/128/256/512.

## 5. Attention and operator terminology

- Transolver-style physics slicing uses actual multi-head scaled-dot-product attention over pooled slice tokens.
- Multi-scale softmax gates are adaptive weighting, not full Transformer self-attention.
- GNO and QuadrANN use graph message passing/aggregation, not standard Transformer attention.
- P-CRAFT uses Fourier/spectral operator layers.
- SQNO provides geometry-once/query-many query evaluation.

The current architecture is therefore a hybrid neural-operator design, not a pure Transformer and not a serial chain of full independent models.

## 6. Phase 2.8.1 code changes in this commit series

Added tests to `tests/test_native_pcraft.py` for:
1. Exact ten-moment and quadric coefficient ordering.
2. Explicit factor-of-two cross-term bounds for yz/xz/xy.
3. Same P-CRAFT encoder instance executing 8^3, 16^3 and 32^3 preserved-resolution inputs.
4. Bounded-odd AD Jacobian consistency across multiple central finite-difference step sizes.
5. Convex-potential symmetry/PSD as a property of that head only, while requiring the bounded-odd Jacobian to remain finite without imposing PSD/symmetry.

No model parameters, preset defaults, checkpoint schema, or training behavior were changed in Phase 2.8.1. The user's existing `xl_ms_gno_transolver_convex` checkpoint remains structurally untouched by these audit tests.

## 7. Explicitly not completed

- A public, independently tested SO(3) transform chain for moments and Jacobians.
- Universal unit/coordinate conversion contract across all teacher adapters.
- A single joint M80+SQNO trainable wrapper and unified checkpoint.
- Cross-scale attention and geometry-physics cross-attention as trained, ablated modules.
- Convex-vs-bounded-odd matched training ablation and resolution of scale saturation.
- Trained resolution-transfer evidence at 512^3.
- Calibrated UQ and trained event classifier.
- End-to-end exact-solver acceleration evidence.

## 8. Phase 2.8.2 next actions

1. Add explicit SO(3) transform utilities for raw moments (mass, first moment vector, second raw moment tensor) and coefficient queries, with round-trip and covariance tests; derive Jacobian chain rule from those transforms.
2. Freeze a physical-unit normalization adapter with reversible moment and Jacobian scaling.
3. Add convex amplitude saturation telemetry and bounded-odd/convex head ablation presets without changing the user's existing preset.
4. Build the M80+SQNO joint wrapper with one geometry encoder, one optimizer and one checkpoint; prove gradients reach the intended trainable SQNO branch and no second P-CRAFT encoder executes.
5. Only then prototype cross-scale and query-conditioned geometry-physics attention behind opt-in configs and matched ablations.
