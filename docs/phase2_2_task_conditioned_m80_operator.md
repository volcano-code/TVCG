# Phase 2.2 — Task-conditioned M80 Unified Operator Adapter v1

## Objective

Extend the frozen M80 native-interval geometry operator with explicit M69 rotation-task conditioning and a material-design Jacobian interface, without modifying the M80 baseline implementation.

## Architecture

```
native geometry / RLE
      |
frozen M80 P-CRAFT + native interval encoder
      |
EncodedNativeGeometry (cached)
      |
M69 TaskSpec -> 31D task feature vector
      |
zero-initialized FiLM (gamma,beta)
      |
conditioned global latent
      |
M80 convex moment-potential head <--- beta_query(alpha)
      |
M, dM/dbeta_query
      |
explicit alpha -> beta_query Jacobian
      |
dM/dalpha = (dM/dbeta_query)(d beta_query/d alpha)
```

The task adapter is a residual conditioning layer on the M80 global latent. It does not replace the P-CRAFT/F-FNO geometry trunk, interval encoder, or convex potential head. The zero initialization guarantees exact M80 identity at initialization. Task effects are learned only after the FiLM parameters are updated.

## M69 task encoding

The feature vector is 31D and has a fixed named order:
- axis (3);
- axis origin divided by outer length L (3);
- contact divided by L (3);
- log density and log2 native-resolution ratio (2);
- shell/protection and physical thresholds (9);
- log-scaled tolerances (9);
- objective, inertia-reference, material-policy and purpose flags (4).

Density remains the registered TaskSpec scalar. No SI unit conversion is introduced. The current v1 encoder intentionally supports only the frozen M69 fixed-axis minimum-retained-mass / original-fixed-axis / single-density-removal task profile. It rejects incompatible task profiles rather than silently mapping them.

## Material Jacobian semantics

The adapter names the base derivative `dM/dbeta_query`. It exposes `dM/dalpha` only through an explicit design-map Jacobian D:
[
J_\alpha=J_{\beta_q}D,qquad D=\partial\beta_q/\partial\alpha.
]
An identity D is accepted only when a caller supplies a `MaterialMapContract` certifying identical frame, retained side, fixed/protected/editable masks, units/moment normalization and derivative-valid status. A 10x9 design map is supported for local S9 coordinates. This API does not itself certify that the M69 binary Boolean map and the M80 learned query map are identical; that still requires the exact teacher/geometry contract.

## Physical consistency

Phase 2.2 requires the M80 convex-potential head. It provides a scalar potential whose gradient generates moments and whose Hessian gives a symmetric PSD query Jacobian in the smooth nonzero-query regime. The wrapper retains this property under FiLM because FiLM changes only the latent parameters of the same convex potential family.

The non-convex query head remains available as a frozen comparison baseline, but is not admitted by this task-conditioned v1 wrapper.

## Tests and limitations

The CPU synthetic fixture builds an actual small M80 model with a synthetic native-interval geometry, no external corpus or checkpoint. Tests cover:
- 31D task feature construction and frame validation;
- exact identity at zero-initialized FiLM;
- task conditioning effect after a controlled parameter update;
- M80 query Jacobian vs central finite differences;
- explicit identity-contract gating;
- 10D and 9D material chain rule;
- frozen-backbone training boundary.

This is a synthetic operator integration test. It is not M69 real geometry evaluation, checkpoint accuracy, training convergence, or generalization evidence. The M80 model is exercised on CPU with a synthetic 8-interval geometry; no RTX 5090 long training is run.

## Next milestone

Train/evaluate the task FiLM adapter only on the already-approved M69 query-label contract when data is explicitly made available; then compare FiLM with conditional normalization and cross-attention under matched parameter and query budgets. Do not claim full geometry-task generalization until geometry-disjoint tests exist.
