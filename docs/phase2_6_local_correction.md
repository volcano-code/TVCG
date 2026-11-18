# Phase 2.6 — Local Correction v1 (Pure Quadric / Hybrid)

## Research question

Can a local material field improve representational capacity beyond the Pure Quadric baseline without changing the original fixed shell, protected region, material policy, or moment convention?

This implementation is an explicitly bounded continuous relaxation. It is not yet a binary topology optimizer or a proof that hybrid designs improve the M69 objective.

## 1. Unified material field

For normalized coordinates x and ten Quadric coefficients beta, the signed query follows the repository's Phase 2.1 convention:
[
q_\beta(x)=\beta_0+\beta_xx+\beta_yy+\beta_zz+
\beta_{xx}x^2+\beta_{yy}y^2+\beta_{zz}z^2+
2\beta_{yz}yz+2\beta_{xz}xz+2\beta_{xy}xy.
]
The Pure Quadric relaxed retention is
[
r_Q(x)=\sigma(q_\beta(x)/\tau).
]
Positive q is the retained side, matching the Phase 2.1 analytic plane-cut fixture.

## 2. Local correction

A compactly supported Wendland C2 basis is evaluated at K local centers:
[
B_k(x)=\begin{cases}(1-s)^4(4s+1),&s=\|x-c_k\|/R<1\\0,&s\ge1.\end{cases}
]
The bounded logit correction is
[
\delta_\psi(x)=\epsilon\tanh(B(x)^T\psi).
]
It is applied only to explicitly editable samples:
[
r_H(x)=
\begin{cases}
1,&x\in\Omega_{fixed}\cup\Omega_{protected},\\
\sigma(q_\beta(x)/\tau+\delta_\psi(x)),&x\in\Omega_{editable}.
\end{cases}
]
When `psi=0`, the hybrid exactly reduces to Pure Quadric. Fixed and protected samples remain fully retained independent of beta/psi. Correction is bounded in logit space, and retention remains in [0,1].

## 3. Moment integration

The sample quadrature integrates the repository's raw-moment ordering:
[
[1,x,y,z,x^2,y^2,z^2,yz,xz,xy].
]
The Quadric query's cross terms carry factor 2; the integrated raw moments do not. The implementation preserves this distinction.

## 4. Material masks and acceptance

Fixed, protected and editable masks must be boolean, pairwise disjoint and form a complete partition of the integration samples. There is no implicit editable-region inference. Binary thresholding is exposed separately. Connectivity, shell preservation on the final voxel solid, and topology validity must be checked by an independent exact binary-material validator; they are not approximated by the differentiable field.

## 5. Pure Quadric vs Hybrid ablation

The module reports pure/hybrid ten moments, retained mass, and correction magnitude on the same sample points and quadrature weights. It does not label either result as physically feasible or claim a hybrid improvement. The next experiment must optimize both representations under identical exact constraints and compare exact retained mass, feasibility, topology, and oracle budgets.

## 6. Tests and evidence boundary

Seven CPU tests cover zero-correction identity, immutable fixed/protected masks, bounded local correction, gradient flow, raw-moment order, hybrid ablation metrics, invalid mask rejection, and fail-closed local-correction contract.

These tests use analytic synthetic samples. They do not establish 512^3 accuracy, real-object topology, M69 optimality, or speedup.

## 7. Next work

1. Connect the field to the registered M69 native geometry mask producer; do not guess masks.
2. Derive/validate the exact alpha-to-beta/local-psi material map and its Jacobian.
3. Add exact binary threshold + connectivity/shell/protection checks.
4. Optimize Pure Quadric and Hybrid under the same M69 exact oracle and report whether local correction actually improves the feasible minimum-mass solution.
5. Keep Hybrid optional unless the ablation demonstrates a real benefit.


## 8. Joint design Jacobians

The module now exposes `hybrid_moments_and_jacobians`, returning the same integrated Moment vector and both derivative blocks:
[
J_\beta=\partial M/\partial\beta,\qquad
J_\psi=\partial M/\partial\psi.
]
This is the local-correction counterpart to the Phase 2.1 Query/Material Jacobian contract. It differentiates the continuous relaxation; it does not claim a derivative through hard thresholding or topology changes.

An optional `local_correction_regularization` adds coefficient L2 and nearby-center smoothness penalties. These are optimization regularizers, not M69 physical constraints.

Two additional tests check selected columns of both Jacobian blocks against central finite differences and verify regularization gradients.
