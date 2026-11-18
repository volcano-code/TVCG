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


## 9. Round 1 CI evidence

GitHub Actions run `44242728144` passed all stages on the Round 1 code revision:
- checkpoint contract: 7 passed;
- Pure Quadric / Hybrid local correction: 9 passed;
- actual M80 integration regression: 7 passed.

Total: **23 CPU tests passed**. The local correction suite includes finite-difference checks for selected columns of both (J_\beta) and (J_\psi), plus regularization gradient validation. This is software and synthetic quadrature evidence only, not a trained M80 or real 512³ physical result.


## 10. Phase 2.6 Round 2 — Hybrid Unified Optimization

Round 2 does not replace the M80 global operator. It makes Local Correction a residual material-response degree of freedom inside the same optimization state.

Let \(M_{80}(\beta;\phi_G,q)\) denote the existing geometry/task-conditioned M80 Moment operator. Let \(M_L(\beta,\psi)\) and \(M_L(\beta,0)\) be the local-field quadrature moments for the same registered geometry. The hybrid surrogate is defined as

\[
M_H(\beta,\psi)
=
M_{80}(\beta)
+
\left[M_L(\beta,\psi)-M_L(\beta,0)\right].
\]

Therefore

\[
M_H(\beta,0)=M_{80}(\beta),
\]

which is an explicit zero-correction identity rather than an informal comparison.

The optimization state is

\[
z=(\xi,\psi),\qquad
\beta=\operatorname{S9Decode}(\xi),
\]

where \(\xi\in\mathbb R^9\) is the existing projective Quadric chart and \(\psi\in\mathbb R^K\) are local correction coefficients. The single differentiable operator now exposes

\[
\frac{\partial M_H}{\partial\xi},
\qquad
\frac{\partial M_H}{\partial\psi},
\]

to the same M69 KKT construction. Neural Newton sees the augmented primal dimension \(9+K\); it does not use a separate optimizer for Local Correction.

### Exact boundary

The learned/hybrid operator is only a proposal mechanism. Exact acceptance receives both \((\beta,\psi)\) and must return exact ten-dimensional moments plus connectivity, topology and event validity. The exact oracle is never replaced by the learned Moment prediction.

### Evidence boundary

Round 2 proves the software contract and differentiable coupling on synthetic quadrature fixtures. It does **not** yet prove a lower exact minimum mass, 512³ physical optimality, or end-to-end speedup. Those require the registered M69 geometry/material oracle and a controlled Pure-Quadric-vs-Hybrid experiment.

### Round 2 tests

The CI now checks:
- exact \(\psi=0\) identity to the M80 baseline;
- joint beta/local Jacobian finiteness;
- augmented KKT dimension \(9+K\);
- independent exact oracle requiring both \(\beta\) and \(\psi\);
- exact-oracle evaluation counting;
- immutable fixed/protected samples.

This keeps Phase 2.6 faithful to its role as a **Hybrid Ablation**, rather than silently turning Local Correction into a second serial optimizer.


## 11. Round 2 CI evidence and audit fix

GitHub Actions run \`44244169573\` passed the full Phase 2.6 CPU workflow:
- checkpoint contract: 7 passed;
- Round 1 Local Correction: 9 passed;
- Round 2 Hybrid Unified Optimization: 6 passed;
- existing M80 integration regression: 7 passed.

Total: **29 CPU tests passed**.

During R2 validation we found and fixed a real edge case in the existing S9 chart: when a Quadric anchor contains zero coordinates, omitting a zero-magnitude coordinate can make the projected coordinate basis rank-deficient. The chart now omits the largest-magnitude anchor coordinate, preserving a full-rank tangent basis. This is a numerical robustness fix, not a relaxation of the KKT event rules.

The R2 synthetic exact-oracle tests deliberately use a relaxed TaskSpec for contract validation so that the test isolates the oracle interface rather than claiming a physical optimum. The production M69 TaskSpec remains unchanged and must be used for the eventual Pure-Quadric-vs-Hybrid ablation.
