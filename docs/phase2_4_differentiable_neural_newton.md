# Phase 2.4 — Differentiable Neural Newton v1

## Objective

Turn the Phase 2.3 fixed-active-set KKT linearization into a safeguarded optimization step, with a nine-dimensional local chart for projective Quadric coefficients, trust-radius control, exact-merit acceptance, and an explicit exact-solver fallback.

## 1. S9 local material chart

The ten Quadric coefficients are projective under positive scaling. Given a unit anchor `beta0 in S9`, construct an orthonormal tangent basis `U in R^(10x9)` satisfying:
[
U^TU=I_9,quad U^T\beta_0=0.
]
The local chart is
[
\beta(\xi)=\frac{\beta_0+U\xi}{\|\beta_0+U\xi\|}.
]
The implementation rejects zero vectors and crossing the local sign chart. This is a local chart, not a global parameterization of projective Quadric space.

## 2. Safeguarded KKT Newton direction

For KKT unknown ordering `z=(x,nu,lambda_A)`, solve:
[
J_F\Delta z=-F.
]
The implementation requires a symmetric KKT Jacobian and regularizes only the primal Hessian block:
[
\begin{bmatrix}H+\mu I&A^T\\A&0\end{bmatrix}
\begin{bmatrix}\Delta x\\\Delta\lambda\end{bmatrix}
=-F.
]
It does **not** add damping to the whole saddle-point matrix, which would alter the constraint equations. Primal and multiplier step radii are controlled separately.

## 3. Exact merit acceptance and fallback

The learned/model-derived step is only a proposal. A caller-supplied exact merit function decides acceptance using backtracking scales. If every trial is rejected, the result explicitly requests the exact solver fallback. The wrapper can invoke a caller-provided exact fallback callback. The exact merit must include the caller's physical residual and feasibility penalties; this module does not invent a replacement for the M69 oracle.

## 4. Differentiability scope

- S9 chart and linear solve are differentiable in ordinary PyTorch execution.
- Backtracking/acceptance and active-set selection are discrete control flow and are not differentiated through.
- The Phase 2.3 fixed-active-set KKT Jacobian remains the local differential model.
- This version does not yet implement implicit differentiation of the converged optimizer solution.
- It does not implement a full active-set update policy, nor does it certify local minima or global optimality.

## 5. Tests

CPU tests verify S9 basis orthogonality and chart behavior, saddle-point KKT solve, primal trust radius, rejection of nonsymmetric KKT matrices, exact merit backtracking, rejected-step fallback signaling, and exact fallback invocation.

These are solver-component tests on synthetic systems, not a real M69 material optimization benchmark or a speedup claim.

## 6. Next gate

Wire the S9 chart directly to the M80 `beta_query -> Moment` map, compose Phase 2.3 M69 KKT with the local chart, and run a synthetic constrained optimization against an exact analytic teacher. Then compare exact Newton and neural-proposed Newton under identical initial states, stopping criteria, and exact evaluation budgets.


## 7. Final CI and S9-to-M69 integration

GitHub Actions run `44119984215` passed the Phase 2.4 CPU suite. The final suite covers:
- deterministic orthonormal S9 tangent basis and local chart decode/encode;
- symmetric saddle-point Newton solve with regularization applied only to the primal Hessian block;
- primal trust-radius clipping;
- refusal of nonsymmetric KKT Jacobians;
- exact-merit backtracking and rejected-step fallback;
- invocation of an exact fallback callback;
- direct composition of the S9 chart with a synthetic Moment map and the Phase 2.3 M69 KKT residual/Jacobian.

The S9-to-M69 test uses an explicitly synthetic differentiable response and validates chain-rule wiring only. It is not an M69 material optimum, trained M80 prediction, or physical speedup result.

**Remaining:** active-set update policy, full M80-to-S9-to-M69 exact-oracle loop, trust-region ratio based on an independently evaluated exact residual, and implicit differentiation of a converged optimizer solution.
