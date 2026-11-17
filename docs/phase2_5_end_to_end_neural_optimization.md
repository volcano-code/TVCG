# Phase 2.5 — End-to-end safeguarded optimization loop (synthetic solver gate)

## Scope and evidence boundary

This phase adds an executable active-set / KKT / trust-region / exact-merit / fallback loop and a reproducible convex-QP solver comparison. The comparison is **not** a physical M69 benchmark and does not use trained M80 weights. It validates solver control flow and fair accounting before attempting the physical material-design problem.

## 1. Active-set convention

All inequalities use `g_i(x)<=0`; multipliers satisfy `lambda_i>=0`. The active-set update:
- adds violated or near-boundary constraints;
- releases active constraints whose multiplier becomes negative;
- preserves multipliers for retained active constraints;
- initializes newly active multipliers to zero.

Complementarity is reported as `||lambda * g_A||_infinity`.

## 2. KKT Newton proposal

For a fixed active set, differentiate the Lagrangian
[
L(x,\lambda_A)=f(x)+\lambda_A^Tg_A(x)
]
and solve the symmetric saddle-point system
[
\begin{bmatrix}H_L+\mu I&J_A^T\\J_A&0\end{bmatrix}
\begin{bmatrix}\Delta x\\\Delta\lambda_A\end{bmatrix}
=-\begin{bmatrix}\nabla_xL\\g_A\end{bmatrix}.
]
Regularization is restricted to the primal Hessian block. The step is trust-radius limited.

## 3. Trust-region acceptance and exact fallback

The model predicts reduction in a quadratic-penalty merit. The exact oracle computes actual reduction using the same physical objective and inequality functions:
[
r=\frac{\Phi_{exact}(x)-\Phi_{exact}(x+\Delta x)}
{\Phi_{model}(x)-\Phi_{model}(x+\Delta x)}.
]
A step is accepted only if predicted and actual reductions are positive and `r>=0.05`. Backtracking scales are fixed. If the surrogate proposals fail, an exact Newton direction is computed and evaluated under the same exact merit. Every exact merit evaluation is counted.

## 4. Fair synthetic baseline

The benchmark solves the same two-dimensional convex QP from the same initial state for:
- `exact_newton`: exact quadratic objective derivatives;
- `surrogate_newton`: a deterministic, slightly perturbed quadratic surrogate proposal.

Both use the same exact objective for acceptance and final reporting, the same linear inequality, penalty, trust radius, stopping tolerances, and iteration limit. The surrogate is a synthetic proposal model, not a trained M80 operator. Results therefore do not imply neural-network speedup.

## 5. Tests and known limitations

Tests cover active-set sign/multiplier behavior, KKT direction construction, exact baseline convergence, exact checking of surrogate proposals, and identical comparison protocol. No dataset or TRELLIS assets are used.

This phase does not yet establish:
- convergence on the true M69 ten-constraint material problem;
- actual M80 checkpoint proposals in the solver;
- real geometry feasibility or connected binary topology;
- a speedup over Spin-It Faster or Exact Newton;
- implicit differentiation through the converged optimizer.

The next integration gate is to replace the synthetic surrogate callback with the trained task-conditioned M80 moment potential and replace the QP exact merit with the independent M69 exact oracle, while retaining the same acceptance and fallback contract.
