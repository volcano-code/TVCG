# Phase 2.3 — Unified differentiable M69 physical constraints and KKT

## Research objective

Connect the shared M80 moment potential to the exact M69 fixed-axis minimum-retained-mass constraints, then form a differentiable fixed-active-set KKT residual and its Jacobian. This phase does not implement the full Neural Newton line search/trust-region solver.

## 1. Computational graph

```
task-conditioned M80 convex moment potential
            |
            v
       M(alpha; G,q)
            |
            v
 exact M69 ten moment slacks s(M,M0,q)
            |
       g(M) = -s(M) <= 0
            |
            v
 min retained mass M[0]
            |
            v
 fixed active set, L=f + lambda^T g + nu^T h
            |
            v
 F_KKT=[grad_alpha L, h, g_active]
            |
            v
 J_F = dF_KKT / d(alpha,nu,lambda_active)
```

## 2. Constraint semantics and sign

The differentiable adapter preserves the ten registered M69 slack values from the source-compatible Phase 1.2 contract. It exposes `g=-slack<=0` for the standard Lagrangian
[
L(alpha,lambda,
u)=M_0(alpha)+lambda^Tg(alpha)+
u^Th(alpha),quad lambdage0.
]
The objective is retained mass. Connectivity remains an independent discrete whole-solid audit and is not fabricated as a differentiable scalar.

The training-only and solver-only polar margins remain separate from final M69 acceptance. They are not substituted into the physical constraints.

## 3. Differentiability and event policy

The adapter returns constraint values even when the derivative is invalid, but marks the branch invalid and refuses KKT construction at:
- zero or nonpositive candidate/reference mass;
- zero-norm transverse COM alignment;
- zero-norm inertia-alignment residual;
- repeated eigenvalues in the transverse inertia block used by principal dominance;
- repeated minimum covariance eigenvalues used by the PSD constraint;
- nonfinite constraints;
- inactive constraints near zero, which are active-set switching boundaries. A constraint that is exactly active remains differentiable on a fixed active set; its derivative is valid unless another spectral/norm event occurs.

No softplus, epsilon smoothing, or fake zero Jacobian is used to conceal these events. Connectivity, binary topology changes, and material-interface topology remain discrete and outside this smooth KKT layer.

## 4. KKT construction

For local feasible coordinates `x` (e.g. a nine-dimensional chart for the normalized Quadric coefficients), equalities `h(x)=0`, and inequalities `g(x)<=0`, the implementation uses
[
L=f(x)+\nu^Th(x)+\lambda_A^Tg_A(x)
]
and
[
F_{KKT}=[\nabla_xL, h, g_A]^T.
]
The unknown vector is `z=(x,nu,lambda_A)`; the Jacobian is obtained by automatic differentiation of the assembled residual. The active set is fixed for this Jacobian. The implementation reports inactive primal and active dual violations. It does not claim semismooth differentiation through active-set switches.

## 5. Implementation and tests

- `m69_constraint_slacks_torch`: differentiable PyTorch evaluation of all ten M69 moment slacks.
- `fixed_active_set_kkt`: generic fixed-active-set KKT residual and full Jacobian.
- `m69_kkt_from_moment_operator`: composes a differentiable Moment function with the M69 objective and inequalities.
- CPU tests compare M69 constraint Jacobians and KKT Jacobians with finite differences, verify signs/dimensions, and confirm derivative-event refusal.

Synthetic tests do not establish real-geometry physical feasibility, learned-operator accuracy, or end-to-end solver speed. The M80 model remains the upstream differentiable Moment potential; no baseline weights or architecture are changed.

## 6. Known limitations and next work

1. The current KKT layer accepts caller-supplied local feasible coordinates. The explicit S9 chart from beta coordinates to nine local coordinates still needs to be wired into the M80 material map.
2. Eigenvalue derivatives are refused at multiplicity; a robust spectral-gap event model or carefully justified invariant reformulation remains future work.
3. Connectivity and binary topology are independent exact acceptance checks, not differentiable constraints.
4. The KKT Jacobian is tested on smooth synthetic objectives and M69 constraint fixtures; no real M69 KKT labels exist yet.
5. Phase 2.4 should add the S9 chart, fixed-active-set trust-region Newton step, exact residual acceptance and fallback to the exact solver.


## 7. Final CI result

GitHub Actions run `44116813811` passed **10 tests in 1.34 seconds**. In addition to analytic constraint/KKT fixtures, the suite instantiates the actual task-conditioned M80 convex-potential wrapper on a synthetic native-interval geometry, composes its moment output into the M69 KKT layer, differentiates the KKT residual/Jacobian, and verifies finite gradients reach the Conditional Normalization adapter. This is a synthetic graph-integration test only; it does not claim physical M69 accuracy or trained-checkpoint performance.

A fixed-active-set correction was made during validation: an exactly active inequality is differentiable while the active set is held fixed. Only an inactive inequality near zero is rejected as an active-set switching event. Spectral multiplicity and zero-norm events remain derivative-invalid.
