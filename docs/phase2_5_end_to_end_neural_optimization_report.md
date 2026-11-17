# Phase 2.5 — End-to-end safeguarded optimization loop

## Scope

Phase 2.5 delivers two pieces:
1. a bridge that composes a caller-supplied task-conditioned M80 Moment function with the Phase 2.4 S9 chart, Phase 2.3 M69 KKT system, safeguarded KKT Newton direction, and exact M69 merit acceptance/fallback callbacks;
2. an executable convex-QP synthetic solver benchmark that tests active-set, KKT, trust-region, exact-merit and fallback control flow.

**Evidence boundary:** the solver comparison is a synthetic convex QP diagnostic. It is not a trained M80 model experiment, not an M69 physical-material optimization result, and not a speedup claim.

## 1. Integrated neural proposal bridge

`propose_m69_neural_newton_step` accepts a differentiable `beta_to_moments` callback. This callback may close over a task-conditioned M80 model and encoded geometry cache. The bridge:
- constructs the local S9 chart;
- evaluates the M69 differentiable constraints and fixed-active-set KKT system;
- rejects invalid M69 derivative branches;
- computes the Phase 2.4 safeguarded KKT Newton direction.

`accept_m69_neural_newton_step` evaluates a caller-supplied independent exact M69 merit function and invokes an exact-solver fallback callback if all neural trials are rejected.

The caller still has to provide a semantically certified Phase 2.1 query-to-material mapping, a real M80 checkpoint/geometry, the correct active set, and an independent exact M69 oracle. This repository milestone provides the executable interface, not those physical inputs.

## 2. Active-set and KKT conventions

All inequalities use `g<=0`; inequality multipliers satisfy `lambda>=0`. The active-set update adds violated/near-boundary constraints, releases active constraints with negative multipliers, preserves retained multipliers and initializes new multipliers to zero. Complementarity is checked with `||lambda*g_A||_infinity`.

For a fixed active set, the KKT direction uses the Lagrangian Hessian and the saddle-point block matrix. Regularization is applied only to the primal Hessian block. The synthetic benchmark starts with its single inequality registered as active at the constrained optimum; both methods use this same declared working set. Active-set discovery is tested separately and is not part of the benchmark comparison.

## 3. Trust-region acceptance and fallback

The model proposal is evaluated with a model L1-penalty merit. Actual reduction is computed with the exact objective and constraints:
[
r=\frac{\Phi_{exact}(x)-\Phi_{exact}(x+s)}
{\Phi_{model}(x)-\Phi_{model}(x+s)}.
]
A proposal is accepted only when both reductions are positive and `r>=0.05`. Backtracking scales and trust radius are fixed. If a surrogate proposal is rejected, an exact Newton correction is tried; the registered 2D benchmark also has a closed-form exact KKT fallback. All exact merit evaluations are counted.

## 4. Fair synthetic comparison

Both methods solve the same 2D convex QP from the same initial state, with the same exact objective, constraint, initial active set, tolerance, trust radius, merit penalty, acceptance ratio and iteration cap:
- `exact_newton`: exact objective derivatives;
- `surrogate_newton`: deterministic perturbed quadratic proposal objective.

Three seeds (2501, 2502, 2503) were run. All six solves converged to the same exact constrained optimum, with zero primal violation to floating-point tolerance and stationarity below (10^{-9}).

| Metric (mean over 3 seeds) | Exact Newton | Synthetic surrogate Newton |
|---|---:|---:|
| Iterations | 2.00 | 2.67 |
| Exact merit evaluations | 4.00 | 11.33 |
| Final objective | 1.2212307692 | 1.2212307692 |
| Max constraint violation | 0 | <3e-17 |
| Fallbacks | 0 | 2 |

The surrogate route requires more exact merit evaluations. This is expected for a deliberately imperfect synthetic proposal and is evidence that the acceptance/fallback system protects correctness, **not** evidence of acceleration. No trained neural network is involved in this benchmark.

Per-seed results are committed in `research/NeuralMoments/results/phase2_5_synthetic_qp_results_2026-11-17.json`.

## 5. CI and tests

GitHub Actions run `44127261155` passed all **7 tests** and completed the three-seed comparison plus artifact upload. Tests cover active-set signs and multiplier release, KKT direction, exact baseline convergence, surrogate exact checking, multi-seed fair protocol, closed-form exact fallback, and the S9-to-M69 KKT-to-Newton bridge.

## 6. Remaining blockers before a physical claim

- Execute the bridge with an actual trained task-conditioned M80 checkpoint and a registered native geometry.
- Establish that the query-to-material map satisfies the full Phase 2.1 contract for the exact M69 binary design semantics.
- Replace the synthetic merit with an independently implemented exact M69 geometry/material oracle.
- Implement and test active-set discovery on the full ten-constraint M69 system, including spectral and topology events.
- Compare against exact Newton/Spin-It Faster with identical physical tasks, accuracy, exact-query budgets and hardware.
- Add implicit differentiation only after local convergence and regularity assumptions are verified.

No TRELLIS data was generated or inspected in this phase; no 5090 long training was started.
