# Phase 2.9.14 — Neural Newton Proposal Boundary

## Implemented

- `neural_newton_adapter.py` computes a damped least-squares proposal from an optimizer-supplied residual and Jacobian.
- Candidate steps are clipped by a trust-radius / maximum-step policy.
- Moment uncertainty, Jacobian uncertainty and Event probability are independent proposal eligibility signals.
- Ineligible proposals are not trial-evaluated.
- Eligible proposals are passed to a caller-supplied exact evaluator at the current and candidate states.
- The adapter accepts only if the exact objective decreases by the configured sufficient-decrease amount and the exact constraint violation is within an explicit feasibility tolerance.
- Rejected proposals return the unchanged current coefficients.
- The adapter does not implement Faster's objective, KKT system, constraints, or exact moment evaluator.

## Important mathematical boundary

The input `predicted_residual` and `predicted_jacobian` must be produced by a Faster-specific optimizer adapter representing the actual local stationarity/residual system. They must not be assumed to be the raw ten Moment outputs and their 10x10 coefficient Jacobian unless the Faster derivation proves that equivalence.

The exact evaluator callback must return:
- `objective`: exact objective at the supplied coefficients;
- `constraint_violation`: nonnegative exact violation metric (zero means feasible);
- optional exact solver diagnostics/payload.

The exact evaluator is authoritative. Neural predictions never accept a step or declare convergence.

## Tests added

`tests/test_neural_newton_adapter.py` covers:
- bounded candidate construction;
- uncertainty and event-risk fallback;
- exact feasible objective-decrease acceptance;
- exact infeasibility rejection;
- no trial evaluation for ineligible proposals.

## Local command

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_neural_newton_adapter -v
```

## Not yet complete

- The callback is a safe generic exact-evaluation boundary, not yet wired to the actual Faster/M69 solver entry point.
- No Faster-specific KKT/residual adapter is claimed.
- No trust-region radius adaptation or multi-step optimizer loop is claimed.
- Tests were added but not executed in this GitHub editing environment.
- No real-data training or performance claim is made.

Next: inspect the actual Faster optimizer API and implement a narrow adapter that constructs its true residual/Jacobian and maps the exact solver's acceptance result into this boundary.
