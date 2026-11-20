# Phase 2.9.14 — Unified M80+SQNO to M69/Faster KKT Bridge

## Implemented in this increment

Added `unified_faster_bridge.py` to connect the current Unified M80+SQNO fused Moment function to the repository's existing Phase 2.3/2.4/2.5 M69/S9 KKT and safeguarded Newton implementation.

The bridge:

1. Builds a `beta -> moments[10]` callback from `UnifiedM80SQNO.evaluate_queries`.
2. Requires `create_graph=True`, because the M69 KKT construction differentiates through the Moment Jacobian to build its stationarity Jacobian.
3. Calls the existing `propose_m69_neural_newton_step` implementation rather than implementing a second KKT solver.
4. Uses the existing optional UQ/Event outputs as conservative proposal eligibility gates only.
5. Delegates exact merit acceptance, rejection and optional exact fallback to the existing `accept_m69_neural_newton_step`.

## Files

- `research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/unified_faster_bridge.py`
- `research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_unified_faster_bridge.py`

## Existing solver components reused

- `research/NeuralMoments/phase2_3_differentiable_m69_kkt.py`
- `research/NeuralMoments/phase2_4_neural_newton.py`
- `research/NeuralMoments/phase2_5_end_to_end_solver.py`

The Phase 2.5 synthetic QP benchmark is not used as evidence of physical Faster integration. The new bridge calls the M69 KKT path and requires the caller to provide the authoritative TaskSpec, original moments, active mask, exact merit callback and exact fallback.

## Local tests

From the repository root:

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_faster_bridge -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_neural_newton_adapter -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_prediction -v
```

## Required caller contract

- The geometry cache must be a differentiable training cache, not an inference-only cache.
- `beta_anchor` must be the raw ten-coefficient vector in the same coordinate convention as the M69 TaskSpec.
- `local` is the 9D S9 chart coordinate used by the existing M69 KKT code.
- `original_moments`, `task`, `length_scale` and `active_mask` must come from the same initial gyroscope/task instance.
- `exact_m69_merit` must evaluate the actual exact Faster/M69 objective and constraints for the decoded trial state.
- `exact_solver_fallback` must call the existing exact solver; it must not return a neural prediction.

## Not claimed

- This is an adapter to the existing M69 KKT abstraction, not proof that the original Spin-It Faster optimizer's complete production entry point has been wired.
- No physical trajectory replay has been run.
- No CUDA test, real-data optimizer update, speedup or convergence claim is made.
- Tests were added but have not been executed in this GitHub-only editing session.

## Next step

Add a narrowly scoped state adapter that maps the exact Spin-It initial gyroscope and Faster optimizer state into the existing M69 TaskSpec/S9 coordinates, then run one analytic/registered M69 fixture through proposal -> exact merit -> accept/reject/fallback. Only after that should the bridge be connected to Work's first real trajectory.
