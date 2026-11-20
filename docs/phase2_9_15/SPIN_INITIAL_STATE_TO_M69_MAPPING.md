# Phase 2.9.15 — Spin-It Initial State to M69 Context Mapper

## Implemented

Added `spin_initial_state_mapper.py`, a fail-closed adapter that maps a provenance-bound, audited Spin-It initial-state record into the existing `M69GyroscopeContext`.

It requires explicit:
- source geometry SHA-256;
- initial-state payload SHA-256;
- TaskSpec payload SHA-256;
- normalized raw ten-coefficient quadric anchor;
- original full-material ten moments;
- length scale;
- active inequality mask;
- axis, axis origin and contact point;
- explicit `task_frame` coordinate declaration;
- fixed-material and design-domain fingerprints;
- audited evidence status.

It checks the source hashes, schema, required TaskSpec fields, task-frame agreement and the existing M69 context contract. It does not infer axes from principal axes, mesh bounds, contact geometry or Spin-It target defaults.

## Source audit finding: why the mapper is deliberately strict

The repository's committed Spin-It demo artifact at
`experiments/spin_it_reproduction/spin2014/modes/spinning-top/diagnostics.json`
is marked `evidence=RECONSTRUCTED_ASSET`. It reports `final_raw_moments`, not the original full-material moments required by the M69 reference constraints. It therefore cannot be mapped into a training/optimization context by this adapter.

The existing Spin-It mode runner creates deterministic 4x4x4 fixtures for mode regression. Those fixtures are software/reconstruction assets, not admitted real OBJ-derived initial gyroscopes.

The M69 task semantics document records a fixed task axis/origin/contact of (0,0,1)/(0,0,0)/(0,0,0), density 1, single-density removal, minimum-retained-mass objective and fixed-axis inertia reference. However, it explicitly states that per-mesh fixed/protected masks and cavity policy are not supplied by TaskSpec and require separate geometry-specific evidence.

The hash-pinned TaskSpec extraction adapter also intentionally leaves semantic mapping pending review. Do not manufacture the missing per-object material masks or original moments.

## Tests added

`tests/test_spin_initial_state_mapper.py` covers:
- valid explicit audited record;
- rejection of reconstructed/demo evidence;
- TaskSpec hash mismatch;
- initial-state hash mismatch;
- non-task-frame coordinates;
- axis mismatch.

## Local command

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_spin_initial_state_mapper -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_m69_gyroscope_context -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_faster_bridge -v
```

## Boundary and next step

This mapper defines and validates the state handoff. It does not yet produce the required audited record from the native Spin-It C++/Python output. The next engineering task is to extend the existing native Spin-It export path to emit the explicit schema, including original full-material moments and fixed/design/protected material fingerprints, then independently verify those fields against the same sealed geometry and TaskSpec.

Do not connect the M69 KKT adapter to a production gyroscope until that export is emitted and verified. Also keep the M69 fixed-axis material-removal task distinct from the Spin-It Faster quadric-query task unless a mathematical equivalence/mapping is explicitly proven; the existing repository documentation treats these as separate task profiles.
