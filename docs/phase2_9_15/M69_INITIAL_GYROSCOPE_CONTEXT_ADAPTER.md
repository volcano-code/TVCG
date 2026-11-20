# Phase 2.9.15 — M69 Initial-State Contract Adapter (increment 1)

## Implemented

The Unified M80+SQNO bridge now includes `M69GyroscopeContext`, a frozen context record that validates the minimum state contract required to invoke the existing M69/S9 proposal path:

- normalized raw quadric anchor `beta_anchor[10]`;
- positive-mass `original_moments[10]`;
- reviewed `m69.fixed-axis-task.v1` task schema;
- positive physical `length_scale`;
- boolean active-constraint mask `active_mask[10]`;
- matching device and dtype for anchor and reference moments.

The proposal preview now evaluates the *current S9-decoded coefficient vector* `chart.decode(local)`, rather than always previewing the chart anchor. This prevents UQ/Event gating from being performed at a different state than the KKT proposal.

## Tests added

`tests/test_m69_gyroscope_context.py` covers valid context and rejection of:
- non-unit projective quadric anchor;
- unsupported TaskSpec schema;
- invalid active-mask shape.

## Local command

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_m69_gyroscope_context -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_faster_bridge -v
```

## Important limits

This is a validated adapter contract, not automatic extraction of a Spin-It initial gyroscope. The caller must still provide the anchor, original moments, TaskSpec, active mask and length scale from the same initial gyroscope/task record.

The repository's hash-pinned TaskSpec adapter deliberately leaves semantic mapping pending review. Therefore this increment does not guess how arbitrary Spin-It fields map to the M69 axis, origin, contact, fixed-shell policy or stability tolerances.

The exact merit and fallback remain caller-supplied. The production Faster entry point and an actual initial-gyroscope replay are not yet wired or tested. No physical optimization or speedup is claimed.

## Next increment

Implement a source-pinned state mapper from the reviewed Spin-It initial-state export and reviewed M69 TaskSpec payload into `M69GyroscopeContext`, with explicit field mapping and round-trip checks. Then execute a registered analytic M69 fixture through proposal, exact merit, accept/reject and fallback before connecting the first real Work trajectory.
