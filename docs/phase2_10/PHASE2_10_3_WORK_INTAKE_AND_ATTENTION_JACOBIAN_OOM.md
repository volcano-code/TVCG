# Phase 2.10.3 — Work delivery intake and Full V1 attention-Jacobian OOM

## Work handoff reviewed

Source: user-provided `CHAT_HANDOFF(1).md`.

The handoff reports:
- data gates: 8/8; `pilot_ready=true` for the operator-pilot scope;
- audit ran on CPU; training steps: 0;
- train: 31 geometries, 21 traceable generated source families, 2,019 Moment queries; 1,179 full Jacobians, 477 three-dimensional restricted Jacobians, 361 eight-dimensional restricted Jacobians, 2 Moment-only;
- strict development: 9 geometries, 8 source groups, 36 queries; 11 full Jacobians, 4 three-dimensional, 3 eight-dimensional, 18 Moment-only;
- locked fixed benchmarks: 5 geometries, 2 paper-reconstruction source groups, 320 queries; 195 full, 65 three-dimensional, 60 eight-dimensional;
- broader development archive: 13 geometries / 64 queries; strict development view excludes source families hit by public TRELLIS-500K; broader operator diagnostics has 10 groups; Urn upstream family remains unresolved;
- 31 train parent meshes include zero physical captures; two original templates remain in train;
- Diamond is one newly sealed real-source engineering example with native replay and independent 512^3 binary-material audit;
- Flowerpot v2 remains external_candidate;
- teacher remains at operator_pilot qualification, not author-equivalent production Faster qualification.

These counts supersede the previous 31 / 11 / 5 split snapshot. Use the new formal manifest and preserve three source roles. The handoff does not include the linked README, manifest, delta or detailed audit JSON files as separate attachments, so their individual hashes and contents have not been independently inspected in this review.

## Training status

`pilot_ready=true` is a Work data-gate decision for the stated operator-pilot scope. It does not mean model training has run. The audit reports zero training steps. Full V1 CUDA smoke remains blocked by GPU memory during the higher-order Jacobian of query-conditioned geometry attention.

## Latest runtime failure

The user reached:
`UnifiedM80SQNO.evaluate_queries -> GeometryPhysicsCrossAttention correction -> autograd.grad(... create_graph=True)`
and OOM occurred on a 2 MiB allocation with only 73 MiB free. The failure is in the higher-order derivative graph for attention corrections, after the analytic Batched Moment/Jacobian operator.

## Implemented mitigation

In `unified_m80_sqno.py`:
- process geometry-attention query Jacobians one query at a time;
- wrap the attention correction plus its ten query derivatives in non-reentrant activation checkpointing when create_graph=True;
- keep autograd enabled inside the checkpointed helper, including evaluation calls that enter from no-grad contexts;
- reuse detached attention diagnostics returned by the checkpointed function rather than recomputing the million-interval attention pass;
- apply the same query-chunk/checkpoint pattern to cross-scale attention Jacobians.

The mathematical query derivative is still explicitly computed for all ten correction channels. No attention branch, Moment/Jacobian loss, geometry resolution, or model module is disabled.

## Required local verification

After pulling latest branch:

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_prediction -v
python -m pytest research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_batched_moment_operator.py -q
```

Then run the exact full manifest preflight and one-batch smoke using the path and absolute-asset option from the Work handoff. Inspect the new manifest's actual split summary first; expected strict split counts from this handoff are 31 train / 9 development / 5 fixed benchmark, not the previous 31 / 11 / 5 snapshot.

## Scope boundary

- The handoff reports the data gate as pilot-ready, but the model smoke and training have not passed.
- The teacher remains operator_pilot qualified only.
- Diamond is a sealed real-source engineering example, not evidence that all training geometries are physical captures.
- The 31 train objects are traceable generated families, not physical captures.
- Do not start multi-epoch Pilot until Full V1 completes one real CUDA update, cache invalidation and checkpoint resume round-trip.
