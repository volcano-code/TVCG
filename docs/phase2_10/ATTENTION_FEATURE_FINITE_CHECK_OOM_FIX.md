# Full V1 Attention OOM follow-up — finite-check temporary allocation

## Observed failure

The latest CUDA smoke reached `GeometryPhysicsCrossAttention.forward` and failed at the input guard:

`torch.isfinite(interval_features).all()`

The failed temporary allocation was 216 MiB with 241.94 MiB free. The interval feature matrix has over one million rows; `torch.isfinite` materializes a boolean tensor with the same element count before reducing it.

This is a validation-temporary allocation, not an attention-score or Key/Value projection allocation.

## Fix

- Keep the full coefficient finite check (only Qx10).
- Keep interval-mass finite/positive checks (one scalar per interval).
- Replace the full interval-feature finite check with a finite check inside the existing interval-chunk loop, immediately before Key/Value projection.
- The validation temporary is now bounded by `chunk_intervals x width`.
- Added a unit test confirming that a NaN in a later interval chunk is still rejected.

No model module, query/Jacobian path, physical mass, or attention operation is disabled or changed.

## Local verification

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_geometry_physics_attention_chunked -v
```

Then rerun the Full V1 CUDA smoke with the current Work manifest.

## Status

Code is committed but not yet run on the user's GPU. The previous OOM is resolved at the source-code level; the complete CUDA update and checkpoint round-trip remain unverified. Other later memory peaks may still appear.
