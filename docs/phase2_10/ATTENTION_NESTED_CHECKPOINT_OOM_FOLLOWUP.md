# Full V1 attention nested-checkpoint OOM follow-up

## Latest user-observed failure

The smoke now reaches the replay of the query-conditioned geometry-attention Jacobian checkpoint. During recomputation, the inner per-interval checkpoint replays `self.key(feature_input)` and fails with a 2 MiB allocation while only 72 MiB remains free.

This differs from the prior finite-check OOM. The finite-check temporary was already changed to chunk scope. The present stack shows nested checkpoint recomputation:
- outer checkpoint wraps attention correction + ten query derivatives;
- inner checkpoint wraps each of roughly 1,150 interval chunks;
- `create_graph=True` builds the higher-order query-Jacobian graph.

The nested replay machinery plus the already-resident million-interval feature/autograd state leaves insufficient headroom.

## Code change

Removed the inner per-interval checkpoint from `GeometryPhysicsCrossAttention.forward`.

The caller in `UnifiedM80SQNO.evaluate_queries` continues to checkpoint the complete attention-plus-query-Jacobian function for each query. Interval processing remains chunked with online log-sum-exp, so no full Key/Value or score tensor is materialized. This avoids nested checkpoint replay while retaining the outer recomputation boundary around the complete higher-order operation.

The attention test now wraps the complete module call in the same outer non-reentrant checkpoint before constructing a query derivative and backpropagating.

## Local verification

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_geometry_physics_attention_chunked -v
```

Then rerun the Full V1 one-batch smoke.

## Important

This is a targeted mitigation, not a verified solution. Removing inner checkpoints may increase transient activations during outer recomputation. If the smoke still OOMs, the next robust step is an analytic query-Jacobian for the attention branch (streaming weighted key/value covariance) so the 10 query derivatives do not require ten autograd traversals over all native intervals. Do not disable attention or Jacobian supervision.
