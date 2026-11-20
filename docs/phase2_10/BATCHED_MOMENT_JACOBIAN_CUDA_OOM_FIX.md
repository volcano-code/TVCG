# Phase 2.10 — Batched Moment/Jacobian OOM: node/query chunking and checkpointing

## Observed failure

The user's Full V1 CUDA smoke passed geometry encoding and reached
`BatchedSpinMomentOperator.evaluate`, then failed while materializing

`nvec[:, :, :, None] * nvec[:, :, None, :]`

The attempted allocation was 800 MiB with 772.5 MiB free. This is the per-node/per-query 10x10 Hessian intermediate used for the exact analytic Moment Jacobian.

## Implemented mitigation

1. `BatchedSpinMomentOperator` defaults to 2,048 integration nodes per block and one query per block.
2. The SQNO v1.2 constructor explicitly caps the operator at 2,048 nodes and one query, even if the inherited single-query moment operator uses a larger node chunk.
3. The entire node-block calculation (basis, smooth-hinge potential, Moment gradient, and analytic 10x10 Hessian/Jacobian contribution) is now wrapped in non-reentrant activation checkpointing whenever autograd is enabled and an input requires gradients.
4. The analytic Hessian formula and global summation are unchanged. Only the order of accumulation and recomputation strategy change.

Approximate largest raw Hessian intermediate per block is now
`2048 * 1 * 10 * 10 * 8 bytes` for float64, about 1.56 MiB, rather than hundreds of MiB for a large node/query block. Additional intermediates and autograd memory remain, so this is not a guarantee of fitting the complete graph.

## Tests added

- Jacobian and Moment outputs agree between 128-node/2-query and 17-node/1-query settings.
- Checkpointed Jacobian backward produces finite, nonzero gradients for weights and query coefficients.
- Existing single-query parity, coordinate-transform, symmetry/PSD and CPU/CUDA checks remain.

## Local verification

```bash
python -m pytest research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_batched_moment_operator.py -q
```

Then rerun:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest /data/xvxinyao/TVCG/datasets/phase2_10_corpus_v1/manifest.json \
  --config full_v1_candidate \
  --device cuda --grid-size 64 --allow-absolute-assets --smoke-update \
  --output runs/phase2_10/smoke/checkpoint.pt
```

## Status and limitations

- Code is committed; it has not yet been run in the user's local environment.
- The full Full V1 smoke has not yet succeeded.
- This reduces the analytic Jacobian integration peak, but model backward through the Jacobian loss may still retain other graphs. If OOM moves to backward, inspect higher-order autograd and consider checkpointing the complete per-query Moment/Jacobian evaluation.
- Do not disable Jacobian supervision, reduce the native 1024 RLE resolution, or remove Full V1 modules as a workaround.
