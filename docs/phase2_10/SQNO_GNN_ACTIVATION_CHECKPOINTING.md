# 2026-11-20 SQNO GNN second-stage CUDA memory mitigation

## User rerun

The previous GNN target chunking moved the OOM from a 1.26 GiB positional-encoding allocation to a 16 MiB MLP allocation, but the process still had only 64.94 MiB free. This confirms that per-edge activations were reduced, but all chunk computation graphs remained retained for backward, alongside the large P-CRAFT graph and native interval feature tensors.

## Additional change

The SQNO GNN now:
- reduces default target chunk size from 8,192 to 1,024;
- applies non-reentrant activation checkpointing to each initial-message target chunk;
- applies non-reentrant activation checkpointing to each feature-propagation target chunk;
- binds each chunk's edge tensor and layer module into the checkpoint closure to avoid Python late-binding errors during backward recomputation.

This trades extra message recomputation for lower saved edge-activation memory. Full source nodes and the global mean context are still used; the topology and global aggregation semantics are not replaced with independent chunk models.

## Verification

The existing `test_quadrann_chunked_gnn.py` remains the small-graph full-vs-chunked forward and gradient contract. Run it after pulling the latest commit.

## Retest

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_quadrann_chunked_gnn -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.spin_pino.tests.test_transolver_chunked -v
```

Then rerun the Full V1 one-batch smoke. If OOM remains, the next target is not to increase chunk size: profile and reduce retained per-interval feature/cache memory and the simultaneous P-CRAFT/SQNO autograd graphs. Do not reduce the native geometry resolution or silently disable GNO/Transolver to make the smoke pass.
