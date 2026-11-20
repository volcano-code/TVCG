# Full V1 second CUDA OOM — SQNO QuadrANN GNN diagnosis and chunked fix

## User-observed failure

After the first Transolver chunking change, the next Full V1 smoke attempt progressed further and failed in the SQNO branch:

`SpinQuadrANNV12.encode_geometry -> SpinQuadrANNV11._gnn_features -> InitialGeometryLayer.message -> positional_encoding`

The failed allocation was 1.26 GiB with 1.29 GiB free. The failing tensor is edge-message positional encoding, not the Transolver assignment tensor.

## Root cause

The original SQNO QuadrANN path constructs a topology graph over all native RLE intervals and invokes PyG MessagePassing for every edge at once. With approximately 1.18M intervals and 16 neighbors, the graph has roughly 18.8M directed edges. The initial message layer materializes absolute/relative positional encodings and MLP activations for all edges simultaneously. This creates a second large peak after P-CRAFT/Transolver has already retained geometry features.

## Code mitigation

- Added bipartite chunk-target message APIs to `InitialGeometryLayer` and `FeaturePropagationLayer`.
- Reworked `SpinQuadrANNV11._gnn_features` to process 8,192 target intervals at a time.
- Source nodes remain the complete global node set; target indices are remapped locally per chunk.
- For each message-passing layer, the global mean context is computed across all nodes before processing target chunks.
- Chunk outputs are concatenated in original node order.
- Topology graph semantics and cross-component filtering remain unchanged.
- Edge slices are copied per chunk rather than retaining a second full list of chunk arrays.
- Fixed a non-writable NumPy origin conversion warning in P-CRAFT.

This preserves full-graph source neighborhoods and global mean context; it is not independent per-chunk GNNs. The current implementation still materializes the CPU topology graph and the full node feature tensors, but it bounds the large per-edge message/positional-encoding activations.

## Test added

`spin_adapter/tests/test_quadrann_chunked_gnn.py`:
- compares chunked output to the original full-graph reference on a small deterministic graph;
- checks finite backward gradients through coordinates and GNN parameters;
- checks invalid chunk size rejection.

The test has been added but has not been run in the user's local environment.

## Local verification

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.spin_pino.tests.test_transolver_chunked -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_quadrann_chunked_gnn -v
```

Then inspect memory and rerun the one-batch Full V1 smoke:

```bash
nvidia-smi

python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest /data/xvxinyao/TVCG/datasets/phase2_10_corpus_v1/manifest.json \
  --config full_v1_candidate \
  --device cuda \
  --grid-size 64 \
  --allow-absolute-assets \
  --smoke-update \
  --output runs/phase2_10/smoke/checkpoint.pt
```

## Current limitations

- No successful rerun is claimed yet.
- P-CRAFT Transolver and SQNO GNN are now chunked over native intervals, but full native interval features, RLE geometry, graph indices, quadrature nodes and optimizer activations still consume memory.
- If OOM moves to GNO, interval encoding or quadrature construction, profile the new stack and chunk that exact stage next.
- Do not start Pilot epochs or long training until the one-batch smoke completes parameter update, cache invalidation and checkpoint resume round-trip.
