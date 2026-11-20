# Full V1 third CUDA OOM — SQNO fusion-stage streaming

## User-observed failure

After the Transolver and QuadrANN GNN chunking/checkpoint changes, Full V1 reached `SpinQuadrANNV12.encode_geometry` and failed at:

`self.fusion(torch.cat((q_features, final_features, p_local, p_global_expanded, local_component_context), dim=-1))`

For 1,177,764 native intervals, the full concatenated fusion input requested 3.09 GiB while only 2.92 GiB remained. This identifies the SQNO per-interval fusion concatenation as the next peak-memory allocation.

## Code changes

1. `SpinQuadrANNV11._gnn_features(..., return_layer_features=True)` can now return the per-layer feature tensors without materializing a full `torch.cat(layer_outputs, dim=-1)` tensor for every native interval.
2. `SpinQuadrANNV12` consumes those layer tensors and forms the concatenated GNN feature only for each 1,024-interval fusion chunk.
3. The existing P-CRAFT local feature, final GNN feature, global context and component context are gathered/concatenated only within that chunk.
4. The pointwise fusion MLP and weight head are checkpointed per chunk during training.
5. The cache stores the small pooled component context rather than a second full per-interval component-context tensor. The full per-interval GNN output remains accessible as the cache's `quadrann_features` field.

The mathematical pointwise fusion function, shared global context, component aggregation and final single global mass-prior softmax are preserved. The change only changes the order/chunking of independent per-interval fusion evaluations.

## Test update

`tests/test_quadrann_chunked_gnn.py` now also checks that streamed layer-feature mode reconstructs the same concatenated feature tensor and final-layer output on a deterministic small graph.

## Local verification

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_quadrann_chunked_gnn -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.spin_pino.tests.test_transolver_chunked -v
```

Then rerun the one-batch Full V1 smoke:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest /data/xvxinyao/TVCG/datasets/phase2_10_corpus_v1/manifest.json \
  --config full_v1_candidate \
  --device cuda --grid-size 64 --allow-absolute-assets --smoke-update \
  --output runs/phase2_10/smoke/checkpoint.pt
```

## Not claimed

- This change has not yet been run on the user's GPU.
- No successful Full V1 smoke update or checkpoint restore is claimed.
- Other retained native interval features, GNO/P-CRAFT graphs, quadrature nodes and autograd caches may still become the next peak.
- No model modules have been disabled and native geometry resolution is unchanged.
