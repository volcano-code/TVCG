# 2026-11-20 Full V1 CUDA Smoke OOM — diagnosis and mitigation

## Observed result

The user's Full V1 preflight passed:
- manifest: 31 train / 11 development / 5 locked benchmarks;
- 2,019 train queries;
- model factory: 4,302,996 trainable parameters;
- all requested Full V1 module flags present;
- first training record N002: 1,177,764 native RLE intervals at 1024^3.

The one-batch CUDA update then failed during native P-CRAFT Transolver PhysicsStateSlicing. The failed allocation was 1.12 GiB; only 474.56 MiB remained free on the 32 GiB RTX 5090 D. PyTorch had 28.42 GiB allocated and the process used 29.71 GiB including non-PyTorch allocations.

This is not a model-factory or manifest failure. It is a native-interval activation-memory failure at the first training geometry.

## Root cause

The original Transolver implementation materialized:
- projected states for all N native intervals;
- logits and temperatures for all N x heads x slices;
- soft assignments for all N x heads x slices;
- weighted assignments and deslicing activations.

For N=1,177,764, H=8 and S=32, a single float32 [N,H,S] tensor is approximately 1.12 GiB. Multiple such tensors and their autograd intermediates coexist. The failure occurred at softmax assignment allocation.

The configured 64^3 coarse P-CRAFT context grid does not reduce the number of native RLE intervals; native intervals are retained for exact query conditioning.

## Code mitigation

`PhysicsStateSlicing.forward` now processes native intervals in point chunks (default 16,384):
1. first pass computes chunked mass-weighted slice mass and token numerators;
2. Transolver attention operates on the small pooled slice-token tensor;
3. second pass recomputes each chunk's assignments and deslices the attended tokens;
4. training uses non-reentrant activation checkpointing on both passes, trading extra compute for lower saved-activation memory;
5. adaptive feature statistics are streamed rather than copying the full feature matrix to float64.

The mathematical mass-weighted global pooling and slice attention remain global across all native intervals. This is not independent per-chunk attention.

Added `tests/test_transolver_chunked.py`:
- chunked vs single-chunk forward equivalence;
- finite backward gradients;
- invalid chunk-size rejection.

## Retest procedure

After pulling the latest branch, inspect GPU processes first:

```bash
nvidia-smi
```

Do not kill unrelated jobs. Once sufficient GPU memory is available, rerun only the one-batch smoke:

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.spin_pino.tests.test_transolver_chunked -v

python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest /data/xvxinyao/TVCG/datasets/phase2_10_corpus_v1/manifest.json \
  --config full_v1_candidate \
  --device cuda \
  --grid-size 64 \
  --allow-absolute-assets \
  --smoke-update \
  --output runs/phase2_10/smoke/checkpoint.pt
```

## Limitations

- This code change is not yet runtime-verified on the user's GPU.
- GNO/native interval encoding and retained interval features still scale with 1.18M intervals; if OOM moves earlier or remains, profile and chunk those stages next.
- The first smoke update is still not a training epoch.
- Do not launch pilot or long training until the smoke report records a successful update, cache invalidation and checkpoint resume round-trip.
