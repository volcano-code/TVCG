# Phase 2.10.2 — Preflight, CUDA Smoke Update and Resume Verification

## Delivered

`training_v13/preflight_unified.py` adds two explicit modes:

### Preflight only

Checks:
- CUDA availability and GPU properties when CUDA is requested;
- strict corpus manifest, hashes and source-group split isolation;
- train-only Moment RMS normalizer;
- named model configuration and successful model construction;
- parameter inventory;
- first training geometry/query/label batch;
- native RLE resolution and interval count.

### One-batch smoke update

Enable with `--smoke-update`. In addition to preflight, it:
- performs one actual Unified M80+SQNO optimizer update on the first train geometry;
- checks finite loss/gradient and a nonzero parameter delta;
- checks that the pre-update geometry cache is rejected after the optimizer update;
- writes a checkpoint containing optimizer, scheduler, data fingerprint and training contract;
- constructs a fresh model/optimizer/scheduler and reloads the checkpoint;
- compares every restored model state tensor against the trained model.

This is an engineering smoke test, not a pilot epoch or performance result.

## Commands

Preflight only:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest /path/to/tvcg_training_manifest.json \
  --config full_v1_candidate \
  --device cuda \
  --grid-size 64 \
  --output runs/phase2_10/preflight.json
```

One-batch CUDA smoke update:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest /path/to/tvcg_training_manifest.json \
  --config full_v1_candidate \
  --device cuda \
  --grid-size 64 \
  --smoke-update \
  --output runs/phase2_10/smoke/checkpoint.pt
```

The second command writes:
- `checkpoint.pt`;
- `checkpoint.report.json`.

## Safety

- The smoke command touches only the first qualified train record.
- It does not iterate epochs or use test labels for selection.
- It never accepts a neural optimization proposal; this is supervised operator training only.
- Do not run against an unreviewed or untrusted manifest.
- Inspect GPU memory before running Full V1 on the 5090; use `reference_v1` for a low-cost factory/loader diagnostic if the full candidate exceeds available memory.

## Tests and status

The existing corpus and normalizer unit tests remain the focused CPU checks. The smoke-update command is the integration test that must be run with Work's admitted manifest on the user's target GPU.

No real-data CUDA smoke result is claimed by this code submission. The user should run the command after the first manifest delivery.
