# Phase 2.10 — Unified Training V13

## Delivered

- `training_v13/unified_corpus.py`: strict SHA-256 manifest-backed RLE + labels reader; source-group split isolation; native P-CRAFT geometry conversion; full and restricted derivative masks preserved.
- `training_v13/train_unified.py`: Unified M80+SQNO trainer using the existing `model.train_step`, Fixed or PCGrad gradient mode, train-only RMS normalization, per-geometry fresh geometry cache, independent validation metrics, cosine schedule, strict model checkpoint/resume.
- Tests for corpus schema/partial derivative supervision and train-only normalizer behavior.

## Training semantics

One optimizer update is performed per geometry record per epoch. The geometry cache is rebuilt before each update and discarded after the optimizer step. All query rows for that geometry are passed together to the existing unified loss implementation, which applies the valid Moment mask and disjoint full/restricted derivative supervision masks.

Normalization is fitted only on valid Moment labels in the train split. No validation/test target contributes to scales.

Validation reports:
- normalized Moment RMSE;
- full Jacobian RMSE on full-valid rows;
- restricted Jacobian RMSE after projecting `J @ basis` over each row's active dimension.

The validation loop uses an inference geometry cache. It does not compute gradients for optimizer updates.

## Required manifest and NPZ

See `docs/phase2_10/UNIFIED_OPERATOR_CORPUS_V1.md`. Manifest schema:
`tvcg.unified-operator-corpus.v1`.

All three splits (train/val/test) must be nonempty and source-group-disjoint. The trainer uses train for updates and val for model selection; test is never used for model selection.

## Local tests

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.tests.test_unified_corpus -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.tests.test_unified_trainer -v
```

## Pilot command (only after Work delivers an admitted manifest)

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.train_unified \
  --manifest /path/to/tvcg_training_manifest.json \
  --output runs/phase2_10_unified_pilot \
  --config full_v1_candidate \
  --device cuda \
  --grid-size 64 \
  --epochs 10 \
  --gradient-mode fixed
```

This is an initial pilot command, not a long-training recipe. Confirm manifest admission, model construction, memory and first-batch CUDA update before running.

## Resume

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.train_unified \
  --manifest /path/to/tvcg_training_manifest.json \
  --output runs/phase2_10_unified_pilot \
  --config full_v1_candidate \
  --device cuda \
  --epochs 10 \
  --resume runs/phase2_10_unified_pilot/checkpoints/latest.pt
```

Resume validates model architecture, training contract, data-manifest fingerprint, optimizer and scheduler before restoring state.

## Explicit limitations

- The Python tests were added but not executed in this GitHub editing environment.
- No real corpus has yet been supplied to this loader in this increment.
- No CUDA update, convergence result, model selection or long training is claimed.
- The loader consumes the current native RLE geometry contract. Work must export this manifest/NPZ schema or provide a reviewed conversion.
- Event/UQ auxiliary heads are not trained by the current `UnifiedM80SQNO.train_step` objective. They remain auxiliary inference heads until their teacher-label losses are explicitly integrated. This is intentional; no fake Event/UQ labels are synthesized.
- Optimizer-level Faster integration remains a separate physics task and is not required for supervised Moment/Jacobian pilot training.
