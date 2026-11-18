# UG-PINO Maximal Trainable Candidate

The maximal currently implemented single-checkpoint operator candidate is the named preset:

`xl_ms_gno_transolver_convex`

It combines, within one `PcraftNativeMomentOperator` and one trainable checkpoint:

- shared-weight P-CRAFT 32/64/128 multiscale geometry context;
- native-interval feature representation and interval residual blocks;
- topology-filtered local GNO branch;
- mass-aware Transolver physical slicing branch;
- convex homogeneous moment-potential readout;
- ten density-weighted raw mass moments;
- coefficient Jacobian obtained by differentiating the same moment function.

This is a unified model with parallel/interactive branches over shared interval features, not a serial composition of separately trained complete networks.

## Important exclusions

- SQNO/QuadrANN remains a separately trained geometry-once/query-many candidate. It is not falsely described as fused into this checkpoint.
- Calibrated uncertainty remains an ensemble/calibration layer; it is not a trained UQ head in this preset.
- Neural Newton remains an outer optimization/controller, not part of the operator checkpoint.
- The convex head is a candidate readout, not proven superior to the bounded odd readout. Both must be compared under matched labels, seeds and budgets.
- This preset has not yet passed a fresh RTX 5090 GPU preflight. CPU integration tests do not establish GPU memory/throughput feasibility.

## Training entry point

Use the existing audited M80 trainer:

`research/NeuralMoments/models/pcraft_ufno/spin_pino/native_train.py`

Commands:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train preflight \
  --manifest "$M72_MANIFEST" --audit "$M72_AUDIT" \
  --out "$OUT/preflight-maximal.json" \
  --preset xl_ms_gno_transolver_convex \
  --device cuda:0 --amp bf16 --batch-queries 16 --chunk 512

python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train train \
  --manifest "$M72_MANIFEST" --audit "$M72_AUDIT" \
  --out "$OUT/train-maximal" \
  --preset xl_ms_gno_transolver_convex \
  --device cuda:0 --amp bf16 --max-steps 300 --max-seconds 7200 \
  --min-queries-per-record 256 \
  --min-train-groups 50 --min-val-groups 7 --min-test-groups 10 \
  --batch-queries 16 --chunk 512 --jacobian-weight 0.05 \
  --val-every 25 --save-every 25 --seed 270201
```

Run the preflight first. The chunk is deliberately reduced from 2048 to 512 for the maximal branch; tune only after measuring peak memory. If GPU preflight fails, lower the chunk and batch size without changing the geometry/query contract. Do not lower the formal corpus coverage threshold to make a formal run appear to pass.

## Today’s acceptance

A bounded 300-step pilot can start today only if:
1. the frozen manifest and audit receipt are present and mutually bound;
2. every qualified record meets the configured 256-query coverage and source-group split gates;
3. maximal preset GPU preflight passes on the user's RTX 5090;
4. forward, Jacobian backward, finite gradients and checkpoint round-trip pass.

If the formal corpus gate is not open, a clearly labelled smoke run may validate the model code path but is not the formal physical training run.

## Promotion plan

The maximal preset is an integration candidate, not an automatically selected winner. Run matched comparisons against:
- `xl_ms_gno_transolver` (bounded odd head);
- `xl_ms_gno`;
- `xl_ms_transolver`;
- `xl_ms`;
- `xl` permanent single-scale control.

Promote the convex head and optional branches only if held-out geometry-group tails and downstream solver outcomes improve at acceptable wall-clock and memory cost. Use at least three seeds before calibration/promotion.


## Existing 32-query corpus pilot

The previously audited `combined70-v2` M72 corpus is known to contain 67 qualified records with source-group split 50/7/10 and 32 queries per object. It is suitable only for an explicitly labelled engineering/architecture pilot, not the formal 256-query release gate. If its local manifest and matching audit receipt still pass the trainer's frozen-manifest checks, use `--min-queries-per-record 32` and record this run as `32q_pilot`. Do not describe it as the final formal training corpus.

Example pilot command after the maximal-preset GPU preflight passes:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train train \
  --manifest "$M72_MANIFEST" --audit "$M72_AUDIT" \
  --out "$OUT/32q_pilot_seed270201" \
  --preset xl_ms_gno_transolver_convex \
  --device cuda:0 --amp bf16 --max-steps 300 --max-seconds 7200 \
  --min-queries-per-record 32 \
  --min-train-groups 50 --min-val-groups 7 --min-test-groups 10 \
  --batch-queries 16 --chunk 512 --jacobian-weight 0.05 \
  --val-every 25 --save-every 25 --val-max-records 7 --val-max-queries 32 \
  --seed 270201
```

This pilot can begin operator training today without waiting for TRELLIS. Its scope is limited to the existing M72 label domain and cannot establish 256-query coverage, real TRELLIS generalization, production-quality UQ or solver acceleration.
