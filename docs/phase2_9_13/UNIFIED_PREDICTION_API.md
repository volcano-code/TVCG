# Phase 2.9.13 — Unified Prediction API

## Implemented in this increment

- Added `SpinMomentPrediction`, a typed public result record.
- Added `predict_full` to compose the existing unified Moment/Jacobian evaluator with configured P-CRAFT UQ and Event heads.
- Added optional query chunking while reusing the same geometry cache.
- Kept Moment UQ, Jacobian UQ, Event logits/probabilities and teacher derivative-validity availability as distinct fields.
- Explicitly reports teacher derivative-validity labels as unavailable at inference; Event prediction is not a substitute for the teacher label or exact solver event checks.
- Added unit tests using a deterministic fake model for output shapes, chunking, raw coefficient Jacobian passthrough, missing-head semantics, invalid inputs and single-query shape preservation.

## Files

- `research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/unified_prediction.py`
- `research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_unified_prediction.py`

## Local test command

From the repository root:

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_prediction -v
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_factory -v
```

## Important limitations

- The tests have been added but were not executed in the remote editing environment. Run the commands above in the project's Python/PyTorch environment before treating them as passing.
- The prediction API composes existing UQ/Event methods. It does not train or calibrate those heads.
- `teacher_derivative_validity_available=False` is intentional: teacher validity is dataset metadata/label, not something inferred from the prediction call.
- Query chunking bounds query-batch evaluation, but does not change the geometry cache's memory footprint.
- This increment does not implement Neural Newton, exact Faster correction, or optimizer acceptance/rejection.
- No real-data training or scientific performance claim is made.
