# Phase 2.8.6 — Architecture Release Gate

## Purpose

Phase 2.8.6 consolidates software-level verification for the architecture
developed in Phases 2.8.3–2.8.5. It is not a trained-model scientific release
and does not freeze the final research architecture.

## Implemented gate

Run from the repository root:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.release_gate.run_release_gate \
  --profile full --output runs/phase2_8_6_release
```

The runner records the Git commit, Python/platform/package/CUDA environment,
architecture feature inventory, compile results, native P-CRAFT test results,
and SQNO Adapter test results. It writes JSON and Markdown reports.

The architecture inventory explicitly tracks:
- global scalar and query-conditioned fusion;
- Geometry-Physics Cross Attention;
- Cross-scale Attention;
- GNO interaction variants;
- adaptive physical slicing;
- the paired fusion ablation harness.

The full profile executes:
1. compileall for native P-CRAFT;
2. compileall for SQNO Adapter;
3. the complete native P-CRAFT pytest suite;
4. the complete SQNO Adapter pytest suite.

Existing test suites provide detailed contracts for raw-coefficient Jacobians,
SO(3)/scale invariance, permutation properties, mass conservation, cache
invalidation, checkpoint migration, optimizer uniqueness, and ablation pairing.
The runner does not claim these subcontracts independently unless their
own tests execute as part of the two complete suites.

## Scientific gates intentionally not passed

The following remain BLOCKED pending Work's dataset construction and
subsequent experiments:
- admission and provenance audit of the additional 100 TRELLIS candidates;
- validation geometries from both Spin-It and Spin-It Faster plus additional
  TRELLIS artifacts;
- a geometry/source-family-isolated locked test set;
- formal training, multi-seed stability and OOD evaluation;
- 64^3 / 128^3 / 256^3 / 512^3 trained-model evaluation;
- matched RTX 5090 D latency and peak-memory profiling;
- full hybrid optimization with independent physical verification.

Data policy:
- TRELLIS Mini7 is a pilot only;
- the additional 100 TRELLIS models are formal training candidates pending
  processing, provenance, quality and leakage audit;
- validation includes eligible models from both papers and additional
  TRELLIS artifacts, not only Heart/Noodles;
- split by geometry and source family, never query rows;
- preserve a separate locked test set.

## Interpretation

A software gate PASS means the commands recorded in the report executed and
passed on that environment and commit. It does not establish Moment accuracy
on formal data, OOD generalization, speedup, memory advantage, or CCF-A
readiness. Do not start long training from this report alone.
