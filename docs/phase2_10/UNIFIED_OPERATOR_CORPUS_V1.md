# Phase 2.10 — Unified Operator Corpus V1

## Purpose

This is the Chat-side ingestion contract for the Work-generated Faster moment corpus. It adapts audited native RLE geometry and query labels into the existing Unified M80+SQNO training batch contract.

It does not run training, relabel data, repair geometry, or promote external candidate materials to Spin-It-certified initial gyroscopes.

## Manifest

JSON manifest schema:

`tvcg.unified-operator-corpus.v1`

Each qualified record contains:
- `id`
- `group_id` (source-family identity; cannot cross splits)
- `split` (`train`, `val`, `test`)
- `status` (`qualified`)
- `source_type`
- `geometry_file`, `geometry_sha256`
- `labels_file`, `labels_sha256`

The loader rejects duplicate IDs, source-group leakage, unsafe paths, hash mismatches and empty required splits.

## Geometry NPZ

Required:
- `columns`: native RLE [N,4], x,y,z_start,z_end, end-exclusive;
- `origin`: [3];
- `h`: scalar voxel width;
- `fixed_moments`: [10];
- `original_moments`: [10];
- `side` or `native_resolution`: scalar lattice side;
- `density`: optional, defaults to 1.

The existing `geometry_from_columns` performs the native RLE checks and verifies fixed plus editable moments against original moments. The loader then creates the conservative P-CRAFT context grid at the configured `grid_size`; native RLE remains the authoritative fine geometry.

## Labels NPZ

Required:
- `queries`: [Q,10], raw quadric coefficient queries;
- `moments`: [Q,10], ten raw moments.

Optional:
- `valid_mask`: bool [Q], defaults to all true;
- `jacobian`: [Q,10,10];
- `derivative_valid`: bool [Q];
- `restricted_basis`: [Q,10,D];
- `restricted_jacobian`: [Q,10,D];
- `restricted_valid`: bool [Q];
- `restricted_dimension`: integer-like [Q];
- `query_status`, `event_state`: audit metadata.

Full and restricted derivative masks must be disjoint. Missing derivative rows remain masked; NaN padding is permitted only in inactive rows or inactive restricted columns, matching `masked_derivative_supervision.py`. The loader does not synthesize full Jacobians from restricted labels.

## API

```python
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.unified_corpus import UnifiedOperatorCorpus

corpus = UnifiedOperatorCorpus(
    "/path/to/manifest.json",
    grid_size=64,
)

print(corpus.split_summary())

for row, batch in corpus.iter_split("train"):
    # batch keys are accepted by unified_training_ablation:
    # geometry, coefficients, target_moments,
    # optional target_jacobian and derivative masks/bases.
    ...
```

## Local tests

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.tests.test_unified_corpus -v
```

## Scope and limitations

- This loader is for the current native RLE + Faster moment label contract.
- It does not imply the Work dataset already conforms; Work must export the manifest/NPZ fields above or provide a reviewed adapter.
- It does not establish 512^3 SDF availability.
- It does not establish Spin-It initial-state sealing or Faster optimization trajectory completeness.
- It does not start or certify model training.
