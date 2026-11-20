# Phase 2.10 — Work corpus intake audit and V13 adapter

## Work delivery reviewed

Snapshot: 2026-11-20T20:34:42+08:00

The supplied `Phase2_10_Chat_Audit.zip` reports:
- 107 intake objects; 100 newly encoded; 4,771 audited query rows packaged.
- 31 complete training geometries / 2,019 train queries.
- 76 partial geometries / 2,752 queries remain in `train_partial`.
- 5 fixed paper-reconstruction benchmarks / 320 queries.
- 11 development geometries / 11 source groups / 25 queries.
- development labels: 11 plane queries, 6 restricted plane Jacobians, 14 ellipsoid Moment-only queries.
- 7 new Objaverse development groups; 4 signed-plane-only records from the original TRELLIS-16 candidates.
- the other 12 TRELLIS-16 candidates are not admitted under the current single-shell teacher.
- all 8 data-quality/split gates except native Spin-It initial-state export and independent reload pass in the supplied `pilot_gates.json`.

The two sample ZIPs each contain one geometry only:
- `train_sample.zip`: N064, 64 train queries;
- `development_sample.zip`: oa_2ca48fc7851b4956b22a3c925409b544, 3 development queries.

They are interface fixtures, not complete train/validation/test corpora. The audit ZIP contains the full manifest and reports but not the full referenced geometry/label asset tree; full assets remain at the absolute paths recorded in the manifest on the user's data machine.

## Important source-derived caveats

1. `PILOT_READY.md` says `pilot_ready=false`: native initial state export + reload for a real geometry is the one failed P6 gate.
2. `initial_state_evidence.json` reports one independently verified `external_candidate` (Flowerpot v2), but `native_sealed_real_geometry_count=0`. Do not relabel this as native Spin-It output.
3. `teacher_contract.json` has `qualified=false`, `qualification=not yet measured`, `physical_task=null`, and describes a local libigl/CGAL Faster-formula reproduction, not the author-released implementation. It records an independent replay/audit path, but does not establish author-equivalence or formal teacher qualification.
4. Resolution contract: RLE1024 is conditioning occupancy; SDF128 is auxiliary distance input; source mesh has no native voxel pitch; teacher integration is a discrete Boolean surface evaluator with no voxel pitch. Do not call SDF128 a 512/1024 physical representation.
5. Development set is suitable for interface checks and limited query comparisons, not generalization claims. Its 25 queries are heterogeneous: plane labels, restricted plane derivatives and ellipsoid Moment-only labels; TRELLIS signed-plane records have no derivative labels.

## V13 adapter changes

`training_v13/unified_corpus.py` now accepts both:
- internal `tvcg.unified-operator-corpus.v1`;
- Work `phase2.10.corpus.v1`.

Work split mapping:
- `train` -> train;
- `development` -> validation;
- `fixed_benchmark` -> locked benchmark/test report partition;
- `train_partial` is not admitted to this snapshot's training run.

The loader maps `geometry_id` / `source_group_id`, accepts Work's relative sample paths, and accepts absolute asset paths only with explicit `allow_absolute_paths=True`. It checks file hashes, source-group separation, RLE native resolution, fixed moments, label query count and mesh-teacher full moments.

Crucially, Work RLE NPZ omits `original_moments`. The loader derives the RLE conditioning geometry's own moments from RLE cells. It does **not** equate these conditioning-grid moments with the continuous mesh teacher's `original_moments/full_moments`; the latter are cross-checked against each other in the labels/manifest. This preserves the reported mesh-vs-raster distinction.

The loader accepts Work's extra provenance/audit arrays without feeding them to the supervised loss. It maps:
- `query_status == accepted_moment` (or `moment_valid`) to Moment validity;
- `derivative_valid` to full Jacobian mask;
- `restricted_valid`, `restricted_dimension`, `restricted_basis`, `restricted_jacobian` to restricted derivative supervision;
- `event_valid` and `restricted_event_valid` remain separate metadata, not fabricated training labels for the current loss.

## Full-corpus preflight command

On the user's data machine, after pulling this commit and confirming the full manifest's absolute asset paths exist:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest /data/xvxinyao/TVCG/datasets/phase2_10_corpus_v1/manifest.json \
  --config full_v1_candidate \
  --device cuda \
  --grid-size 64 \
  --allow-absolute-assets \
  --output runs/phase2_10/preflight_full.json
```

Expected normalized split counts:
- train: 31 geometries / 2,019 queries;
- val: 11 geometries / 25 queries;
- test: 5 fixed benchmarks / 320 queries.

The train sample ZIP and development sample ZIP can be checked separately with the included Work `corpus_loader.py`, or with `UnifiedOperatorCorpus(..., required_splits=("train",))` and `required_splits=("val",)` respectively. They cannot individually start the full trainer because they do not contain all required splits.

## Next gate

Do not start long training yet. First run the full-corpus V13 preflight on the data machine, then run a one-batch CUDA smoke update. This checks whether the new adapter, Full V1 factory, P-CRAFT geometry encoder, joint Moment/Jacobian loss, optimizer update, cache invalidation and checkpoint restore all work with the real Work package.

A successful supervised pilot would establish that the model can learn the frozen local teacher's geometric moment-query function on this corpus. It would not yet establish native Spin-It initial-gyroscope closure, author-equivalent Faster teacher status, or end-to-end optimizer acceleration.
