# M90.3 sparse topology diagnostics and guarded acceptance entry

Isolated CPU follow-up to M90.2. No production, Oracle, dataset/checkpoint,
M90.2 source or earlier report changes. New source code only in this directory.

- `sparse.py`: strict integer canonical RLE, subset-checked difference, sparse
  six-face union-find, bounded triangle-event mesh reconstruction. No dense n^3
  array, compiler, model, optimizer or GPU use.
- `guarded_validator.py`: recommended entry point for subsequent M90.2-schema
  acceptance. Calls the immutable M90.2 validator, then guards any would-be pass
  with conservative sparse surface checks. Does not promote pending inputs or
  extend acceptance to native512, undeclared SI units or mixed representations.
- `test_sparse.py`: 11 deterministic tests, including dense small-grid comparisons,
  strict face-versus-edge/corner adjacency, cavities and disconnected retained
  solids, boundary ambiguity, support errors, missing mesh and resource limits.
- `run_native.py`: predeclared analytic512 and first eligible M83 training record
  diagnostic. All RLE outputs saved/reopened; no optimized designs generated.
- `replay_controls.py`: 33 pinned prior controls through new guard plus a saved
  counterexample proving vertical-boundary ambiguity changes pass to pending.

## Always run under the existing budget wrapper

From `/data/xvxinyao/Spin_test2`:

```bash
python research/NeuralMoments/experiments/m90_2_saved_result_validator/bounded_run.py \
  --out reports/NEW_UNIQUE_M90_3_RUN -- \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/.venv/bin/python \
  research/NeuralMoments/experiments/m90_3_sparse_topology/run_native.py \
  --mode analytic512 --out reports/NEW_UNIQUE_M90_3_RUN/evidence
```

Use `--mode m83` for the pinned first training record only, or use
`guarded_validator.py REGISTRY RESULT --registry-sha256 HASH --result-sha256 HASH
--out NEW_REPORT.json` for the guarded M90.2 schema. Hashes must be registered
independently. Every output folder must be fresh. Caps: two CPU/thread affinity,
120 seconds process-group deadline, 1 GiB address space, 1 GiB output directory,
16 MiB per file. Mesh work preflights <=2,000,000 projected face-ray candidates.
Projected parallel-face checks share that cap; ambiguity returns pending.

## Important boundary correction

A z ray cannot detect centres lying on a vertical face from crossings alone.
This defect was reproduced both in initial M90.3 and immutable M90.2 mesh helper.
The new sparse guard tests z-parallel faces in their dominant-normal projection
and refuses centre-on-surface inputs. Old reports/controls stay unchanged; the
33 prior controls retain their expected statuses. The new counterexample has
legacy status pass but guarded status pending, with claim and TaskSpec complete
pass cleared. Do not use the old unguarded entry for new acceptance claims.

## Scientific meaning

Native512 analytic cube/cavity and one actual training mesh complete within the
resource budget. Actual original/fully retained solid has one six-face component.
This is topology evidence only. Full retention still fails the frozen TaskSpec,
no nontrivial M83 design was attempted, and acceptance remains pending.

New mesh rays use phase (.5,.5,.5). Frozen mesh68 uses
(.500000173,.500000317,.5), legacy stream_decode uses
(.50000017,.50000031,.5). The backend difference is explicit. Matching historical
voxel totals does not establish identical membership or full backend equivalence.

Whole-mesh validity relies on the pinned source registration (or analytic fixture).
Two faces per edge and hit deduplication do not certify absence of self-intersection,
coincident sheets, tangent ambiguity, or exact continuous CAD validity. Arbitrary
unregistered meshes are not newly admitted by these diagnostics.

The selected GSO record explicitly declares source OBJ coordinate units unknown;
its existing task uses registered scalar density and dimensionless thresholds.
No metre/kg physical-scale evidence was invented. Homogeneous density is the
registered mathematical material hypothesis, not a measured physical property.
No cross-representation error threshold is selected from the observed discrepancy.

Next scientific decision: whether the main evaluation is an explicitly declared
source-unit/dimensionless track, with SI physical claims kept separate; and what
preregistered convergence/error-admission protocol permits mixed mesh/voxel
references. A Chat handoff is prepared in the new report directory. No training,
new optimizer budget, manufacturing or dynamic-stability claim is authorized.
