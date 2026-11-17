# M90.2 independent saved-result validator

Isolated CPU research implementation. No model, optimizer or training imports.
Production M69, SQNO, PINO Material and all original data remain unchanged.

`validator.py` implements A0–A6; `geometry.py` implements strict canonical RLE,
explicit editable-minus-retained complement, float64 prism antiderivatives,
independent cell integration, mesh-centre reconstruction and six-face components.
`test_validator.py` provides analytic controls, including actual frozen oracle69
retained export. `run_controls.py` persists each audited fixture and JSON report.
`m83_preflight.py` checks one actual training record without native topology work.
`bounded_run.py` imposes two-CPU affinity/thread environment, no visible GPU,
120-second process-group timeout, 1 GiB address-space cap, 16 MiB per-file cap,
and 1 GiB output-directory watchdog. Run all workloads through this wrapper.

## Reproducible entry point

From `/data/xvxinyao/Spin_test2`, use the existing interpreter:

```bash
python research/NeuralMoments/experiments/m90_2_saved_result_validator/bounded_run.py \
  --out reports/NEW_UNIQUE_M90_2_RUN -- \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/.venv/bin/python \
  research/NeuralMoments/experiments/m90_2_saved_result_validator/run_controls.py \
  --out reports/NEW_UNIQUE_M90_2_RUN/cases
```

Never reuse an output folder. To audit a registration/result pair, replace the
child command with `validator.py registry.json result.json --registry-sha256 HASH
--result-sha256 HASH --out NEW_REPORT.json`. Hash pins must be established by the
registrar independently of result claims; a supplied self-endorsed hash is not
external provenance. The validator checks consistency against those pins, not
the honesty of the authority that registered them. CLI exit 0=pass, 1=fail,
2=pending. `m83_preflight.py` process success means a report was produced; its
report remains pending, not a design acceptance.

## Registration and saved-result contract

See persistent fixtures under the run's `cases/` directory for full schemas.
Registry has original NPZ/hash, original mesh NPZ/hash, complete TaskSpec/hash,
pinned task69.py hash, grid origin/side/width, physical SI units, uniform rho0,
explicit model/checkpoint provenance (null when unused), and original max extent.
All grid, mesh, moments and TaskSpec vectors must already be in the same task
frame, declared `coordinate_space=registered-task`. The rigid world-to-task
transform is provenance only; this adapter does not apply or independently
reproduce that source conversion. World-frame inputs needing transformation
remain pending. A trusted registered task-frame original is required.

Original NPZ fields: occupied/editable/protected boolean masks, density on every
cell, original_moments, fixed_moments. Mesh NPZ: vertices/faces. The original
length is independently measured from occupied prism extents before normalizing
any residual. Masks and references are hash-bound before reading result content.

Saved JSON declares role=deleted, encoding=canonical-integer-rle.v1, integer runs
[x,y,z_start,z_stop), rho0, identity/grid/frame/unit/task/original/registry bindings,
and saved removed/retained/fixed ten moments. No coercion, clipping or repair.
Adjacent intervals must be merged by the producer before registration. Optional
final_density, if present, must exactly equal rho0 on retained original cells and
zero elsewhere. A fractional representation cannot be accepted as a binary one.

`oracle69.export_rle` is RETAINED EDITABLE. Convert with `complement_retained`;
record source_role=retained_editable, operation=editable-minus-retained and a
hash-bound retained JSON file. The validator reopens that source and independently
checks the complement. Never pass raw retained RLE to audit69.audit_saved.
RLE-only results with no material-field claim represent homogeneous binary
material by definition; continuous model fields are not implicitly certified.

All ten moments use float64, tolerance abs(delta)/(M0*L^degree)<=1e-10, separate
from unchanged TaskSpec tolerances. Direct occupied-cell, removed-prism and
retained-cell computations cross-check immutable original minus deleted and
fixed plus editable retained exactly once. No original-mesh/voxel equality is
claimed: only the exact-voxel-prisms.v1 adapter can pass this version.

Whole-solid topology revoxelizes the supplied outer triangle mesh by float64
odd/even z-ray intersections at registered centres, checks equality with original
occupancy, then labels the entire retained mask including fixed shell with six
face neighbours. Missing/open/ambiguous meshes and surface-centre ambiguity are
pending. Mesh not fully covered by this lattice requires an extended adapter.
No cavity-only or caller-supplied connectivity boolean can certify success.
The mesh operation is a represented-centre reconstruction, not robust exact CAD
intersection, self-intersection certification or an original-source pose audit.

## Scope and limits

Only <=64^3 analytic/native-grid fixtures and <=5,000,000 face-ray tests are
supported by this bounded dense adapter. Preflight rejects larger work as pending;
no full512 topology compile/voxelization was attempted. M83 scalar density and
original mesh are available, but full original occupancy/material metadata,
explicit SI registration and mixed-reference discretization admission remain
pending. No benchmark case is newly admitted and no optimized design is claimed.

A pass means represented-domain rotation/material acceptance only. It does not
certify manufacturing, wall robustness, cavity accessibility, exact continuous
CAD, dynamic stability, optimizer convergence or population generalization.
Fixtures use explicitly registered engineering TaskSpecs (resolution 8); existing
research TaskSpecs are never reduced or relaxed. The intact positive control has
its own declared zero-minimum-removal task; full retention under the unchanged
positive-removal task correctly fails.

Evidence: `/data/xvxinyao/Spin_test2/reports/m90_2_saved_result_validator_20261116/`.
Failed attempts and review regression RED runs are retained there. Only the
isolated validator and frozen task69 CPU suite were tested, not a GPU/training
project-wide suite. Phase5/M90 prior diagnostics remain historical evidence.
