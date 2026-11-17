# Rotation/material v1

Additive experimental functions around unchanged PINO material v1. No new teacher,
core network, trained weights or M83 replacement. Model weights remain untrained.

`bounded_material(base, logits, smoothing=True)` maps sigmoid logits to physical
density `rho_base * beta` inside editable support. A positive3³ averaging kernel
is normalized over editable neighbors, preserving constant fractions at edges.
Noneditable logits have exactly zero gradient; fixed fields remain unchanged.
`base` is the immutable original density upper bound, not an optimization variable.
The material fraction and the model's query retention are separate factors. Do
not multiply beta twice or recompute the original reference from reduced density.

Mechanism provenance: [official DensiCrafter obj.py](https://github.com/idvxlab/DensiCrafter/blob/21e6d8dc646129a9b21a38c2d5bee5bef2704d98/obj.py)
and [loss.py](https://github.com/idvxlab/DensiCrafter/blob/21e6d8dc646129a9b21a38c2d5bee5bef2704d98/loss.py), MIT2025 Shengqi Dang(SII).
Read-only snapshots and license are in the Phase5 report. This is an independent
adaptation of sigmoid design variables, smoothing and spatial restrictions. It
does not reproduce upstream basal-region growth, contact-area maximization,
2000-step optimizer or TRELLIS pipeline. No upstream code is imported/executed.

`rotation_objective(moments, original, task, length)` exposes per-query losses,
all10 physical task69 moment constraint values and metrics. `task` is frozen
`task69.TaskSpec`, with arbitrary axis/origin/contact. `original` is constant
full-material [10] moments; current moments are [Q,10]. The function rejects
nonpositive current/original mass and differentiable original references.

Losses are dimensionless squared hinges for COM alignment, centered-inertia axis
alignment, original-axis polar retention, principal dominance, centroid rise,
minimum removal, no-added mass/inertia and second-moment feasibility. `mass` is
the retained/original mass ratio. Caller selects term weights; none are tuned
here. Training polar loss includes TaskSpec's training margin, reported physical
constraints do not. Inactive penalties correctly have zero gradients. At
repeated eigenvalues principal/eigenvalue losses may be nondifferentiable;
finite-difference tests use nondegenerate violating fixtures.

These functions do NOT certify feasibility. Variable-density material is labeled
`bounded_variable_density_relaxation.v1`; M69's single-density removal-only policy
is unchanged. Connectivity, binary conversion, printability, shape accuracy and
rotational dynamics require separate audits. Smoothing cannot certify connectivity.
PINO's smooth-cut quadrature still differs from hard-cut reference labels.

## Example

```python
from task69 import TaskSpec
from research.NeuralMoments.experiments.rotation_material_v1.material import bounded_material
from research.NeuralMoments.experiments.rotation_material_v1.physics import rotation_objective

original = model.encode(base).full.detach().clone()  # once, before design changes
design = bounded_material(base, logits)
moments = model(design, raw_coefficients)
result = rotation_objective(moments, original, TaskSpec(), base.length)
loss = result['losses']['com'].mean() + result['losses']['polar'].mean()
loss.backward()  # no optimizer is automatically run
```

The caller must retain the same original reference throughout a design run.
Change TaskSpec only by explicit task/version choice. No `complete_pass` flag is
returned; use the frozen independent audit for applicable material policies.

## Data and reproduction

`data/synthetic_density_v1/` has12 deterministic8³ records and60 analytic hard-cut
moment labels:8 train,2 validation,2 held_out. Labels use independent cuboid
integrals, not PINO outputs. Full original and current full moments are distinct.
This is a diagnostic schema/label corpus, not a representative dataset or test
of GSO/Procedural generalization. No split has been evaluated by the model.

From `/data/xvxinyao/Spin_test2`, use the existing QuadrANN `.venv/bin/python`, with
`PYTHONPATH=/data/xvxinyao/Spin_test2:/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src`,
`CUDA_VISIBLE_DEVICES=''`,2 CPU threads and `PYTHONDONTWRITEBYTECODE=1`:

```
python -m pytest research/NeuralMoments/experiments --rootdir=research/NeuralMoments/experiments -p no:cacheprovider -q
python reports/rotation_material_v1_20261116/validate.py --device cpu --output /path/to/new/results
python -m research.NeuralMoments.experiments.rotation_material_v1.corpus --output /path/to/new/data
```

Outputs must be fresh directories. Validator temporarily perturbs/restores only
its new in-memory model for finite differences, verifies state restoration and
runs no optimizer. GPU mode uses10% allocator cap; check shared availability and
use a90-second external timeout. TF32 is disabled for Phase5 derivative checks.
