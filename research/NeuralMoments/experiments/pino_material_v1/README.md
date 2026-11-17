# PINO material v1 (experimental, untrained)

This independent baseline preserves SQNO v1.2, M83 and M69. It reuses the existing
PINOFeatureEncoder topology without editing it; no QuadrANN network, teacher or
SQNO checkpoint is used. Version string: `pino-material-v1`.

## Input and units

`MaterialGrid` contains [X,Y,Z] Cartesian cell fields and physical world bounds:
editable density, editable mask, immutable fixed density, fixed mask, positive
length L and density reference. Densities are physical and nonnegative, not
clamped to [0,1]. Masks are binary, disjoint and constant. Fixed density cannot
require gradients. Empty cells have both masks zero. Fields are piecewise
constant inside their cells. Dimensions must be >=8 and divisible by4.

The four encoder channels are [rho_edit*editable/rho_ref, editable,
rho_fixed*fixed/rho_ref, fixed]. This is a NEW contract using the frozen
encoder's tensor pass-through interface, not its legacy mapping conversion.
The encoder uses float32/complex64; physical integration supports float32/64.
No geometry resampling or inferred occupancy occurs. World origin stays fixed;
L is explicit. The local head predicts a per-cell width in [.02,.20].

Query shape is [Q,10], order [constant,x,y,z,x²,y²,z²,yz,xz,xy]. The last three
coefficients multiply 2yz,2xz,2xy. Moment order uses yz,xz,xy without the factor2.
At X=x/L, normalize raw coefficients c*L^degree before evaluating the polynomial.
Positive gauge scaling preserves retention; negative scaling complements it.
Length rescaling must transform bounds and L by s, coefficients by s^-degree;
moments then transform by s^(3+degree), at fixed physical density.

## Physical guarantees and limitations

Integration uses positive tensor-product Gauss volume weights, physical density
and a bounded C2 quintic retained fraction. Fixed moments are added exactly once.
Thus fixed mass <= total mass <= original mass, to floating-point precision.
Full and empty constant queries are exact. Full degree-two grid moments are exact
for Gauss order>=2. Full retention conserves physical mass and all ten grid
moments; removal intentionally lowers mass. Complement sum is original+fixed.

Partial-cut quadrature is approximate, and smoothing changes the hard-indicator
target. Increasing Gauss order removes neither smoothing bias nor voxel geometry
error. No convex potential, Hessian symmetry or exact hard-cut Oracle is claimed.
`jacobian_input` differentiates raw coefficients, including normalization and
units; `create_graph=True` keeps its neural gradient path. Ordinary forward is
autograd-compatible for editable density, raw queries and PINO parameters.
Density gradients include the bandwidth dependency. Fixed material is immutable.
Saturated queries have zero query/neural gradients; do not train only endpoints.

Single geometry with batched queries. `encode` caches a graph for repeated
queries; re-encode after parameter/density updates or after freeing its backward
graph. No chunked production evaluator, RLE adapter, training pipeline, rotational
loss or new dataset ships in v1. Memory scales with cells*order³*queries. The
validated 8³ fixtures are not a production-capacity benchmark. Uniform cells are
not a lossless replacement for native M83 RLE; do not silently rasterize M83.

## Minimal use

```python
from research.NeuralMoments.experiments.pino_material_v1.model import MaterialGrid, PINOMaterialV1

# Supply all physical fields/bounds explicitly; keep fixed fields constant.
grid = MaterialGrid(rho, editable, fixed_rho, fixed_mask,
                    (-1., -1., -1.), (1., 1., 1.), 2., 3.)
model = PINOMaterialV1(width=2, modes=2, layers=2)
cache = model.encode(grid)
moments = model.evaluate(cache, raw_c)
jacobian = model.jacobian_input(cache, raw_c, create_graph=True)
```

From `/data/xvxinyao/Spin_test2`, use the existing interpreter at
`research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/.venv/bin/python`.
Set `PYTHONPATH=/data/xvxinyao/Spin_test2`, `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1` and run:

```
python -m pytest research/NeuralMoments/experiments/pino_material_v1/test_contracts.py --rootdir=research/NeuralMoments/experiments/pino_material_v1 -p no:cacheprovider -q
python reports/pino_material_v1_20261116/validate.py --device cpu --output /path/to/a/new/result-directory
```

The validator rejects an existing output directory, performs no optimizer step,
and reports derivative, unit and integration errors. Use a shell timeout of90s.
GPU smoke is optional and must check available shared-device memory first.
