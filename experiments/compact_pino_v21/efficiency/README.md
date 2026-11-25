# Compact PINO V2.2 — operator execution efficiency

This release implements the five execution improvements agreed after the V2.2
integration release (`879233cdc4b843661c569022a9fe9d1ecd4f1be4`). It preserves
the V2.1 conservative physical response, native256 material, shared 32³/64³
P-CRAFT/U-FNO, Token16 default, and all learned parameter names and shapes.
It does not change the physical bandwidth, teacher, sampling distribution,
objective, derivative eligibility, or the Pilot promotion thresholds.

## Implemented changes

| Change | What is reused or simplified | Invalidation / limits |
|---|---|---|
| Static encoder inputs | Token moments/statistics/coordinates and 32³/64³ material channels | Material content/mapping, coordinates, density, fixed material, token/grid configuration and implementation version |
| Development encoding reuse | One learned geometry encoding per matching geometry/task within a fixed-weight evaluation pass | A bounded inference-only cache is destroyed at the end of every pass; never used across optimizer updates |
| Token-context projection | Compute the first local layer's context contribution once per token, then add the per-point coordinate contribution | A learning graph exists only for the current forward/backward; owner and parameter-version checks reject stale projections |
| Analytic local slope | Compute the derivative of the same cubic retention formula and contract with current native vertex motion | Keep the parameter/context graph; custom response classes retain an automatic-differentiation fallback |
| Frozen-query preparation | Reuse the finite surface/BVH, whole-block classification and original-order boundary stencils | Exact material/cut/side/bandwidth/chunk identity, bounded LRU, oversize bypass; differentiable vertex tensors bypass preparation caching |

The learned response is still one field. There is no independent Moment or
Jacobian head. The first-layer projection is an algebraic rearrangement of
existing weights. The analytic slope retains differentiation with respect to
the local network and the geometry encoder, so Jacobian supervision continues
to update both.

FP32 projection changes floating-point addition order. Equivalence to the
reference is checked with explicit numerical tolerances, not a claim of
bitwise equality to the previous implementation. Exact checkpoint continuation
is tested within this new implementation separately.

## Reference and optimized modes

```python
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21

reference = CompactPINOv21(token_grid=16, efficiency=False)
optimized = CompactPINOv21(token_grid=16, efficiency=True)
optimized.load_state_dict(reference.state_dict(), strict=True)
```

`efficiency=False` uses the unprojected local network, its original local AD
slope, and uncached static/query preparation. It remains available for paired
comparisons and diagnosing a regression. Static construction follows the same
numerical reduction order in both modes. The reference uses this commit's
material-identity checks and input extraction: its timing is not a literal
replay of the previous commit's executable instructions.

Default retained cache budgets per model/session:

- Static NumPy encoder inputs: 128 MiB, also limited to 128 entries.
- Frozen query preparation: 128 MiB, including conservative buffer/container
  accounting, plus at most 256 small oversize-query keys.
- Evaluation learned features: 64 MiB of retained tensor storage, also limited
  to 128 entries. Tensors live on the model device.

These budgets exclude model parameters, material memmaps, temporary encoding
activations, a newly prepared candidate awaiting insertion/eviction, and one
normal streaming preparation chunk. They are cache limits,
not a bound on total process or GPU memory. Oversized entries use the uncached
path. A first oversized query may build a bounded preparation prefix before
falling back; the negative cache prevents repeating that prefix while its key
is retained.

```python
model = CompactPINOv21(
    token_grid=16,
    efficiency=True,
    static_cache_bytes=128 * 1024**2,
    query_cache_bytes=128 * 1024**2,
)
print(model.efficiency_statistics())
model.clear_efficiency_caches()
```

`GeometryFeatures` binds the material signature as well as model owner and
parameter versions. A material edit after encoding requires a new encoding.
Cache checks are not teacher admission: real inputs still pass the existing
frozen-source and SHA256-qualified data handoff.

For a Newton trajectory that produces a new operand at every call, set
`query_cache_bytes=0` to stream preparation without retaining one-use stencils.
Static input caching, token projection and analytic slopes remain enabled.
The same fixed-weight geometry encoding can still be reused through the
existing `encode_geometry(..., inference=True)` / `evaluate_operand` interface.
Frozen-query benchmark speed ratios do not describe this changing-operand use.

## Training configuration and checkpoints

The existing runner now records these additional `TrainConfig` fields:

```json
{
  "efficiency": true,
  "static_cache_bytes": 134217728,
  "query_cache_bytes": 134217728,
  "evaluation_cache_bytes": 67108864
}
```

No sampler, loss weight, learning rate, exposure budget, or selection rule is
implicitly changed. Caches contain no labels and no cached learned outputs
across training updates. New cache modules are included in execution-source
fingerprints. Cache contents are never saved in a checkpoint.

Existing operator `state_dict` weights remain strictly loadable. Exact resume
across changed execution source/configuration is intentionally rejected by
the existing runner. To compare new runs, initialize both arms from the same
weights and freeze their new configuration identities. The runner's
`--initial-weights` interface accepts its own checkpoint format; this release
does not claim compatibility with an unseen private Work checkpoint wrapper.

## Verification and performance evidence

The controlled benchmark uses the actual default Token16 encoder, the same
initial state in both arms, the same finite operands and the same material.
It checks all ten Moment values, qualified Jacobians and every parameter's
joint-loss gradient before interpreting timing.

Three timing scopes are kept separate:

1. A single M+J query with input/query caches empty, after runtime/kernel warmup.
2. A repeated frozen-query evaluation pass, with learned feature caches empty
   at each pass and static/query preparation reusable.
3. A batch-one Moment+Jacobian forward/loss/backward, with a fresh learning
   graph on every iteration. This timing does not include an optimizer update.

The first scope is **not** a cold process or a raw-mesh-to-qualified-solution
measurement. Repeated frozen queries are **not** a changing Newton trajectory.
Small-sample P95 values are descriptive timings, not project-level P95 evidence.
CPU runs report retained cache bytes; CUDA runs additionally synchronize timing
and record allocated GPU memory. No CPU measurement is presented as a 5090 D
result.

Run the integrated tests from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  experiments/compact_pino_v21/efficiency \
  experiments/compact_pino_v21/training \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v21_response.py \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v22_response_efficiency.py \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v22_geometry_inputs.py \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v22_query_cache.py \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v22_compact_integration.py
```

Recheck actual updates and exact continuation with a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m \
  experiments.compact_pino_v21.training.smoke \
  --output /abs/new_v22_efficiency_resume_check
```

For Work-side performance comparisons, use the benchmark CLI in this directory
with the frozen admitted manifest and the target device. Keep the previous
Pilot checkout/results intact; do not splice new execution code into an active
old run. GPU block sizes/cache budgets and actual physical acceptance remain
Work-side measurements. This software release does not promote the previous
NO-GO Pilot or authorize long training on missing physical evidence.
