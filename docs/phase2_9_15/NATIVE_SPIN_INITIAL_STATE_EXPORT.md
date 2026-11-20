# Phase 2.9.15 — Native Spin-It Initial Material-State Export

## Implemented

The existing `api::SpinIt2014Artifact` diagnostics export now includes a separate `initial_state` object with schema:

`spin-it-2014-initial-material-state-v1`

It records the *input source domain* rather than confusing it with the optimized final material:

- source voxel-domain artifact identity;
- grid dimensions, bounds, spacing and canonical x-fast cell order;
- source material mask;
- source carveable mask;
- derived fixed-material mask (`material && !carveable`);
- derived design-domain mask (`material && carveable`);
- initial geometric raw ten moments in model coordinates;
- the same moments transformed into the registered target frame;
- model-to-target linear and translation transforms;
- target origin and axes;
- explicit unit-density geometric-volume semantics;
- explicit declaration that SI physical scale is not registered;
- explicit declaration that a quadric anchor is not part of this Spin-It export.

The source moments are integrated analytically per occupied axis-aligned voxel box. For a cell with center c, spacing h and volume V:
- mass = V;
- first moment = V c;
- second raw moment = V (c c^T + diag(h^2/12)).

This produces geometric volume moments at unit density in the source coordinate units. It must not be described as kilograms or SI mass without a separately registered scale/density.

## Why this export is separate from M69/Faster

The Spin-It 2014 voxel initial-state export and the M69 fixed-axis TaskSpec / Faster quadric-query task are distinct contracts. This change does not invent a quadric anchor, does not infer an M69 TaskSpec, and does not claim their objectives or variables are equivalent.

The existing reconstructed spinning-top fixture remains marked `RECONSTRUCTED_ASSET`; this export does not upgrade its evidence class.

## Files changed

- `src/api/SpinIt2014.cpp`
- `tests/spin_it_2014_api_regression.cpp`

## Verification

The existing C++ API regression now asserts that the diagnostics include the initial-state schema, source/fixed/design masks, model/target initial moments and the non-SI declaration.

This GitHub editing session cannot compile the C++ target or run CTest. Run the repository's existing build and focused test locally before treating the native exporter as runtime-verified.

## Local verification

Use the existing configured build directory (do not overwrite a user's active build configuration):

```bash
cmake --build build --target spin_it_2014_api_regression -j2
ctest --test-dir build -R '^spin_it_2014_api_regression$' --output-on-failure
```

If the build uses a different target name, inspect `ctest -N` and the generated build system rather than guessing.

## Remaining work

1. Add a canonical JSON serializer/hash for the nested initial-state payload if downstream manifests require a standalone payload digest.
2. Add a Python reader/validator for this exact schema and verify mask counts, coordinate transforms and moment recomputation independently.
3. Add an explicit Faster-specific initial-state adapter. Do not route this voxel Spin-It export through M69 unless the objective/state mapping is mathematically justified.
4. Run the C++ regression and independent Python moment audit on a real OBJ-derived object after Work's data pipeline produces one.
