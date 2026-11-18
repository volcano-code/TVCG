# R3.1.3 — Native request assembly

## Native build preflight correction

The local user report shows the compiled extension is importable as top-level `_make_it_stand`, while the preflight previously attempted only `make_it_stand._make_it_stand`. The preflight now tries both layouts by default, matching the existing Python facade fallback. The direct import and symbol test from the user is positive evidence that the native module itself compiled and exports the full-rank entry point.

## Request builder

r3_native_request_builder_v1.py assembles the native request envelope from explicitly named source NPZ outer/design triangle meshes, a hash-pinned TaskSpec JSON record, and reviewed request fields. It validates triangle indices, source hashes, mode, equation profile, payload capture, density, units and target-frame dimensions. It performs no geometry transformation, voxelization, mask inference, or solver execution.

The resulting envelope is accepted by r3_native_solver_replay_v1.py. The replay runner unwraps the request while retaining the envelope hash as provenance.

## Required source fields

Both outer_surface and design_domain must be source-backed triangle meshes in the exact solver coordinate frame. The fixed-shell ten raw moments, target frame, material density, unit scale, spinning-top objective and complete SpinSolverConfig must be explicit. Geometry/task hashes must bind to the admitted case. No missing values are silently defaulted by this builder.

## Evidence boundary

Passing request assembly tests proves serialization and schema handling only. It does not admit a physical mesh, prove mesh containment, establish pin/support equivalence, or execute the native solver. The current local preflight issue is an import-layout mismatch, not a failed C++ build.
