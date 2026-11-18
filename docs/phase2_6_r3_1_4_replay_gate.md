# R3.1.4 — Native Exact Replay safety and provenance gate

## Local ABI diagnosis

The user directly imported `_make_it_stand` and found the full-rank symbol, proving the extension file exists and exports the entry point. The subsequent import failure reports `GLIBCXX_3.4.32 not found` from the Conda NumPy-bundled libstdc++. This is a dynamic runtime selection/ABI issue, not a missing C++ symbol or failed compilation. Prefer a process-local `LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6)` after verifying that system library exports GLIBCXX_3.4.32; do not replace Conda or system libraries.

## Physical replay gate

The native replay runner now requires a separate physical-admission certificate (`tvcg.r3.physical-admission.v1`) with status `physically_admitted`. Metadata-only admission is insufficient. The certificate must bind case, geometry, TaskSpec, outer/design mesh hashes, pin/support hash, native geometry evidence hash, reviewer/source, and evaluator configuration hash. The request envelope and Faster initialization must match those identities. The evaluator file is hashed again before execution.

This gate is deliberately fail-closed. The current metadata admission report does not meet this certificate schema and cannot authorize a native exact run until native geometry/topology/containment and mask correspondence have passed.

## Evidence boundary

The new tests use controlled contract fixtures only. They prove fail-closed identity checking, not physical feasibility. TRELLIS generation remains the user's separate Work workflow. No real M69 Exact Replay is claimed in this change.
