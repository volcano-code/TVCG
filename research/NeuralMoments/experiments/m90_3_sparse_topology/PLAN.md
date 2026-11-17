# M90.3 bounded sparse topology diagnostic

This is the next isolated research step authorized by the user's continuation.
M90.2 acceptance code and every prior report remain immutable. No production,
Oracle, dataset/checkpoint or Git mutation; no GPU, optimizer or training.

Goal: remove the dense full512 bottleneck for independent topology diagnostics,
while retaining pending status for scientific registration/representation gates.
Implementation choice: Python/NumPy sparse interval union-find and triangle-event
rasterization, no C++ builds, dense n^3 mask, model imports or source repairs.
Budget remains 2 threads/CPU cores, 120 s per subprocess, 1 GiB address space and
<=1 GiB total output. At most two million projected face-ray candidates; reject
before expensive allocation. Register the same first eligible M83 training case
as M90.2, never select by topology outcome; no labels or held-out loading.

- [x] Tests: face-only connectivity, strict RLE/difference, disconnected retained,
      cavity-vs-whole distinction, small dense differential checks, mesh/lattice
      validation, bounded mesh reconstruction, budget and original mesh absence.
- [x] Independent sparse interval component counter and subset-checked difference.
- [x] Bounded closed-mesh triangle-event reconstruction on the same task lattice;
      duplicate shared-face intersections deduplicated, ambiguity => pending.
- [x] Native512 analytic cuboid/cavity control; no dense512 allocation.
- [x] Read-only actual training mesh/native512 diagnostic if preflight permits;
      compare whole-solid voxel count and moment discrepancy without an acceptance
      threshold selected from outcomes. Persist exact hashes and failures.
- [x] Independent review, final tests, preservation and append handoff/control docs.

Scientific decision deferred to Chat: explicit dimensionless source-unit track
versus SI-scale physical claims; preregistered discretization/error/convergence
admission. The source policy already declares unknown OBJ physical units and
uses dimensionless TaskSpec thresholds. Never invent metres or relax TaskSpec.
Routine coding is authorized now; no skill permission/commit step overrides the
user's continuation instruction and preservation requirements.

Review ledger: vertical-boundary ambiguity reproduced in review_red, fixed in sparse guard, 11 tests pass. Guarded M90.2 wrapper added without modifying prior code; 33 prior control statuses preserved, new ambiguous boundary downgraded to pending. Native512 runs repeated after the guard fix. Exact-centre versus frozen epsilon-shifted backends remain distinct. No scientific tolerance/units decision or production integration made.
