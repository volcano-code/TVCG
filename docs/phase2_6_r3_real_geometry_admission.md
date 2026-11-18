# R3.1.2 — Reviewed real geometry admission gate

## Scope

This gate joins an already source-backed geometry adapter manifest, canonical TaskSpec record, and pin/support artifact. It requires explicit coordinate frame, length unit, meters-per-unit scale, grid-to-world transform, rotation axis/origin/contact, material policy, shell/design semantics, pin/support semantics, and a human-reviewed source reference.

It checks SHA-256 for geometry manifest, TaskSpec record and pin/support artifact; validates that contact lies on the declared axis; and checks axis/origin/contact against the canonical TaskSpec. It does not generate masks, infer CAD semantics, or alter coordinate frames.

## Output status

A successful report sets metadata_admission=true only. It deliberately keeps physical_mesh_admission=false and native_exact_geometry_checks_passed=false until native topology, containment, mesh/mask correspondence, and exact geometry checks pass.

## Work handoff

TRELLIS generation remains owned by Work. Once its output is ready, create source-backed geometry and pin/support artifacts, a reviewed TaskSpec record, and the admission manifest. This repository provides the validator; it does not generate or replace those assets.

## Tests and remaining gap

Contract tests cover hashes, axis/contact consistency, TaskSpec agreement, and explicit review. Test fixtures are synthetic software fixtures only. They do not count as TRELLIS/M69 physical admissions. R3.1 Exact Replay remains blocked until a real case passes the native geometry checks and carries a source-backed Faster initialization record.
