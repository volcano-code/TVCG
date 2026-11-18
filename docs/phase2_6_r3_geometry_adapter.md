# R3.0.3 — Explicit geometry adapter

## Scope

The adapter binds a source-backed original triangle mesh, fixed-shell mask, and editable design-domain mask. It performs no mesh repair, voxelization, shell offsetting, cavity inference, coordinate recentering, or mask generation.

## Input manifest

The JSON manifest names:
- NPZ mesh artifact and exact vertex/face array keys;
- explicit fixed-shell mask artifact/key;
- explicit design-domain mask artifact/key;
- registered grid shape and origin;
- explicit unit token;
- SHA-256 for all three source artifacts.

Masks must be boolean arrays and have the registered shape. Shell and design masks must be nonempty and disjoint. Paths resolve relative to the manifest.

## Deliberate non-inferences

TaskSpec fractions (shell_fraction and protect_radius_fraction) are not sufficient to reconstruct per-mesh masks. The adapter does not generate them. Pin/support and Faster beta initialization remain separate required R3 fields and are not synthesized here.

The unit token is required, but SI conversion is not inferred. A token such as source_unit_unscaled explicitly records that physical scale remains unresolved.

## Tests

Unit tests use temporary arrays only to verify schema, hashes, boolean-mask policy, and partition checks. These fixtures are not M69 data and are not counted as physical admissions.

## Exit criteria / remaining gap

R3.0.3 software adapter: implemented.
Real TRELLIS/M69 geometry bundle: pending upstream Work handoff.
Physical geometry admission: zero until real source files and independently reviewed masks are supplied.
