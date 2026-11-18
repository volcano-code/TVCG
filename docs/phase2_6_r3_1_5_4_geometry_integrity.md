# R3.1.5.4 — Independent geometry integrity audit

## What is computed

The independent geometry audit validates the combinatorial tetrahedral complex: finite vertices, valid indices, four distinct vertices per cell, duplicate-cell rejection, strictly positive signed tetrahedron volumes, face incidence no greater than two, opposite orientation across interior faces, closed boundary edge incidence, and connected-component counts. It records boundary/interior face counts and signed volume.

## What is not inferred

These checks do not prove tetrahedron interiors are non-overlapping and do not prove equivalence to a native triangle boundary. Equal low-order moments are not a geometry equivalence proof. Therefore the manifest requires hash-bound assets for native final boundary, source outer mesh, fixed shell, design domain, cavity, and independent tetrahedralization, plus a separate reviewed mapping certificate binding all those hashes, case/task/geometry identities and coordinate frame. The certificate must explicitly say `reviewed_equivalent` and name reviewer, source and method. This is a provenance gate, not an automatic geometric intersection proof.

The audit report states this limitation explicitly. A real certificate must be backed by an actual geometric correspondence procedure (e.g. boundary comparison plus volume coverage/intersection checks) performed on the exact assets. CI fixtures only test contract and topology logic.

## Run

    PYTHONPATH=research/NeuralMoments python research/NeuralMoments/r3_independent_geometry_integrity_v1.py --manifest reports/r3/geometry-integrity-manifest.json --output reports/r3/geometry-integrity-report.json

All input paths are relative to the manifest directory and every asset is re-hashed before use.

## Exit criteria

R3.1.5.4 software gate is implemented. Real geometry integrity remains pending until Work supplies the exact native final boundary, outer/shell/design/cavity assets, a corresponding independent tetrahedralization, and a reviewed mapping certificate with evidence. No M69 physical admission or CCF-A geometry-equivalence claim is made by CI.
