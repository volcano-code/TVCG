# R3.1.5.2 — Native vs independent raw-moment comparison

The comparison layer requires the native replay ledger to carry explicit `moment_metadata`: case_id, geometry_hash, coordinate_frame, length_unit, meters_per_unit, density, and canonical moment order. The request builder now requires this metadata and verifies density agrees exactly with the native request's material_density. The native replay runner persists the metadata into the ledger.

The independent tetrahedral backend reports the same metadata plus a distinct backend identity and raw moments. The comparator rejects case/geometry/frame/unit/density/order mismatches and never applies an implicit coordinate or density conversion.

Component-wise comparison uses `abs(native-independent) <= atol + rtol * max(abs(native), abs(independent))`. This avoids the prior incorrect global denominator `max(1, abs(moment))`, which can hide substantial relative errors in small SI second-moment components. The report includes absolute/relative errors and pass/fail per component.

Tests use controlled numerical fixtures only. They establish comparison semantics, not a real TRELLIS/M69 match. A real comparison still requires an independently tetrahedralized copy of the exact retained material volume and matching source geometry hash.
