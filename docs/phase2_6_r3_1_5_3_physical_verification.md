# R3.1.5.3 — Independent spinning-top physical verification

## Native moment semantics audited

`physics::RawMassMoments10` defines density-weighted raw mass moments in this exact order: (M,LX,LY,LZ,QXX,QYY,QZZ,QYZ,QXZ,QXY). Component 0 is mass, not geometric volume. All ten components include material density and are expressed in the evaluator target frame. The independent tetra backend and comparator now use this same order. Geometric volume is separately recorded in m^3 so mass-density-volume consistency can be checked without confusing mass with volume.

## Original spinning-top constraints

`SpinMomentProblem::evaluate` for `SpinningTop` defines four equality constraints in its target frame: `LX=0`, `LY=0`, central inertia `I_c,xz=0`, and `I_c,yz=0`. The independent verifier reproduces these four equations from the independent mass moments after transforming into the explicit right-handed task frame. It does not label generic CoM-axis offset or principal-axis alignment as original solver constraints.

## Additional diagnostics

The report also gives CoM offset from the declared spin axis, central inertia tensor, principal inertia eigenvalues/eigengaps, and nearest principal-axis alignment. These remain diagnostics unless a user supplies explicit limits. All hard tolerances are mandatory inputs with no invented defaults; the report records them.

Mass consistency is checked as `M = density * geometric_volume_m3`. Density, units, coordinate frame, case and geometry hash must agree. A non-orthonormal or left-handed task frame is rejected.

## Validation

Tests use one analytic tetrahedron with its principal inertia frame and negative/mismatched metadata cases. They test formula plumbing only and are not TRELLIS/M69 results.

## Remaining evidence gap

No real case is physically admitted yet. The independent tetrahedralization must be shown to represent the exact same retained material region as the native final material boundary, including cavities and fixed shell. The real case must then pass native/independent moment comparison and this physical constraint verifier. No Hybrid speedup claim follows from this CI.
