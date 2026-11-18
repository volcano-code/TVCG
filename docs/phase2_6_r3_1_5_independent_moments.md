# R3.1.5.1 — Independent exact tetrahedral raw moments

## Mathematical contract

The independent backend integrates an explicitly supplied tetrahedral decomposition of the retained material solid. It does not invoke the native quadric evaluator, native residual, or native moment code. It does not generate tetrahedra, classify material, or infer the retained side.

For each positively oriented tetrahedron T with vertices x_0..x_3 and volume V_T, exact simplex identities are used:

- integral_T 1 dV = V_T
- integral_T x dV = V_T/4 * sum_i x_i
- integral_T x x^T dV = V_T/20 * ((sum_i x_i)(sum_i x_i)^T + sum_i x_i x_i^T)

Coordinates are converted to metres using explicit meters_per_unit before integration. Density is explicit and multiplies all ten raw mass moments. The order is (V,Lx,Ly,Lz,Qxx,Qyy,Qzz,Qyz,Qxz,Qxy), where Qij = integral rho*x_i*x_j dV.

The backend rejects non-finite coordinates, degenerate tetrahedra, and negative signed orientation. It requires a source-backed tetrahedralization that represents exactly the retained material domain. Tetrahedralization provenance and mesh-to-tet equivalence remain an external admission requirement.

## Derived physical quantities

Mass and CoM are derived from the ten mass moments. The second moment tensor Q yields I_origin = tr(Q)I-Q and I_com = I_origin - m((c^T c)I-cc^T).

## Validation

Tests use the analytic unit tetrahedron and verify exact closed-form moments, unit conversion, density scaling, orientation rejection, degeneracy rejection, finite-value checks, and positive-definite central inertia. These tests validate the independent integrator only; they are not M69/TRELLIS results.

## Real-data gate

To compare with a native replay, Work must provide a reviewed tetrahedralization of the same final retained solid, tied to the admitted geometry hash, coordinate frame, unit scale and material density. The independent backend then emits a separate record for `r3_deterministic_replay_audit_v1.py verify-moments`. No physical comparison is claimed until that real record exists.
