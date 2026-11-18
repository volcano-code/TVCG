# Phase 2.8.2 — SO(3) physical transform and Jacobian chain rule

## Rotation convention

Use column vectors and an active proper rotation:

`x' = R x`, where `R^T R = I` and `det(R)=+1`.

The rotated query is defined by preserving the same scalar field in rotated coordinates:

`f'(x') = f(R^T x')`.

Write the native quadric as

`f(x) = x^T A x + b^T x + c0`,

where the native coefficient order is
`(1,x,y,z,xx,yy,zz,2yz,2xz,2xy)`.
Because each native cross coefficient multiplies twice the product, the symmetric matrix stores that coefficient directly in each off-diagonal entry:

`A_yz = A_zy = q_2yz`, similarly for xz and xy.

The rotated coefficients therefore obey:

`A' = R A R^T; b' = R b; c0' = c0`.

This is implemented by `transform_native_coefficients`.

## Raw moment transform

The raw moment order is:

`(mass, x, y, z, xx, yy, zz, yz, xz, xy)`.

Let `m0 = integral rho dm`, `p = integral x rho dm`, and `S = integral x x^T rho dm`. Under `x'=R x`:

`m0' = m0; p' = R p; S' = R S R^T`.

The second moment tensor is not the centered covariance or inertia tensor. The implementation reconstructs the symmetric raw second-moment matrix from the ten-vector, rotates it, then serializes it in the native order. This is implemented by `transform_raw_moments`.

Fixed and editable material moments are additive raw moments. Rotate each partition with the same transform and then sum; linearity guarantees the result equals rotating their sum.

## Jacobian chain rule

Let `r = M(c)`. Define the ten-dimensional linear transforms by:

`r' = T_R r`, and `c' = S_R c`.

Then:

`J' = d r' / d c' = T_R J S_R^{-1}`.

The implementation `transform_native_jacobian` uses this exact convention. This is a coordinate-change identity, not a learned equivariance guarantee.

## Validation added

The SO(3) tests now include:

- Rotated finite-quadrature scalar-field identity: `M_rot(c') = T_R M(c)`.
- Jacobian chain rule: `J_rot = T_R J S_R^{-1}`.
- Independent central finite differences on the rotated query map.
- Fixed/editable partition rotation and recomposition.
- Identity, inverse round-trip and composition properties.
- Rejection of reflections (determinant -1) and nonfinite physical payloads.
- Existing coefficient covariance and arbitrary SO(3) augmentation tests remain.

The finite-quadrature test uses a smooth sigmoid indicator as a differentiable mathematical fixture. It validates the transform algebra and chain rule; it is not a Faster teacher accuracy result or a real-mesh physical validation.

## Scope and limitations

- This phase adds no model parameters and does not change checkpoint schema.
- It does not make the learned FNO/GNO/Transolver model exactly SO(3)-equivariant. It supplies exact label/query transforms and testable covariance contracts.
- Arbitrary voxel rotation still uses trilinear resampling; voxel interpolation error is separate from the exact moment/query transform.
- Translation and general unit/density conversion are not part of this SO(3) rotation-only contract.
- Rotating a task with a fixed external spin axis/contact/support requires rotating those task conditions consistently too; this phase's functions transform geometry moments and quadric queries only.
- SO(3) covariance of the full learned operator still requires training/evaluation evidence with transformed geometries and task conditions.
