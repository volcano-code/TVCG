# Phase 2.1 — Pure Quadric Material Operator v1

## Research question

Can the existing M80 quadric-query Jacobian be used as the Pure Quadric material-design Jacobian?

**Answer:** only under an identity (or explicitly differentiated) design map and identical physical contracts. The equality is not automatic from both arrays being 10x10.

## 1. Canonical query convention

M80 uses the ten query basis
`[1,x,y,z,x^2,y^2,z^2,2yz,2xz,2xy]`.
The corresponding coefficient vector is
`q=(c,bx,by,bz,Axx,Ayy,Azz,Ayz,Axz,Axy)`.
The associated symmetric quadratic matrix stores off-diagonal entries once; the polynomial contains twice each cross coefficient.

For a fixed geometry and a binary retained-side rule `q(x)>=0`, define
[
M_i(q)=\rho_0\int_{\Omega_G}1[q(x)\ge0]m_i(x)dx.
]
On a regular smooth branch, the shape derivative is
[
\frac{\partial M_i}{\partial q_j}
=\rho_0\int_{\partial\Omega_G\cap\{q=0\}}
m_i(x)\frac{p_j(x)}{|\nabla q(x)|}dA,
]
where `p_j` is the matching query basis function (including factor 2 for cross terms). This is a boundary shape derivative; it is not a voxel derivative and it is not valid at singular/tangent/topology events.

## 2. When is query Jacobian the material Jacobian?

Let material design `alpha` map to query coefficients `q=Q(alpha)`. Then
[
J_\alpha=\frac{\partial M}{\partial\alpha}
=J_q\frac{\partial Q}{\partial\alpha}.
]

Thus `J_alpha = J_q` only if:
1. `alpha` is exactly the same ten coefficient coordinates as `q` (identity map);
2. the coefficient order and cross-term convention match;
3. the same coordinate frame and length normalization are used;
4. retained-side/sign convention is identical;
5. fixed, protected and editable masks are identical;
6. density and moment units/normalization are identical;
7. the exact teacher certifies a regular derivative branch.

If alpha is represented by nine local coordinates on S^9, then `J_alpha = J_q U`, where U is a 10x9 orthonormal tangent basis. If beta is normalized inside the model, its normalization derivative must also be included. If a smooth sigmoid/hinge replaces the hard indicator, the derivative is that surrogate's derivative and is not automatically the hard Boolean material derivative.

## 3. Analytic fixture

For the box `[xmin,xmax]×[ymin,ymax]×[zmin,zmax]` and
[
q(x)=c+b_xx,\quad b_x<0,quad a=-c/b_x,
]
the retained domain is the box portion `x<=a`. For each raw moment exponent `(r,s,t)),
[
M_{rst}=\rho_0
\frac{a^{r+1}-x_{min}^{r+1}}{r+1}
\frac{y_{max}^{s+1}-y_{min}^{s+1}}{s+1}
\frac{z_{max}^{t+1}-z_{min}^{t+1}}{t+1}.
]
The full ten-by-ten coefficient Jacobian is the exact face integral above. An independent high-order Gauss rule over (y,z), with exact x-interval integration, is used to evaluate central finite differences for all ten coefficient directions.

The fixture is analytic and synthetic. It does not exercise M80 weights/checkpoints or certify new-geometry performance.

## 4. Positive scale invariance and S9

For k>0, `M(kq)=M(q)`. Therefore
[
J_q(kq)=k^{-1}J_q(q),\qquad J_q(q)q=0.
]
Negative scaling reverses the retained side and is not an invariance.

For unit representative `u=q/||q||`, the tangent projector is
[
P_u=I-uu^T.
]
A Euclidean gradient is projected as `g_T=P_u g`. Newton updates are solved in an orthonormal tangent basis U:
[
(U^THU+\lambda I)\eta=-U^Tg_T,
quad q^+=\frac{u+U\eta}{||u+U\eta||}.
]
The implementation uses a damped tangent-space Newton step and rejects steps crossing the local sign chart.

## 5. Event semantics

- A regular plane has rank-zero quadratic matrix but nonzero spatial gradient. Rank zero alone is not a derivative failure.
- A plane coincident with a box support face is a topology/support event; derivative is `None`, not a zero matrix.
- A repeated-root surface such as `q=(x-a)^2` has zero boundary gradient at the root and is singular; smooth shape Jacobian is invalid.
- General quadric topology changes, multiple roots and axis-selection switches require oracle/event metadata. The current analytic classifier is deliberately limited and returns unsupported/unknown for unclassified cases.

## 6. Implemented evidence

The module `pure_quadric_material_operator_v1.py` provides:
- exact ten-moment integration for a plane-cut box;
- exact shape derivative for all ten M80 query coefficients;
- independent high-order finite-difference evaluator for nearby quadratic perturbations;
- positive-scale invariance checks;
- identity-gated query-to-material chain rule;
- S9 tangent projection, retraction and damped tangent Newton;
- explicit invalid event outputs.

CPU tests exercise these mathematical contracts. They do not instantiate M80 because the M80 native operator requires a registered native RLE geometry/cache and model checkpoint/training configuration; this phase deliberately does not fabricate those inputs. No M80 forward pass or checkpoint evaluation is claimed.

## 7. Next research gate

1. Verify the analytic derivative against the actual M80 AD Jacobian on a registered synthetic native-interval fixture with matching hard-query semantics.
2. Extend the exact teacher to the same binary retained-side and fixed/protected mask contract.
3. Add KKT residual derivatives with the M69 fixed-axis constraints.
4. Only then build the unified task-conditioned neural operator and differentiable optimizer.
