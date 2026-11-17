# Phase 1 — Unified mathematical contract (v1)

**Status:** mathematical contract frozen for implementation; primary objective selection remains an explicit decision gate.  
**Scope:** geometry-conditioned rotation-stable material optimization.  
**Normative implementation references:** `SpinDesignRequest`, `SpinMomentProblem`, `RawMassMoments10`, `QuadricCoefficients10`, and `docs/spin_it_faster_equations.md`.

This contract separates the new material-optimization problem from the existing Spin-It Faster quadric-query problem. It is intentionally conservative: no objective, tolerance, physical unit, contact convention, shell thickness, or event threshold is inferred where the source contract does not determine it.

## 1. Symbols and dimensions

| Symbol | Meaning | Shape / domain |
|---|---|---|
| `G` | immutable input solid and its provenance | mesh / registered voxel or native-interval representation |
| `phi_G` | geometry encoding | model-dependent tensor |
| `q` | rotation task | structured task record |
| `beta` | normalized quadric coefficients | `R^10`, `||beta||₂=1` |
| `psi` | optional local-correction parameters | `R^n_psi`; absent in Pure Quadric |
| `alpha` | material design variables `(beta, psi)` | `S^9 × R^n_psi` |
| `rho(x;G,alpha)` | physical density field | scalar, kg/m³ when SI is registered |
| `M` | total ten raw mass moments | `R^10` |
| `dM/dalpha` | material-variable moment derivative | `10 × n_alpha_local` |
| `g,h` | inequality/equality task constraints | `R^m, R^p` |
| `F_KKT` | active-set KKT residual | dimension equals active-set primal/dual unknown count |

The coefficient-query variable in the existing M80 operator is denoted `beta_q` in future code when needed. It is a query coordinate, not automatically the material design variable `alpha`.

## 2. Geometry and coordinate frames

### 2.1 Geometry record

The geometry record binds:
- immutable geometry identity and SHA-256;
- source object/family identity and split group;
- outer boundary and any inner/cavity boundaries;
- registered coordinate frame and transform provenance;
- coordinate units and physical scale status;
- native representation (mesh, voxel, RLE/interval) and resolution;
- source/representation conversion identity and error budget, if one is approved.

A geometry is not physically admitted merely because it is watertight or voxel-connected. The existing M90.2/M90.3 scope limits remain in force.

### 2.2 Target frame

Use a right-handed orthonormal target frame `(o,R)`, where columns of `R` are the target basis axes expressed in model coordinates, matching `SpinTargetFrame`. For a model-space point `x_m`,

[
x_t = R^T(x_m-o).
]

For spinning-top tasks, `e_3` is the requested spin axis in target coordinates. The task record stores the axis, origin, contact point and their provenance separately. The contract must explicitly state whether the target origin equals the physical contact point; equality is not assumed from field names alone.

Raw moments, task constraints and solver coordinates must be expressed in the same registered target frame before evaluation. Affine moment transforms must be applied to all ten moments; rotating only the first moments is invalid.

### 2.3 Units

When SI is registered:
- `x`: m;
- `rho0`: kg/m³;
- `M0`: kg;
- `M_i` for first moments: kg·m;
- `M_ij` for second moments: kg·m²;
- inertia: kg·m².

A dimensionless/source-coordinate track is permitted only as a separately named track. It must not be silently relabelled SI or mixed with SI labels. Normalization scales and their powers must be recorded per moment degree.

## 3. Material domain and design variables

Partition the source solid into disjoint:
- `Omega_fixed`: immutable fixed shell/material;
- `Omega_protected`: immutable protected material, if distinct from the fixed shell;
- `Omega_editable`: the only region the optimizer may modify.

The partition is geometry-owned and hash-bound. The three sets must be pairwise disjoint; any uncovered region and its meaning must be declared. Fixed material is included exactly once in the total moments.

For the strict single-material track, `rho0` is one positive constant. The primary binary material field is

[
ho(x;G,alpha)=rho_0[
  1_{Omega_fixed}(x)+1_{Omega_protected}(x)
  +1_{Omega_editable}(x) r(x;alpha)].
]

The masks are disjoint, so this sum does not double-count material. If protected material is already part of the fixed shell, the contract must use one canonical mask rather than overlapping masks.

### 3.1 Pure Quadric (primary representation)

[
beta(x)=a+b^T x+x^T A x,
]

with coefficient order

[
(a,b_1,b_2,b_3,A_{11},A_{22},A_{33},A_{23},A_{13},A_{12}).
]

`A` is symmetric. The stored cross coefficients are matrix entries, so the polynomial contains `2 A23 yz + 2 A13 xz + 2 A12 xy`. Material is retained on the declared positive side `beta(x)>0` (the sign convention must be bound to the teacher/task identity). Since positive scaling does not change the set, `beta` is represented on `S^9`; the sign is not quotiented out.

[
r(x;beta)=1[beta(x)>0], quad x\in Omega_{editable}.
]

No fractional density is part of the final strict single-material result.

### 3.2 Quadric + Local Correction (optional extension)

`psi` parameterizes a local correction field `delta_psi(x)` whose support is strictly contained in `Omega_editable`. A candidate hard representation is

[
r(x;beta,psi)=1[beta(x)+delta_psi(x)>0].
]

This representation is a separately versioned ablation, not silently substituted for Pure Quadric. The basis, support, amplitude bounds, regularizer and discretization must be declared before training. A smooth surrogate may be used during optimization, but its temperature/filtering parameters and the final hardening rule must be recorded. Smooth density is not binary-material acceptance.

## 4. Ten raw mass moments

Use the canonical order from `RawMassMoments10`:

[
M=(m,l_x,l_y,l_z,Q_{xx},Q_{yy},Q_{zz},Q_{yz},Q_{xz},Q_{xy})^T.
]

For target-frame coordinates `x=(x,y,z)`,

[
M_k = \int_{Omega_G} rho(x;G,alpha) m_k(x) dx,
]

where the monomial basis is

[
m(x)=(1,x,y,z,x^2,y^2,z^2,yz,xz,xy)^T.
]

The total moment vector is

[
M_{total}=M_{fixed}+M_{protected}+M_{editable}.
]

This is a density-weighted mass-moment vector. If a legacy path uses volume moments `v,l,Q`, the density conversion must be explicit; mass and volume moments must not be mixed.

For constant density:
[
c_{COM}=l/m,qquad
I_o=tr(Q)I-Q,qquad
I_c=I_o-m[(c^Tc)I-cc^T].
]

## 5. Material derivatives — do not conflate query and design derivatives

The new target derivative is

[
J_alpha = \frac{\partial M_{total}}{\partial alpha}
\in R^{10\times n_alpha}.
]

For Pure Quadric, on a smooth fixed-topology branch with a regular boundary and transversal intersections, the formal shape derivative is

[
\frac{\partial M_k}{\partial beta_j}
= rho_0 \int_{\partial Omega_{editable}(beta)}
 m_k(x)\frac{\partial beta(x)/\partial beta_j}{|\nabla beta(x)|} dA,
]

with orientation/sign adjusted consistently to the chosen retained side and outward normal. This expression is valid only where the boundary motion is regular and the Boolean/material topology is locally unchanged. The implementation must verify its sign against analytic fixtures and the exact teacher.

For a local correction parameter `psi_j`, the corresponding regular-boundary expression replaces `partial beta/partial beta_j` by `partial delta_psi/partial psi_j`, with the same support and event restrictions.

The existing M80 derivative
[
J_beta^{query}=\partial M/\partial beta_{query}
]
is a query-response derivative. It is not `J_alpha` unless an explicit, validated chain rule maps the query coefficients to the material variables and the teacher certifies the composed derivative.

For a parameterized map `beta=beta(alpha)`, a chain rule may be used only after the map and both derivative factors are independently validated:
[
\partial_alpha M=(\partial_beta M)(\partial_alpha beta).
]
The existence of a 10×10 M80 Jacobian alone does not establish this condition.

## 6. Rotation task and constraint profiles

The task record `q` contains:
- mode/profile identifier;
- target frame, spin axis, origin and contact point;
- task-specific physical parameters and units;
- fixed-shell/protection policy identity;
- exact TaskSpec/source hash and evaluator identity.

The current C++ implementation exposes several distinct problem modes. They must remain distinct profiles; do not merge their objectives or constraints.

### 6.1 Existing spinning-top moment profile

The current `SpinMomentProblem::spinningTop` evaluates, in the target frame:
- equality constraints `l1=0`, `l2=0`, `Ic13=0`, `Ic23=0`;
- for the inertia-and-lowering endpoint, objective `f=yoYoObjective(Ic)+wc*(l3/m)`;
- for the lower-COM endpoint, objective `f=l3/m` with the same four equalities.

The current Spin-It Faster equation note expresses the same top task in its own notation as minimizing the yo-yo inertia ratio plus `wc*c3`, with `l1=l2=0` and `Ic13=Ic23=0`. These are equality-constrained moment-space optimization profiles, not a minimum-mass-under-inequality formulation.

### 6.2 New minimum-mass material-design profile

The research target may instead be written as

[
\min_{alpha} M_0(G,alpha)
\quad\text{s.t.}\quad
g_i(M_{total}(G,alpha),q)\le 0,\quad
h_j(M_{total}(G,alpha),q)=0,
]

plus immutable-region, support, single-density, binary-material and topology requirements.

This is a different outer optimization problem from the existing Faster moment-space objective. Its actual stability inequalities, tolerances, signs and task semantics must be taken from the selected frozen TaskSpec / physical oracle. They are not invented in this contract. The primary objective profile therefore remains an explicit Phase 1 decision gate until the user selects the authoritative objective/TaskSpec for the main experiments.

### 6.3 Constraint records

Every constraint record stores:
- stable identifier and version;
- relation (`eq` or `le_zero`);
- formula/source implementation identity;
- units and normalization scale;
- value, derivative validity and event applicability;
- tolerance from the frozen task, not fitted from observed outcomes.

Do not encode topology/connectivity, manufacturing or dynamic simulation as a smooth scalar inequality unless an independently justified differentiable surrogate and separate exact acceptance rule are both specified.

## 7. Events and derivative validity

Every query/label has a derivative status: `valid`, `invalid_event`, or `unknown`. Unknown is never converted to a zero Jacobian.

At minimum classify:
- regular smooth boundary;
- tangent / repeated-root contact;
- Boolean topology or component-count change;
- quadric rank/stratum change;
- source truncation/cap branch change;
- ambiguous or non-manifold geometry;
- fixed/editable/protected partition change;
- unsupported derivative implementation / missing provenance.

A derivative label is admitted only when the teacher certifies the local branch and an independent directional finite-difference audit passes on that branch. Event samples remain available for event/risk modelling but are masked from smooth-Jacobian regression.

## 8. KKT contract

For the new minimum-mass profile, use the sign convention

[
L(alpha,lambda,nu)=M_0(alpha)+lambda^T g(alpha)+nu^T h(alpha),
\quad lambda\ge0,\quad g\le0.
]

For a fixed active set `A`, the KKT conditions are:
[
\nabla_alpha L=0,\quad h=0,\quad g_A=0,
]
with inactive primal feasibility `g_I<0`, active dual feasibility `lambda_A>=0`, and complementary slackness `lambda_i g_i=0`.

For Pure Quadric, `beta\in S^9`. Stationarity is evaluated in the tangent space:
[
P_beta\nabla_beta L=0,\quad P_beta=I-beta beta^T.
]
Local-correction variables use their declared feasible tangent/box constraints. If an explicit multiplier for the sphere constraint is used instead, that formulation and its sign must be versioned; do not combine both forms.

For a fixed active set, with unknowns `(alpha,nu,lambda_A)`, define

[
F_{KKT}=
\begin{bmatrix}
\nabla_alpha L\
h\
g_A
\end{bmatrix}
]

and apply the beta tangent projection to its beta-stationarity block. The residual dimension equals `n_alpha + n_h + |A|` when `alpha` is represented in local feasible coordinates and the active multipliers are included as unknowns. The KKT Jacobian is the derivative of this assembled residual with respect to `(alpha,nu,lambda_A)`; it is not the existing `dM/d beta_query`.

For the full inequality system, use a separately versioned complementarity formulation (e.g. semismooth Fischer–Burmeister) or an explicit active-set algorithm. Do not treat an active-set switch as a smooth event.

## 9. Acceptance and numerical tests

Phase 1 tests must establish:
1. canonical moment and coefficient order;
2. affine frame transform consistency for all ten moments and dual coefficient pairing;
3. fixed material is added exactly once;
4. analytic material derivatives match central finite differences on regular fixtures;
5. derivative validity is refused at branch endpoints/events;
6. KKT sign convention, active-set residual dimensions and beta tangent projection;
7. invalid units, overlapping masks, non-finite values and stale identities fail closed.

The committed analytic fixture uses a box clipped by the plane `x<=a`. Its exact moments and boundary derivative are independent of the neural model. It is a mathematical contract test, not a substitute for M69/Cherchi labels or a real-object admission.

## 10. Open decisions (must not be guessed)

1. Which frozen objective/TaskSpec is authoritative for the main paper: strict minimum-mass under the M69 rotation constraints, or a faithful Spin-It Faster moment-space objective? They can be reported as distinct tracks, but must not be conflated.
2. Exact contact-point/origin relation for each benchmark task.
3. The precise primary stability inequalities and their frozen tolerances for the minimum-mass track.
4. Whether fixed shell and protected material are one mask or two disjoint masks for each source family.
5. Approved physical-unit policy for each dataset family.
6. Local-correction basis, support and hardening policy (only needed after Pure Quadric baseline).

Until these are resolved, implement the common math and tests, but do not freeze a benchmark objective or generate purported labels for unknown constraints.
