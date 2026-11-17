# TVCG code audit: Geometry-conditioned Quadric PINO
Date: 2026-11-17

## 1. Audit scope and reproducibility boundary

Audited the GitHub snapshot at branch research/tvcg-initial-import, commit 3094ddcc9e0b5ea008b43d10142de8e402874d27. The recursive Git tree contains 4,118 entries. The repository currently exposes one branch. This report describes that committed snapshot, not an uncommitted local Spin_test2 working tree.

The committed WORK_HANDOFF.md contains later local research status and paths under /data/xvxinyao/Spin_test2 and /data/xvxinyao/Spin_M68_Trellis_Data_Kit. Those local-only files, reports, datasets, checkpoints, and M69 source are not automatically part of this GitHub commit. The research_control directory referenced by WORK_HANDOFF.md and the proposed quadric_pino directory are absent from the audited Git tree. Therefore the audit distinguishes committed implementation from handoff-reported local evidence.

No source code was changed in the audit. No Python/C++ test, model inference, optimizer step, GPU benchmark, or training run was executed by this read-only repository audit.

## 2. Executive assessment

The repository already contains substantial building blocks:
- SpinPINO: geometry-conditioned positive quadrature weights, ten raw moments, and coefficient Jacobians.
- PcraftNativeMomentOperator: native RLE interval conditioning, P-CRAFT spatial context, optional GNO/Transolver/multiscale paths, and moment/Jacobian query evaluation.
- M80 training infrastructure: manifest/audit-bound corpus loading, source-group split checks, resumable training interfaces, and GPU preflight.
- Neural Solver Bridge: a fail-closed moment/Jacobian callback with geometry/topology/cache identity checks, admission callback, and exact fallback.
- C++ Spin-It Faster and QuadricMomentEvaluator implementation, including explicit derivative-method identities and fixed-topology discrete differentials.
- Experimental fixed/editable material representation and TaskSpec-based rotation objective functions.

However, these pieces do not yet constitute the requested unified geometry-conditioned constrained optimizer. In particular, the current learned models primarily approximate the map from a geometry and a quadric query to ten moments and their query Jacobian. They do not yet provide a validated material-parameter optimizer, a coupled KKT system, or a differentiable safeguarded Newton layer. The newer material relaxation is explicitly untrained and not a final single-density removal-only design.

Current maturity: strong research infrastructure and several mathematically meaningful components; unified optimization algorithm and end-to-end evidence remain incomplete.

## 3. Component-by-component findings

### 3.1 SpinPINO backbone and moment head

Committed files:
- research/NeuralMoments/models/pcraft_ufno/spin_pino/pino_backbone.py
- research/NeuralMoments/models/pcraft_ufno/spin_pino/spin_pino.py
- research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/

Verified from source:
- The backbone is the P-CRAFT factorized Fourier / U-shaped 3D encoder.
- SpinPINO predicts nonnegative geometry-conditioned quadrature weights at fixed quadrature nodes.
- It evaluates ten raw moments in the order mass, x, y, z, xx, yy, zz, yz, xz, xy.
- It supports smooth_hinge and logistic_atom moment parameterizations.
- It provides an analytical coefficient Jacobian for its parameterized moment map.
- Its physics losses encode complement, projective radial invariance, Jacobian symmetry/PSD checks, mass bounds, and optional admitted Jacobian supervision.

Limit:
- The learned map is principally geometry-to-quadrature-measure followed by a quadric coefficient query. It is not yet a general optimizer over material parameters alpha and task q.
- Its fixed quadrature-node representation and smooth parameterization are not themselves a certificate for the hard Boolean material region.
- Existing convex-potential identities apply only to the declared parameterization and must not be generalized to unsupported geometries, task constraints, or active-set events.

### 3.2 M80 P-CRAFT native operator

Committed file:
- research/NeuralMoments/models/pcraft_ufno/spin_pino/native_pcraft.py

Verified from source:
- Geometry enters through native M72 RLE intervals, physical interval masses, extents, moment bases, fixed moments, and original moments.
- A conservative coarse grid provides P-CRAFT global spatial context; native intervals remain represented in the interval feature path.
- Configurations expose optional GNO, Transolver-style physics-state slicing, multiscale context, and a convex moment-potential head.
- Query evaluation returns moment information and supports automatic differentiation for the query Jacobian.
- The code includes explicit geometry/cache identities and bounded-memory neighbor-graph caching.

Limit:
- The coarse context grids are not equivalent to a dense 512-cubed learned field. Native sparse intervals are retained, but this is a hybrid sparse-native representation and must be benchmarked and described as such.
- The model still represents a geometry-conditioned quadric query/moment operator, not a complete alpha-optimizer.
- There is no coupled task-KKT output or active-set-aware optimizer in this module.

### 3.3 Existing trainer and data contracts

Committed file:
- research/NeuralMoments/models/pcraft_ufno/spin_pino/native_train.py

Verified from source:
- M80 has a formal-corpus reader requiring a frozen M69/M72 manifest and a valid audit receipt.
- It verifies source hashes, geometry/label shapes, derivative-valid masks, and source-group separation across train/validation/test.
- It includes training and preflight infrastructure for the M80 moment operator.

Limit:
- This trainer supervises the existing moment-query task. It does not train a unified optimizer or demonstrate that a new geometry can be optimized to a feasible minimum-mass design.
- WORK_HANDOFF.md reports that the latest TRELLIS-only dataset policy is not yet physically admitted. The current 24 ordinary TRELLIS objects have zero TaskSpec-ready cases, zero physical geometry admissions, zero feasible material states, and zero Oracle labels. Five passing a surface gateway is only a preliminary surface result.
- The latest handoff reports that the earlier 20 direct spinning-top meshes are diagnostic-only and excluded from formal training. Do not count generated meshes as admitted training samples.
- No new formal training should start until geometry, material, task, units, shell/protection, and representation-error gates pass.

### 3.4 Neural Solver Bridge

Committed file:
- research/NeuralMoments/models/pcraft_ufno/spin_pino/neural_solver_bridge.py

Verified from source:
- The bridge transforms native coefficient and moment frames consistently.
- It checks geometry, topology, model-version, and cache identities.
- It refuses neural use without an admission callback, while the model is in training mode, or when the cache is stale.
- It checks finite values and the declared convex-Jacobian conditions.
- It falls back to the exact evaluator on rejected or failed neural calls.

Limit:
- This is a moment/Jacobian callback adapter, not the requested differentiable constrained optimization layer.
- The callback does not itself solve for alpha, manage inequality active sets, differentiate through a safeguarded Newton trajectory, or certify KKT convergence.
- Its admission and fallback behavior is a valuable safety boundary and should be preserved.

### 3.5 Material representation and rotation physics

Committed files:
- research/NeuralMoments/experiments/pino_material_v1/model.py
- research/NeuralMoments/experiments/rotation_material_v1/material.py
- research/NeuralMoments/experiments/rotation_material_v1/physics.py
- research/NeuralMoments/experiments/rotation_material_v1/corpus.py

Verified from source and README:
- Fixed and editable support are represented separately.
- DensiCrafter-inspired bounded sigmoid density fractions and editable-neighbor smoothing are implemented as an experimental mechanism.
- TaskSpec-based code computes ten rotation/material constraints and per-term losses.
- The synthetic corpus has 12 deterministic 8-cubed records and 60 analytic hard-cut labels, split 8/2/2.

Explicit limitations documented by the project:
- These models are untrained.
- No optimizer is automatically run.
- Variable density is a relaxation, not proof of M69 single-density removal-only compliance.
- Smoothing does not certify connectivity.
- Binary conversion, printability, shape accuracy, and rotational dynamics are not certified.
- The small synthetic corpus validates schema and labels only; it is not evidence of geometry-population generalization.
- Smooth-cut quadrature differs from hard-cut reference labels.

### 3.6 C++ quadric evaluator and Faster mathematical core

Committed files include:
- include/spin/QuadricMomentEvaluator.h
- include/spin/QuadricDiscreteMomentDifferential.h
- include/spin/QuadricCoefficients.h
- src/spin/QuadricMomentEvaluator.cpp
- src/spin/QuadricDiscreteMomentDifferential.cpp
- docs/spin_it_faster_equations.md
- the Spin-It Faster solver sources and regression tests.

Verified from source:
- Quadric coefficients use the ten-dimensional degree-two polynomial convention and normalized projective representation.
- The value evaluator uses a CGAL mesh Boolean and closed-boundary moment integration.
- Derivative identity distinguishes LegacyContinuousImplicitShape, DiscreteFullRankQuadric, and DiscreteReducedStratum.
- The reduced-stratum differential remains lower-dimensional and is explicitly not an ambient ten-dimensional Newton Jacobian.
- The discrete full-rank differential is fail-closed around unsupported topology/feature transitions.
- The Faster equation document records fixed-shell moments, target-frame transformation, quadric material regions, the equality-constrained optimality system, tangent updates, and exact represented-boundary moment evaluation.

Limit:
- The default evaluator derivative method is LegacyContinuousImplicitShape; the discrete fixed-topology methods are explicit opt-ins with restricted applicability.
- A value/Jacobian evaluator is not yet a Python Exact Teacher Adapter for the training pipeline.
- The exact evaluator's represented-domain result and the separate M69 final acceptance policy must remain distinct.
- The archived build configuration defaults the Spin-It Faster solver and Python extension off; a reproducible native build profile is still required for the training-label adapter.

### 3.7 Missing unified components

The audited Git tree contains no quadric_pino package and no research_control directory. The following are not present as an integrated, tested implementation:
- a versioned unified geometry/material/task contract for (phi_G, alpha, q);
- a Pure Quadric alpha parameterization with fixed-shell/protected-region semantics wired into the neural model;
- an exact teacher adapter that produces typed, provenance-bound labels for moments, material derivatives, and valid KKT systems;
- a shared potential/physics model that enforces consistency among M, dM/dalpha, F_KKT, and J_F;
- active-set and nonsmooth-event handling;
- a differentiable, trust-region safeguarded Newton optimization layer;
- a local material correction interface with explicit support, boundedness, and single-density hard-export semantics;
- an end-to-end comparison that demonstrates feasible minimum-mass designs on held-out geometries.

## 4. Gap matrix

| Target capability | Existing evidence | Gap / required work | Priority |
|---|---|---|---|
| Geometry conditioning phi_G | P-CRAFT encoder, RLE/native intervals, SDF-related inputs | Canonical geometry contract and representation/error budget; TRELLIS task-ready cases | P0 |
| Quadric material alpha | Ten-coefficient query and C++ Faster parameterization | Explicit alpha semantics, fixed shell, protected support, hard sign convention and frame transforms | P0 |
| Moment map M | SpinPINO and M80 operator | Exact hard-material teacher alignment and cross-geometry calibration | P0 |
| dM/dalpha | Query Jacobians / AD and C++ derivatives | Prove coordinate/gauge consistency and derivative applicability; typed labels | P0 |
| Rotation task q | Task69 objective functions and target-frame transforms | Unified versioned task contract and complete admitted real cases | P0 |
| KKT F and J_F | C++ Faster optimality solver; task losses | Derive from the same physical map; valid active-set labels and differentiable residual Jacobian | P1 |
| Differentiable optimizer | Native solver bridge and fallback | Implement alpha update layer, trust region, exact line search/check, fallback, implicit differentiation gates | P1 |
| Local correction | Experimental bounded density relaxation | Restrict to explicit editable support; compare against Pure Quadric; hard single-density export | P2 |
| Exact verification | C++ evaluator and M69 references | Python teacher adapter, independent saved-result validation and real-case acceptance | P0 |
| Dataset/generalization | Synthetic labels and TRELLIS diagnostic assets | TaskSpec-ready geometry/material states, independent family splits, exact labels | P0 |
| Runtime advantage | Existing neural callback timing instrumentation | Same-query/same-tolerance speed accounting and full solve comparison | P2 |
| 512+ validation | Native sparse interval representation and high-resolution audits | Explicit memory/error budget; no claim that a coarse context grid is a dense 512 model | P1 |

## 5. Overall conclusion

The project is not starting from zero. The moment-query operator, native geometry encoder, Faster mathematical core, exact represented-boundary evaluator, and fail-closed callback are valuable existing assets.

The central missing contribution is the coupling of those assets into one optimization problem. The next implementation should not add another independent SDF network or a serial DensiCrafter stage. It should define alpha and q rigorously, connect the existing moment operator to the Faster optimality system, derive KKT residuals from a shared physical representation, and implement a safeguarded differentiable optimization layer.

The Pure Quadric-PINO path is the mandatory core baseline. Quadric plus local correction is a separate extension and must demonstrate a measurable gain under the same frozen task and acceptance policy.

## 6. Cloud evaluation status

This audit used GitHub source reads and metadata only. No code execution was available through the read-only repository inspection path. The repository contains CPU GitHub Actions workflows for M80/PINO contracts, but the audited branch had zero workflow runs. Those workflows target other research branches or main, not the initial-import branch. Therefore no new cloud test result is claimed in this report.
