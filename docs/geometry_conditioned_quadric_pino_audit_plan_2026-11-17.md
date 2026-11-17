# TVCG code audit and unified neural-operator roadmap

**Audit snapshot:** 2026-11-17  
**Repository:** `volcano-code/TVCG`  
**Branch:** `research/tvcg-initial-import`  
**Audited HEAD:** `3094ddcc9e0b5ea008b43d10142de8e402874d27`

This document records a targeted source audit of the committed repository and the implementation plan for the agreed **Geometry-conditioned Quadric Physics-Informed Neural Optimization** framework. It does not claim that the proposed unified optimizer is already implemented.

## 1. Executive assessment

The repository contains substantial reusable infrastructure:

- Spin-It 2014 and Spin-It Faster C++ implementations and regression tests;
- a P-CRAFT/F-FNO U-shaped geometry encoder;
- Spin-PINO moment and coefficient-Jacobian heads;
- the larger M80 P-CRAFT native-interval operator, including optional GNO, Transolver, multiscale and convex-potential components;
- a strict M72 corpus reader/trainer and a fail-closed Faster callback bridge;
- C++ quadric moment evaluation and explicitly separated full-rank/reduced-stratum differential contracts;
- M90.2/M90.3 independent saved-result and sparse-topology validators.

However, the central research objective is **not yet implemented**. The current learned models are geometry-conditioned *quadric-query-to-moment response operators*. They do not yet optimize a material design variable jointly conditioned on the rotation task, and they do not produce a valid KKT system for the minimum-mass design problem.

The largest gap is therefore not a missing backbone. It is the missing mathematical and software layer that connects:
1. material design variables and the density field;
2. the rotation task and its constraints;
3. exact/learned material derivatives;
4. KKT residuals and active-set logic;
5. a safeguarded differentiable optimization layer.

**Recommendation:** retain the existing PINO/M80 and SQNO implementations as frozen reusable baselines. Build the unified optimization framework as a new, isolated research package; do not replace or destabilize the baselines.

## 2. Repository and reproducibility audit

### 2.1 Repository state

- The audited branch exists and points to the audited commit above.
- The repository tree contains 3,795 tracked file blobs at this snapshot.
- The repository default branch is `research/tvcg-initial-import`.
- The repository is writable by the connected GitHub integration.
- The latest commit's combined status contains no reported status checks, and the Actions runs endpoint returned zero runs. This is **not evidence of a passing CI run**.

### 2.2 Reproducibility blockers

1. **M69 and M83 are external dependencies.** The M80 data specification describes the formal M69/M72 corpus, but the actual manifest, audit receipt, NPZ geometry/label files and M83 source corpus are not present in this repository tree. Large NPZ/mesh/model artifacts are ignored by `.gitignore`.
2. **M90 validators are not portable as committed.** The M90.2 validator hardcodes an absolute local path to `/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src/task69.py` and a source hash. This is suitable for the originating machine but cannot run unchanged in a generic GitHub runner.
3. **QuadrANN submodule metadata is incomplete.** `.gitmodules` declares libigl only, while the tree also contains a QuadrANN gitlink. A clean checkout cannot assume that the QuadrANN gitlink will be initialized from a declared submodule URL.
4. **P-CRAFT source licensing remains unresolved.** The Spin-PINO validation report explicitly records pending source ownership/license clearance. This must be resolved before external redistribution or publication claims relying on source reuse.
5. **Existing CI does not automatically cover the current branch.** `.github/workflows/m80-pcraft-native.yml` is configured for pushes to the historical M80 branch, pull requests to `main`, or manual dispatch. It does not declare a push trigger for `research/tvcg-initial-import`. No CI result was found for the audited HEAD.

## 3. Detailed code audit

### 3.1 Existing Spin-PINO

Relevant files:

- `research/NeuralMoments/models/pcraft_ufno/spin_pino/pino_backbone.py`
- `research/NeuralMoments/models/pcraft_ufno/spin_pino/spin_pino.py`
- `research/NeuralMoments/models/pcraft_ufno/spin_pino/dataset.py`
- `research/NeuralMoments/models/pcraft_ufno/spin_pino/train.py`
- `research/NeuralMoments/models/pcraft_ufno/spin_pino/equivariance.py`

**What exists**

- P-CRAFT-inspired factorized Fourier U-shaped 3D encoder.
- Four geometry channels: density, occupancy, normalized SDF and internal boundary, plus coordinate channels.
- Geometry-conditioned positive quadrature weights with support/density masking.
- Ten raw moments and an analytic Jacobian with respect to the ten quadric coefficients.
- Smooth-hinge and logistic-atom moment parameterizations.
- Rotation transforms, augmentation and covariance tests.
- CPU training/evaluation code for a restricted analytic P0 vertical-plane task.

**What it actually learns**

The current interface is effectively
[
  (phi_G,eta)longmapsto (widehat M,partial_etawidehat M),
]
where (eta) is the ten-dimensional quadric coefficient/query vector. The current dataset labels are exact continuum integrals for boxes, spheres and ellipsoids cut by strictly interior vertical planes. They are explicitly not W2/Cherchi labels and do not certify general 3D quadric Boolean derivatives.

The reported M5 logistic-atom result (three seeds) is useful evidence for this restricted task: moment relative L2 0.2060 and coefficient-Jacobian relative L2 0.3147. The same report explicitly says the model is not Newton/Cherchi certified and is not validated on the real Heart/Teapot/Noodles cases. These numbers must not be presented as accuracy on the target optimization problem.

**Gap**

There is no material design variable (alpha), no explicit rotation-task input (q), no material-density decoder (r(x;alpha)), no task-conditioned stability constraints, and no KKT training target.

### 3.2 M80 P-CRAFT native operator

Relevant files:

- `spin_pino/native_pcraft.py`
- `spin_pino/native_train.py`
- `spin_pino/neural_solver_bridge.py`
- `M80_DATASET_V2_SPEC.md`

**What exists**

- P-CRAFT spatial encoder and native RLE interval response representation.
- Coarse multiscale context with native intervals retained for local feature sampling.
- Optional GNO and Transolver++ physical-slicing blocks.
- Optional convex moment-potential head.
- Batched ten-moment predictions and automatic-differentiation Jacobians with respect to quadric coefficients.
- Strict M72 manifest/audit/hash reader, geometry-group split checks, query-count checks, resumable training, checkpoint round-trip checks and evaluation diagnostics.
- A fail-closed bridge to the native Faster evaluator. It checks geometry/topology/cache identities, model mode, finite values and moment envelopes; an explicit admission callback is required. Without it, the bridge uses the exact evaluator.

**Important distinction**

The M80 model's query variable is still the quadric coefficient vector (eta). Its `moments_and_jacobian` returns
[
  widehat M(phi_G,eta),qquad
  partial_etawidehat M.
]
This is not (
abla_alpha M), where (alpha) controls material retention. The M80 training loss currently supervises moments and derivative-valid coefficient Jacobians. Optional event metadata is parsed by the corpus reader but is not currently a training target. The trainer does not construct rotation-task KKT residuals.

The M80 data specification describes 67 qualified geometries and a 256-unique-query minimum per geometry, with expansion targets of 150 and 300+ independent parent groups. Those are corpus specifications/targets, not proof that the corresponding files are committed or that all coverage has been re-audited in this repository.

### 3.3 Native quadric moment teacher

Relevant files:

- `include/spin/QuadricMomentEvaluator.h`
- `include/spin/QuadricDiscreteMomentDifferential.h`
- `src/spin/QuadricMomentEvaluator.cpp`
- `src/spin/QuadricDiscreteMomentDifferential.cpp`

**What exists**

- Exact mesh Boolean/material-boundary moment evaluation.
- Separate derivative identities for legacy continuous implicit-shape, discrete full-rank and discrete reduced-stratum derivatives.
- Full-rank discrete differential support for supported ellipsoid/hyperboloid cases.
- A separate reduced-dimensional differential for supported rank-deficient strata.
- Fail-closed behavior when provenance, topology, source motion or derivative conditions are not satisfied.

**Gap and safety rule**

The C++ evaluator's derivative is principally with respect to quadric coefficients, not a general local material field. Its default derivative method is the legacy continuous method; selecting a discrete method is an explicit setting. Reduced-stratum Jacobians must never be promoted to an ambient 10x10 Newton Jacobian. Some authority-driven value paths deliberately report `derivativeDefined=false` until source motion is independently validated.

The value-side libigl/CGAL Boolean evaluator is not automatically the same implementation as the separate Cherchi reference teacher. Teacher identity, derivative identity and applicability must remain explicit in every label.

### 3.4 Solver bridge and optimization layer

The existing `M80ExternalMomentEvaluator` is a guarded *evaluator adapter*, not the requested neural optimization layer.

It can supply a candidate moment/Jacobian response to the native Faster callback contract after admission. It does not currently:
- optimize (alpha) as a material variable;
- construct the task-conditioned KKT system;
- update active inequality constraints;
- perform a learned Newton/trust-region optimization loop;
- differentiate through a complete constrained optimization solve.

The exact fallback is valuable and should be retained. The future optimizer must report learned-proposal time, exact correction time and final independent verification time separately.

### 3.5 Material optimization and M90 validators

The repository contains M90.2 saved-result and M90.3 sparse-topology validation code. These provide valuable independent checks for saved binary deletion results, fixed/editable support, moment consistency and represented-domain six-face connectivity. The M90.3 README explicitly limits its result to topology evidence; it does not establish optimizer convergence, manufacturing validity, dynamic stability or a successful nontrivial M83 design.

The `WORK_HANDOFF.md` also records Phase4/Phase5 material-field prototypes: explicit density, fixed/editable masks, bounded retention fractions and differentiable ten-moment task constraints. The Phase5 corpus described there has 12 synthetic 8^3 records and 60 analytic hard-cut labels; no model evaluation on those splits was recorded. These are useful components, but they are not yet integrated with the M80 native operator and do not establish single-density binary removal compliance.

## 4. Gap matrix against the target

| Target capability | Current evidence | Gap |
|---|---|---|
| Geometry conditioning (phi_G) | P-CRAFT U-FNO, four-channel geometry; M80 native RLE | Extend/compare representations, freeze geometry/frame contract and test new geometry holdouts |
| Rotation task conditioning (q) | Rotation transforms and augmentation utilities | No explicit task encoder for axis, origin, contact and task parameters in the learned operator |
| Material design (alpha) | Separate Phase4/5 density prototypes; quadric coefficient query (eta) | No unified parameterization or density map joining fixed shell, editable domain and Quadric |
| Ten physical moments (M) | Implemented for restricted analytic and M80 query tasks | Need exact labels for the unified material/task domain |
| Material derivative (
abla_alpha M) | Not implemented as the operator's target derivative | Derive from exact material map; do not confuse with (partial_eta M) |
| KKT residual (F_{m KKT}) | Existing Spin-It/Faster/SLQP code and KKT-related C++ utilities | No task-conditioned KKT construction/labels in PINO training |
| KKT Jacobian (J_F) | AD for moment-vs-quadric query Jacobian | Must derive by AD/analytic differentiation of the assembled KKT residual |
| Active-set events | Event metadata fields and native derivative fail-closed contracts | Event labels/logic are not trained or integrated into optimizer |
| Differentiable neural Newton | Guarded moment callback bridge | No unified differentiable constrained optimizer, trust region or active-set solve |
| Local correction | Density prototypes and DensiCrafter-inspired mechanisms | No constrained Quadric+LocalCorrection formulation or fair ablation |
| Exact final verification | C++ evaluator and M90 validators | External M69/M83 dependency, incomplete portable teacher/data packaging |
| Generalization and speed | Restricted M5 results and M80 training infrastructure | No verified full-task multi-geometry optimization benchmark or end-to-end speed result |

## 5. Complete neural-operator construction plan

### Phase 0 — Freeze contracts and make the baseline reproducible

1. Freeze the original Spin-It 2014, Spin-It Faster, SQNO v1.2, Spin-PINO M5 and M80 configurations/checkpoints as immutable baselines.
2. Define versioned contracts for geometry, units, coordinate frame, task, material, moment order, coefficient order, fixed shell and derivative semantics.
3. Resolve submodule metadata and P-CRAFT source licensing.
4. Make M69/M72 data and teacher dependencies available to CI through immutable, hash-verified artifacts or an explicitly configured external data mount. Never silently substitute synthetic labels.
5. Add a CPU CI workflow for the current TVCG branch; retain separate local RTX 5090 gates.

**Exit gate:** clean checkout can run the declared CPU regression suite; all external dependencies are explicit and hash-bound.

### Phase 1 — Derive the unified optimization problem

Use non-overloaded notation:

- (phi_G): geometry representation;
- (q): rotation task (axis, origin, contact and task settings);
- (alpha=(eta,psi)): material design variables, with (eta) the Quadric coefficients and optional (psi) local-correction variables.

Define disjoint fixed/protected and editable domains and
[
ho(x;phi_G,alpha)=ho_0chi_{Omega_G}(x)
left[chi_{Omega_{m fixed}}(x)+
chi_{Omega_{m editable}}(x)r(x;alpha)ight].
]
The exact partition and any additional protected regions must be specified in the contract; fixed material is included exactly once.

Define
[
min_alpha M_0(phi_G,alpha)
quad	ext{s.t.}quad
g_i(M(phi_G,alpha),q)le 0,
]
plus the original fixed-shell, single-material, support and task constraints.

Derive:
- (M), the ten raw moments;
- (
abla_alpha M);
- constraint Jacobians and active-set conditions;
- the KKT residual for regular smooth active sets;
- explicit event classes for topology changes, tangencies, rank changes and invalid derivatives.

**Exit gate:** symbolic derivation, units/frame checks and finite-difference tests on analytic fixtures; no invalid smooth KKT labels at event samples.

### Phase 2 — Exact material/physics teacher and data contract

1. Extend the exact teacher interface from ((phi_G,eta)mapsto(M,partial_eta M)) to the material variables actually optimized.
2. For pure Quadric, use a verified chain rule such as (
abla_alpha M=(partial_eta M)(partial_alphaeta)) only where the parameter map and derivative applicability are proven.
3. For local correction, provide exact cell/prism or equivalent integration of density and its derivatives.
4. Keep Cherchi, libigl/CGAL, analytic references and independent moment integration separately identified; they are not interchangeable teachers.
5. Build a versioned dataset containing geometry/task/material parameters, exact moments, material derivatives, constraints, active-set state, derivative-validity reason, event margin, provenance and hashes.
6. Split by parent geometry identity and task families; rotations/augmentations and near-duplicates must not leak across splits.

**Exit gate:** exact moment and derivative cross-checks, independent finite-difference audits, event masking and reproducible manifests.

### Phase 3 — Unified Geometry-conditioned PINO

Retain the P-CRAFT UFNO/F-FNO trunk and M80 native-interval path as the initial backbone candidates. Do not assume SDF is mandatory; compare it with occupancy/density/boundary and native interval representations.

Add shared conditioning for:
- geometry (phi_G);
- rotation task (q);
- material state (alpha), including normalized Quadric coefficients and optional local correction descriptors.

The operator should expose a coupled physical map. A preferred consistency strategy is to predict a differentiable moment/material potential and derive dependent quantities by analytic differentiation or automatic differentiation, rather than independently regressing mutually inconsistent outputs:
[
widehat{mathcal F}_	heta(phi_G,alpha,q)
ightsquigarrow
(widehat M,
abla_alphawidehat M,
widehat F_{m KKT},widehat J_F).
]
Construct (F_{m KKT}) from the predicted/derived objective gradient, constraint Jacobian, multipliers and active-set state. Derive (J_F) from the assembled residual where smooth. Use explicit event classification or masked objectives for nonsmooth cases.

Architecture ablations:
- existing M80 query-to-moment operator;
- task-conditioned Quadric-PINO;
- shared trunk with material-gradient/KKT consistency;
- alternative geometry encodings;
- optional multiscale/GNO/Transolver blocks only when they improve matched-budget results.

**Exit gate:** held-out geometry/task prediction of moments, material derivatives and KKT residuals with calibration and consistency tests.

### Phase 4 — Differentiable constrained optimization layer

Implement a safeguarded solve over (alpha):
1. initialize from the exact Spin-It Faster Quadric solution or a declared common initializer;
2. obtain (widehat F_{m KKT}) and (widehat J_F);
3. solve a damped Newton/SQP step with trust region and merit/filter acceptance;
4. update active constraints using explicit event handling;
5. periodically evaluate exact residuals and reject unsafe neural steps;
6. fall back to exact Newton/SLQP when admission fails or the event/uncertainty gate is crossed;
7. retain the best exact-feasible iterate.

Differentiate through fixed-active-set smooth solves using an implicit or unrolled formulation. Treat active-set switches and topology events as piecewise/nonsmooth operations; do not claim globally smooth differentiation through them.

**Exit gate:** matched initial states and exact evaluation budgets against exact Newton/SLQP and Spin-It Faster; report convergence, feasibility, rejected steps, exact calls and fallback rate.

### Phase 5 — Pure Quadric baseline and optional local correction

**Pure Quadric-PINO is the primary method.** It must first reproduce the exact baseline task and achieve reliable convergence.

Only then evaluate
[
r(x;alpha,psi)=r_{m quadric}(x;eta)+
delta r_{m local}(x;psi)
]
with strict bounds:
- (delta r_{m local}) is zero outside the editable region;
- fixed shell/protected regions are immutable;
- density/material policy is explicit;
- binary finalization and connectivity are independently checked;
- correction complexity and regularization are reported.

Compare Pure Quadric and Hybrid at equal task constraints, geometry, initialization, exact-teacher budget and final acceptance criteria. Keep the extension only if it provides a reproducible mass or manufacturability benefit.

### Phase 6 — Geometry/task data expansion and 5090 training

1. Re-audit the current M83 corpus and the 24-geometry TRELLIS cohort; verify actual availability, provenance, physical frames and teacher coverage.
2. Start with a geometry-disjoint 16/4/4 train/validation/holdout pilot, while also holding out task families and rotations where applicable.
3. Expand query coverage to the preregistered unique-query target; positive rescalings and equivalent queries do not count as new coverage.
4. Run at least three seeds and report per-geometry distributions, not only pooled means.
5. Scale native RLE/interval processing and chunked multiscale features to 512-native-resolution cases. Do not equate a 128-grid context encoder with a dense 512^3 learned field; explicitly report native resolution, context resolution and interval count.
6. After CPU contracts and small GPU smoke pass, conduct long RTX 5090 training with resumable checkpoints, fixed data fingerprints, peak memory, throughput and wall-clock logs.

**Exit gate:** stable multi-seed held-out training and optimization results; no test-set tuning.

### Phase 7 — Final benchmark and paper evidence

Compare, under the same geometry, rotation task, constraints and initial conditions:

- Spin-It 2014 voxel optimization;
- Spin-It Faster exact Quadric optimization;
- SQNO v1.2;
- existing Spin-PINO/M80 query-to-moment operator;
- Geometry-conditioned Quadric-PINO;
- Geometry-conditioned Quadric-PINO + Local Correction;
- exact Newton/SLQP and neural-accelerated Newton variants.

Report separately:
- proposal/model inference time;
- optimization time and exact correction calls;
- final independent oracle/FEM/dynamic verification time;
- total end-to-end time;
- mass, ten-moment error, constraint residuals, feasibility and convergence;
- geometry/task generalization, event failure rate and fallback rate;
- parameter count, memory and resolution.

The primary speed claim must be accompanied by end-to-end cost and exact verification cost; a proposal-only speedup must never be presented as total system speedup.

## 6. Immediate implementation order

The first implementation increments should be:

1. Add a versioned mathematical/units/frame/material contract and analytic fixtures.
2. Add exact material-derivative teacher adapter tests for the supported pure-Quadric branch.
3. Add task/material conditioning to a copy/adapter of M80 without changing the frozen M80 baseline.
4. Construct KKT residuals from the common physical map and test derivatives on smooth fixtures.
5. Add the safeguarded differentiable optimizer layer and exact fallback.
6. Only after Pure Quadric passes, add the local-correction branch and its ablations.

The previously proposed `quadric_pino/` sampler/teacher/trainer directory is **not present** in the audited commit. It should be created only after the Phase 1 contract defines the correct variables and derivative semantics; otherwise it risks sampling the wrong query space.

## 7. Current decision and status

- **Keep:** P-CRAFT PINO/F-FNO, M80 native interval encoder, SQNO v1.2, C++ Quadric evaluator, M69/M90 independent verification.
- **Reposition:** SDF as optional geometry representation; DensiCrafter as optional local material mechanism; Neural Newton as the constrained optimization layer.
- **Do not claim yet:** general geometry-to-optimal-material mapping, task-conditioned KKT learning, material-variable Jacobian accuracy, neural Newton convergence, 512^3 end-to-end training, manufacturing validity or real-case speedup.
- **Next engineering milestone:** freeze Phase 1 contracts and build the exact material-derivative teacher adapter for the pure-Quadric branch.

