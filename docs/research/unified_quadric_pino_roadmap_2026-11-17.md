# Unified Quadric-PINO: complete implementation and validation roadmap
Date: 2026-11-17
Target repository: volcano-code/TVCG
Core method: Geometry-conditioned Quadric Physics-Informed Neural Optimization

## 1. Scientific objective

For a registered original geometry Omega_G, rotation task q, fixed shell/protected region, and one physical material, solve for the minimum-mass admissible internal material:

minimize mass(rho)
subject to the frozen rotation-stability and material constraints.

Use rho(x) = rho0 * chi_OmegaG(x) * r(x; alpha), with fixed/protected regions immutable. The Pure Quadric path is the core baseline. Local correction is an optional extension, not a mandatory stage.

The research claim is acceleration of the same optimization problem solved by Spin-It Faster, not a change in task, material policy, or acceptance threshold. The theorem/assumptions that justify quadric optimality must be stated per task; adding learned geometry conditioning does not extend those assumptions automatically.

## 2. Design principles

1. One optimization problem, one shared geometry-conditioned physical operator, one safeguarded optimization framework.
2. Reuse SpinPINO, M80 PcraftNativeMomentOperator, SQNO v1.2, C++ Faster solver, and exact evaluator where their contracts match.
3. Keep Spin-It and Spin-It Faster as immutable, reproducible baselines.
4. Treat SDF as one candidate geometry representation alongside occupancy, native RLE, surface/point features, or hybrid encodings. Select by controlled ablation.
5. Derive mutually consistent physical outputs. Do not independently regress M, dM/dalpha, F, and J_F without integrability and derivative-consistency checks.
6. Preserve exact evaluator fallback, typed derivative applicability, topology identities, and independent M69 verification.
7. Separate smooth training relaxations from final hard single-density material acceptance.
8. Never label nonsmooth active-set transitions as ordinary smooth KKT samples.

## 3. Mathematical contract to freeze before model work

### 3.1 Inputs and variables

- Geometry phi_G: original outer geometry, coordinate frame, units, occupancy/boundary representation, and immutable reference hash.
- Task q: rotation axis, axis origin, contact point/region, task family, target frame, and every task-specific parameter.
- Pure Quadric alpha: normalized ten-coefficient projective parameter on S9, with explicit sign/inside convention and target-frame transform.
- Fixed/protected material: immutable support and its ten moments, computed once and added exactly once.
- Local correction psi: introduced only in the hybrid ablation, confined to a preregistered editable support and unable to add material outside the original design domain.

### 3.2 Material semantics

For the primary single-density path, final material is binary and removal-only. Fixed shell and protected regions are always retained. A smooth indicator may be used for differentiable training, but its outputs are not final acceptance artifacts. Hard export must be independently integrated and checked.

The optional hybrid parameterization may be expressed as a quadric baseline plus a bounded local correction. Its precise map must guarantee support containment, no material addition, immutable fixed regions, and a deterministic hard-export rule. Any relaxation that permits fractional density must be labeled separately and cannot be passed off as the primary single-density result.

### 3.3 Physical map and optimality system

Define one shared operator conditioned on geometry and task. The preferred consistency-first design is a shared latent/potential representation whose moment output and coefficient/material derivatives are obtained through analytic differentiation or automatic differentiation where the mathematical map is differentiable.

The coupled outputs are:
- M(phi_G, alpha, q): ten physical raw moments;
- dM/dalpha: derivative in the declared tangent/reduced coordinates;
- F_KKT: task-specific stationarity, feasibility, complementarity, and normalization residuals;
- J_F: derivative of that same residual with respect to the active optimization variables.

F_KKT and J_F should be assembled from the same M and dM/dalpha representation. If auxiliary learned heads are added, train them as consistency-checked proposals, not independent authorities. For the projective quadric variable, enforce the S9 gauge/tangent condition and the exact coefficient/moment dual pairing.

### 3.4 KKT and nonsmooth events

For a smooth active set, construct labels only when the required regularity checks pass: finite derivatives, stable active set, LICQ or a documented alternative, strict complementarity where required, and a second-order sufficient condition for a local minimum. Store active-set identity and event margin.

At topology changes, contact switches, rank-deficient quadric strata, repeated eigenvalues, or other nonsmooth events:
- mark the ordinary smooth KKT derivative invalid;
- preserve valid moment-value supervision where available;
- route to a reduced-stratum derivative, event-specific model, or exact solver;
- do not fabricate a full ten-dimensional Jacobian.

## 4. Implementation phases and deliverables

### Phase 0 — Repository and scientific contract freeze
Deliverables:
- audited source map and protected-file manifest;
- versioned GeometryTaskMaterialContract;
- exact coordinate, units, density, shell, support, moment-order, coefficient-order, and sign conventions;
- explicit mapping from each target task to the Spin-It Faster theorem assumptions and M69 constraints;
- benchmark policy defining the common primary objective and separately reported original-paper objectives.

Exit gate: one test fixture passes geometry/task/material contract validation without implicit defaults or guessed physical values.

### Phase 1 — Pure Quadric material parameterization
Deliverables:
- alpha on normalized S9 with gauge-aware tangent updates;
- exact mapping from alpha to the hard quadric material region;
- immutable fixed-shell/protected masks and fixed moments;
- independent hard-material export and moment recomputation;
- tests for sign convention, scale invariance, coordinate transforms, fixed-shell invariance, and support containment.

Exit gate: analytic simple geometries pass exact hard-material value checks; no density relaxation is misreported as final material.

### Phase 2 — Exact teacher and typed derivative adapter
Deliverables:
- C++ evaluator adapter for geometry + alpha + task-frame requests;
- value labels for ten moments;
- derivative labels with explicit method, dimension, coordinate basis, topology identity, Boolean backend, and applicability;
- event/topology/rank diagnostics and error bounds;
- repeatability and independent quadrature/analytic cross-checks;
- dataset writer with geometry/task/solver/source hashes.

Exit gate: repeated labels are deterministic within declared tolerances; invalid derivatives are masked, never coerced into full-rank labels.

### Phase 3 — Shared Geometry-conditioned Quadric PINO
Deliverables:
- shared geometry encoder with ablations for native RLE, occupancy, SDF, and hybrid geometry features;
- quadric/task query conditioning;
- shared potential or consistency-first moment head;
- outputs M and dM/dalpha with derivative consistency;
- derived F_KKT and J_F through the same differentiable physics layer;
- rotation-frame and SO(3) equivariance tests;
- calibration/uncertainty estimates and fail-closed applicability output.

Architecture comparisons:
A. Existing SQNO v1.2 / SpinPINO baseline.
B. M80 PcraftNativeMomentOperator baseline.
C. Shared Geometry-conditioned Quadric PINO.
D. Potential-head versus direct moment-head with integrability regularization.
E. Geometry representation ablation: RLE, occupancy, SDF, hybrid.

Exit gate: held-out geometry and query tests meet preregistered moment and derivative error thresholds; finite differences, equivariance, and physical identities pass.

### Phase 4 — Differentiable constrained optimization layer
Deliverables:
- equality/inequality constrained alpha solver;
- active-set-aware KKT residual assembly;
- damped Newton or trust-region updates with tangent projection;
- exact residual evaluation at every accepted candidate;
- line search, rejection, and exact fallback;
- implicit differentiation only on verified regular active sets;
- exact Newton and native Spin-It Faster solver baselines.

Exit gate: neural-assisted solves converge to the same acceptance region as exact solves on the pilot, with no increase in false-feasible results. Report solver calls, exact fallbacks, iterations, and wall time separately.

### Phase 5 — Optional Quadric + Local Material Correction
Deliverables:
- bounded correction variable psi restricted to an explicit editable support;
- no-added-material and fixed-shell invariants;
- deterministic binary hard-export and connectivity checks;
- regularization ablations for smoothness/boundary behavior;
- direct comparison against Pure Quadric at equal task, accuracy, and compute budget.

Exit gate: retain this extension only if it improves feasible mass or manufacturing quality without invalidating the physical task or overwhelming the speed advantage.

### Phase 6 — Geometry/task dataset admission
Deliverables:
- task-ready TRELLIS ordinary-object cases with registered axis, origin, contact, units, density, fixed shell, protected region, and initial-solid semantics;
- exact mesh topology and representation-error admission;
- feasible nontrivial material states and typed Oracle labels;
- source-family-disjoint Train/Development/Holdout manifests;
- leakage and provenance audits.

Current blocker from the latest handoff: 24 ordinary TRELLIS objects exist, but TaskSpec-ready cases, physically admitted cases, feasible material states, and Oracle labels are all zero. Five surface-gateway passes do not equal physical admission. No formal training should be started from these objects yet.

Exit gate: a minimum pilot of independently admitted geometries and nontrivial feasible material states, with no use of diagnostic-only assets as formal samples.

### Phase 7 — Training and scaling
Deliverables:
- small CPU/GPU smoke, then bounded pilot, then long training only after Phase 2–6 gates;
- RTX 5090 single-GPU training profile;
- native sparse/RLE and high-resolution scaling path, with explicit memory and discretization error budgets;
- at least three seeds and geometry-family holdouts;
- checkpoint, source, dataset, environment, and command fingerprints.

512-cubed requirement: distinguish a dense 512-cubed representation from sparse native intervals with a coarse P-CRAFT context grid. Report actual representation and peak memory. Do not call a 32/64/128 context grid a dense 512-cubed neural field.

Exit gate: stable training, no split leakage, reproducible checkpoints, and measured generalization on unseen geometry families.

### Phase 8 — Full evaluation and paper evidence
Compare under identical input geometry, task, material, shell, precision, and final acceptance:
- Spin-It 2014;
- Spin-It Faster 2024;
- exact Faster/Newton;
- SQNO v1.2 / SpinPINO;
- M80 operator without optimization layer;
- Geometry-conditioned Quadric PINO;
- Geometry-conditioned Quadric PINO + differentiable optimizer;
- optional local-correction hybrid.

Report:
- final mass and relative mass reduction;
- every physical constraint and independent M69 pass/fail;
- moment/Jacobian/KKT errors;
- exact-evaluator calls and fallback rates;
- optimizer iterations and wall-clock breakdown;
- geometry/query generalization;
- memory and resolution;
- connectivity, hard-export, and fabrication checks where claimed;
- ablations and failure taxonomy.

Speed claim must include neural inference, encoding, optimization, exact correction, and fallback. A secondary speed number may exclude final independent verification only if the same accounting convention is applied to every method and the excluded verification time is separately reported.

## 5. Required experiment gates

| Gate | Required evidence | Stop condition |
|---|---|---|
| G0 Contract | Fully registered geometry/task/material/units | Any unknown physical input or ambiguous material region |
| G1 Parameterization | Exact hard region and fixed-shell tests | Sign/frame/support mismatch |
| G2 Teacher | Repeated exact moments and typed derivatives | Unstable labels or invalid derivative typing |
| G3 Operator | Held-out M and derivative accuracy | Failed physical identities or equivariance |
| G4 Optimizer | Exact residual convergence and no false-feasible states | Neural step accepted without exact check |
| G5 Hybrid | Equal-budget ablation | No material/manufacturing gain |
| G6 Data | Independent admitted geometries and labels | Diagnostic-only or unregistered objects |
| G7 Scale | Reproducible single-GPU training and 512+ representation evidence | Memory/precision/error budget not met |
| G8 Paper | Fair baselines and independent acceptance | Different tasks or tolerances across methods |

## 6. Immediate next actions

1. Resolve the source-of-truth mismatch: archive the current local Spin_test2 source/control state into TVCG without overwriting protected work; include the missing research_control documents and the latest dataset handoff manifests as versioned artifacts.
2. Create the versioned geometry/task/material contract and map it to Spin-It Faster and M69.
3. Implement and test the Pure Quadric hard-material adapter before adding any new neural head.
4. Implement the C++ Exact Teacher Adapter and typed label schema.
5. Connect the existing SpinPINO/M80 operator to the contract, retaining the old models as immutable baselines.
6. Derive the task KKT system and active-set labels from the same physical moment map.
7. Implement the safeguarded differentiable optimizer only after the teacher and derivative gates pass.
8. Admit TRELLIS geometries and construct the first real pilot before long training.
9. Add a branch-targeted CI workflow and run CPU contracts on each commit; use the local RTX 5090 only for GPU training and high-resolution tests that are not available in the cloud runner.

## 7. Implementation policy

Each increment must record:
- branch and parent commit;
- changed files and content hashes;
- tests and exact commands;
- measured CPU/GPU results, with hardware/runtime identity;
- training steps and checkpoint identity, or an explicit zero;
- dataset split and fingerprint;
- exact limitations and next gate.

No code path may silently substitute smooth labels for hard labels, reduced-stratum derivatives for ambient Jacobians, diagnostic assets for admitted training cases, or predicted feasibility for independent exact acceptance.
