# Phase 2.7.1 — Neural Operator Architecture & CCF-A Readiness Audit

Audit date: 2026-11-18
Repository: volcano-code/TVCG
Branch: research/phase2-6-r3-m69-hybrid-ablation
Audited HEAD: 29c7104a92d06241897afb5210a63c4b538da92e

## Executive assessment

The repository contains substantial research prototypes and several functioning software paths, but it does not yet contain one formally unified, fully trained, production-ready UG-PINO checkpoint. The most defensible current foundation is the M80 task-conditioned P-CRAFT native moment operator, with SQNO/QuadrANN as a geometry-once/query-many candidate, optional GNO and Transolver branches, optional convex moment potential, and a separate M69 KKT / safeguarded Newton / exact-oracle bridge.

The system should be developed as a shared geometry/task representation with jointly trained physics heads and an outer safeguarded optimization loop. It should not be implemented as a blind serial stack of every candidate architecture. Optional branches must remain switchable until matched-budget ablations demonstrate value.

A “trainable today” engineering milestone is achievable as a reproducible pilot entry point using existing formal corpus contracts, but a “perfect model” or final-quality model is not an honest same-day claim. Current M80 status explicitly records long training not started, formal 256-query corpus not confirmed, multiscale GPU preflight pending, and solver end-to-end validation false.

## 1. Audited implementation inventory

### 1.1 Primary Spin-PINO
- `research/NeuralMoments/models/pcraft_ufno/spin_pino/spin_pino.py`
- `pino_backbone.py`, `train.py`, `dataset.py`, `equivariance.py`, `loss_conditioning.py`
- P-CRAFT UFNO backbone with geometry-conditioned quadrature/moment head.
- Existing analytic training path predicts moments and Jacobians; M5 logistic-atom report records a three-seed analytic benchmark (seeds 131/137/139, 120 steps).
- M5 reported moment relative L2 0.20602 and Jacobian relative L2 0.31466 on the restricted analytic box+sphere / held-out ellipsoid task family. Labels are analytic continuum vertical-plane cuts, not Cherchi/M69 physical labels. It is not Newton-admitted and does not establish arbitrary 10D quadric/CAD event generalization.
- `spin_pino/STATUS.json` explicitly says trained_on_spin_labels=false, Cherchi labels admitted=0, Newton derivative admitted=false, real Spin end-to-end validated=false.

### 1.2 M80 production-oriented operator path
- `models/pcraft_ufno/native_pcraft.py`, `native_train.py`, `M80_ARCHITECTURE_CONTRACT.md`, `M80_EXPERIMENT_PROTOCOL.md`, `M80_RELEASE_GATE.md`, `M80_STATUS.json`.
- M80.1 single-scale P-CRAFT + native M72 RLE intervals is the permanent control.
- M80.2 GNO and M80.3 Transolver are optional branches.
- M80.4 shared-weight multi-resolution P-CRAFT context pyramid (32/64/128) is opt-in.
- M80.5 convex homogeneous moment potential is opt-in; bounded odd head remains default.
- Fixed and editable moments are separated; native moment order is mass,x,y,z,xx,yy,zz,yz,xz,xy; coefficient order is 1,x,y,z,xx,yy,zz,2yz,2xz,2xy.
- Native training reader requires a frozen formal M69/M72 manifest and audit receipt, fingerprints and per-file hashes, source-group split isolation, and a qualified train/val/test split.
- M80 status (dated 2026-11-13) says: 104 registered / 67 qualified; split groups 50/7/10; formal 256-query corpus not confirmed; multiscale GPU preflight false; long training not started; native solver bridge end-to-end false; real geometry Newton replay false; exact terminal geometry audit false; calibrated UQ model unavailable.
- The 512-resolution preflight was a one-step random-initialization resource/graph check, not trained-quality evidence. Its Jacobian scaled MSE was 220.06, indicating that this preflight is not evidence of a useful trained Jacobian.
- Single-scale Large preflights passed. GNO/Transolver factorial preflights were resource checks only. Multiscale/GNO GPU preflights remain pending.

### 1.3 SQNO / QuadrANN
- `candidate_projects/quadrann_reproduction/spin_adapter/spin_quadrann_v12.py`
- `training_v12/train.py`, dataset/trainer/checkpoint/normalization modules.
- Implements geometry encoding once, then batched coefficient queries, with P-CRAFT and QuadrANN/GNN features fused in a geometry cache.
- This is a strong candidate for the query-many geometry operator role, but it remains a separately trained model path and is not yet merged into a shared M80 production checkpoint.
- Training entry point uses M83Corpus and a preset; formal corpus and label admission still need auditing before production promotion.

### 1.4 M69 KKT / Neural Newton / Phase 2.6
- `phase2_3_differentiable_m69_kkt.py`, `phase2_4_neural_newton.py`, `phase2_5_end_to_end_solver.py`, `phase2_6_hybrid_unified_optimization.py`, `phase2_6_unified_operator_integration.py`.
- Actual M80 wrapper and P-CRAFT native moment operator are wired through S9, M69 constraints, KKT and safeguarded Newton proposal code.
- Exact candidate acceptance is a separate callback with fallback.
- The Phase 2.6 document explicitly states CPU tests use a randomly initialized small M80 model and synthetic native-interval geometry. Production exact material integration for arbitrary 512^3 quadric-clipped designs is not complete. No trained-checkpoint accuracy, real physical feasibility or speedup is established.
- Phase 2.5 results are a synthetic convex QP integration test using a deterministic perturbed quadratic surrogate, not a trained neural model or M69 physical task. It reports more exact merit evaluations for the surrogate proposal than exact Newton (11.33 vs 4.00 mean), so it does not demonstrate acceleration.

### 1.5 R3 native exact replay / independent verification
- The R3 line contains native build preflight, request assembly, native replay, result acceptance, independent tetra moments, physical verification, geometry integrity and double-replay evidence audit.
- These establish software contracts and evidence schemas. They do not establish a real TRELLIS/M69 native replay or neural acceleration.

## 2. Architecture decision

### 2.1 Unified system, not a serial chain
Use one shared geometry/task representation, then parallel/interactive operator branches and joint heads:
- Shared geometry encoder: P-CRAFT/M80 native interval representation (primary).
- Query-many branch: SQNO/QuadrANN candidate operating on the shared geometry cache or a compatible adapter.
- Local/topological branch: GNO, opt-in.
- Global physical interaction branch: Transolver, opt-in.
- Physical response parameterization: bounded odd baseline; convex potential as a controlled candidate.
- Joint outputs: ten mass moments, coefficient Jacobian (preferably AD-consistent with the same moment function), derivative-valid/event status, uncertainty, optional response features.
- Outer optimizer: M69 KKT / safeguarded Neural Newton.
- Exact authority: native exact evaluator, exact acceptance and independent terminal verification.

Do not force all optional branches into the final model. Promote each only on matched data, seeds, query budget and wall-clock ablations.

### 2.2 Mathematical contracts that must be frozen
1. Moment and coefficient ordering and factor-of-two cross terms.
2. Fixed/editable decomposition: total moments = fixed + editable; Jacobian only from editable contribution.
3. Coefficient-to-model and target-frame moment transform chain rule.
4. Projective coefficient normalization / sign complement semantics.
5. SO(3) transformation of geometry, moments and Jacobian.
6. Smooth derivative validity vs non-smooth Boolean/topology events; event-adjacent derivatives must be withheld.
7. Moment/Jacobian consistency: J must be derivative of the same predicted moment function, with independent teacher/finite-difference checks.
8. Units, density, coordinate frame and immutable geometry/cache identity.
9. Exact solver remains authoritative; uncertainty/admission gate must fail closed.

## 3. CCF-A readiness assessment

### Current position
The project has enough engineering breadth to support a serious paper project, but the current evidence is not yet enough for a CCF-A submission. The present strongest candidate contribution is not “combining U-FNO, SQNO and Transolver”; it is a derivative-consistent, geometry-conditioned moment operator with reliable event-aware Newton coupling that measurably reduces expensive exact solver calls while preserving terminal physical feasibility.

### Candidate contributions
A. Geometry-once / query-many physical moment operator for quadric-based material optimization.
B. Derivative-consistent moment/Jacobian learning with explicit event validity and independent derivative audits.
C. Reliability-aware neural-Newton hybrid with exact correction/fallback and a measurable reduction in exact evaluations.

These become publishable only if the full method has a clearly novel technical mechanism and controlled evidence against strong baselines. Architectural assembly alone is insufficient.

### Missing decisive evidence
- A trained M80/UG-PINO checkpoint on admitted physical labels.
- Formal, diverse, source-group-separated corpus with adequate query coverage and derivative-valid labels.
- Geometry OOD, parameter OOD, SO(3), near-event and topology stress tests.
- Three or more seeds for promotion and uncertainty calibration.
- Matched-budget ablations of P-CRAFT, SQNO, GNO, Transolver and convex potential.
- Exact native parity and real geometry Newton replay.
- Same-accuracy and same-budget comparisons against Spin-It Faster / exact Newton.
- End-to-end wall-clock including preprocessing, encoding, neural inference, exact corrections, restarts and terminal verification.
- Independent physical/geometry verification on real cases.
- Ablation proving each novel mechanism matters.

### Venue positioning
- SIGGRAPH/TOG: strongest fit if the contribution is a new geometry/material optimization method with compelling 3D design/fabrication results.
- NeurIPS/ICML: strongest fit if the contribution is a broadly useful operator-learning or derivative-consistent reliable hybrid optimization method, validated beyond one application.
- AAAI: plausible if the main story is general physical reasoning/optimization methodology, not merely an application-specific acceleration wrapper.

No acceptance guarantee is possible. A CCF-A target is reasonable as a research ambition, but submission readiness is currently not established.

## 4. Same-day deliverable: trainable unified baseline

The same-day target should be a reproducible training-capable baseline, not a perfect model:
- Reuse M80.1 P-CRAFT as the stable encoder/operator baseline.
- Reuse its same-function AD Jacobian path.
- Keep GNO, Transolver, multiscale context and convex potential behind config flags.
- Add/verify a unified trainer that loads only a frozen, audited formal corpus, performs geometry-group splits, trains Moment + Jacobian with explicit scaling and event-valid masks, validates P50/P95/P99/max tails, saves best and resumable checkpoints with source/data/config hashes.
- Provide a CPU synthetic smoke test for wiring only, then a local RTX 5090 GPU preflight and short pilot. Synthetic smoke must never be presented as physical model quality.
- Do not launch multi-day 512^3 training until corpus, derivative labels, GPU memory/throughput and 200–500 step pilot gates pass.

## 5. Phase 2.7 implementation sequence

### Phase 2.7.1 — Audit (this report)
Complete the code inventory and freeze the architecture contract.

### Phase 2.7.2 — Unified trainable baseline
Create a single configuration/CLI around the M80 native P-CRAFT operator and joint Moment/Jacobian training, preserving the existing M80.1 control and checkpoint compatibility. Optional branches are independently switchable. No forced fusion until ablation.

### Phase 2.7.3 — Mathematical gate
Add automated tests for finite-difference Jacobian, chain rule, fixed/editable moment split, q/-q complement, positive scale invariance, SO(3) transform, event masking, and stale cache rejection.

### Phase 2.7.4 — Training gate
Formal manifest/audit validation, train/val/test group isolation, best checkpoint, exact resume, deterministic seeds, AMP policy and tail metrics.

### Phase 2.7.5 — RTX 5090 pilot
GPU preflight, 200–500 step pilot, at least three seeds for promotion. Compare baseline and optional branches under matched budgets.

### Phase 2.7.6 — Architecture freeze
Promote only branches with demonstrated held-out or downstream benefit. Freeze a single UG-PINO configuration before long training.

### Phase 3.1 — Large training
Only after all release gates, extend 128^3 -> 256^3 -> 512^3 and then long training.

### Phase 3.2 — Neural-Newton hybrid evidence
Run exact-vs-hybrid matched-accuracy and matched-budget studies with independent terminal verification.

## 6. Exit criteria for “trainable today”
The model is called trainable only when:
- CLI can load a valid audited dataset;
- one optimizer step updates the intended trainable parameters;
- Moment and Jacobian losses are finite and gradients are finite;
- validation executes without leakage;
- checkpoint save/load and exact resume work;
- source/data/config fingerprints are recorded;
- local GPU preflight passes.

This milestone is not equivalent to final-quality training, 512^3 success, real TRELLIS generalization or CCF-A readiness.
