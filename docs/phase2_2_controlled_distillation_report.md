# Phase 2.2 — Three-seed controlled synthetic distillation comparison

**Experiment date:** 2026-11-17  
**Protocol:** 3 seeds × 3 conditioning architectures × 60 optimizer steps  
**Geometries:** 2 deterministic synthetic native-interval geometries  
**Queries:** 8 per task/geometry  
**Train task IDs:** 0–9  
**Held-out task IDs:** 10–13  
**Backbone:** frozen, independently copied M80 convex-potential operator  
**Optimizer:** Adam, learning rate 0.002  
**Teacher:** fixed deterministic nonlinear task–geometry latent interaction  
**GitHub Actions:** run 44111792099 (success)

## 1. Purpose and interpretation

This experiment answers a narrow question: under a controlled synthetic task-conditioning distillation problem, how well do FiLM, Conditional Normalization and Cross-Attention adapters approximate the same deterministic task/geometry-conditioned M80 response?

It is **not** an M69 physical-oracle experiment. The synthetic teacher is a deliberately constructed nonlinear latent interaction. It does not represent exact material moments, M69 constraint labels, real geometry, or a validated physical law. Therefore these results must not be used as evidence of physical accuracy or real-geometry generalization.

## 2. Controlled protocol

For each seed (22021, 22022, 22023):
- initialize one small M80 convex-potential backbone;
- create two deterministic synthetic native-interval geometry fixtures;
- derive 14 task-feature vectors by controlled variation around the M69 reference profile;
- use task IDs 0–9 for training and 10–13 for held-out validation;
- use the same eight normalized Quadric query vectors for all architectures;
- freeze each copied M80 backbone and train only the task-conditioning adapter;
- run 60 Adam steps per architecture at learning rate 0.002;
- measure held-out Moment RMSE and held-out query-Jacobian RMSE against the same fixed synthetic teacher.

All variants are parameter-matched by selecting hidden width; no dummy parameters are inserted.

## 3. Parameter budget

| Architecture | Hidden width | Trainable parameters | Gap to FiLM budget |
|---|---:|---:|---:|
| FiLM | 128 | 6,222 | 0.00% |
| Conditional Normalization | 128 | 6,238 | 0.26% |
| Cross-Attention | 136 | 6,342 | 1.93% |

The maximum adapter budget gap is 1.93%.

## 4. Three-seed results

Values below are mean ± sample standard deviation across the three seeds.

| Architecture | Held-out Moment RMSE | Held-out query-Jacobian RMSE |
|---|---:|---:|
| FiLM | 1.1693e-5 ± 2.9139e-6 | 1.1998e-5 ± 3.0262e-6 |
| Conditional Normalization | 1.1274e-5 ± 2.6508e-6 | 1.1663e-5 ± 2.8674e-6 |
| Cross-Attention | 1.2686e-5 ± 3.8024e-6 | 1.2897e-5 ± 3.8203e-6 |

Conditional Normalization has the lowest mean error in this particular synthetic distillation task. Its mean advantage over FiLM is small relative to the across-seed standard deviation. Cross-Attention is worse on this small task/geometry setup. These data do **not** establish a statistically reliable architecture winner.

The complete per-seed records and protocol are stored in:
`research/NeuralMoments/results/phase2_2_controlled_distillation_2026-11-17.json`.

## 5. Engineering interpretation

1. Conditional Normalization is a promising next default candidate because it is nearly parameter-identical to FiLM and achieved the lowest mean errors in this synthetic test.
2. FiLM remains the simpler baseline and should not be removed.
3. Cross-Attention is not justified as the default by this experiment. It may still help with richer multi-geometry context, but that hypothesis is untested.
4. The observed differences are small compared with seed variation. More steps, more synthetic task families and more varied analytic geometry fixtures are needed before a selection.
5. The query-Jacobian metric is a derivative of the learned M80 response with respect to query coefficients. It is not a material-design Jacobian unless the Phase 2.1 mapping contract is separately satisfied.

## 6. CI verification

GitHub Actions run 44111792099 completed successfully. It ran:
- the original task-conditioned M80 tests;
- the three-architecture functional/parameter-budget tests;
- a controlled-distillation smoke test;
- the full three-seed, 60-step synthetic distillation experiment;
- upload of the machine-readable results artifact.

No RTX 5090 long training was started. The run used GitHub-hosted CPU resources.

## 7. Decision and next step

**Decision:** keep FiLM and Conditional Normalization as co-primary candidates. Do not promote Cross-Attention to the default. Do not claim a definitive winner.

The next meaningful research step is Phase 2.3: construct a shared differentiable physical potential and derive Moment/constraint/KKT differentials consistently. The architecture comparison can continue later with stronger analytic teacher families, but it should not delay the KKT formulation.

Before any physical optimization claim, the exact M69 material/query semantics and independent oracle derivatives still need to be matched.
