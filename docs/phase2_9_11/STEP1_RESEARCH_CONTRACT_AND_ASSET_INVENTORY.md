# Phase 2.9.11 — Step 1: Research Contract and Asset Inventory

Date: 2026-11-20
Repository: `volcano-code/TVCG`
Branch: `research/phase2-6-r3-m69-hybrid-ablation`
Baseline used for the prior Phase 2.9.5 software report: `696c753701ae3bf475530297ce599eaf79c4a8db`

## 1. Purpose and scope

This document freezes the intended system boundary before further model changes. It is an inventory of evidence visible in the repository and handoff records, not a claim that every end-to-end capability has been executed.

The research objective remains:

1. Use the existing Spin-It reproduction to construct an initial gyroscope from an input object and its fixed/design material specification.
2. Use the existing Spin-It Faster-compatible quadratic-surface optimization and exact moment/Jacobian evaluation as the optimization reference.
3. Build a physics-informed neural operator that amortizes repeated moment and parameter-derivative evaluations over gyroscope geometry/material states and quadric queries.
4. Feed neural predictions/proposals into the optimizer only as proposals; exact evaluation, acceptance, rejection, fallback and final material verification remain authoritative.
5. Evaluate whether the complete hybrid reduces end-to-end optimization cost without weakening the original task constraints or final acceptance.

The network is not an OBJ-to-gyroscope generator and does not directly replace Spin-It. The neural model is not allowed to accept its own physical proposals.

## 2. Frozen responsibility boundaries

| System | Responsibility | Boundary |
|---|---|---|
| Spin-It reproduction | Input geometry processing and initial gyroscope/material state | Preserve the existing reproduction; do not reimplement it in the neural model |
| Faster-compatible exact solver/teacher | Quadric queries, exact/reference moments and derivatives, optimizer state and correction | Must be tied to the same geometry/material/query contract |
| Spin Neural Moments | Predict moments, raw-input coefficient Jacobians, uncertainty and event-risk outputs | Prediction/proposal only; no authoritative physical acceptance |
| Work data pipeline | Source geometry selection, provenance/topology audit, initial-state generation, teacher labels and trajectories | Must preserve source-family splits and immutable label provenance |
| Independent verifier | Recompute final material properties and task constraints | Must not rely on neural predictions as ground truth |

Important qualification: repository evidence establishes substantial Spin-It/Faster-related assets and a local teacher pipeline, but this document does not assert that the complete original papers' implementations have been independently reproduced in every feature. The exact runnable entry points and task-level equivalence remain an audit item.

## 3. Current model asset inventory

Evidence reviewed at the Phase 2.9.5 baseline includes:

| Asset | Repository evidence | Status supported by evidence | Still unverified |
|---|---|---|---|
| P-CRAFT native geometry encoder | `spin_pino/native_pcraft.py` | Implemented; used by unified wrapper | Formal-corpus convergence and target-resolution throughput |
| M80 moment branch | `spin_adapter/unified_m80_sqno.py` plus native operator | Integrated in unified wrapper | Best head on independent geometries |
| SQNO query-many branch | `spin_quadrann_v12.py`, unified wrapper | Integrated with external shared P-CRAFT features | Real-corpus accuracy and cost |
| Shared geometry cache | `UnifiedGeometryCache`, encode/evaluate methods | Implemented with ownership/version checks | Production data loader integration |
| Global scalar fusion | unified wrapper | Implemented baseline | Whether it is best |
| Query-conditioned fusion | unified wrapper | Implemented; Jacobian includes gate derivative | Real-data benefit |
| Geometry-physics cross-attention | `geometry_physics_attention.py` | Implemented, opt-in | Real-data benefit and production resource profile |
| Cross-scale attention | `cross_scale_attention.py` | Implemented, opt-in | Real-data benefit |
| GNO / Transolver-style physics slicing | native P-CRAFT and architecture docs | Existing implementation/configuration paths documented | Which exact full configuration is stable and beneficial |
| Moment/Jacobian UQ | Phase 2.9 docs and native model | Opt-in head and evaluation primitives implemented | Calibration on held-out geometry families |
| Event validity | Phase 2.9 docs and native model | Optional teacher-labelled head/loss implemented | Coverage and calibration on audited event labels |
| Mixed full/restricted derivative supervision | Phase 2.9.5 implementation and handoff | Software path implemented and regression-tested | Real corpus CUDA update and resumed training |
| Fixed/PCGrad training | unified training/state utilities | Software mechanisms implemented | Real-data convergence and compute-normalized comparison |
| Neural Newton / optimizer proposal | planning/interface work | Design target | Complete executable integration and exact acceptance loop |

### Configuration policy

A feature being implemented does not mean it is enabled in every model instance. In the current unified wrapper, geometry cross-attention and cross-scale attention default to disabled; interaction variants and fusion modes are explicit constructor choices. Preserve this capability for controlled configuration and later ablations.

Do not silently change every optional module to enabled by default in this step. First define one explicit, named `full_v1_candidate` configuration in a later implementation step, validate its shape/gradient/memory contracts, and retain a minimal reference configuration. The full candidate is not declared scientifically superior until measured.

## 4. Current data evidence and boundary

The Phase 2.9.5 handoff records a frozen pilot snapshot of 19 geometries and 1,251 queries:
- 719 full Jacobian labels;
- 323 restricted 3-direction labels;
- 208 restricted 8-direction labels;
- 1 moments-only row.

This is a geometry/query supervision corpus. It is not automatically an optimization-trajectory corpus, and its rows are not independent geometry samples.

The handoff reports coefficient-coordinate and finite-difference checks. It also reports a PLY teacher / RLE representation-domain discrepancy for some cases. Consequently:
- preserve the original records and labels;
- do not relabel PLY targets as OBJ targets;
- do not silently treat RLE moments as equivalent to teacher moments;
- admit each geometry/query only under an explicit same-domain contract;
- keep ordinary query supervision distinct from actual optimizer trajectory supervision.

The independent development set was reported as having zero admitted supervised geometries at that handoff. That blocks reliable model selection/generalization claims, but does not prevent architecture/software contract work using analytic synthetic fixtures.

The proposed Objaverse + local TRELLIS source mix is a data-generation plan, not yet an admitted corpus. Generated or downloaded geometry must pass provenance, topology, source-family deduplication, Spin-It initialization and teacher-domain checks before admission.

## 5. Unified inference contract to implement in subsequent steps

The intended model-level operation is:

`predict(geometry_cache, raw_quadric_coefficients) -> Prediction`

The stable prediction record must distinguish:
- `moments`: [Q, 10], fused ten raw moments;
- `jacobian_coefficients`: [Q, 10, 10], derivative of the fused moments with respect to the raw coefficient input;
- `moment_log_std`: [Q, 10], optional predictive scale;
- `jacobian_log_std`: [Q, 10, 10], optional predictive scale;
- `event_logits` / `event_probability`: [Q], optional teacher-defined event-risk output;
- `validity`: derivative validity, event-label availability, and model-output validity kept as separate fields;
- `diagnostics`: branch outputs, fusion weights, attention diagnostics, cache identity and timing.

The Jacobian must differentiate the *same fused moment function*, including all query-dependent gates and attention corrections. Uncertainty and event outputs must not be conflated with derivative validity or exact solver acceptance.

This is a target contract for subsequent implementation; it is not a claim that one complete public API already exposes every field.

## 6. Optimizer integration contract

The model may return a proposal containing:
- candidate parameter step/direction;
- predicted moment/Jacobian at the queried state;
- uncertainty and event-risk diagnostics;
- reason for requesting exact evaluation or fallback.

Only the exact Faster-compatible solver may:
- evaluate authoritative trial states;
- check constraints and event transitions;
- accept or reject a step;
- shrink trust regions or fall back to exact Newton/reference calculations;
- declare convergence;
- validate the final saved material.

No neural-only acceptance path is permitted in the primary method.

## 7. Step sequence after this contract

| Step | Scope | Exit evidence |
|---|---|---|
| 2.9.11.2 | Inspect actual Spin-It/Faster/M69 entry points and define state/coordinate adapters | Runnable entry-point map, task identity and round-trip tests |
| 2.9.12 | Implement one typed unified prediction API | Output-shape, raw-coefficient Jacobian and optional-head contract tests |
| 2.9.13 | Implement optimizer proposal/fallback adapter | Exact solver owns acceptance; replay tests |
| 2.9.14 | Complete analytic/synthetic integration tests | Finite differences, invariants, batching, cache and checkpoint tests |
| 2.9.15 | Freeze V1 configuration and release contract | Reproducible config, source fingerprints, release report and local commands |
| 2.10+ | Connect Work's admitted corpus and train | Real-batch CUDA updates, validation and pilot evidence |

## 8. Explicit non-goals for Step 1

- No new neural backbone.
- No forced activation of every optional module.
- No change to Spin-It/Faster task definitions.
- No mutation or regeneration of existing pilot labels.
- No Objaverse/TRELLIS bulk download or generation.
- No formal or long training.
- No claim of cross-geometry generalization, 512^3 performance, speedup or paper-level success.

## 9. Acceptance checklist for Step 1

- [x] Research objective and component responsibility boundaries documented.
- [x] Existing model assets inventoried from repository implementation/docs.
- [x] Implemented versus unverified capabilities explicitly separated.
- [x] Existing pilot data scope and domain caveats recorded.
- [x] Unified prediction and optimizer contracts specified for subsequent implementation.
- [ ] Spin-It/Faster runnable entry points verified on the user's machine.
- [ ] Full V1 config and executable API implemented.
- [ ] Exact optimizer feedback loop tested.
- [ ] Formal training corpus admitted and CUDA training executed.

The unchecked items are intentionally not marked complete by this document.
