# Compact PINO V2.4 — Cut-conditioned Conservative Geometry Operator

**Status: experimental research branch. Not production default; no trained or physically accepted V2.4 checkpoint is claimed.**

## Decision and scope

The objective is not to maximize grid resolution or add attention modules indiscriminately. It is to reduce the wall-clock time to a **native-accepted optimization result** while preserving conservative ten-moment prediction and derivative correctness.

V2.4 is organized as two gated steps:

- **V2.4-A (first candidate):** native 256^3 material, real P-CRAFT 64^3 + 128^3 context, Token16, one active-band cut-conditioned cross-attention layer, bounded local retention, unified M/J.
- **V2.4-B (only if A shows local-resolution error):** add sparse 256^3 local patches (512^3 only as a limited sensitivity experiment) around active cut bands, thin walls, caps and certified high-error regions. Do not build a dense global 256^3/512^3 feature volume.

The sparse local level is a conditional extension, not a pre-approved default. It must demonstrate accuracy or correction-call gains that justify its memory and query cost.

## Evidence motivating the design

The supplied 2026-11-26 speed audit reports:
- V1 E13 at native RLE 256^3: 0.4774 s median hot query; at 1024^3: 7.4445 s.
- V2.2 Noodles v8 seed 0: 34.26 s neural proposal (16 dynamic calls), 38.42 s native correction, 21.00 s independent native acceptance, 21.36 s other cold-start overhead; total 115.04 s.
- Static geometry encoding was only 0.72 s in that run. Thus merely making the static encoder smaller is not the primary direct latency lever.
- V2.2/E16 improves fixed-development Moment metrics and provides one successful neural-proposal + native-correction Noodles seed-0 solve, but direct neural termination still failed native stationarity. The single seed-0 hybrid result is not a generalization claim.
- V1's hot query is not an accepted-solution speed result: the cited E34 run exhausted 480 evaluations without an accepted solution.

The architecture history confirms that a 32^3/64^3/128^3 P-CRAFT context pyramid existed before Compact-Lite reduced the context to 32^3/64^3. V2.4 restores a **real 64^3+128^3 context candidate**, without confusing context resolution with native 256^3 material.

## Research hypothesis and contribution

The proposed contribution is a **cut-conditioned conservative moment operator**:

1. Geometry is encoded once by a P-CRAFT-derived multi-resolution trunk (64^3+128^3); native material/RLE and positive quadrature remain the source of conservation.
2. A compact descriptor of the current finite, oriented cut (signed-distance band, local position/normal/curvature/cap features with a certified coefficient-motion chain) becomes a small set of cut tokens.
3. Only active-band material tokens are compacted and query those cut tokens through one zero-initialized cross-attention residual block. Inactive tokens do not enter the attention operation.
4. The same final retention field produces all ten moments and its complete derivative. Any cut-dependent attention path must remain in the autograd graph; no detached conditioning and no independent J head.
5. A solver-aware handoff policy proposes cheaply, then calls exact native correction/acceptance. The scientific metric is wall-clock to a native-accepted solution, not neural callback latency alone.

Potential paper novelty is the combination of (a) cut-conditioned local geometric interaction, (b) positive-measure conservative moment integration with derivative-consistent output, and (c) solver-aware exact correction evaluated under paired accepted-solution time. GINO, DINO and geometric attention are inspirations, not claims of novelty by themselves.

## Architecture contract

- Native material: 256^3 RLE / positive-measure material atoms; fixed material is added exactly once.
- First context candidate: exactly 64^3 and 128^3; 128^3 must be a real encoder forward and sampled into local token features, not interpolation from 64^3.
- Optional local level: sparse 256^3 patches only after profiling demonstrates a local geometry representation gap; 512^3 is not a default.
- Material tokens: Token16 initially retained for controlled comparison.
- Dynamic cut interaction: one 4-head, width-192, zero-initialized residual cross-attention block; 16 cut tokens initially. Active-band material tokens are compacted before attention.
- Local response: existing monotone_warp as warm-start baseline; cross-attention conditions its two response channels before the bounded parameter mapping. Bounds must hold; global monotonicity under arbitrary coefficient-dependent conditioning is not claimed.
- Physical head: one positive retention measure; M in fixed ten-component order; J obtained by differentiating the same M graph.
- Native correction and independent original-mesh acceptance remain mandatory; no threshold is relaxed.

## Required ablations

All candidates use the same geometry/query manifests, native teacher and evaluation protocol. E16 is an initializer where shapes are compatible; migration must be explicit.

A. V2.3 E16 (32/64, no dynamic cut attention) — reference.
B. 64/128 encoder only, no dynamic attention — isolates scale change.
C. 32/64 + cut-conditioned MLP — isolates added cut information.
D. 32/64 + cut-conditioned cross-attention — isolates attention.
E. 64/128 + cut-conditioned MLP — scale plus conditioning without attention.
F. 64/128 + cut-conditioned cross-attention — combined first candidate.
G. Only if B–F show a local geometry error: add sparse 256^3 active patches to the best candidate; compare against the same candidate without patches.

Do not compare only A vs F; that confounds resolution, information and attention. Report parameter count, peak memory, cold encoding, per-new-cut preparation, response+M/J, native correction, final acceptance, accepted-solution success rate and total wall time.

## Self-audit / risks

- 128^3 activation memory can be large; profile peak allocated/reserved on RTX 5090 D before training. Use checkpointing and chunked local feature sampling, not silent downsampling.
- Do not jump to dense global 256^3/512^3. Sparse local patches are conditional and require a measured local-resolution error.
- Attention must not be applied to all 4096 Token16 positions. The primitive compacts active material tokens before MHA; Work B must verify real active counts and timings.
- The cut-token descriptor must preserve finite surface, cap and chart semantics. Never use Boolean-result material, true cut moments, or teacher-only information as model inputs. Vertex motion is derivative metadata; feeding coefficient-dependent vertex velocities as learned Moment inputs would require the additional higher-order construction chain. Compute one canonical inside-retention field and route operation side at the output.
- The new attention output depends on cut parameters. Its derivative contribution must be included in J. The current analytic-slope shortcut is insufficient unless the additional chain term is implemented and independently verified.
- Positive weights and 0<=retention<=1 remain invariant. Cross-attention may only modulate bounded local response parameters, not directly emit ten moments.
- E16 checkpoint compatibility is partial: geometry trunk may initialize, but new attention parameters are new and zero-gated; 128^3 context has a different input path. Report mapped/missing/unexpected tensors; never silently use strict=False.
- The V2.3 shared-CPU surface-executor microbenchmark did not show stable speedup. Only same-GPU, same-query, idle-window Work B tests can establish runtime benefit.
- One Noodles seed-0 hybrid success is feasibility evidence only. Independent geometries, paired initial conditions and confidence intervals are required for a paper-level claim.

## Implementation slice in this branch

This branch currently adds:
- a standalone zero-gated cut-conditioned attention primitive;
- actual active-token compaction before attention (inactive material tokens do not enter Q/K/V);
- an explicit 64^3/128^3 + native-256^3 configuration contract;
- unit tests for identity initialization, inactive-token pass-through, cut-token gradients, active-token attention accounting, masking and configuration semantics;
- default higher-order-safe math attention and a ten-moment / 10-by-10 Jacobian loss backward regression (7 CPU tests passed with PyTorch 2.8.0+cpu; target CUDA integration remains untested);
- Work A/B integration and acceptance handoff.

It is **not yet wired into production `evaluate_stencil`**, and the actual 64^3/128^3 multiscale encoder is not yet integrated into the Compact V2.3 execution path. The current `v24_config.py` is a strict architecture contract, not proof that a full encoder is already built. The next integration change must extend the stencil descriptor and the full Jacobian chain together; wiring only the forward path would be scientifically invalid.

## Current action guides (2026-11-27)

- [Verified update summary and limitations](UPDATE_SUMMARY_20261127.md)
- [Work A: data and teacher actions](WORK_A_ACTION_GUIDE.md)
- [Work B: integration, training, timing and acceptance](WORK_B_ACTION_GUIDE.md)
- [Proposed training contract, not yet a runnable Trainer config](TRAINING_RECOMMENDATION.yaml)

These guides supersede earlier handoff wording about unconditional E16 output identity across grid changes, allowing Boolean-derived inputs, and treating every native-J error increase as an implementation derivative bug.

