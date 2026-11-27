# Compact PINO V2.4 research candidate — Cut-conditioned Conservative Geometry Operator

**Status: experimental branch. Not production default; no trained or physically accepted V2.4 checkpoint is claimed.**

## Evidence motivating the design

The supplied 2026-11-26 speed audit reports:
- V1 E13 at native RLE 256^3: 0.4774 s median hot query; at 1024^3: 7.4445 s.
- V2.2 Noodles v8 seed 0: 34.26 s neural proposal (16 dynamic calls), 38.42 s native correction, 21.00 s independent native acceptance, 21.36 s other cold-start overhead; total 115.04 s.
- Static geometry encoding was only 0.72 s in that run. Thus merely making the static encoder smaller is not the primary direct latency lever.
- V2.2/E16 improves fixed-development moment metrics and one E16 hybrid path reached a native-accepted solution, but direct neural termination still failed native stationarity. The single seed-0 hybrid result is not a generalization claim.

The supplied architecture history also confirms that a 32^3/64^3 Compact-Lite context replaced an earlier P-CRAFT 32^3/64^3/128^3 multiscale plan. This V2.4 candidate explicitly restores a real 128^3 context scale while retaining the 256^3 native material representation. These are different resolution layers.

## Research hypothesis

The proposed paper contribution is **cut-conditioned conservative moment operator learning for fast, derivative-consistent geometry optimization**, not simply a larger PINO.

1. **Geometry branch:** P-CRAFT-derived factorized Fourier/U-shaped encoder at 64^3 + 128^3, computed once per geometry. Native 256^3 RLE/material atoms remain the conservation source.
2. **Cut branch:** encode a small number of descriptors of the current finite, oriented cut (signed-distance band, local normal, local curvature/shape, cap/side identity and certified motion-chart features) into cut tokens.
3. **Local interaction:** only active-band material tokens query the cut tokens through one zero-initialized cross-attention residual block. This is not all-token self-attention; interaction cost scales with active material tokens × cut tokens.
4. **Conservative output:** the interaction modulates the existing bounded local retention field. All ten moments still come from one positive weighted material integral.
5. **Derivative consistency:** the full Jacobian is differentiated from that same final moment graph. Cut-token dependence must remain attached to autograd; no detached conditioning and no independent J head.
6. **Solver-aware use:** cheap neural proposal followed by exact native correction and unchanged independent native acceptance. The primary metric is wall-clock to a native-accepted solution, not callback latency alone.

GINO motivates geometry-aware local-to-global mappings; DINO/DIFNO motivates derivative-aware operator training. Neither alone is the novelty claim. The proposed contribution is their task-specific combination with finite-cut conditioning, positive-measure moment conservation and exact solver handoff.

## Candidate architecture contract

- Native material: 256^3 RLE / positive-measure atoms; fixed material is added exactly once.
- P-CRAFT context: exactly 64^3 and 128^3. The 128^3 feature map must be produced by a real encoder forward and sampled into local material features, not interpolated from 64^3.
- Material tokenization: Token16 for the first controlled comparison.
- Dynamic cut interaction: one 4-head, width-192, zero-initialized residual cross-attention block; 16 cut tokens initially; active-band queries only.
- Local response: existing monotone_warp warm-start; cross-attention conditions its two response channels.
- Output: ten moments in the existing order; J from the same M graph.
- Exact native correction/acceptance unchanged.

The initial configuration contract is in `v24_config.py`; the attention primitive is in `cut_conditioned_attention.py`.

## Required ablations

All arms must share the same E16 source checkpoint where compatible, same geometry/query manifest, native teacher, optimizer budget and evaluation protocol:

| Arm | Encoder context | Dynamic cut interaction | Purpose |
|---|---|---|---|
| A | 32/64 | none | V2.3 reference |
| B | 64/128 | none | isolate resolution |
| C | 32/64 | cut-conditioned MLP | isolate added cut information |
| D | 32/64 | cross-attention | isolate attention |
| E | 64/128 | cut-conditioned MLP | scale + conditioning without attention |
| F | 64/128 | cross-attention | combined candidate |

Do not compare only A vs F; that confounds resolution, information and attention.

## Self-audit before integration

- **128^3 memory risk:** profile peak allocated/reserved memory on RTX 5090 D first. Use checkpointing and chunked sampling; never silently downsample the 128^3 input.
- **Attention cost:** never apply dense self-attention over all 4096 Token16 cells. Restrict queries to active-band tokens and keep cut token count small.
- **Cut descriptor cost:** finite caps, orientation, operation side and chart semantics must be represented. If descriptor construction requires the expensive Boolean path, the speed hypothesis may fail; time descriptor creation separately.
- **Jacobian correctness:** because attention depends on cut parameters, the chain term through cut tokens must be included in dM/dc. The existing analytic-slope shortcut is insufficient unless this extra derivative is added and verified.
- **Conservation:** attention may modulate only bounded local retention parameters; it must not directly emit ten moments. Positive weights and 0<=retention<=1 remain invariant.
- **Checkpoint migration:** E16 is an initializer, not a strict drop-in checkpoint. Report every mapped/missing/unexpected tensor; do not silently use strict=False.
- **Performance uncertainty:** the V2.3 shared-CPU surface-executor microbenchmark did not show stable speedup. Only same-GPU, same-query, idle-window Work B tests can establish runtime benefit.
- **Scientific scope:** one Noodles seed-0 hybrid success is feasibility evidence only. Independent geometries, multiple paired initial conditions, success rate and confidence intervals are required for a paper-level claim.

## Implementation boundary in this commit

This commit adds:
- a standalone zero-gated cut-conditioned attention primitive;
- a strict 64^3/128^3 + native-256^3 architecture contract;
- unit tests for identity initialization, inactive-token pass-through, cut-token gradients, masking and configuration semantics.

**It is intentionally not wired into production `evaluate_stencil` yet.** The current response API does not carry a complete differentiable cut-token descriptor into the local response. Wiring only the forward path would produce an incomplete Jacobian and would be scientifically invalid. The next integration change must extend the stencil descriptor and the full chain rule together, then compare the analytic/autograd Jacobian with native finite differences before any training.
