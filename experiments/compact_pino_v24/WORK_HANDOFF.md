# Compact PINO V2.4 — Work A / Work B execution handoff

Base: `cd49802ee480bc168b9902d453280e334658a1c7`
Branch: `research/compact-pino-v24-cut-conditioned`
Current candidate commit: `8f689316ca52ca71f4a052e792ca8bd95c51ee59`

## Work A — data and teacher assets

### A0. Freeze identity before generating anything
- Keep historical R1 train split and fixed 222-row development set unchanged.
- Keep E16 as warm-start checkpoint; record SHA256, architecture config, source manifest and optimizer-state availability.
- Create a separate V2.4 data snapshot and supplemental development set. Never append new rows into the old 222-row denominator.
- For each case bind native mesh/RLE, density, coordinate frame, fixed/editable material split, finite operand (faces, caps, orientation, retained side), raw coefficient chart, vertex motion, teacher build/version, operation result, ten moments and provenance hashes.

### A1. Required coverage
Prioritize:
1. Legal task endpoints and accepted trajectory tails with fixed support material present.
2. Medium/high retained-mass queries near convergence, not only arbitrary cuts.
3. Cross-source curved cuts with qualified R8 and actual Newton-direction Jv.
4. Thin walls, cavities, disconnected components, caps, closest-feature ties and topology/refusal events.
5. Ordinary non-terminal queries with their true nonzero native residuals.

Moment, R3, R8, Full-J and Jv qualifications are separate. Never promote an unqualified derivative to Full-J or fill missing labels with zeros. A row with valid M but invalid J remains eligible for M-only supervision.

### A2. Cut-token teacher representation
For every finite cut, generate a compact token sequence with a fixed schema:
- oriented local surface patch centroid and normal;
- principal curvature / local quadratic patch coefficients where defined;
- cap-vs-side identity and retained-side sign;
- signed-distance band statistics for the associated active material block;
- chart/vertex-motion basis features and validity mask;
- topology/event/closest-feature ambiguity flags.

The descriptor must be computed from the same finite, oriented, capped operand used by the deployed teacher. Preserve exact query-to-token provenance. If descriptor extraction calls an expensive Boolean or full Jacobian path, time and report it separately.

### A3. Work A deliverables
- V2.4 train and supplemental development manifests with frozen source-group splits.
- Teacher/mesh/operand hashes and per-label qualification receipts.
- Coverage table: source × geometry × cut family × retention bin × trajectory phase × derivative rank.
- Explicit fixed-material audit; fixed moments are added once and only once.
- Rejected/ambiguous rows with reason codes, not silently dropped.
- A small balanced smoke subset for Work B integration, plus the full training snapshot.

## Work B — architecture integration, numerical correctness and speed

### B0. Checkout and environment
```bash
git fetch origin research/compact-pino-v24-cut-conditioned
git worktree add --detach ../TVCG-v24 8f689316ca52ca71f4a052e792ca8bd95c51ee59
cd ../TVCG-v24
python -m pytest -q tests/research/test_compact_pino_v24_cut_attention.py
```
Use the validated CUDA/PyTorch environment; record GPU, driver, CUDA, PyTorch, CPU affinity and native build IDs.

### B1. Integrate 64^3 + 128^3 correctly
- Add a named V2.4 encoder config that performs real P-CRAFT forwards at 64^3 and 128^3.
- Keep native material 256^3; do not describe context resolution as material resolution.
- Sample both scale feature maps at actual material-token / active-node coordinates.
- Preserve native interval features and static material moments; 128^3 must not be synthesized by upsampling 64^3.
- Record feature-map shapes, activation bytes, encoder time and peak allocated/reserved memory.
- Compare 32/64 and 64/128 with matched width/modes/layers and same query/teacher split.

### B2. Integrate differentiable cut tokens
- Extend prepared stencil with cut descriptors carrying local oriented patch/cap features and derivatives with respect to raw solver coefficients.
- Build cut tokens from this descriptor; compact active-band material tokens before attention. Verify measured attention work equals sum(N_active × N_cut), not N_all × N_cut.
- Feed conditioned context and two-channel response delta into existing bounded monotone retention parameterization.
- Do not detach cut tokens, descriptor tensors, attention outputs or coefficient-dependent inputs.
- Keep positive quadrature weights and 0<=retention<=1. All ten moments still come from one final retention field.
- Include both derivative paths:
  `dr/dc = (dr/ds)(ds/dc) + (dr/dz)(dz/dc)`.
- If cap/topology/closest-feature validity is ambiguous, preserve native-refusal/fallback behavior.

### B3. Mandatory numerical gates before training
1. Zero-gated initialization reproduces V2.3 E16 Moment and J within frozen tolerances.
2. Cut-token MLP and cross-attention both receive finite, nonzero gradients; the zero-initialized response projection receives a useful first-step gradient against nonzero targets.
3. Analytic/autograd J agrees with independent multi-step finite differences on valid Full/R3/R8 charts.
4. Same-cut side complementarity, fixed-material add-once, positive weights and retention bounds pass.
5. Near ties, caps, topology events and invalid charts reject/fallback exactly as registered.
6. Compare full Moment, J, Jv, native stationarity vector and next Newton direction against native teacher.
7. Check checkpoint migration explicitly: mapped, new, missing and unexpected tensors; no silent `strict=False`.

### B4. Ablations — do not collapse factors
Use same E16-derived starting state where shape-compatible, data, exposure schedule, optimizer, loss coefficients and evaluation:
- A: 32/64, no dynamic attention (V2.3 reference).
- B: 64/128, no dynamic attention (scale only).
- C: 32/64 + cut-conditioned MLP.
- D: 32/64 + active-band cut-conditioned cross-attention.
- E: 64/128 + cut-conditioned MLP.
- F: 64/128 + active-band cut-conditioned cross-attention.
- G (only if B–F show local geometry representation error): add sparse 256^3 active patches to the best candidate; compare with the same candidate without patches. 512^3 is a sensitivity experiment only, never a default dense grid.

Do not compare only A vs F; that confounds resolution, information and attention. Report parameter count, peak memory, cold encoding, per-new-cut preparation, response+M/J, native correction, final acceptance, accepted-solution success rate and total wall time.

### B5. Performance protocol
Measure on RTX 5090 D in an idle, fixed-affinity window:
- cold geometry/material preparation;
- one-time 32/64 or 64/128 encoding;
- per-new-cut descriptor generation;
- active token selection / cut-token construction;
- attention and local response;
- Moment-only and joint Moment/Jacobian;
- CPU/GPU transfer and synchronization;
- native correction and final independent acceptance.

Run callback microbenchmarks and full cold-start solve. Full solve timing starts at original mesh read and includes all failures, rejected steps, fallback, correction and final acceptance. Compare against a freshly re-run optimized pure-native baseline using identical input, seed, tolerances and hardware.

### B6. Physical and paper-level gates
- Keep original native stationarity and balance thresholds unchanged.
- Report direct neural proposal acceptance separately from hybrid accepted-solution success.
- Use multiple independent geometries and paired initial conditions; Noodles seed 0 is diagnostic only.
- Report success rate, restarts, function evaluations, Newton iterations, accepted objective, constraint residuals, native stationarity residuals, cold/steady time, P50/P95 and paired time-ratio confidence intervals.
- Include training/teacher construction cost and estimate amortization over repeated optimization tasks.

## Decision rules
- If 128^3 improves accuracy but increases cold solve time, keep it only if it reduces correction calls or improves accepted-solution time.
- If cut attention adds no gain over the same cut descriptor with an MLP, keep the MLP.
- If descriptor generation is as expensive as native surface preparation, the cut-attention route has not achieved the intended amortization; redesign the descriptor or use a cheaper chart-native descriptor.
- If new attention improves M but worsens J/Jv, do not train further; fix the complete derivative path.
- Add sparse local 256^3 only when the 64/128 model demonstrates a repeatable local error that the patch is intended to resolve.
- Do not start long training until B3 passes and the Work A smoke manifest is admitted. Production V2.3/E16 remains untouched.
