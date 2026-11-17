# Phase 2.2 second round — Conditioning architecture comparison

**Repository:** volcano-code/TVCG  
**Branch:** research/tvcg-initial-import  
**Code under test:** `998647ee70c22c42bad6f229768f80b2ff1188d8`  
**CI run:** `44109272340` — success

## 1. Compared variants

All variants wrap the same frozen `PcraftNativeMomentOperator`, use the same 31D M69 task feature vector, the same synthetic 8-interval native geometry, the same query vector, and the same convex moment-potential head.

1. **FiLM:** task MLP emits multiplicative/additive latent modulation.
2. **Conditional normalization:** residual conditional LayerNorm modulation, preserving exact identity at zero initialization.
3. **Cross-attention:** task embedding is the query; keys/values are four geometry context tokens (global latent, 36D statistics, normalized fixed moments, normalized design moments). A zero-initialized output projection gives exact identity at initialization.

The frozen M80 trunk and original M80 implementation are not modified.

## 2. Parameter-budget matching

The builder targets the FiLM adapter at hidden width 128 and searches hidden widths 8..256 for the closest parameter count in the other variants. It reports actual trainable parameters and rejects a mismatch above the configured 20% ceiling. No dummy parameters are added.

| Variant | Hidden width | Trainable adapter parameters | Relative gap |
|---|---:|---:|---:|
| FiLM | 128 | 6,222 | 0.00% |
| Conditional normalization | 128 | 6,238 | 0.26% |
| Cross-attention | 136 | 6,342 | 1.93% |

The measured maximum budget gap is 1.93%, so this is a close parameter-matched architecture comparison.

## 3. CPU integration tests

GitHub Actions run `44109272340` passed all seven tests in 2.41 seconds:

- all three variants exactly reproduce the frozen M80 output at zero initialization;
- all three respond to a controlled task-conditioning parameter update;
- all three retain query-Jacobian versus central finite-difference consistency;
- all three preserve symmetric positive-semidefinite query Jacobians from the convex-potential head on the tested smooth fixture;
- parameter counts are reported and within the matched-budget tolerance;
- gradients reach the trainable conditioning output layers;
- the M80 backbone remains frozen and in evaluation mode.

A separate Phase 2.2 integration workflow `44108845369` also passed its eight task-conditioned M80 tests.

## 4. What this comparison does and does not establish

**Established:** the three conditioning mechanisms are implemented; they share the same M80 operator and synthetic input/query fixture; their adapter parameter budgets are closely matched; they preserve tested identity, task-response, derivative and convex-potential invariants.

**Not established:** empirical accuracy superiority, training convergence, robustness across many geometries/tasks, M80 checkpoint quality, 512^3 GPU performance, or end-to-end optimization speed. The tests do not train the variants to convergence. No architecture is declared the winner from random/untrained weights.

## 5. Research decision

Keep all three variants as controlled candidates. FiLM remains the simplest baseline. Conditional normalization is almost parameter-identical and is a low-risk alternative. Cross-attention adds explicit interaction between the task query and geometry/moment context tokens at only 1.93% more adapter parameters in this configuration.

The next fair experiment is a deterministic synthetic calibration/training comparison with fixed initialization, query set, optimizer, step budget and parameter budget, followed by KKT-consistency tests. Only that experiment can provide an initial learning-performance ranking; real-geometry claims still require later geometry-disjoint validation.
