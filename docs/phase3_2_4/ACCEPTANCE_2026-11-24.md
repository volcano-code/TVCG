# Compact PINO V2.0 acceptance, 2026-11-24

Baseline PR #1 head `ac8ab8c7d5333f0d18cea698e99b0a9f0f567e83`. The scientific decision is **NO-GO for native256 training under the current positive-quadric contract**.

## Architecture and physical domain

Instantiating both V2 factories on CPU gives one encoder and one cut-response module, 2,862,536 trainable parameters and 15,284,000 parameter bytes each. Compact-Lite has 3,351,256 trainable parameters. The token partition changes spatial resolution without changing learned parameter count. Both V2 configurations differentiate the final 10D physical moment in raw-coefficient coordinates. An independent direct AD check on the same synthetic partial cut reproduced the reported derivative. Production geometry conditioning uses native RLE256 and 32/64 encoder contexts.

A scalar fraction per token ties its mass and first/second moments together. For example, a uniform segment of length L cut at its midpoint retains half its mass and moves its center of mass by L/4. Multiplying the segment's original ten moments by one half cannot reproduce that shift. This is an approximation even for a plane. Keep within-token physics and learned query response as distinct error sources.

A fresh production native evaluator on TRELLIS train geometry T2 returned these editable masses for opposing raw coefficients of a partial hyperbolic cut:

| Positive | Negative | Independently integrated uncut mesh | Positive + negative − uncut |
|---:|---:|---:|---:|
| 0.0008542237848092942 | 0.0010491946674687014 | 0.0019032189653552037 | +1.99486922792e-7 |

This test uses the same mesh for both signs and the uncut reference. Thus changing the voxel resolution cannot remove this measured mesh-level discrepancy. The relative discrepancy is 1.048e-4. An ellipsoid control on the same geometry satisfied mass complementarity to 1.13e-17. Separately constructing the finite capped surface for opposite input signs can produce different triangulations. The network enforces complementarity for all weights. A universal exact equivalence with this teacher is refuted. The counterexample does not set an accuracy bound for every surface or imply that every mesh query fails.

No training receipt issued. No real optimizer preflight, paired screen, benchmark on trained V2 weights or native Newton acceptance attributed to V2. Revisit the physical contract with paired native operands, including the same-cut reverse Boolean, before training. Any change to native semantics or the frozen response needs its own review. Freeze numeric scientific accuracy limits before inspecting future trained predictions; the checklist names the metrics but does not specify all absolute limits.

## Software and operational changes

The old expected-acceptance test paired a proposed negative step with an exact objective minimized at a positive coefficient. The exact gate correctly rejected the uphill trial. The corrected fixture tests feasible descent and independently covers uphill and infeasible rejection with the production Armijo and feasibility settings. The legacy release workflow lists eight scientific gates separately from the software exit code.

The V2 screen launcher now records the plan, all three smoke reports, round-trip checkpoints and memory profiles after preflight completion. It verifies these digests before starting any training arm. A native256 smoke geometry must carry both Moment and derivative supervision. Single-query benchmark output now includes warmup/repeat counts, synchronization scope, parameter/checkpoint bytes and P99 as a descriptive sample percentile.

The diagnostic worktree preserves the historical Full V1 checkpoint source policy. Its parameter shapes still match Epochs 11 and 13. Use the historical source pairing for strict checkpoint restore. CPU contracts and source compatibility are separate from 5090 D physics validation. PR #1 remains draft.
