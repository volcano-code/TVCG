# Phase 2.8.3 — Moment head optimization and convex saturation diagnostics

## Motivation

The 2026-11-18 RTX 5090D 300-step pilot used \`xl_ms_gno_transolver_convex\`. Its report recorded \`convex_potential_scale_fraction ~= 0.999999463\`, indicating that the geometry-conditioned total-amplitude gate was operating very close to its configured upper bound.

This is a warning signal, not by itself proof of an incorrect moment map. The effective amplitude also depends on the geometry-specific \`amplitude_bound\`; therefore a fraction alone is insufficient to interpret the actual output scale. Validation now records:
- amplitude fraction;
- raw gate logit;
- absolute amplitude bound;
- effective amplitude = fraction × bound.

The metrics are reported per geometry and as validation distributions.

## Head candidates

The model now has explicit matched preset names:

| Preset | Readout | Amplitude mapping |
|---|---|---|
| \`xl_ms_gno_transolver_bounded_odd\` | Bounded Odd | existing odd tanh response |
| \`xl_ms_gno_transolver_convex\` | Convex Potential | legacy sigmoid gate |
| \`xl_ms_gno_transolver_convex_atan\` | Convex Potential | bounded atan gate |

The original \`xl_ms_gno_transolver_convex\` remains unchanged and defaults to sigmoid. The original maximal training checkpoint remains structurally compatible: the new \`convex_amplitude_mode\` config field defaults to \`sigmoid\` when loading older checkpoint configs, and the head parameter names/shapes are unchanged.

## Amplitude parameterizations

Let \(z\) be the geometry-conditioned scalar gate logit.

Legacy sigmoid:
\[
a_\sigma(z)=\sigma(z).
\]

New ablation candidate:
\[
a_{\mathrm{atan}}(z)=\frac12+\frac{1}{\pi}\arctan(z).
\]

Both maps are strictly increasing and map finite logits into \((0,1)\). The atan map has algebraic tails; the sigmoid has exponential tails. At large positive logits the atan map approaches one more slowly, preserving a larger derivative over part of the high-logit range. This is a hypothesis for ablation, not a claim that atan is universally better.

The total potential amplitude remains:
\[
A_{\mathrm{total}}=A_{\mathrm{bound}}a(z).
\]

The positive mixture weights and SPD correlation-matrix construction are unchanged. Therefore the convex-potential head retains its parameterization-specific smoothness, positive homogeneity, and symmetric-PSD Hessian properties away from the zero query, subject to the existing numerical regularization.

## Validation and diagnostics

The trainer's validation report includes a \`convex_amplitude\` section with distributions for fraction, gate logit, bound, and effective amplitude. Each geometry detail contains the same scalar diagnostics.

This makes it possible to distinguish:
- a saturated fraction with a tiny physical bound;
- a saturated fraction with a large effective amplitude;
- a non-saturated gate whose amplitude is still small because the geometry bound is small.

No amplitude penalty was added to the default loss. We avoid introducing an arbitrary regularizer before a matched ablation demonstrates that it improves held-out moment/Jacobian accuracy rather than merely moving the gate away from one.

## Required fair ablation

Use the same:
- frozen dataset manifest and audit;
- geometry-family train/validation/test split;
- seed list;
- optimizer and learning-rate schedule;
- query batches and training steps;
- geometry trunk and GNO/Transolver settings;
- Moment/Jacobian loss weights.

Compare all three heads. Report:
- Moment scaled MSE and per-object tails;
- Jacobian scaled MSE and per-object tails;
- finite-difference agreement of the predicted Jacobian;
- convex fraction, logit, bound and effective amplitude;
- gradient norms and nonfinite/overflow counts;
- parameter count, peak GPU memory and wall time;
- exact-solver replay outcomes when the solver evaluation phase is available.

Do not compare a bounded-odd model with a different trunk or training budget and attribute the difference to the head.

## What this phase proves and does not prove

Proved by code/CPU tests:
- sigmoid and atan amplitude modes share exactly the same parameter/state-dict contract;
- both mappings remain in (0,1);
- the atan derivative is larger than the sigmoid derivative at the tested high positive logit;
- the atan convex head preserves positive-scale invariance, q/-q complement, and its own symmetric-PSD Jacobian parameterization;
- old checkpoint configs lacking the new field load with legacy sigmoid behavior;
- validation emits fraction/logit/bound/effective-amplitude diagnostics.

Not proved:
- atan improves trained Moment or Jacobian accuracy;
- the convex head is more physically correct than bounded-odd;
- the saturation observed in the previous pilot has been fixed in a trained checkpoint;
- the exact physical moment Jacobian is universally symmetric or PSD;
- any GPU training, memory, speed, or solver improvement.

## Next step

Run the matched head ablation only after the data/teacher workstream supplies the approved manifest. Then use the evidence to select the head for the M80+SQNO unified model. Until then, preserve the existing \`xl_ms_gno_transolver_convex\` checkpoint and treat atan as an opt-in research candidate.
