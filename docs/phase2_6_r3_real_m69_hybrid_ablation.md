# Phase 2.6 R3 — Real M69 Hybrid Ablation

## Purpose

R3 is the first physical experiment phase of the TVCG hybrid solver. It connects the
existing Phase 2.6 M80/S9/Hybrid contracts to real, provenance-pinned M69 records.

The scientific question is:

> At matched M69 physical accuracy and acceptance criteria, can a neural proposal plus
> exact correction reduce exact moment/Jacobian work and total solver cost relative to
> the exact Spin-It Faster baseline?

R3 does not change the M69 problem definition, objective, fixed shell, protected region,
single-density removal policy, or final exact acceptance contract.

## Four required arms

1. exact — Spin-It Faster-style exact Moment/Jacobian + safeguarded Newton.
2. neural_only — M80 Moment/Jacobian proposal without exact correction; retained as a
   stress-test/negative control and never used as the final physical authority.
3. neural_correction — neural proposal followed by exact local correction and exact
   acceptance.
4. full_hybrid — unified M80 + local residual correction + Neural-Newton initialization
   + exact acceptance/fallback.

All arms use the same registered M69 TaskSpec, geometry records, stopping tolerances,
candidate parameterization, and independent exact validator.

## Ground-truth authority

The exact oracle owns:

- ten raw mass moments;
- M69 physical constraints;
- connectivity/topology/event checks;
- final feasibility;
- final retained mass;
- exact evaluation accounting.

A neural prediction is never counted as a physical acceptance certificate.

## Required measurements

Every run must record:

- wall-clock time, split by neural/exact/correction/Newton;
- neural query count;
- exact moment count;
- exact Jacobian count;
- Newton iterations;
- line-search/backtracking trials;
- restarts/fallbacks;
- ten-moment relative errors;
- Jacobian relative errors where available;
- retained mass and mass fraction;
- every M69 constraint residual/check;
- connectivity/topology result;
- final pass/fail.

## Matched comparisons

### Same-accuracy comparison

Compare exact and hybrid methods only after exact validation passes the same M69 gates.
Report speedup as T_exact / T_method and separately report reduction in exact oracle calls.

### Same-budget comparison

Fix an exact-query budget and compare the best exact-verified feasible result returned
by each method. Report retained mass, feasibility, and failure rate.

## Data admission

R3 must consume real M69 records through an adapter. It must not invent geometry masks,
TaskSpec semantics, alpha-to-beta mappings, or topology flags.

The authoritative M69 TaskSpec remains hash-pinned by
research/NeuralMoments/m69_task_spec_adapter_v1.py and
m69_fixed_axis_contract_v1.py.

The existing Phase 2.6 documentation explicitly states that the production arbitrary
512^3 exact material oracle is still an open gate. R3 therefore begins with exact replay
and oracle admission before any acceleration claim.

## Evidence boundary

Passing R3 software tests is not evidence of physical acceleration. A valid R3 result
requires real M69 geometry, a verified exact oracle, and exact final validation.

Only after R3 establishes a positive and stable hybrid effect should Phase 3 scale the
operator/training system toward 512^3 and long RTX 5090 runs.
