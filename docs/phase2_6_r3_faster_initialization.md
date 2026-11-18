# R3.0.4 — Faster numerical initialization audit

## Findings from the repository implementation

The current C++ request-backed path exposes the following actual behavior:

- Quadric polynomial: beta(x) = a + b dot x + x^T A x.
- Coefficient order: (a,b1,b2,b3,A11,A22,A33,A23,A13,A12).
- Cross coefficients are symmetric-matrix entries; polynomial evaluation contributes the factor two.
- Coefficient vector is normalized to unit norm on S^9; positive scaling preserves the material side, negative scaling reverses it.
- For SpinningTop, deterministic initialization modifies the random coefficient vector by forcing A33 negative with a minimum offset, then normalizes it.
- Deterministic restart stream is generated from deterministicSeed XOR 0x5350494e49544641, with one guess per maximumRestarts.
- Settings enforce lineSearchReduction=0.5 and maximumGeodesicStep=0.5.
- The strict dual-monomial profile caps function evaluations per restart at 60.
- Solver result stores restart count, per-restart function-evaluation counts, event stream, accepted iteration trace, event hash and trace hash.

These are facts about the current TVCG reconstruction. They are not proof of bitwise equivalence with the unreleased author Matlab implementation.

## Important limitation

The default generated guesses are produced from the registered seed and mode. If an explicit initialGuesses vector is supplied, it replaces generated guesses. Therefore every replay must serialize the actual complete vector, not merely the seed.

A Faster numerical initial guess is not the physical initial object. The original mesh, shell/design masks, task frame and material policy remain in the separate geometry/problem contract.

## New contract

The R3.0.4 record binds:
- mode and equation profile;
- seed and all solver budgets/tolerances;
- line-search constants;
- exact normalized 10-vectors supplied to the solver;
- initialization source;
- solver source commit;
- evaluator identity;
- geometry and task hashes.

No default or random values are silently filled in.

## Evidence boundary

Tests use normalized vectors to verify serialization and validation only. They do not claim a real M69 initial guess or exact trajectory. The real initialization record can only be produced from a selected admitted geometry and the exact native run.

## Next gate: R3.0.5 / R3.1

Capture a real source-backed initialization record, run the exact native evaluator, and persist the full event/trace ledger. Only then compare neural proposals against the same starts.
