# R3.0.5 — Exact solver event capture infrastructure

## Native evidence already present

The native C++ solver already exposes:
- deterministic restart sequence from mode and seed;
- an event sink for finalized solver events;
- attempt_start, evaluation, attempt_terminal, and newton_linear_solve events;
- optional evaluation payload containing solver coordinates, raw moments, variable moments, Moment Jacobian, derivative validity and evaluator identities;
- optional assembled Newton system payload and linear solve diagnostics;
- separate evaluation timing sidecar;
- scientific event hash and iteration trace hash.

Evaluation payload capture is opt-in through SpinSolverSettings.captureEvaluationPayload; it does not trigger additional evaluator calls.

## New R3 ledger validator

The new Python validator checks:
- exact case/geometry/task/evaluator/source/initialization identities;
- ordered zero-based restart ordinals;
- one start and one terminal event per restart;
- evaluation event count equals reported function evaluations;
- per-restart evaluation counts match;
- no restart follows convergence;
- terminal status agrees with final restart.

Scientific hashes exclude timing fields and bind scientific decisions and payloads.

## Important scope boundary

This module validates serialized solver evidence. It does not yet provide a Python binding that invokes the C++ solver, and no real M69 event ledger is claimed. Native C++ event capture is already present in the repository; R3 adds strict packaging and validation around it.

## Next execution gate

After an admitted geometry and Faster initialization record exist, invoke the native request-backed solver with capture enabled, serialize its events/trace and timing sidecar, then validate this ledger. Only a native run on a real TRELLIS/M69 case can satisfy R3.1 Exact Replay.
