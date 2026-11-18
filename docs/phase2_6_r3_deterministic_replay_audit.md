# R3.0.7 — Deterministic replay and independent verification

## R3.0.6 CI

Native Replay Binding CI completed successfully (run 44263633843). This proves Python facade/request contract tests, not execution against a real M69 object.

## Deterministic replay comparison

r3_deterministic_replay_audit_v1.py compare compares two replay bundles across case, geometry and TaskSpec identity; evaluator and solver source identity; initialization hash and equation profile; native event, trace, input and output hashes; status, function evaluations and per-restart counts; and final normalized quadric coefficients. Timing sidecars are excluded. Scientific outputs are compared exactly.

## Independent physical verification

verify-moments consumes a separately generated raw-moment record and requires canonical ten-moment order, independent backend identity, source geometry hash matching the replay input, and explicit material density. It independently derives mass, center of mass, central inertia tensor and normalized spinning-top constraints (l1,l2,Ic13,Ic23). It compares all ten raw moments against native replay output. It does not read or trust the native solver's independent_raw_residual field.

Tests use analytical fixtures to test formula plumbing only. They are not real M69 geometry or a real M69 run.

## Local use

    PYTHONPATH=research/NeuralMoments python research/NeuralMoments/r3_deterministic_replay_audit_v1.py compare --first reports/r3/replay-1.json --second reports/r3/replay-2.json --output reports/r3/replay-determinism.json

    PYTHONPATH=research/NeuralMoments python research/NeuralMoments/r3_deterministic_replay_audit_v1.py verify-moments --native reports/r3/replay-1.json --independent reports/r3/independent-moments.json --density 1.0 --output reports/r3/independent-verification.json

Density must come from the admitted task/material record. R3.1 remains blocked until real geometry admission and an independent moment record exist.
