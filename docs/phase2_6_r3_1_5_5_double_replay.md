# R3.1.5.5 — Double native replay evidence audit

## Execution receipts

The native replay runner now writes a unique UUID run receipt with UTC start/end times, host, platform, Python version and process ID. These fields establish that Replay A and Replay B are separate invocations; they are excluded from scientific equality checks.

## Audit contract

The double replay auditor consumes two native replay bundles, two independently generated mass-moment records, two independent physical verification reports, and two geometry integrity reports. It requires distinct execution IDs, identical case/geometry/task/evaluator/source/init/profile/request identities, convergence and native final geometry verification, exact equality of scientific event/trace/input/output hashes, final quadric, mass moments and final geometry hashes. For each replay it reruns the strict native-vs-independent moment comparator and requires a match, then requires physical and geometry reports to pass with reviewed native/independent equivalence.

Timing is excluded from scientific identity. Evidence file hashes are recorded in the output report.

## Important limitation

The auditor does not execute the native solver; it verifies evidence from two separate executions. The CI fixtures test the auditor contract only and are not M69 experiments. A passing CI cannot be described as a real double replay.

## Local invocation

    PYTHONPATH=research/NeuralMoments python research/NeuralMoments/r3_double_replay_audit_v1.py --replay-a reports/r3/replay-a.json --replay-b reports/r3/replay-b.json --independent-a reports/r3/independent-a.json --independent-b reports/r3/independent-b.json --physical-a reports/r3/physical-a.json --physical-b reports/r3/physical-b.json --geometry-a reports/r3/geometry-a.json --geometry-b reports/r3/geometry-b.json --output reports/r3/double-replay-evidence.json

## Exit criteria

Software evidence auditor is implemented. Real R3.1.5.5 remains pending two physically admitted native executions on the same TRELLIS/M69 object, separate independent tetrahedral integrations, reviewed geometry mappings and passing physical verification reports.
