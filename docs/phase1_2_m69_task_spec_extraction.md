# Phase 1.2 — M69 TaskSpec extraction and semantic binding

## Why this stage exists

The rotation-stability task must be sourced from the frozen M69 TaskSpec, not reconstructed from memory or replaced with the Spin-It Faster equality constraints. The existing C++ `SpinDesignRequest` and `SpinMomentProblem` expose several problem profiles; the M90.2 validator independently loads a pinned `task69.py` and requires a complete serialized TaskSpec.

The M69 source file is not committed in TVCG. The validator records this exact source SHA-256:

`5463dcfe57f3aea21f577fd77d26867ef9e7ec18098b7ff25e7bde5209fc3389`.

This phase therefore adds a hash-pinned extraction adapter. It does not fabricate a task instance or infer field meanings from names.

## Implemented adapter

`research/NeuralMoments/m69_task_spec_adapter_v1.py`:

1. verifies the source file against the pinned SHA-256 before importing it;
2. imports sibling dependencies from the same source directory;
3. instantiates the authoritative `TaskSpec`;
4. obtains the complete default schema from `TaskSpec().to_dict()`;
5. optionally validates a supplied complete serialized payload using `TaskSpec.from_dict()`;
6. checks the task's `task_hash`, positive scalar `density` and integer `native_resolution`;
7. emits every canonical field and a canonical payload SHA-256;
8. leaves axis, origin, contact, fixed/protected masks, constraints and tolerances explicitly unmapped until the actual source fields and semantics are reviewed.

The output is marked `extracted_source_verified_semantics_pending_review`. It is not physical admission, an optimization label, an SI declaration or a training record.

## Known facts from the committed M90.2 adapter

The existing M90.2 source requires a complete TaskSpec key set derived from `TaskSpec().to_dict()`, then calls `TaskSpec.from_dict()`. It checks:
- `task.task_hash` against the registered task hash;
- `task.native_resolution` against the registered grid side;
- `task.density` against the registered positive scalar density.

The exact remaining field names and their semantic meaning must come from the pinned local source, not this document.

## Local extraction command

From the TVCG repository root:

```bash
python -m research.NeuralMoments.m69_task_spec_adapter_v1 \
  --task-source /data/xvxinyao/Spin_M68_Trellis_Data_Kit/src/task69.py \
  --output reports/phase1_2_m69_task_spec.json
```

The adapter writes only the extraction report under the selected output path. It does not edit the source TaskSpec or any dataset. The user should inspect the complete field list/payload before the semantic mapping is implemented.

## Phase 1.2 exit criteria

- [x] Pin and verify the authoritative TaskSpec source.
- [x] Implement complete TaskSpec extraction and payload hash.
- [x] Fail closed on source mismatch, missing fields, nonfinite values and invalid density.
- [ ] Run against the user's actual M69 source file.
- [ ] Review actual field names, formulas, units and defaults.
- [ ] Bind axis/origin/contact/fixed/protected region semantics without inference.
- [ ] Map the exact M69 constraint values, signs and tolerances to the unified contract.
- [ ] Cross-check the adapter output against an independently registered M90 task record.

Only after the remaining items pass can the M69 task be called semantically bound. The Spin-It Faster moment-space objective remains a separate baseline profile.
