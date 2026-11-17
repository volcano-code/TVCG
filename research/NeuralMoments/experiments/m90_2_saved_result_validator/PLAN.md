# M90.2 independent saved-result validator implementation plan

Goal: enforce A0–A6 on immutable registered originals and reopened deletion results.
Architecture: standalone CPU NumPy validator, frozen task69 for physical conditions only;
independent prism integrals, cell cubature and mesh-centre whole-solid reconstruction.
Spec: reports/rotation_admission_20261116/STRICT_REMOVAL_ACCEPTANCE.md plus current user instruction.

Constraints: no production edits, reset, deletion, staging, commit, push, GPU, optimizer or training.
Two CPU threads; each subprocess <=120 seconds; <=1 GiB memory/output budget. Preserve history.
Explicit execution instruction supersedes skill approval/commit/worktree steps: execute here in a new directory.

- [x] Write saved-file positive/negative controls; observe missing validator failure.
- [x] Implement pinned registry/result provenance, scalar original material, strict canonical RLE and explicit retained-editable complement.
- [x] Implement independent float64 ten moments, immutable original subtraction and fixed-shell composition cross-checks.
- [x] Evaluate frozen TaskSpec and reconstruct original mesh occupancy / 6-face retained connectivity; unknown/budget => pending.
- [x] Run deterministic suite, frozen task69 tests, actual M83 metadata preflight and machine report.
- [x] Review independently; verify preservation and append handoff/control state.

Review focus: self-endorsed metadata must not pass; changed original or result bytes must fail;
missing mesh must not pass; unsigned/float RLE coercion must not repair inputs;
full-retention material control must fail a positive-removal TaskSpec.
No full project suite: it includes GPU/training/build paths outside the explicit authorized budget.

Execution ledger: no commits/worktrees/deletion under explicit user restrictions. Full project GPU/training suite deliberately excluded. Independent review fixes and RED→GREEN evidence are in review_resolution.json. Native512 preflight and missing metadata conservatively remain pending, not implementation success on real data.
