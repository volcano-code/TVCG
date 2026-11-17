# M90.4 convergence Implementation Plan

Spec: reports/m90_4_convergence_20261116/protocol.json and M90_4_PROTOCOL.md, frozen before execution. Current user A–F are sole scientific authority.

Execute inline with executing-plans/test-driven-development and fresh final reviewer. User prohibits commit/delete; use new isolated directory on existing research branch and keep all artifacts. No worktree, stage, commit or cleanup.

1. Capture baseline; freeze decisions, units, metrics and gates before runs.
2. Test then implement sparse Mesh68 wrapper, native small-grid differential, RLE membership comparison, independent moments and topology controls. Exact-centre source remains unchanged.
3. Seal implementation source hashes; run each family/backend/grid in its own bounded subprocess, all five preregistered phases; save and reopen RLE.
4. Aggregate errors, convergence, phase and admission screens without threshold changes. Fresh review and meaningful corrections/tests if needed.
5. Verify all baseline hashes and Git; append handoff/control docs; deliver machine report, protocol, tests and artifact manifest.

Review focus: frozen native parity/endpoints versus sparse canonical merging; accurate phase units; missing/ambiguous original geometry; common reference and all ten moment orders; immutableTaskSpec512; strict no post-hoc thresholds; whole retained connectivity not deleted connectivity; conservative status aggregation; memory/time caps.
