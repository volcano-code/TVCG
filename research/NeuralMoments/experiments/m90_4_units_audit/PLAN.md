# M90.4 evidence and units audit Implementation Plan

> Execute inline under the user's existing authorization. No commits or worktree changes.

**Goal:** verify immutable M90.3 evidence and identify actual M69 units before any new protocol is frozen.

**Architecture:** a separate characterization test module reads pinned TaskSpec/Oracle APIs; reports record facts separately from unavailable Chat decisions. No production changes.

**Tech Stack:** Python, NumPy, unittest; existing bounded CPU runner.

**Spec:** current user request for M90.4.1–M90.4.3. Six frozen Chat decisions are a required missing input; read_thread returned placeholders.

## Global constraints

Two CPU threads, 120 seconds per process, 1 GiB memory; no GPU, optimizer, training, frozen-source edits or Git mutations. Keep prior reports and checkpoints. Do not choose admission thresholds after observing experiments.

## Review focus

Length references may differ between audit, Oracle and encoder; absolute floors may break scale invariance; radius may mean a sphere; source coordinates do not imply SI; same voxel counts do not prove backend equivalence.

## Tasks

- [x] Capture Git, hashes, disk and M90.3 evidence before edits.
- [x] Read actual TaskSpec, callers, geometry and numerical tolerance definitions.
- [ ] Characterize scale covariance and length-reference differences without invoking an optimizer.
- [ ] Recheck existing TaskSpec and M90.3 sparse tests under caps.
- [ ] Write machine-readable audit and append status to the three requested handoff/state files.
- [ ] Obtain six frozen decisions; only then freeze the physical-scale and backend protocol with a pre-run hash.
- [ ] Implement and run the specified 128/256/384/512 convergence experiment under that protocol. Until decisions are available, all four remain not run, pending.
