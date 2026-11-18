# Phase 2.6 R3.0 — Problem Reconstruction Audit

## Status

R3.0 now distinguishes two states:

1. contract-bound: the case records all physical and numerical inputs needed for replay;
2. physically admitted: the real M69 files, hashes, semantics, and exact solver replay have independently passed.

This commit implements the first state without pretending the second has been achieved.

## Frozen inputs

A real R3 case must bind:
- original 3D mesh;
- fixed shell;
- design domain;
- rotation axis and origin;
- contact point;
- pin/support record;
- initial material policy;
- Faster numerical quadric coefficients;
- provenance/type of the Faster numerical initialization;
- SHA-256 for every geometry artifact.

The physical initial object and Faster numerical initial state are represented separately.

## Evidence boundary

Existing repository documentation states that the authoritative M69 source is not committed and that actual pilot mesh paths are not yet formally admitted. Therefore this stage does not fabricate a mesh, pin, cavity, shell mask, or Faster initial guess.

The M69 constraint adapter remains the constraint layer; it is not geometry admission.

## Next gate: R3.0.2

Bind one real M69 case from the user's local M69 data kit, then:
1. verify all geometry hashes;
2. verify fixed shell and design-domain provenance;
3. verify axis/contact/pin semantics;
4. verify initial material policy;
5. bind the exact Faster numerical initialization;
6. replay the exact moment/Jacobian/Newton trajectory;
7. compare the replay record with independently stored M69 evidence.

Only then can the case become physically admitted.

## Distance to final target

- Problem definition fidelity: contract scaffold complete; real-case admission not complete.
- Exact baseline reproducibility: not yet demonstrated for an admitted M69 case.
- Neural acceleration evidence: none yet; it is blocked by exact replay admission.
- 512 cubed long training: intentionally blocked until baseline fidelity is proven.
