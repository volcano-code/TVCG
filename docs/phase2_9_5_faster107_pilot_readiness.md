# Phase 2.9.5 — Faster107 pilot supervision and data handoff

## Decision and evidence boundary

Proceed with engineering integration for a bounded, training-only fitting
experiment on the 19 complete-quota geometries. Actual corpus training starts
after the consumer interface, coefficient convention, geometry representation,
and a real-batch backward/update check have been verified. Independent
development selection, generalization, GPU speedup, and hybrid optimizer
acceptance remain unestablished.

The user reported a full software release PASS at commit
`09266f62fa3b0e35fa288c903457c45ede373ecf`: seven checks passed and eight scientific
gates remained blocked. That result applies to that commit. This increment
changes the mathematical coordinates of the native convex head and introduces
partial derivative supervision, so its verification is recorded separately.

Dataset evidence is the user-supplied `CHAT_REVIEW.md`, dated 2026-11-19 Beijing,
SHA-256 `78022106a29876e4b8b6c0a5c3863b77cba99d644ad879e2f84aa475a56e53ce`.
The source-machine dataset, loader, index, audit, and teacher archives are not
available in the coding environment. Counts and hash-audit claims below are
reported by that review, not an independent reread of those archives.

| Scope | Geometries | Queries | Derivative coverage / permitted use |
|---|---:|---:|---|
| Initial training pilot | 19 | 1,251 | 719 full, 531 restricted, 1 moment-only |
| Remaining training snapshot | 88 | 1,797 | A separately registered later experiment |
| Entire reported snapshot | 107 | 3,048 | 1,790 full, 1,256 restricted, 1 direction-evidence-only, 1 moment-only |
| Reconstructed paper proxies | 5 | 320 | One report after configuration, budget and checkpoint rule freeze |
| Independent development | 0 admitted | 0 usable | 16 candidate original meshes are not admitted supervision |

Mini7 contributes 483 queries and the additional 12 complete-quota objects
contribute 768. Exact geometry IDs and lineage must come from the frozen index
and audit; this document does not infer an ID list from the counts. Formal
scale remains a target of 200 qualified training geometries and 40 independent
development geometries, not the status of this pilot.

## Implemented supervision contract

`UnifiedM80SQNO` now accepts the following training fields. They describe the
model API, not an assumed schema of `Faster107Corpus`.

| Field | Shape / meaning |
|---|---|
| `coefficients` | Raw native coefficients `[Q,10]`, or `[10]` for one query |
| `target_moments` | Raw moments `[Q,10]`; `valid_mask` selects supervised moment rows |
| `target_jacobian` | Optional raw-coefficient Jacobian `[Q,10,10]` |
| `jacobian_valid_mask` | Boolean `[Q]`, selecting full derivative rows |
| `restricted_basis` | Raw-coefficient directions `[Q,10,D]` |
| `restricted_jacobian` | Corresponding directional derivatives `[Q,10,D]` |
| `restricted_valid` | Boolean `[Q]`, disjoint from full derivative rows |
| `restricted_dimension` | Active prefix length per row; e.g. 3 or 8 in this review |
| `jacobian_scale` | Positive scalar or `[10,10]`, for full derivatives |
| `restricted_scale` | Independent positive scalar, `[10]` or `[10,1]`, for restricted output rows |

Teacher moments, Jacobians, bases, and normalizers are detached. Active rows
and active restricted columns are selected before subtraction, squaring, or
matrix multiplication in the derivative loss. Excluded NaN padding is never
converted into a zero target. Complex teacher labels are rejected before real
dtype conversion.

For a single query, restricted tensors still have explicit shape `[1,10,D]`.
Training-contract normalizers must always be finite and positive, including
when a particular batch has no active derivatives; the saved configuration
must remain valid for a later supervised batch.

Write the raw predicted Jacobian as `J_i`, restricted basis as `R_i`, and
restricted dimension as `d_i`. A full row contributes

```
full_i = mean_100(((J_i - Y_i) / full_scale)^2)
```

A restricted row contributes

```
restricted_i = mean_(10*d_i)(((J_i @ R_i[:, :d_i] - YR_i[:, :d_i]) / restricted_scale)^2)
```

The derivative task is the mean of these per-query losses over the disjoint
union of supervised full/restricted rows. Consequently an 8-direction row
does not receive 8/3 times the weight of a 3-direction row merely because it
has more recorded directions. No QR normalization or basis rotation is
performed. Projection uses the raw Jacobian before normalization.

With all derivative masks off, the derivative loss is a detached zero and
PCGrad omits that task; the moment and moment-containing branch auxiliary
objectives still train. `target_jacobian=None` also supports restricted-only
or moment-only batches. Event labels remain an independent contract; derivative
masks never create an event target.

The existing three-task objective remains

```
L = L_moment + w_J * L_derivative + w_aux * L_branch
L_branch = 0.5 * (L_M80_moment + L_SQNO_moment
                 + w_J * (L_M80_derivative + L_SQNO_derivative))
```

Both fixed and PCGrad modes use these same scalar tasks. PCGrad sum reduction
remains the default. Diagnostics report full/restricted branch losses,
supervised row counts, and the sum of active restricted dimensions. The paired
short-run utility uses the same loss implementation for initial/final
evaluation, preventing evaluation from silently dropping restricted labels.

The training contract is now `tvcg.unified-m80-sqno.training.v2`; it records
restricted normalization, raw coefficient coordinates, and per-query
supervision semantics. Strict resume includes the new loss source fingerprint
and rejects changed normalization or supervision semantics.

## Corrected native convex coordinates

The native input basis is
`(1,x,y,z,x^2,y^2,z^2,2yz,2xz,2xy)`, whereas reported moments end in
`(yz,xz,xy)`. Let

```
B = diag(1,1,1,1,1,1,1,2,2,2)
```

The scalar polynomial is `m(x).T @ B @ c`. Raw moments differentiated with
respect to native `c` produce `J_c`, which need not be symmetric. The convex
Hessian coordinates are `J_c @ B^-1`.

The native convex head now evaluates `t = D @ B @ c` and returns raw moments
`fixed + 0.5*design + D @ grad(Psi)(t)`. Hence
`J_c = D @ Hessian(Psi) @ D @ B`; its weighted Hessian is symmetric PSD. The
native regularizer, preflight/validation diagnostics, and solver bridge
convexity checks now use that same weighted matrix. Moment bounds,
positive-scale invariance, and complement checks remain in raw moment units.

If raw moments transform as `m_target = T @ m_model`, coefficients transform
as `c_model = B^-1 @ T.T @ B @ c_target`, and
`J_target = T @ J_model @ B^-1 @ T.T @ B`. Bridge tests check the doubled-cross
polynomial pointwise and check its raw coefficient Jacobian by finite
differences after rotation and translation.

Native semantics advance to v7. Legacy convex checkpoints are rejected rather
than silently reinterpreted in the corrected coordinates. Legacy bounded-odd
native checkpoints do not have this convex-head semantic change. Unified
strict source/architecture checks continue to apply.

These properties apply to the native convex head. A query-dependent fusion
gate or optional attention correction is not automatically a convex potential;
no global PSD claim is made for every unified configuration.

## Real-data checks still needed

1. **Actual coefficient storage.** The review says cross coefficients are
   doubled. Inspect the loader and teacher definition to establish whether the
   stored last three entries are native matrix entries `c`, or already-doubled
   monomial coefficients `B*c`. Do not multiply by two twice. Full Jacobians
   and restricted bases must follow the same chain rule as the inputs.
2. **Physical coordinates.** Confirm bbox-center translation, longest edge 1,
   density 1, positive retained side, moment order, and zero fixed moments.
   `geometry_from_columns` does not itself normalize a geometry to this frame.
3. **Encoder identity.** The present default uses RLE-derived volume fraction,
   occupancy, a coarse-occupancy EDT distance feature, and a boundary indicator,
   while retaining native intervals. It does not automatically read the
   snapshot's independently stored SDF. An external-SDF experiment needs its
   own alignment, sign, units, resampling, and channel contract.
4. **Teacher versus RLE representation.** M80 endpoint/moment bounds and SQNO
   mass are anchored to the RLE voxel-union domain. Verify that teacher total
   moments refer to that domain, or quantify the representation discrepancy.
   Do not call a mismatch between teacher mesh and RLE a pure learning error.
   Keep SQNO smoothing bias and any continuous-reference discrepancy separate.
5. **Real backward/update.** Use training-only records spanning full, 3D
   restricted, 8D restricted, and moment-only supervision. Check finite active
   losses/gradients, true parameter updates, teacher isolation, derivative-off
   moment training, fresh encoding after updates, and checkpoint next-step
   equivalence. Synthetic tests cannot replace this real-batch acceptance.

## Proposed bounded protocol, pending interface and physical preflight

This is a concrete next-run specification to freeze into a run manifest after
the checks above. It is not an executed experiment or an already-frozen
architecture. No real Faster107 optimizer step has been run in this increment.

| Setting | Proposed first exploratory comparison |
|---|---|
| Cohort | Exact 19 complete-quota IDs from the pinned index; no partial cohort expansion |
| Arms | Fixed and PCGrad, from identical weights and fresh AdamW state |
| Budget | 100 optimizer steps per arm; equal steps, not equal compute cost |
| Seeds | Initialization 830501; sampling 830502; per-step RNG starts at 830503 |
| Sampling | Geometry-balanced shuffled cycles; within each geometry two queries per each of five cut classes |
| Query budget | 10 queries per step, same precomputed batches in both arms; fail if a class cannot supply two |
| Optimizer | AdamW, learning rate `1e-5`, weight decay `1e-4`, gradient clip 1 |
| Loss weights | Moment 1, derivative 0.1, branch auxiliary 0.25; restricted supervision enabled |
| Normalization | Freeze train-only normalizers and hashes after reading actual labels; no benchmark-derived scales |
| Model | Freeze the complete native/SQNO/encoder configuration and dtype after physical/memory preflight |
| Checkpoint rule | Record initialization and final step 100; any intermediate recovery snapshots cannot select the result |
| Stopping / selection | Stop at the fixed budget; no validation-based early stopping or best checkpoint |
| Reporting | Initial/final training losses, full/restricted counts, gradient conflicts, module deltas, runtime and peak memory |
| Excluded uses | No paper-proxy tuning, held-out testing, OOD claim, or M69 feasibility claim |

For a geometry-balanced cycle, permute all 19 IDs before starting the next
cycle; do not sample globally from the 1,251 rows. Record the complete sampled
query IDs and class assignments in the run manifest. Report teacher validity
coverage alongside loss; never choose rows based on model prediction error.

The five paper proxies are Noodles v8, Heart H8/M21, 2014 Ellipsoid, Fig.9
Teapot and Fig.7 Cupid Heart. They remain available for one prespecified report
after configuration, budget and checkpoint rules are frozen, grouped by
geometry, cut class and derivative dimension. They are reconstructed proxies;
original CAD and teacher value equivalence are not established. None is an
iterative development set. Do not split queries from a training geometry or
move its template/repair family into development to fill the current gap.

## Obtain the missing consumer evidence without training

The new standard-library-only collector reads five fixed evidence paths,
requires the exact reviewed index SHA-256, copies their bytes, and statically
describes `Faster107Corpus` through Python AST. It never imports the consumer,
follows paths contained in JSON, opens NPZ/mesh archives, or runs the model.
Required files are index, audit and loader; smoke and benchmark coverage are
optional diagnostics. A missing required file or hash mismatch fails before
output creation. Existing output directories are not overwritten.

```bash
cd /data/xvxinyao/TVCG
ADAPTER=research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter

python "$ADAPTER/faster107_handoff.py" \
  --dataset-root /data/xvxinyao/Spin_test2/datasets/Spin_PINO_Rotation_Benchmark_v2 \
  --output runs/phase2_9_5/faster107_handoff_v1
```

Upload the resulting `runs/phase2_9_5/faster107_handoff_v1/handoff.json` to
continue the real consumer adapter. It embeds the actual index, audit, loader,
optional receipts, hashes, and static interface in one file. Its
`integration_ready` is explicitly false; successful collection establishes
an evidence handoff, not corpus or scientific admission.

Pinned index SHA-256:
`bafed1a784d66fa91f5044131f62abc2513e3cf7b6b594e9e196d960705501af`.

## Verification scope

Regression coverage includes mixed real/complex PCGrad parameters, weighted
native convex coordinates, affine coordinate changes, partial derivative
projection, ignored NaN padding, moment-only updates, strict resume,
same-input paired training, and read-only evidence collection. All executed
training in these tests uses synthetic fixtures. The full release gate keeps
all eight scientific gates blocked and now describes the reported pilot and
independent development limitation without claiming dataset admission.

The continuation review also corrected the optional native event-conditioned
Jacobian loss: it now selects positive-weight rows before subtraction and
squaring. Derivative-invalid rows, unknown event labels, and zero-weight event
classes can retain NaN placeholders without contaminating the active loss or
gradients. Event validity and derivative validity remain separate inputs.

Additional unified tests exercise independent Moment and derivative masks:
masking a Moment target does not discard that row's valid restricted derivative.
They also save a model after AdamW state exists, then compare uninterrupted and
resumed `mixed -> moment-only -> mixed` updates in fixed and PCGrad modes,
including real/complex optimizer state, scheduler, RNG and training metrics.
Exact CPU agreement in these fixtures is not a cross-device CUDA guarantee.

## Work-side guidance and local verification

The actionable Chinese handoff and remediation specification is
[Work data guidance, 2026-11-20](phase2_9_5_work_data_guidance_20261120.md).
It separates existing pilot evidence, model-side fixes, required source-data
evidence, and later formal dataset expansion. It does not ask Work to replace
valid restricted derivatives with fabricated full Jacobians.

After updating the development branch, run the full software gate from the
repository root in the existing training environment:

```bash
python -m research.NeuralMoments.models.pcraft_ufno.release_gate.run_release_gate \
  --profile full --output runs/phase2_9_5/release
```

The gate runs all native and adapter tests, including the mixed-supervision,
resume and handoff regressions. A successful report is a software result;
the source-machine consumer handoff and real-batch CPU/CUDA checks remain
necessary before the proposed 100-step-per-arm pilot.
