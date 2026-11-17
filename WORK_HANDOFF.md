# WORK_HANDOFF — Spin Neural Moments

Updated2026-11-16. Primary repository `/data/xvxinyao/Spin_test2`. Phases1–5 experimental/audit gates complete within recorded scope; Phase4 pending GPU smoke completed in Phase5. No optimizer steps, trained weights or generalization improvement established.

## Direction and preserved state

PINO is the sole learned core. Preserve SQNOv1.2 hybrid baseline. DensiCrafter supplies verified material/density mechanisms, not a teacher; TRELLIS supplies geometry/data. New experiments remain separate from frozen production code, datasets and checkpoints.

Branch `research/m80-pnmo-large-train-20261112`; HEAD `a1e7b097fa011fd62b93f0977d2005f18f7e4f43`. Existing four-line xl_ms_convex edit in `research/NeuralMoments/models/pcraft_ufno/spin_pino/native_pcraft.py` remains user work. Existing QuadrANN/upstream/runs untracked trees remain. No install, staging, commit or production edit.

Latest evidence `/data/xvxinyao/Spin_test2/reports/rotation_material_v1_20261116/preservation.json`:6442 repository files,246 external M83/M69 files,12 checkpoints and8 Phase4 source-manifest entries unchanged; HEAD and user diff unchanged. Protected baselines in `reports/work_audit_20261116/{baseline_files,external_preservation_baseline}.json`. Git/cache files and intentionally new reports/plan/handoff are outside the protected list.

## Prior audit and source gap

All report folders below are under `/data/xvxinyao/Spin_test2/reports/`:

- `work_audit_20261116/`:1956 C/C++ sources/headers (634 project/adapted,1322 libigl),0 CUDA sources,111 build files,44 binaries. Authorship not inferred. Frozen SQNO combines PINO and QuadrANN, scalar density; all67 qualified M83 records inspected usedrho=1.
- `m69_reference_recovery_20261116/`:11/11 test_oracle69.py tests, including finite meshes. Explicit root fixes ancestor collection delay; not whole M69 project coverage.
- `material_contracts_20261116/`:11 diagnostics, raw-query/density/fixed-shell math, established legacy smooth mass/scale violations; rotation prototypes then standalone.
- `source_gap_20261116/`:57 train/val geometries×32 queries, six within-object scale controls, frozen model/base smooth/base hard predictions. No held-out evaluation. Fresh train relativeL2 GSO0.218502 vsProcedural0.022364; base smooth similar, hard0.040994/0.005855. Native length/mass supports disjoint; no causal source attribution. All57 full-query mass violations concern legacy smooth model. Labels axis_adaptive; failed initial native_z diagnostic retained/corrected only in diagnostic runner.

## Phase4 and Phase5 implementation

Unchanged Phase4 code: `/data/xvxinyao/Spin_test2/research/NeuralMoments/experiments/pino_material_v1/`. Explicit physical density, disjoint spatial fixed/editable masks, world bounds and length; positive physical Gauss measure, bounded smooth retained fraction, learned per-cell width. Fixed moments added once; raw Jacobian/density paths verified. Uniform8³ cells are not a lossless M83 adapter. Untrained PINO, no graph/teacher, no homogeneous-convex-potential claim. Evidence `reports/pino_material_v1_20261116/`.

New Phase5 code: `/data/xvxinyao/Spin_test2/research/NeuralMoments/experiments/rotation_material_v1/`:
- `material.py`:sigmoid fraction times frozen physical base density; optional3³ filter normalized over editable neighbors. Fixed/outside gradients zero; no support expansion.
- `physics.py`:differentiable all10 task69 moment constraints and per-term losses; arbitrary axes/origins/contact. Immutable original mass/inertia reference, explicit training polar margin. Caller chooses weights; no optimizer or feasibility certificate.
- `corpus.py` and `data/synthetic_density_v1/`:12 deterministic8³ variable-density records,60 independent analytic hard-cut labels,8/2/2 train/validation/held_out. Full original and current full moments distinct. No model evaluation on any split.
- README and19 new tests; Phase4 tests remain unchanged.

Official DensiCrafter pinned to `21e6d8dc646129a9b21a38c2d5bee5bef2704d98` at `https://github.com/idvxlab/DensiCrafter` (MIT2025 Shengqi Dang). Read-only8-file source snapshot, license, blob/local hashes and mechanisms in Phase5 report. Two local files add final newline; explicitly recorded. Upstream sigmoid/smoothing/restricted updates adapted; contact-area growth/static support losses not imported as rotational objectives. No upstream code executed or teacher added. Upstream optimizer hardcodeslr0.05 despite configlr0.01; no reproduction claim.

## Validation and limits

Phase5 evidence: `/data/xvxinyao/Spin_test2/reports/rotation_material_v1_20261116/`:RESULT, commands, validation_summary, CPU/GPU per-term JSON, source audits, review, preserved failures and file hashes.

Full experimental suite40 passed (21 Phase4+19 Phase5); frozen legacy40 passed/1 skipped; frozen test_task69.py6 passed. Independent reviewer repeated40/40 plus batched/degenerate/invalid-input checks, no substantive finding. Seven active objectives reach material logits, raw queries, head and PINO spectral parameters; finite differences pass. All10 constraints matchtask69 within1.11e-16 CPU/GPU. Temporary in-memory finite-difference perturbations restored and checked,0 optimizer steps.

ActualRTX5090D: Phase4 smoke now passed12 diagnostics (peak90.95MiB,~0.56s); new physics audit peak18.25MiB,~0.91s.10% allocator cap/90s timeout; shared-device fixture timings are not performance benchmarks. Phase5 TF32 disabled; float64 physical math withfloat32/complex64 encoder.

Decisions/limits: dirty primary checkout retained with separate experiment directories; normalized editable smoothing deliberately differs from upstream; variable density is an experimental relaxation, not M69 single-density material-policy compliance. Connectivity, binary conversion, manufacturing, optimizer convergence and population generalization are unverified. Smooth-cut bias and coarse geometry error remain. Inactive losses correctly have zero gradients; eigenvalue derivatives may be nonsmooth at degeneracy. Small corpus validates schema/labels only.

## Next actions

Phase6 first predeclare shared geometry/query distribution, independent hard-reference and quadrature checks, seeds, small compute budget and baseline controls. Separate continuous density relaxation from physical hard-removal evaluation. Then bounded optimization with per-term monitoring and independent connectivity/material checks; hold out test for declared evaluation. No long training starts automatically from passed derivative tests. GSO/Procedural gap still needs common-support controls and representative data.

Design/ledger: `/data/xvxinyao/Spin_test2/docs/superpowers/plans/2026-11-16-rotation-material-v1.md`; staged plan sibling `2026-11-16-spin-pino-roadmap.md`; mathematical contract sibling `2026-11-16-spin-material-contract.md`.

Interpreter `/data/xvxinyao/Spin_test2/research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/.venv/bin/python` (Python3.13.5,torch2.8.0+cu128). Use2 CPU threads, PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=/data/xvxinyao/Spin_test2:/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src. Validators require fresh output folders; exact commands saved.

M83 `/data/xvxinyao/Spin_M68_Trellis_Data_Kit/runs/m83/m83-dataset-v2-full-20261113T124505Z/operator-manifest.json`.
M69 `/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src/oracle69.py`, physical reference sibling `task69.py`.
Frozen checkpoint root `/data/xvxinyao/Spin_test2/research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/training_v12/runs/moment_v12_memory_optimized/`.


## 2026-11-16 — Unified research-control initialization (current entry point)

Current control source of truth: `/data/xvxinyao/Spin_test2/research_control/PROJECT_STATE.md`; read `NEXT_ACTIONS.md` for Chat↔Work protocol and required branch/commit/changed-files/tests/training/checkpoint/dataset-fingerprint packet. Companion pages: `RESEARCH_ROADMAP.md`, `EXPERIMENT_LEDGER.md`, `DECISIONS.md`, `DATASET_STATUS.md`, `ARCHITECTURE.md`. Historical content above is preserved byte-for-byte.

Primary scope is now explicitly frozen to 3D rotation-stability topology optimization: canonical 2014/2024 spinning tops, TRELLIS, M83/GSO and Procedural. Exclude yo-yo, 2D fidget spinner, static/buoyant/deformation tasks from primary scope. PINO sole core; SQNO v1.2 immutable baseline; M69 exact represented-domain reference; DensiCrafter mechanism reference. Strict single-density removal-only is a target, not a property established by the variable-density relaxation/M90 audits.

Phase 1–5 remain completed only within recorded experimental/audit gates; Phase 6 / Spin-PINO Rotation Benchmark v1 is planned. M90 removal and moment reports are present, but exact M90.1.2↔M90.1.3 report naming is unverified. No new tests of model behavior, training, production edits or checkpoint/data changes in this documentation session. Git HEAD `a1e7b097fa011fd62b93f0977d2005f18f7e4f43` on `research/m80-pnmo-large-train-20261112`; existing dirty files retained. Before/after Git, evidence hashes and fresh preservation verification: `/data/xvxinyao/Spin_test2/research_control/verification.json`. Next: case admission and strict material/benchmark protocol, with no automatic compute launch.


## 2026-11-16 — Rotation candidate inventory and strict-removal design

Completed asset/provenance preparation in `/data/xvxinyao/Spin_test2/reports/rotation_admission_20261116/`: 11 reconstructed canonical task candidates (2014:9, 2024:2), 9 distinct geometry file hashes; zero formal canonical admissions. Three tea-pot task variants share one geometry. M83 has 67 qualified operator records, all designs not_attempted; frozen splits/groups retained. Inspected TRELLIS/TRELLIS2 manifests have 256 pending_reference entries, zero mesh paths/attempts. Scope exclusions include deformation and multi-density top fixtures as well as nonrotation tasks.

`STRICT_REMOVAL_ACCEPTANCE.md` is a proposed validator design, not an implementation/test pass. It requires one original density, binary deletion contained in editable support, frozen shell/reference, independent saved-file moments and whole-solid connectivity; retained-editable export and deleted-RLE audit inputs are distinguished. Historical paper-resolution/cage flags are not new reproduction evidence. No production edits, training, data/checkpoint changes, commit or push. Evidence hashes, builder review and preservation receipt remain alongside the result.

Current source of truth: research_control/PROJECT_STATE.md and NEXT_ACTIONS.md. Suggested Chat decision: use common M69 objective for primary results and original paper objectives separately; choose first adapter before bounded validator implementation. A self-contained packet is `/data/xvxinyao/Spin_test2/reports/rotation_admission_20261116/CHAT_HANDOFF.md`. Switching is optional; no chat has been created.


## 2026-11-16T19:12:44.017462+08:00 — M90.2 independent saved-result validator

Current execution supersedes the earlier implementation-pending entry for the bounded validator only. New code: `/data/xvxinyao/Spin_test2/research/NeuralMoments/experiments/m90_2_saved_result_validator`. Evidence: `/data/xvxinyao/Spin_test2/reports/m90_2_saved_result_validator_20261116/RESULT.md`, `machine_report.json`, `preservation.json`. Independent A0–A6 adapter for hash-pinned saved deletion JSON/RLE, homogeneous original material, strict support, float64 ten-moment cross-checks, unchanged complete TaskSpec and independently reconstructed whole-solid six-face topology. Explicit retained-editable complement tested against actual oracle69; caller connectivity booleans never certify acceptance.

Fresh tests: 24 deterministic validator tests passed; 33 saved audits (4 pass, 22 expected fail, 7 expected pending); frozen task69 6 passed. No model/training regression claimed. Review counterexamples and fixes preserved. Engineering 8-cubed fixtures only; SI units and all coordinates already in registered task frame are required. Frame conversion is provenance, not independently reproduced. Same-representation tolerance 1e-10; no TaskSpec relaxation.

Actual first eligible M83 training record `gso70-bird_rattle-ac8b085942-54a52d35971a` checked read-only. All 10 actual TaskSpec conditions evaluated on a new empty-deletion full-retention diagnostic; failed minimum_removal, axis_alignment_com, axis_alignment_inertia and principal_dominance. No optimized design or benchmark admission. Native512 topology/build not started: historical workspace estimate 1,342,177,280 bytes exceeds 1 GiB; dense adapter preflight also refuses. Native original occupied-cell density evidence, explicit SI registration and mixed mesh/voxel discretization budget remain pending.

CPU only, <=2 threads/CPU affinity, 120 s process-group deadline, 1 GiB address space/output caps. Final controls 0.308 s, peak RSS 88981504 bytes. GPU was observed read-only (RTX 5090 D, 5990 MiB used, 12% utilization at initial check); this session did not run GPU/optimizer/training. Existing background activity not attributed to this task. No reset/delete/stage/commit/push, no production/Oracle/data/checkpoint edits. Branch `research/m80-pnmo-large-train-20261112`, HEAD `a1e7b097fa011fd62b93f0977d2005f18f7e4f43`; pre-existing native_pcraft.py edit preserved.

Next bounded work: register a real input's task-frame units, complete original material/outer reference, and a same/mixed-representation error budget; design a native sparse/extended-grid topology adapter within an explicitly approved resource budget before claiming real-case acceptance. No automatic optimization or training. Only represented-domain rotation/material acceptance is permitted; manufacturing, dynamic stability, exact CAD and generalization remain unverified.


## 2026-11-16T19:35:18.306703+08:00 — M90.3 sparse native topology and Chat decision handoff

User requested continued research and proactive notice when Chat discussion is appropriate. Routine engineering continues in Work; recommend Chat for changes to scientific objectives/comparison policy, interpretation, numerical admission thresholds, or new compute/training budgets. Supply evidence, open decisions and a return-to-Work instruction; do not force a switch or imply automatic context transfer.

New isolated source `/data/xvxinyao/Spin_test2/research/NeuralMoments/experiments/m90_3_sparse_topology`; evidence `/data/xvxinyao/Spin_test2/reports/m90_3_sparse_topology_20261116/RESULT.md`, `machine_report.json`, `preservation.json`, `CHAT_HANDOFF.md`. 11 fresh deterministic tests pass; 33 pinned M90.2 control outcomes preserved through new guarded entry, plus a saved vertical-boundary counterexample correctly downgraded from old pass to pending. Old M90.2 code/reports are unchanged. New acceptance entry is `guarded_validator.py`, which refuses ambiguous vertical-surface centres and clears stale pass claims.

Native512 analytic cube/cavity passed in 7.383s, 137.57MiB RSS; max normalized analytic ten-moment error 4.96e-13. Actual fixed first M83 training case `gso70-bird_rattle-ac8b085942-54a52d35971a` completed in 3.215s, 98.16MiB: original/full-retained 7,620,873 voxels, one six-face component, editable subset confirmed. No dense512 or compilation, GPU, optimizer/training; CPU caps unchanged.

This closes one native512 full-retention topology diagnostic, not real-case acceptance. Full retention still fails frozen TaskSpec; no nontrivial saved M83 design attempted. Source `origin.source_coordinate_units` explicitly declares unknown OBJ physical units and dimensionless task use; this corrects any impression that no unit-policy metadata existed. New rays use exact centres; frozen mesh68/legacy stream_decode use epsilon offsets. Equal voxel totals are not a membership equivalence certificate. Observed mesh/voxel original ten-moment discrepancy max 1.51298e-05 is descriptive only; no admission threshold chosen after outcomes.

Current recommended Chat decision: explicitly separate a dimensionless source-unit primary track from SI-scale physical claims, then preregister mixed-reference convergence/error admission. No SI unit inference, TaskSpec relaxation, training, manufacturing or dynamic-stability claim. Preserve branch/HEAD/user diff/checkpoints/datasets; full receipt below report root.


## 2026-11-16T20:11:56.640340+08:00 — M90.4 source audit; protocol pending missing Chat decisions

User requested M90.4.1–M90.4.3 under six Chat-frozen decisions. read_thread returned assistant placeholders, and the available M90.3 handoff contains proposals rather than those decisions. Requested exact six-decision text; do not infer it or freeze thresholds from old observations.

Actual evidence in `/data/xvxinyao/Spin_test2/reports/m90_4_protocol_audit_20261116`; new isolated characterization source `/data/xvxinyao/Spin_test2/research/NeuralMoments/experiments/m90_4_units_audit`. All TaskSpec fields and listed geometry/caller units audited. Key findings: final audit uses outer L, Oracle default uses1.02L, encoder uses editable extent; COM pass/fail can differ at the boundary. Mesh68 protect_radius is a contact-centred 3-D ball. Absolute numerical floors prevent a blanket claim of pipeline scale invariance. Source units remain undeclared and density registered, not measured.

4 new unit tests +6 frozen TaskSpec tests +11 M90.3 sparse tests passed under2 CPU/120s/1GiB caps. M90.3 source manifest and12 checkpoints reverified. No GPU/optimizer/training or new convergence experiment. Requested128/256/384/512 rows are all not_run/pending. `M90_4_PROTOCOL.md` exists as an explicitly NOT FROZEN pending record, SHA256 `35f44aa8c0c287653b775fdfe7759aac3f09a61b266943328990b9d80247cf00`; this is not a frozen-contract hash.

Next action: recover the six exact decisions, then freeze the protocol before experiments, without changing TaskSpec or fitting thresholds. User authorization to execute remains active; only missing scientific content blocks dependent work. Preserve all prior evidence and user edits. See machine_report.json/preservation.json for actual Git and hashes.


## 2026-11-16T20:37:39.610615+08:00 — M90.4 executed under exact user-frozen A–F

The previous missing-decision blocker is resolved by the user's exact A–F message. Sole authority: `/data/xvxinyao/Spin_test2/reports/m90_4_convergence_20261116/protocol.json`; freeze SHA256 `97ca9c5a1d2d384841f740e6e1e9ad34585c7685f9af89b1f7279884859014e3`. The old pending protocol/report remains immutable historical evidence. Current scale contract is Source-coordinate + Registered-density + M69; SI track remains separately pending genuine physical size/density evidence. Final audit uses original outer L, Oracle default1.02L and encoder editable extent are distinct; actual run path explicitly passes outer L.

Completed two backends × two families × four grids128/256/384/512 × five ray phases =80 phase records. Sparse wrapper reuses frozen Mesh68 Tree/ray outer classification, not the full dense importer; Exact-centre remains separately identified.400 saved RLE artifacts verified; all10 moments, volumes, six-face topology, XOR/Jaccard and adjacent-grid changes reported. Analytic intact/internal-cavity controls have one retained component; disconnecting slab negative control has two. Real case is only pinned first eligible M83 training record, full retention.

Nominal Mesh68/Exact-centre memberships match on every tested grid; this is bounded evidence, not arbitrary-mesh equivalence. M83 moment error is nonmonotone:128 0.000268246549,256 1.7862244e-05,384 5.70558982e-05,512 1.51297718e-05. All are evaluated against unchanged pre-run precision screen1e-11 derived from0.1*min(TaskSpec mass/inertia tolerance); both families fail precision. Real512 negative Mesh68 offset flips1 voxel; ±1e-4-cell diagonal offsets flip32/40 voxels. Real phase-robustness screen fails; topology remains stable. No threshold or TaskSpec relaxation.

60 final relevant tests passed (15 new,24 validator,11 sparse,4 units,6 frozen TaskSpec). Longest experiment23.900s; peak199.09MiB; CPU<=2. No GPU/optimizer/training. Fresh reviewer found2 aggregation missing-evidence issues, fixed RED→GREEN without changing measurements; native multi-interval subcell-gap differential remains a deferred minor.

Current outcome: bounded M90.4 protocol/convergence deliverables complete; geometry representation screen FAIL, real M69 material-design acceptance PENDING. Frozen full-retention TaskSpec still fails minimum_removal/COM/inertia-alignment/principal-dominance. Do not advance to design/dataset/training under decisionF. Next appropriate Chat discussion: representation/error propagation strategy under the frozen task; any alternative must be prospectively registered, not tune the failed threshold to observed errors. Reuse current evidence in `/data/xvxinyao/Spin_test2/reports/m90_4_convergence_20261116/CHAT_HANDOFF.md`.


## 2026-11-16 — bounded Rotation Benchmark v1 delivery

This entry supersedes earlier “inventory/design complete, implementation pending” status for this bounded dataset task only. Dataset: `/data/xvxinyao/Spin_test2/datasets/Spin_PINO_Rotation_Benchmark_v1`. Read `DATASET_README.md`, `CHAT_HANDOFF.md`, `admission_report.json` and `evidence/preservation.json`; the complete canonical training corpus remains incomplete. No frozen model/Oracle or existing dataset/checkpoint changes; no optimizer, training, GPU generation, Git stage/commit/push.

374 catalog records: 12 reconstructed canonical records (11 3D candidates, one 2D Flower reference), 64 exact upstream GSO sources, 42 inspired/procedural records, 256 pending TRELLIS references. Exact original-author paper assets: zero. Noodles remains a 3D primary candidate; Flower is explicitly excluded from 3D labels. Canonical axes/contact/material registrations remain pending.

Two new explicitly preregistered analytic box rotors are geometry-admitted and have feasible nontrivial removal-only binary states under all ten frozen M69 TaskSpec checks plus A0–A6 guards. Native512, one train family, removal 12.5% / 15%; 6 query labels and 2 material Jacobians. These are numerical controls, not fabricated tops, optimized mass minima or generalization results. No arbitrary real-mesh admission is implied.

23 new tests + 60 frozen regressions passed; 7 schema artifacts and 3 negative controls passed; fresh independent numerical reconstruction produced 86 byte-identical arrays and matching label values. Independent read-only review cleared important findings after fixes. M83 104 source records/splits remain intact, 67 task-compatible operator inputs, zero new M83 admissions. Historical M90.4 real-case precision failure remains in force. TRELLIS preflights remain blocked, zero new generated meshes.

Next work requires verified original paper assets/licensing and explicit canonical TaskSpecs; a preregistered real-mesh representation error policy and nontrivial saved states for M83; pinned TRELLIS dependencies/weights and verified reference inputs. See dataset handoff for exact hashes, files, resource receipts and reproduction commands. Do not start training from the 374-row catalog; only the two-state training_manifest is admitted.


## 2026-11-16 — category-by-category construction and proactive Chat handoff

User requests building the previously defined source categories one at a time and proactively flagging appropriate Chat research discussions. Latest worklist/decision packet: `/data/xvxinyao/Spin_test2/reports/rotation_category_workflow_20261116/CATEGORY_HANDOFF.md`. M83 is a corpus version, not a separate source: its 32 training Procedural + 8 held-out design-panel records belong to Procedural; its 64 GSO records belong to GSO. Suggested processing order: canonical paper cases, Procedural, GSO, TRELLIS/TRELLIS2. Keep Flower 2D reference-only. A source/task blocker is recorded per object and does not block independent categories. This turn organizes 374 existing records into four disjoint worklists; no new samples or training claimed.

Track per category actual meshes, unique geometry/family coverage, task readiness, geometry admission, feasible material states, failures and splits. Continue routine engineering in Work. Proactively recommend Chat for coverage/comparison decisions, physics/acceptance-policy changes, contradictory findings, meaningful compute/training expansion, and category-level scientific review. Supply evidence, exact questions/options/recommendation and return-to-Work instructions; do not assume automatic context transfer or a background monitor. Current Chat-worthy blocker: real-mesh representation error under frozen M90.4 constraints. Preserve failed evidence and thresholds; do not silently redefine the primary physical reference or loosen tolerances.


## 2026-11-16 — category 1 reconstructed geometry pack delivered

User explicitly requested proceeding with canonical paper geometry while Chat separately handles real-mesh acceptance, and permits transparent reconstruction when author models cannot be obtained. New isolated pack: `/data/xvxinyao/Spin_test2/datasets/Spin_PINO_Canonical_Geometry_v1`; see DATASET_README.md, RESULT.md, catalog.json, mesh_audits.json and evidence/preservation.json. The earlier Rotation Benchmark v1 release is unchanged.

Expanded read-only discovery found better historical inputs under `/data/xvxinyao/moment-guided-spin-v1/code/make-it-balance-paper-sim/`, omitted by the earlier primary-repo proxy catalog: actual self-built Heart+shaft, Utah-source Teapot reconstruction, and video-referenced Noodles V8. Referenced by verified stable path/hash, not copied. Newly generated a figure-guided tilted Ellipsoid+shaft and a multiring petal Flower polygon/SVG; Flower's OBJ is display-only 2D auxiliary. Total 5 selected identities (4 3D +1 2D), covering 8 of 12 canonical rows; Teapot variants/shared Heart do not increase geometry count. Original-author assets0. Armadillo2poses, Elephant, Dancing Couple remain4pending rows; source-family leads and historical failure evidence retained.

ALL canonical benchmark geometry/material admissions remain false; new Oracle labels0. Outer-surface preflights pass for5selected identities. Complete selected-role checks: Heart/Noodles pass within limited combinatorial preflight; Teapot fixed.obj fails with2degenerate triangles; Ellipsoid/Flower have no material-region pair and remainpartial. Historical Noodles wall policy ZERO_WALL_RESEARCH_V1 cannot be inherited as current fixed-shell acceptance. Surface face components are not solid connectivity; no new arbitrary-mesh self-intersection or native512/M69 certificate claimed.

19 tests pass; catalog schema12records +3negative controls pass. Four generated geometry files independently reproduce byte-identical hashes. Fresh read-only review recomputed11mesh audits and checked42source pins; one important per-role reporting issue fixed RED/GREEN and re-reviewed clean. Actual mesh previews inspected. CPU only, build peak about314.45MiB; no optimization/training/GPU inference. Preserve tracked user diff, model/oracle, old datasets/checkpoints and prior reports; no reset/delete/stage/commit/push.

Next category1 work: fix Teapot fixed-region representation only in a new version with explicit verification; register complete fixed/editable/protected regions for new Ellipsoid; resolve paper-specific animal poses/self-intersections and Couple geometry. Keep Chat's representation/physics decision separate, then bind complete M69 TaskSpecs and run guarded feasible-state admission. The geometry pack is usable for inspection/preparation, not a training corpus.


## 2026-11-16 Canonical fidelity correction (user visual review)
Prior geometry-pack selections were not adequate evidence of paper fidelity. Prefer the new selection overlay in reports/canonical_fidelity_correction_20261116/selection_correction.json over the old catalog selection conclusions. Later 2024 Heart rounded-clearance-v3, 2014 Teapot lower-spindle 20261019, and image-derived Flower 48-hole/three-layer reference were located. Withdraw shared 2014/2024 Heart matching: 2014 Cupid-arrow remains pending. Noodles V8 remains candidate-only: opaque envelope obscured its strands, but geometry differences remain. No appearance acceptance, M69 admission, or labels added. Flower stays 2D reference-only. 107 historical source identities matched; existing release checksums 201+54 passed. Immutable dataset files preserved. See RESULT.md for evidence and limits.


## 2026-11-16 Heart rebuild and four-model paper-fidelity audit
User requested rebuilding visibly inaccurate Faster Heart and evaluating four objects. New isolated dataset-preparation pack: datasets/Spin_PINO_Canonical_Heart_Rebuild_v2. Final is generated/smooth/heart-reference-v2.obj, hash fb981c393e6555d9d7720c8c5925c41298931be98c9e4f2df71aef6655cfcba9. Reference-conditioned 2.5D solid with explicitly assumed Poisson depth and upper valley, NOT recovered author CAD. Draft nominal is superseded. Same annotation IoU old bounded-camera fit .79946/new .99691 is FIT ONLY; no independent 3D fidelity claim. Depth/width .24/.48 gives identical projected outline but volumes .09197/.18225, demonstrating non-identifiability. 2014 Cupid-arrow Heart remains separate pending.
Four-model measurements refer to latest selection correction, not original v1 proxies: Teapot lower-spindle saved-mask IoU .93962; Flower image-derived profile 48 visible holes, roundtrip IoU .89731 and boundary P95 1px (thin-feature raster sensitivity); Noodles ratios measure preview-02 source components including editable envelope, NOT final optimized material. All appearance/M69 flags false, new labels0; Flower stays2D reference-only. Seven tests pass and final mesh independently byte-reproduces. Fresh read-only reviewer verified mesh/projection. Old release files and user model change preserved.
Chat research handoff: define scientific reconstruction error/uncertainty acceptance and separate reconstructed comparison group if author CAD is unavailable. Do not relax frozen M69 thresholds or infer task metadata. Work continues routine geometry/measurement independently. See pack RESULT.md for quantitative evidence and exact limitations.


## 2026-11-16 Heart grip visual rejection / actual connection checked
User reports apparent gap between grip and heart. Read-only actual OBJ checks: one vertex-connected component, watertight/winding-consistent, one closed central X=0 section, 11 selected horizontal sections each single-loop. No disconnected grip found. Apparent floating: only few-pixel neck and shaft occlusion in opaque solid; paper reference uses translucent shell/design display. Do not infer all paper-visible shaft is physically exposed. Image-inflated depth remains unverified; user visual acceptance FAILED/pending. No geometry change and no admission. See reports/heart_connection_diagnostic_20261116/RESULT.md and actual section evidence. All 47 v2 release checksum entries unchanged. Next correction needs true depth/occlusion constraints, not a higher same-view silhouette score.


## 2026-11-16 — category 1 executable input dataset v2

Current category-1 input-preparation entry: `/data/xvxinyao/Spin_test2/datasets/Spin_PINO_Canonical_Dataset_v2`. Read DATASET_README.md and admission_report.json. Consolidates latest selection correction, projected Heart-v2 plus coherent earlier Heart alternative, lower-spindle Teapot, Noodles initial/fixed/design, Ellipsoid and image-derived Flower. Does not overwrite previous releases or claim their old selections are current.

12 paper rows,10 case identities,8 source families;5 reconstructed candidate cases (4 3D+1 2D),5 pending.6 coherent version bundles reference13 files without copying meshes.12 fresh OBJ audits:9 pass limited preflight;3 Flower display layers fail with2 degenerate triangles each. Native 48-hole Flower planar JSON/SVG exported separately; dimensionality remains2D. No new3D meshes, original-author assets0, appearance/M69 geometry/material admissions0, new Oracle labels0; training manifest empty.

Heart's outward shape is input CAD, not optimization output. Projected Heart-v2 remains unaccepted2.5D geometry; never inherit earlier fixed/design roles. Old Heart/Noodles fixed+design ten-moment closure residuals2.55e-15/8.64e-15 are diagnostics only, not spatial partition or fixed-shell certificates. Noodles ZERO_WALL_RESEARCH_V1 remains incompatible by inheritance. All canonical families stay reference-only; M83 split unchanged.

17 tests pass; schema plus3 negative controls pass;23 deterministic build artifacts independently reproduce byte-identically. Fresh reviewer cleared2 guard defects after3 observed red tests then green.315 prior release/control/user files checked before append,12 historical checkpoints and frozen M83 manifest rehashed unchanged. Existing native_pcraft.py edit and Git branch/HEAD preserved. CPU-only build2.62s,peak547.06MiB; no GPU/optimizer/training/stage/commit/push.

Remaining external inputs: author CAD or adequate multi-view/dimensional evidence, paper-specific animal/Cupid-arrow poses, matching fixed/design/protection regions, explicit TaskSpecs and native representation acceptance. Existing M90.4 block remains. Routine input engineering stays in Work; scientific reconstruction/representation decisions continue in the already active Chat, with CHAT_HANDOFF.md for concrete context.


## 2026-11-17 — Procedural category 2 bounded dataset
Category1 input-preparation passes only; paper-fidelity and physics gates remain pending. New isolated datasets/Spin_PINO_Procedural_Dataset_v1:46catalog records=40M83 frozen32/4/4 +2existing controls +4newmeshes. Four geometry/material admissions total, two new; curves2 and M8340 pending. Cross/stepped removal11.1111%/17.3077%, polar retention98.3539%/94.7809%, unchanged fullM69 and A0-A6 pass. Shared generator family train; no generalization/dynamic/fabrication claim. No GPU/training/optimization; no held-out tensor decoding.
20tests+3schema negative controls pass;90mesh/RLE files byte-reproduce;6queries+2Jacobians numerical-identical. Independent guard review fixes revalidated.370previous files and12checkpoints plusM83manifest unchanged before this append. Existing user model edit/Git branch/HEAD preserved. See dataset RESULT.md, DATASET_README.md, evidence/preservation*.json, CHAT_HANDOFF.md. Next: explicit curved representation/shell acceptance and scientific family coverage decisions in existing Chat; do not inflate this4-state control pilot into final training corpus.


## 2026-11-17 — user rejects Heart v2 and simplistic category2; v3 visual review
User rejected the projected Heart and simple Procedural pilot as formal corpus. Do not call either category scientifically complete. Current selection: datasets/Spin_PINO_Canonical_Heart_Rebuild_v3/catalog.json and dataset_selection_correction.json. Heart v2 is withdrawn; new true3D intrinsic-heart v3 is a candidate awaiting user visual review, not originalCAD or M69-admitted. One closed connected welded exterior; two shaft sections single closed loops;3OBJs independently byte-reproduce;10tests pass. No new TaskSpec/fixed-shell/design masks/Oracle labels.
Six simple Procedural objects (2oldboxes+cross+step+2lathes) excluded_control_only. Four numeric feasible states remain valid pipeline evidence but contribute0formal samples.40M83records remain individually unreviewed/pending. formal_training_manifest.json is empty for these reviewed benchmark sources; production datasets untouched.
First-category actual-mesh browser and direct local files are delivered in outputs/CLASS1_MODEL_VIEWER.html and CLASS1_MODELS;Teapot display-only +Y-to+Z transform,Noodles structural view hides envelope,Flower stays2D. Full OBJ sources unchanged.568previous files+12checkpoints+6frozen dependencies preserved before append. Old datasets/reports remain immutable; only this latest selection overrides suitability claims. See RESULT.md for assumptions, failed resource-cap attempts, successful2CPU/2GiB bounded reproduction and source hashes.


## 2026-11-17 — 最新：TRELLIS-only v2 + 普通对象先行

本节覆盖旧分类中关于训练来源的规划；保留旧文以供审计。研究标题仍为 Physics-Informed Neural Operators for Accelerating Rotation-Stable Topology Optimization。继续 Dataset Construction，不改核心架构。

正式训练与 Development Validation 只接受 TRELLIS/TRELLIS2。2014 六例仅 Canonical Validation；Faster Heart/Noodles 仅 Locked Test；GSO 仅原始 Test；Procedural 仅诊断；Flower 二维参考。

用户最新纠正：先生成普通动物、原创角色/生物等三维对象，再独立准备旋转任务，最后优化内部材料。已实际生成的 20 个完整陀螺形 TRELLIS 网格全部降为诊断资产，正式训练/开发候选 0，不以更名计入。新的普通对象生成队列 24 条，真实普通对象网格 0。历史 256 条 reference 仍然是 0 网格。

诊断资产：20 个不同几何哈希/20 个名义家族，家族独立性未认证；表面网关通过 4，多边界壳拒绝 15，自交拒绝 1。TaskSpec-ready 0、物理几何准入 0、可行材料状态 0、Oracle 标签 0。f07 的 512 稀疏表示十矩归一化误差 4.14e-4，既有 1e-11 筛查未通过；64³ SDF 仅诊断。同种子单例推理复跑不位级一致，原始输出均固定哈希。

目录 datasets/Spin_PINO_Rotation_Benchmark_v2；交付 README、全部正式/外部/锁定清单、来源、注册表、失败/泄漏报告、16 项通过测试、查看器、DATASET_HANDOFF 与 CHAT_HANDOFF。没有启动正式训练。Final 仍为 200 Train+40 Dev，每几何128–256状态；Pilot 必须先物理合格，不能通过诊断模型或重复状态补数。

下一步：普通对象小批生成及家族复核；由现有 Chat 明确输入准备（轴/接触点/可选附件）、完整 TaskSpec/壳与保护区域、多边界语义及表示精度。禁止默认猜测或放宽。未自动发送交接文件、未创建新 Work。正式材料接入尚未完成，验证器明确拒绝无证据准入。


## 2026-11-17 最新批次更新：24 个普通对象已真实生成

普通对象不再是待生成：ordinary_pilot_v1 中已完成24个参考图和24个真实TRELLIS OBJ，零生成失败，24个不同几何哈希；20个训练方向候选＋4个开发验证方向候选。2个通过现有表面网关（貘、河马），19多壳、2相交、1不闭合被拒。物理任务就绪、正式训练准入、材料状态和Oracle标签仍全部为0。旧20直接陀螺继续诊断排除。

本段覆盖上文关于普通对象“0个/未生成”的历史状态，详见 ordinary_pilot_v1/README.md、summary.json 和 DATASET_HANDOFF.md。研究标题和TRELLIS-only策略不变。已记录所有来源/哈希/种子与20项通过测试，生成图未冒充网格。下一步是外形/组件复核及明确旋转任务准备；不能按24个已合格Pilot计算。


## 2026-11-17 — ordinary24 conservative refinement completed; physical Pilot still pending

Latest overlay: datasets/Spin_PINO_Rotation_Benchmark_v2/ordinary_refinement_v1/catalog.json. 24 original TRELLIS objects preserved; 12 locally modified derivatives and12unchanged source references, no new independent objects/families.11boundary components locally repaired,43tiny external fragments removed with retained evidence. F01 floating specks removed, cavities preserved; F10 three small openings explicitly patched. Whole meshes watertight24/24; original single-shell gateway2→5 (F02,F03,F04,F07,F23).22diagnostic SDF/RLE datasets completed; F16/F18 intersection issues remain. Cat F05 legitimate shared-vertex contacts exact-refined using existing M70 code, no production changes.

192 diagnostic occupancy single6-face component7/22; these are finite-grid diagnostics, not exact source topology. Tapir384 becomes1component versus5at192; rabbit384has2versus7. Both remain0internal cells under2%longest-extent clearance sweep. Sweeps measure distance to empty-cell centers, not certified safe masks or assigned physical shell thickness. Actual sections reveal thin shells/cavities. Do not label the batch as physically admitted or start bulk labels/training. TaskSpec-ready,geometry-admitted,feasible material states,Oracle labels all0.

New original/refined comparison viewer in outputs/ORDINARY_3D_24_REFINED_VIEWER.html. Tests16new+20existing pass; independent read-only review verified22SDF/RLE roundtrips and all artifact bindings/counts. Initial trimesh split implicit-hole-fill issue isolated toF10 and corrected with repair=False + explicit provenance. Existing raw assets/model code/checkpoints/frozen manifests/Git changes preserved.

Next: existing Chat must define initial-solid preparation (which generated cavities may be filled, which designed cavities must remain, permitted thickening/shape error), explicit axes/origin/contact/units/rho0/shell/protection, legitimate multi-boundary semantics and native representation precision. Concrete CHAT_HANDOFF.md in refinement pack; not automatically sent. Continue Dataset Construction Work; no Architecture Research Work and no physical-assumption change.
