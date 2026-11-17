# M69 fixed-axis TaskSpec: verified registration and semantic contract

Source SHA-256: 5463dcfe57f3aea21f577fd77d26867ef9e7ec18098b7ff25e7bde5209fc3389  
Task hash: 54a52d35971aacf83e5ab3dc52dcf11e686724ad0199201340009a2bd342b38d  
Schema: m69.fixed-axis-task.v1; native resolution: 512.

## Verified task values

The complete TaskSpec returned by the user's hash-pinned local extraction is stored in m69_fixed_axis_task_registration_v1.json.

| Field | Value | Interpretation |
|---|---:|---|
| axis | (0,0,1) | Fixed source/task-frame axis |
| axis_origin | (0,0,0) | Fixed-axis origin |
| contact | (0,0,0) | On prescribed axis; coincides with origin here |
| density | 1.0 | Registered TaskSpec scalar, not evidence of kg/m³ |
| shell_fraction | 0.03 | Frozen parameter forwarded to mesh import |
| protect_radius_fraction | 0.04 | Contact-centred protection radius fraction per M90 task notes |
| protect_height_fraction | 0 | No additional axial-height protection fraction |
| polar_retention | 0.90 | Minimum retained fixed-axis polar inertia ratio |
| min_removal_fraction | 0.10 | Minimum removed mass fraction |
| max_centroid_rise | 1e-7 | Maximum centroid rise divided by outer length |
| com_tolerance | 1e-4 | Transverse COM offset divided by outer length |
| alignment_tolerance | 1e-4 | Normalized fixed-axis inertia alignment error |
| principal_gap_min | -1e-8 | Minimum normalized principal gap |
| require_connected | true | Independent whole-solid connectivity required |
| mass_tolerance | 1e-10 | Relative mass tolerance |
| inertia_tolerance | 1e-10 | Relative polar-inertia tolerance |
| moment_psd_tolerance | 1e-9 | Covariance PSD tolerance |
| solver_tolerance | 1e-9 | Solver residual tolerance |
| solver_polar_margin | 1e-5 | Solver-only polar margin |
| training_polar_margin | 5e-4 | Training-only polar margin |
| objective | minimum_retained_mass | Minimize retained mass |
| material_policy | single_density_removal_only | One material density; removal only |
| inertia_reference | original_fixed_axis | Original fixed-axis polar inertia is the reference |
| inertia_reference | original_fixed_axis | Original fixed-axis inertia is reference |

The source comments specify that positional tolerances are fractions of frozen outer length L, and inertia errors are divided by original inertia about the prescribed fixed axis. No recentering or axis optimization occurs.

## Exact constraint adapter

m69_fixed_axis_contract_v1.py ports task69._physical and task69.evaluate_moments. It evaluates ten named scalar conditions, reports moment_pass separately from complete_pass, and never infers connectivity from moment feasibility. The complete gate requires an independently supplied connectivity result when require_connected is true.

## Units and region policy

TaskSpec density=1.0 is retained as a registered scalar for the source/task coordinate track. The actual source-mesh-to-metre scale is not registered, so neither kg nor kg/m^3 claims are permitted. L is the frozen outer length in the supplied coordinate units. SI conversion requires independently registered source physical scale.

TaskSpec provides shell_fraction and protection fractions, but not the actual per-mesh masks. The exact shell offset algorithm must be bound to the pinned mesh68 importer. The M90 notes identify protect_radius as a contact-centred 3D ball; its discrete mask still requires per-mesh evidence. The TaskSpec does not authorize adding a tip or shaft. Cavity fill/preserve remains source-mesh-specific.

## Pilot decisions from currently available evidence

- F01 and F22 remain the first two priority review candidates.
- F02 is not a clean third pilot: its seven boundary witnesses remain, including two exact coplanar cases. Retain it as a boundary-ambiguity stress test only, with all witnesses preserved and no arbitrary grid-phase selection.
- F11 is the least discrepant review lead among F11/F12/F23 by supplied P99 first-hit depth-difference/L values: 0.01974, 0.03171, 0.42996. However the handoff explicitly retains local-fill doubts. F11 is a provisional third-candidate review lead, not an admitted pilot.
- F12 and F23 remain pending. These values are not surface-displacement ratios.

The accessible handoff does not contain the actual F01/F22/F11 mesh paths, source SHA-256 values, cavity annotations or complete mesh audits. No initial mesh, cavity fill, or cavity-preservation operation is therefore falsely marked approved. Formal geometry admission, feasible states and pilot Oracle labels remain zero.

## Continuous/binary pairing and Jacobians

Continuous material relaxation is a proposal, not a final single-density binary result. Each proposal must be paired with deterministic hardening on the same registered 512-native support and identical fixed/protected masks. The hard binary output is independently checked for all ten moments and whole-solid six-face connectivity.

M69 query labels M(q) and dM/dq supervise quadric-query response. They are not material-design dM/dalpha. Derivatives are admitted only on smooth branches when the oracle marks them valid and directional finite-difference audits pass. Tangencies, topology/axis-selection switches and ambiguous boundaries remain invalid/unknown, never zero Jacobians.

Freeze the query distribution before teacher calls; group splits by parent geometry; retain full/empty endpoint controls; bind seeds, geometry hashes, query hashes and audit hashes. SDF sign convention and exact mesh-boundary policy remain pending per selected mesh. Do not silently reuse the quadric polynomial sign as the SDF sign.


## Registration completeness correction

The machine-readable registration fixture includes all 28 fields emitted by the pinned TaskSpec, including `inertia_reference=original_fixed_axis`. Its source/task hashes are copied from the user's local extraction. The JSON record remains a task registration, not a per-geometry physical admission.
