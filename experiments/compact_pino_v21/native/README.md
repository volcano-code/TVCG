# Finite-operand and native-physics replay

The conditioning entry in `spin_adapter/v21_native.py` calls the deployed native source constructor, side selector and physical-coordinate boundary audit before the Boolean. The returned `FiniteOperand` contains the native target-coordinate vertices, triangle indices, per-face cap flags, Boolean operation and optional vertex motion. The operation is `intersection` when the native selector identifies the positive side inside the solid and `difference` otherwise. The archive contains actual cap indices even for multiply capped cutters.

`spin_adapter/v21_native_physics.py` provides the independent `evaluate_native_physics(evaluator, q, chart=...)` label service. The full entry calls the deployed evaluator. The typed entry requests the binary's reduced-stratum source motion, checks the supplied raw basis against its own unit-coordinate basis, propagates moving vertices through Cherchi constructions, and integrates actual material moments and transport. `NativePhysicalTargets` has independent numerical availability flags for Moment, ambient J and reduced J. A successful typed call provides its own retained geometry and a 10×3 or 10×8 signed-tetrahedron derivative witness in the requested raw basis. A simultaneous ambient call uses its own 10-column source velocity. Before sharing moments across the two scopes, the replay verifies the actual retained geometry hash and both moment vectors. The two event scopes retain their own unknown value, −1. A geometry-only consumer can import the operand module on a NumPy machine without loading a Boolean or integrating any material labels.

All numeric coordinates use the registered normalized solver target frame. The raw polynomial coefficient order is constant, three linear coefficients, three diagonal quadratic coefficients, then the coefficients of 2yz, 2xz, 2xy. Returned moments end in the undoubled yz, xz and xy integrals. For raw q, the normalization differential is `(I − uuᵀ)/‖q‖`, with `u=q/‖q‖`. For a raw chart with basis B and a unit native tangent frame U, the reduced velocity multiplies `Uᵀ (I−uuᵀ) B / ‖q‖`. The same physical fixed-shell moments are added once in the target frame.

## Evidence

The replay includes all 25 queries from the archived converged seed-0 Noodles journal and the registered 256³ conditioning grid. Strict development contains 64 plane R3 and 25 paraboloid R8 probes, covering every registered chart of these types on maskB, combB, R01 and R02. The preserved source chart and compact view are joined by query ID and checked for identical raw coefficients. The compact view withholds the 25 R8 targets and retains the 64 R3 targets. Each deployed chart source call and each independent full production call has its own recorded decision.

| Native experiment | Result |
|---|---|
| Production replay across all 192 probes and controls | 90 finite pre-Boolean operands, 89 numeric moments and 86 numeric full Jacobians |
| Frozen Noodles trace | 24 accepted with identical post-Boolean geometry hashes and moments. Maximum raw-J discrepancy 1.78×10⁻¹⁵. Query 3 repeated the frozen source rejection. |
| Independent oriented tetrahedron calculation on Noodles | Maximum differences 1.91×10⁻¹⁶ for moments and 1.95×10⁻¹⁴ for Jacobian entries |
| Registered strict development R3 / R8 | The native typed source rejected every one of 64 / 25 queries under the deployed algebraic finite map. All 89 ambient calls also rejected. |
| Registered T2 and combB hyperboloids | Reconstructing the opposite query changes summed mass relative to full editable mass by +1.0482×10⁻⁴ and +8.4985×10⁻⁶, respectively. Toggling the Boolean on the identical source restores the editable ten moments to maximum errors 1.13×10⁻¹⁷ and 4.21×10⁻¹⁷. |
| Raw velocity on a non-unit registered hyperboloid | Independent source-vertex finite differences in all ten columns at two step sizes. Largest relative error per column 1.12×10⁻⁹. Material-J Frobenius relative error below 4.50×10⁻¹¹. |

Changing the native `algebraic_finite_map` setting to false yields a separately identified instrument on the external development candle. Its measured 10×3 and 10×8 differentials agreed with the two-step, rank-preserving finite-difference experiment. At the smaller step the relative errors were 3.34×10⁻¹¹ and 5.43×10⁻¹¹. These measurements live under `alternative_chart_diagnostic` and carry the alternative evaluator identity. Deployment eligibility is false.

The numeric case ledger is in `evidence/native_replay.json`. Its `runtime` object identifies the actual loaded native modules and arithmetic backend. `evidence/native_targets.npz` stores raw queries, finite moments and Jacobians, and their independently differentiated signed-tetrahedron counterparts. Every numeric array has its own row-ID index. Rejected and unavailable targets have no numeric entry. `host_inputs.json` pins the archive readers, solver setup, production source modules, native libraries, original trajectory and V2.0 control records used by this replay. The local Work A encoder is pinned independently in the cross-domain grid report.

## Reproduction

For a portable numerical check, run from the repository root with Python and NumPy.

```bash
python experiments/compact_pino_v21/native/verify.py
```

For a complete fresh physical replay on the archive machine, pass a new result directory and an available CPU number. The command reads the existing archive and writes geometry, operands, rows, diagnostics and a fresh review under the supplied directory. The prior acceptance report is read-only input.

```bash
bash experiments/compact_pino_v21/native/reproduce.sh /tmp/tvcg-v21-native-fresh 17
```

For an individual physical geometry, run the entry with the pinned production Python and native library paths.

```bash
align=/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1
script=$(realpath experiments/compact_pino_v21/native/replay.py)
"$align/run.sh" 17 "$script" --asset noodles --out /tmp/tvcg-v21-native-fresh
"$align/run.sh" 17 "$script" --asset combB --prefix combB_R8_ --out /tmp/tvcg-v21-native-fresh
```

`geometries/<id>.npz` contains target-frame design vertices and faces, fixed moments, density, RLE columns, grid origin, pitch and native side, and both transforms back to physical coordinates. `operands/<row_id>.npz` is the geometry-only input. `rows/<row_id>.json` is a separate physics record. For the six historical Objaverse assets the registered grid side is 1024. `diagnostic256/<id>/geometry.npz` holds a fresh 256³ direct-ray encoding of the same signed mesh using Work A's native encoder. The manifests give distinct hashes and scopes. For the recorded execution the common transient output directory was `/tmp/tvcg-v21-native`.
