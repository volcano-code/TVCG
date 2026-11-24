# Registered physical replay

The population contains every original plane R3 and paraboloid R8 raw chart on the four development meshes. Each compact query and its historical source basis are joined by the preserved query ID and by exact equality of the ten raw coefficients. The sampler uses both signed directions at raw steps 0.00002 and 0.00001. The numerical decisions and limits are fixed in [protocol.json](protocol.json).

| Development mesh | R3 | R8 |
| --- | ---: | ---: |
| maskB | 16 | 9 |
| combB | 16 | 7 |
| R01 | 16 | 4 |
| R02 | 16 | 5 |

The versioned typed teacher calculates each center on the actual retained design material. The independent calculation sums oriented tetrahedra in the registered target frame, adds the fixed shell and differentiates those tetrahedra using the reported motion of every retained boundary vertex. The perturbed forward path calls the separate native value-only source, authenticates the physical operand, runs the original physical Cherchi Boolean and integrates the newly returned exact boundary with the registered MPFR/GMP arithmetic. Each centered column uses its original scale and sign.

For R8 the path moves along an original raw tangent and maintains the center's signed near-zero quadratic eigenvalue. Its other two eigenvalues keep the registered signature. The center is returned bitwise at zero. The receipt gives the actual binary64 spectrum, the tangent residual and the measured error of the centered path derivative for each column.

Retained vertices are identified by their original design/source vertex indices or the geometric support of the Cherchi LPI, TPI or LNC construction. The support preserves the line/plane groups required by each construction. The incidence hashes relabel faces by sorted vertex support and record oriented and unoriented connectivity. Thus changes to output ordering and changes to actual triangle connections can be distinguished. A velocity comparison over a common subset carries the matched count. The full sampled gate requires a complete correspondence, identical oriented triangle incidence and all three velocity/material comparisons at both steps. The separate row and column counts preserve cases with excellent numerical material agreement during retriangulation.

## Receipts

The compressed registered ledger embeds the protocol, per-geometry input/runtime identities, every row's actual raw basis, source construction witness, cap and side facts, physical refusal reasons, native and independent matrices, exact-rational integration status for perturbations, both raw finite-difference matrices when available, and per-column endpoint and retained-topology diagnostics. The physical experiment assigns the unknown value to both neighbourhood event flags. A separate constructive neighbourhood test can examine the same center and raw coefficients.

The two historical, nonunit external chart controls reproduce the archived instrument's ten moments and raw matrices bitwise. The independently manufactured oblique-plane cut also has a closed-form mass, first and second moments and a closed-form Jacobian for all three raw directions. The empty manufactured cut records the physical rejection. [Historical control summary](evidence/control_summary.json) · [Manufactured plane](evidence/manufactured_plane.json).

## Development results

All 89 centers returned the ten retained material moments and their raw chart Jacobian. Independent signed tetrahedra agreed at every center, with maximum component errors of 5.12 × 10⁻¹⁷ in the moments and 1.13 × 10⁻¹⁵ in the Jacobians. The 784 pairs of signed offsets produced 1,568 physical cuts. Every pair preserved the source side, faces and cap flags and passed all three numerical tests for source motion, material moments and motion of corresponding retained vertices.

| Raw chart | Centers | Direction and step pairs | Same retained vertex constructions | Same unoriented triangle incidence | Largest relative Jacobian matrix error at 10⁻⁵ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plane R3 | 64 | 384 | 126 | 0 | 4.29 × 10⁻⁷ |
| Paraboloid R8 | 25 | 400 | 28 | 0 | 6.44 × 10⁻⁶ |

Across both signs, the median fraction of uniquely matched center vertices was 99.9885% and the minimum was 99.6976%. Triangle incidence changed in every pair, so 0 of 89 rows met the preregistered full sampled gate. The closed-surface Euler characteristic agreed at 783 of 784 pairs. At `combB_R3_021`, raw column 0 and step 2 × 10⁻⁵, it was −202 at the center and negative offset and −200 at the positive offset. On R8 the largest observed drift of the signed near-zero eigenvalue was 6.80 × 10⁻¹⁶. The largest component error between the original tangent and the centered raw path derivative was 1.68 × 10⁻⁹. [Compressed per-row receipts](evidence/registered_physical_replay.json.gz) · [Count and error distributions](evidence/summary.json).

To verify the portable registered receipts and the exact frozen roster, run from the repository root.

```bash
python experiments/compact_pino_v21/stratum/physical_replay/archive.py verify
python -m pytest -q experiments/compact_pino_v21/stratum/physical_replay
```

The verifier recomputes the two summary maxima involving the near-zero quadratic eigenvalue with the installed binary64 symmetric eigensolver. For those two numbers, the comparison allows four machine epsilons times the Frobenius norm of the raw symmetric matrix. Drift uses the sum of the center and endpoint norms because it subtracts two independently computed eigenvalues. Plane quadratic blocks have a zero allowance. Every other summary field is compared exactly. The archive passes with NumPy 2.1.3 and 2.4.3.

To run fresh physical measurements on the archive host, first build the pinned versioned stratum extension. The launcher performs all 89 centers, then the two-step finite differences. Its default worker layout is for the archive host's CPU topology.

```bash
bash experiments/compact_pino_v21/stratum/physical_replay/run_all.sh /tmp/tvcg-stratum-physical-replay
python experiments/compact_pino_v21/stratum/physical_replay/archive.py package --out /tmp/tvcg-stratum-physical-replay
```

Individual assets also accept deterministic row shards or a list of registered IDs. For example, the following command rebuilds a single R8 chart and every signed perturbation using the archive machine's interpreter. Re-running a command validates the scientific context and both chart and source fingerprints before any completed measurement is reused.

```bash
runner=$(realpath experiments/compact_pino_v21/stratum/physical_replay/replay_registered.py)
align=/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1
"$align/run.sh" 12 "$runner" --asset combB --only combB_R8_032 \
  --integrator fast --out /tmp/tvcg-stratum-individual
```

The scratch directory contains each original pre-Boolean operand and each actual retained boundary, its full motion when available, indexed construction keys, per-column checkpoints, and the per-geometry source and physical library hashes. The portable ledger links these large, reproducible scratch witnesses by SHA-256.

The launcher distributes costly Boolean calls by pair of signed raw offsets. Individual pairs can also be assigned to another worker using `assist_columns.py --asset R02 --pairs R02_R8_032/1/7 --out /path/to/scratch` through the same archive interpreter. The components of the final argument are the registered row ID, zero-based step index and original zero-based basis column. A completed, matching checkpoint can be consumed by the row finalizer.
