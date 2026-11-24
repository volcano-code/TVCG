# Native R8 admission on a represented coefficient box

Two preselected rows use the actual compiled candidate-03 rank classifier in an exhaustive binary64 cover. Each certificate concerns value-source construction for the finite graph on its stated raw coefficient box. This is a machine-scale result. Its use for material incidence requires the separately bound, positive geometric enclosure on a containing box. Only the enumerated raw radius receives native-source admission.

| Registered row | Raw radius | Largest-coordinate ULPs | Candidate products | Distinct quadratic matrices | Elapsed |
|---|---|---:|---:|---:|---:|
| R02_R8_032 | 2⁻⁵² | 2 | 5,019,165 | 2,667,229 | 9.21 s |
| maskB_R8_035 | 2⁻⁵² | 4 | 36,807,210 | 17,748,522 | 60.23 s |

Both certificates accompany the exact attained rank-eight chart witness and classifier margins. [R02 result](evidence/R02_R8_032.json) · [maskB result](evidence/maskB_R8_035.json)

## Which inputs the computation covers

For a row with ten raw coefficients and a radius `r`, all binary64 inputs in each exact interval `[qᵢ−r,qᵢ+r]` are included. The bounds are calculated as exact fractions, then converted inward to the first and last representable input. Every component spans multiple representable queries. The row's original R8 basis is bound by digest. A separate witness finds two-sided attainable displacements in all eight supplied directions and verifies the eight chord vectors are linearly independent using rational elimination.

Only six normalized coefficients affect the eigensolver. The native norm uses nonnegative squares, positive additions and square root, each monotone in the absolute input components. The two vectors of extremal component magnitudes therefore attain the smallest and largest norm. The original compiled normalization function is called at both. Each returned component is the nearest-even direct division of its corresponding raw coefficient by the binary64 norm. Intersecting the exact rounding cells across nonzero components gives inclusive binary64 bounds on the denominator. The cover enumerates every representable denominator between those bounds.

For each denominator, all values of each raw quadratic entry are divided in binary64. The six resulting lists define a Cartesian product. The enumerator examines the union of those products, omitting a duplicate tuple only if it appears in an earlier product. This set can include tuples that cannot occur at that denominator for any full query. That overcoverage is conservative. The first four coefficients are covered by intervals, including their effect on normalization, the canonical coefficient pivot, the linear transverse term, and the validation of the normalized coefficient norm.

## Which native instructions the computation verifies

The verification build retains the triangulator object from a full rebuild that reproduces the registered extension byte for byte. A copy changes only the visibility of its original file-local classifier symbol. The verifier checks the entire object text section before and after this change. The offline classifier executes the shipped 3-by-3 eigensolver, including all QR deflation, iteration, eigenvalue sorting and sign decisions. The source module used by physics is separate. The evidence includes digests for the two build recipes, probe and source modules, unmodified compiler object, machine code section, compiler, linker, symbol tool and applicable Eigen headers.

For every distinct matrix the verifier requires native successful convergence, the same R8 classification, null index, separated spectrum, separated eigenvector sign pivots, the native roundoff leaf test, an increasing transverse axis throughout the bounds on the linear coefficients, and margins for the exact-rational residual and orthogonality tests. In the verified range `|Aᵢⱼ|, |vᵢⱼ| ≤ 1.01` and `|λⱼ| ≤ 2`. A three-term residual uses three products, two additions, the eigenvalue product and a subtraction. The sum of absolute terms is below 5.1. With unit roundoff 2⁻⁵³ the absolute error is less than 2⁻⁴⁶, including subnormal errors. The verifier adds the full 2⁻⁴⁶ allowance when checking the native 2⁻⁴² threshold. The transverse dot-product and orthogonality checks use the same conservative allowance.

The center's native normalized coefficients, raw query and raw basis digests, source map, evaluation context, triangulation hash, selected Boolean operation, and actual target-frame vertices/faces/cap hashes must agree with the containing geometry certificate. The resulting geometric neighborhood is limited to the intersected raw radius. Output triangulation identity, solver convergence, neighboring returned SVD bases, and admission at the two finite-difference step sizes require separate evidence.

A separate direct interval execution follows the reference QR formulas. It reproduces the exact native center eigenvalues and unsigned eigenvectors on the 25 registered rows. Enclosing normalized components by 2⁻⁵⁰ makes the deflation comparison unresolved. Its per-row diagnostics are recorded in [interval_qr.json](evidence/interval_qr.json). The exhaustive certificate is independent of this diagnostic.

## Independent reproduction

Run `reproduce.sh CENTERS GEOMETRIC_PROOFS NEW_OUTPUT_DIRECTORY` after acquiring the physical replay center records and the conditional geometric proofs. On the archive host the first two inputs are usually `/tmp/tvcg-stratum-physical/new/centers` and `/tmp/tvcg-stratum-neighbourhoods-final`. The script rebuilds the offline probe, exercises full-rank/false-leaf/transverse and duplicate-count negative controls, and repeats both exhaustive checks. Individual results record the wall time of their matrix enumeration. File hashes and numerical results can be compared directly, excluding the timing field.

The two exact inputs for each positive certificate are included in [replay_inputs.zip](evidence/replay_inputs.zip). An independent verification of both positive cases can be run from this directory.

```bash
mkdir -p /tmp/tvcg-r8-independent/inputs
unzip -qo evidence/replay_inputs.zip -d /tmp/tvcg-r8-independent/inputs
./reproduce.sh /tmp/tvcg-r8-independent/inputs/centers /tmp/tvcg-r8-independent/inputs/geometry /tmp/tvcg-r8-independent/output
```

Using this four-file bundle repeats the point-trace diagnostic on the two positive cases. The complete physical centers directory repeats the 25-row interval diagnostic.
