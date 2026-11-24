# Versioned finite R3/R8 source

`ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1` defines the plane and paraboloid branches for the registered algebraic finite-map profile. The historical module raises a rank-deficient source error for both branches. The versioned extension uses its own entry points and scientific identity. The deployed full-rank evaluator, Cherchi Boolean and certified integration retain their individual hashes in the encompassing operator.

## Source and side

1. The native backend normalizes the supplied ten coefficients once. The lowest-index coefficient with largest magnitude fixes the canonical sign. The closed mesh uses this canonical polynomial. Its interior represents the increasing side of the finite piecewise-planar graph. Negating all coefficients yields the same vertices and faces and exchanges intersection with difference.
2. R3 requires six identically zero quadratic coefficients. It uses the archived plane frame and square patch. R8 uses the native canonical eigenframe, the ordered two non-null axes and the transverse axis oriented in the direction of increasing polynomial. The registered 24-by-24 grid supplies two triangles per square with the native diagonal. The classifier and an additional bound of 64 binary64 epsilons times the largest absolute eigenvalue admit a roundoff-size least eigenvalue. All supplied quadratic coefficients enter every root calculation. The source returns that eigenvalue, together with the exact-rationally checked frame residual and orthogonality bounds. A rounded rotated R8 record may have a nonzero exact determinant.
3. For each parameter pair, let `y = s*u + t*v`. The finite source point is the positive-slope root of `q(y + w*n) = 0`. The calculation uses exact binary rational coefficients for the constant, linear, and quadratic terms and an exact linear or quadratic algebraic root. Each coordinate rounds once to the nearest binary64 value with ties to even. The projected lower disk must enclose the design bounding box projection with a strict clearance.
4. The archived closure procedure duplicates the interface along the positive transverse axis and stitches its rim. The base extrusion depth is four times the native radius. The depth doubles within the supplied budget until the actual rounded far-cap vertices lie beyond the design box. The interface grid keeps its original radius throughout. Each lateral triangle needs strict support separation, all lower faces need consistent outward winding, and the complete represented boundary must pass the native exact outward audit. The construction witness records the cap depth, doubling count, support margins, normalized source coefficients, frame and audit digest.
5. The reduced motion differentiates each constructed vertex, including the selected extrusion, using the native normalized-coefficient tangent basis. R8 differentiates the eigenframe and all terms in the implicit graph equation. A caller with raw coefficient coordinates composes these columns with the native unit basis and the derivative of normalization exactly once. Canonical sign/frame seams, ill-conditioned axial roots and non-simple R8 eigenframes reject motion independently of value. The motion assumes the recorded frame and cap-growth decisions. Event-neighborhood certification belongs to a separate result.

This extension implements a finite eigenframe grid on R8 and a planar square on R3. The archive's fixed-normal paraboloid and its prepared exact-rank-two arrangement have distinct source definitions. The witnesses describe the typed graph, represented geometry and its pointwise legality. Evidence for the material integral and its motion is collected by the physical replay.

## Build and verify

The patch appends the versioned entry to the archived candidate-03 triangulator. This allows direct use of its internal radius, eigensystem convention, plane frame, cap builder, orientation normalization and hash. The build reads the archived compiler and linker recipes and static archives. Every compiler-opened upstream dependency is checked against the original receipt. Two drifted headers on this host come from the receipt-identical files in the TVCG checkout. A module-local export list exposes only the new Python initializer.

From this directory, the following commands build and check the extension.

```bash
/home/lin/anaconda3/bin/python build.py
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/lin/anaconda3/bin/python test_guards.py -v
align=/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
"$align/run.sh" "$cpu" "$PWD/smoke.py" --historical --out /tmp/tvcg-stratum-build/old.json
"$align/run.sh" "$cpu" "$PWD/smoke.py" --out /tmp/tvcg-stratum-build/new.json
```

`load_native(build_dir=None)` accepts the default build directory `/tmp/tvcg-stratum-build` or the `TVCG_STRATUM_BUILD` override. It checks the built module against the versioned reference receipt, then exposes `source_value(q, design, settings)`, `source_motion(q, design, settings)`, and the full-rank regression entry. The settings argument is the frozen evaluator's `triangulation`. The bridge's `backend_identity` includes the loaded module digest, patch/recipe hashes, original module and compiler-dependency identity.

The two independent builds on this host produced identical module bytes under the digest in [build_receipt.json](build_receipt.json). [historical_smoke.json](historical_smoke.json) preserves the two original rejections and full-rank acceptance. [source_smoke.json](source_smoke.json) records both new physical boundary digests, all raw-direction source-motion differences at nonunit scale, opposite polarity controls and the bitwise full-rank comparison.
