# Supported-space Newton callback

The public `supported_stratum_callback` in `spin_adapter/v21_stratum.py` connects the typed physical evaluator to `solve_spin_it_faster_with_external_moment_evaluator`. A caller supplies an orthonormal fixed dual support and a function that returns a raw chart at the query the solver actually submits.

```python
callback = supported_stratum_callback(operator, current_chart, support_basis=dual_support)
result = spin.solve_spin_it_faster_with_external_moment_evaluator(
    request, callback, initial_quadrics=[initial_q], support_basis=dual_support)
```

At each query, let `u = q / ||q||` and let `B` contain the current raw chart directions. The reduced QR factorization of `(I − uuᵀ) B / ||q||` gives `U R`. The callback returns `native_tangent_basis = U` and solves `J_native R = J_raw` using the same physical call's `10 × d` derivative. It checks both the current dual point and the geometry tangent obtained from the entire fixed support, including the factor of two on the mixed dual monomials. Missing chart derivatives, changed physical identities and missing geometry or topology digests stop the call. Native assembly also checks the objective, constraint gradients and their differentials on the support.

## Native examples

`receipt.json` contains the compiled C++ `supported_space` attachments for three calls. All use one initial query, one restart and a budget of one function evaluation. The compatible examples use the strict `DUAL_MONOMIAL_RAW_V1` equation profile, `STATIC_BALANCE` and `MAXIMUM_MASS`.

- The oblique plane cuts the three positive axes of a tetrahedron. Its four-dimensional dual support contains the constant and linear monomials. Its three raw columns have distinct lengths and nonzero off-diagonal mixing. A closed-form subtraction of two tetrahedra checks the returned moments and the derivative pulled back to the original raw chart.
- The registered `R02_R8_032` paraboloid uses the nine-dimensional dual span of the stored coefficients and chart at the center. The chart provider asks the compiled source for its current tangent. The saved physical preparation, raw chart and independent signed-tetrahedron integral check the first eight directional columns. This support is local to the initial query. Following a curved R8 leaf through later Newton steps requires a moving support and a rank-preserving update.
- The same plane, geometry and support are also passed to `FIDGET_SPINNER`. Its objective uses quadratic moment equations outside the four-dimensional support. The C++ attachment retains the defined physical derivative and the assembly rejection.

| Initial call | Native Eq.14 matrix | Maximum raw-Jacobian difference | Returned solver status |
| --- | --- | --- | --- |
| Oblique tetrahedron, R3 | 6 × 6 | 9.71 × 10⁻¹⁷ against the closed form | `OPTIMIZATION_DID_NOT_CONVERGE` |
| Registered `R02_R8_032` | 11 × 11 | 4.68 × 10⁻¹⁷ against the retained tetrahedra | `RESOURCE_LIMIT_REACHED` |
| Oblique tetrahedron, fidget objective | Assembly rejected | — | `GEOMETRY_EVALUATION_FAILED` |

The compiled R8 callback received the registered coefficient bits exactly. Its geometry and topology digests both match the archived center.

The receipt verifies initial Eq.14 assembly. Convergence, spinning-top performance, neighborhood certificates and learned-model substitution are outside the scope of these three calls.

## Replay

Run from the repository root on the archival host after the versioned finite source has been built as described in `../native_backend/README.md`. The native launcher selects the pinned CPython and libraries. Choose a CPU allowed by the current host.

```bash
root=$(pwd)
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1/run.sh "$cpu" \
  "$root/experiments/compact_pino_v21/stratum/solver_bridge/replay.py" \
  --out /tmp/stratum-solver-bridge-replay.json
python -m pytest -q experiments/compact_pino_v21/stratum/solver_bridge/
```
