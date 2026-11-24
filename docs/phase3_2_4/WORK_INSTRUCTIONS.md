# Phase 3.2.4 — Compact PINO V2: single physics response

Base: 4764c59fe06bb04e96688596b0dcd10cf6777028 (Phase 3.2.3).
This is an implemented research candidate, not a validated native solver replacement.

## 1. Architecture

Native RLE256 -> encoder-only P-CRAFT (32/64 shared U-FNO context grids)
-> mass-weighted spatial tokens (native interval feature pooling and exact
RLE ten-moment aggregation) -> one geometry/cut cross-attention response
-> bounded retained fractions -> one physical moment integral and its raw
coefficient AD Jacobian.

The V2 factory does not construct M80, SQNO, SQNO-GNN, query fusion, shared GNO,
Transolver, cross-scale attention, UQ or Event heads. The encoder retains the
Compact-Lite backbone width/depth for matched comparison. V2 attention width
64, heads 4, token grid 16 (at most 4096 occupied groups); 8-grid candidate at
most 512. A token groups whole RLE intervals by center, including their full
support for moment aggregation and bounds. This is a lossy within-token
cut-response representation: it is NOT exact native cut integration and
does not claim the accuracy of native RLE256 evaluation at every query.
Long runs spanning bins and thin structures are known risks.

For editable group k, store the FP64 exact sum m_k of raw moments and its
conservative normalized monomial box. The model calculates a scalar p_k in
[0,1] using a smooth conservative quadric enclosure and a learned odd cut
correction from sign-invariant query cross-attention. No unconstrained
ten-moment residual is added.

M(c)=M_fixed + sum_k p_k(c) m_k; J(c)=dM/d(raw c).
The cross terms in the native polynomial use 2yz,2xz,2xy; reported moments
remain undoubled. Positive rescaling of c leaves M unchanged. Sign reversal
complements p; thus M(c)+M(-c)=2 M_fixed + M_editable for this response domain.
Certified full/empty groups have constant 1/0 fractions and zero derivative.
The response preserves mass bounds and feasible moment mixtures relative to
the RLE conditioning measure; it does not guarantee native mesh accuracy,
exact differentiability at native topology events, or a symmetric positive
semidefinite monomial-coordinate Jacobian.

The additional native256 receipt field response_domain must be
"positive_quadric_on_editable_domain". Work A must independently establish
that native finite-cut/cap/fixed-shell conventions actually reduce to that
domain for admitted queries. If not, NO-GO for this model on those labels;
do not just add the string. Complementarity is a model assumption and must
be checked against native teacher, including signs, empty/full, and valid
event masks. No exact-solver-validity claim comes from the model.

## 2. Tests and validation boundary

The pure torch/NumPy response module was executed in a CPU environment
(PyTorch 2.10.0+cpu). Six parametrized cases passed on synthetic interval
fixtures: exact additive ten-moment conservation, mass bounds, complement,
positive-scale invariance, zero endpoint derivatives, raw coefficient AD vs
FD, radial null J*c, one-sided local slopes for zero planar coefficients,
and mixed Moment/Jacobian gradients into token features and learned attention.
The initial endpoint-gradient failure was corrected by using a safe
denominator only in the crossing region. The final core Git blob should be
dc3086859e561a30216fe770bf8eb8a4f2e965f7.

These are isolated core tests. The full repository, actual P-CRAFT GPU update,
native teacher and checkpoint integration tests have NOT been executed in
this environment. The added integration tests must run independently on Work.

## 3. Work B commands

Pin the exact published commit in a new worktree. Keep 3.2.3 V1 artifacts.
Run the 3.2.3 regressions, and additionally:

```bash
QROOT=research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q "$QROOT/spin_adapter/tests/test_compact_v2_response.py" \
  "$QROOT/spin_adapter/tests/test_compact_pino_v2.py" \
  "$QROOT/training_v13/tests/test_native256_readiness.py"
```

First Work A issues a genuine matching receipt with the V2 response-domain
review; otherwise only synthetic code tests are permitted. Use the actual
approved manifest and receipt paths, a representative partial-cut training
geometry, and a NEW output directory.

```bash
TVCG_MODULE=research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13
python -m "$TVCG_MODULE.simplification_plan" \
  --manifest "$TVCG_MANIFEST" --native-cut-contract "$TVCG_CUT_CONTRACT" \
  --output "$TVCG_RUN_ROOT" --stage v2_screen256 --epochs 3 --seed 830251 \
  --lr 3e-5 --device cuda --allow-absolute-assets
bash "$TVCG_RUN_ROOT/preflight.sh"
# Run training only after software, native domain and GPU checks pass:
bash "$TVCG_RUN_ROOT/train.sh"
```

Arms: compact_lite_256 (V1), compact_pino_v2_256 (16-grid tokens),
compact_pino_v2_tokens8_256. Same train/dev labels, query order, seed,
encoder initialization, normalizers, optimizer, and number of updates;
branch_aux_weight=0 for all three so it does not confound the comparison.
V2-only response tensors are freshly initialized, explicitly logged, not
copied from V1. The historical V1 with auxiliary loss 0.25 is a separate
secondary comparator. No resume across architectures. The standard old
V1 plan and defaults remain.

## 4. Acceptance and remaining research

Report actual parameter and memory storage counts; number of occupied tokens
vs native intervals; P50/P95 cold encoding and warm M / M+J latency; total
GPU time including updates. V2 always uses its intrinsic token memory; the
existing benchmark_inference CLI reports its native two query paths, not a
fabricated cached-vs-uncached speedup. Run on matching geometries and same
trained budget. The model's internal AD/FD is not native correctness.

Compare source-macro Moment, Full-J, R3, R8, near-zero/low-mid-high retained
mass, per-query actual Newton directions, and time-to-feasible native solve.
Use fixed cut contract and independent mesh acceptance. Check teacher
complementarity, normalization, and actual finite-domain rules first; compare
8/16 group resolution, and test long runs, disconnected/thin/cavity geometry.
Reject compression if any protected physical metric deteriorates beyond the
predeclared tolerance. Relative noninferiority to a poor V1 is insufficient
for final training or deployment.

If token approximation is inadequate, next research is splitting long
intervals at physical token boundaries, more adaptive tokens, or higher-order
within-token response, keeping one physically constrained response instead
of restoring independent output branches. Do not infer a speedup from group
count alone; no native solver integration, no real-data accuracy, no target
GPU validation, and no long training are claimed here.
