# UG-PINO system positioning and architecture

## Research positioning

**Physics-Informed Unified Neural Operator for Rotation-Stable Topology Optimization**

The project targets acceleration of the Spin-It Faster moment-based optimization loop while retaining the same physical task and exact solver authority. The learned system approximates geometry-conditioned moment responses and their coefficient derivatives; the original analytic/native solver and independent physical checks remain responsible for final correction and acceptance.

The project is not a generic 3D reconstruction model and not a replacement for physical validation.

## Three system layers

1. Geometry representation and operator encoding.
2. Physics response, derivatives, event-validity risk and uncertainty.
3. Exact solver coupling, safeguarded Newton/KKT, and terminal geometry/material audit.

## Current operator components

### Geometry trunk

- P-CRAFT UFNO3d/F-FNO style U-shaped encoder.
- Four-channel geometry input: density, occupancy, normalized SDF and internal boundary; coordinate channels are appended internally.
- Factorized 3D spectral convolution using FFTs along x/y/z, local 1x1 convolution skip, normalization, U-shaped down/up paths and skip connections.
- Shared-weight multi-scale context at 32^3, 64^3 and 128^3.
- Native RLE interval descriptors and local feature sampling.

### Native interval interaction

- GNO local graph branch with geometry-only neighbor graphs.
- Optional six-connected component filtering prevents graph edges across disconnected material components.
- Mass-aware Transolver-style physical state slicing.
- The slicing block learns per-head soft assignments of interval tokens to physical slices, pools slice tokens with mass weights, applies scaled dot-product attention over the slice-token axis, then deslices updates to intervals.
- The current XL preset uses 32 slices and 8 heads.
- Residual interval MLP blocks provide local nonlinear refinement.

### Moment response

- Convex Moment Potential Head: positive mixture of smooth SPD quadratic norms, differentiated with automatic differentiation to produce the ten-moment response and its Jacobian.
- Bounded-odd M72 response head remains an alternative/control.
- Ten-dimensional output order: mass, first moments x/y/z, second moments xx/yy/zz, and cross moments yz/xz/xy.
- Fixed and editable material moments are kept separate and recombined under the native moment contract.

### Event and derivative supervision

- Moment loss remains active on all valid moment labels.
- Jacobian loss only uses teacher derivative_valid labels after optional event-margin, finite-difference-audit and error-bound gates.
- The optional Event Validity Head is trained only from a separate explicit teacher event_valid boolean. It is not inferred from derivative_valid.
- Event prediction is a risk signal, never a replacement for exact solver event checks.

### SQNO / QuadrANN path

- SQNO v1.2 provides geometry-once/query-many processing using topology-aware QuadrANN interval GNN features, positive mass-prior quadrature weights and a batched moment operator.
- A shared P-CRAFT feature export/import contract now allows SQNO to consume the UG-PINO geometry encoding without recomputing P-CRAFT.
- The shared feature adapter is an interface milestone, not yet a single jointly optimized M80+SQNO model/checkpoint/trainer. The joint wrapper, optimizer ownership, checkpoint schema and matched ablation remain open.
- For the current XL multiscale width 192, instantiate the SQNO adapter with `preset="xl_ms"` so feature dimensions match.

## Attention: explicit answer

Yes, the current maximal M80 architecture uses attention.

The primary explicit attention mechanism is the Transolver-style physics-state slicing block:
- learned soft interval-to-slice assignments;
- mass-weighted pooling;
- multi-head scaled dot-product attention over 32 slice tokens;
- deslicing back to native interval features.

There is also softmax multi-scale gating over 32/64/128 geometry contexts and learned local scale gating. These are attention-like adaptive weighting mechanisms, but should not be conflated with full Transformer self-attention over every geometry interval.

GNO uses local graph message passing and weighted aggregation, not standard Transformer self-attention. QuadrANN uses topology-aware message passing with mean/std aggregation and global context, not a conventional Transformer block. The PINO backbone is Fourier/spectral operator learning, not attention.

Thus the system deliberately combines spectral operators, graph message passing, physical slicing attention, multi-scale gating and structured moment potentials rather than relying on one generic Transformer.

## Completion status

Implemented in the repository:
- P-CRAFT PINO/F-FNO geometry trunk and multi-scale contexts.
- GNO and topology filtering.
- Transolver-style mass-aware slice attention.
- Convex and bounded-odd moment response alternatives.
- Automatic-differentiation Jacobian.
- Event-aware derivative-label masks and optional explicit event-validity head.
- M72 trainer, GPU preflight, validation tails and checkpoint/resume.
- SQNO shared P-CRAFT feature contract and external cache input.
- CPU CI for the maximal candidate and moment-only preflight contracts.

Still open:
- Dedicated importer for the newly handed-off seven-object Faster moment labels.
- Valid independent Jacobian supervision for those new objects.
- A single trainable wrapper/checkpoint that jointly owns M80 and SQNO branches.
- Multi-seed UQ training, held-out calibration and risk-coverage.
- GPU preflight and trained-quality evaluation for the new event-enabled preset.
- Audit of the large per-object Jacobian errors from the first 300-step pilot.
- Matched ablations for each branch and the convex-head amplitude saturation.
- 512-resolution trained-quality/OOD validation.
- Native solver parity, safeguarded Newton replay, exact terminal material audit and end-to-end speedup.

## Data status at the 2026-11-18 handoff

The two TRELLIS releases contain 200 geometry candidates. Seven first-release objects have 448 accepted Faster moment labels (64 each). The handoff marks Jacobian-valid false, and reports zero newly qualified Jacobians, zero new protected M69 feasible material states and zero training steps. The second 100 are geometry candidates, not labelled Faster tasks.

The reported 300-step RTX 5090D pilot used the previous M72 corpus (67 qualified objects, 32 queries each), not the new 448-label Faster release. Its Moment validation changed from 0.002413 to 0.002419, while Jacobian scaled MSE changed from 1449.55 to 1426.80. This demonstrates a successful training execution path, not satisfactory learned accuracy or solver acceleration.
