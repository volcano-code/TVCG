# R3.0.6 — Native C++ solver replay binding

The repository already contains the native pybind entry point solve_spin_it_faster_fullrank_quadric and the Python facade make_it_stand.spin.solve_spin_it_faster_fullrank_quadric.

The facade returns solver status, input/output/event/trace hashes, function evaluation counts, restart counts, accepted Newton trace, full events, timing sidecar, final quadric, raw moments and final geometry verification status. The full-rank entry requires DUAL_MONOMIAL_RAW_V1 and is distinct from the legacy formula-reproduction path.

## New runner

r3_native_solver_replay_v1.py loads a typed SpinItFasterRequest, validates the initialization record, binds geometry/task hashes, mode, seed, budgets and tolerances, requires evaluation payload capture, invokes the existing native C++ solver through pybind, writes a replay bundle and validates the event ledger. It does not implement Newton in Python or synthesize initial quadrics.

## Inputs

- Request JSON: SpinItFasterRequest fields plus case_id, geometry_hash and task_hash.
- Initialization JSON: tvcg.r3.faster-initialization.v1 with explicit normalized initial quadrics.
- Evaluator JSON: SpinEvaluatorConfig fields.

The native extension must be built with MAKE_IT_STAND_BUILD_PYTHON=ON and MAKE_IT_STAND_BUILD_SPIN_IT_FASTER=ON.

## Example

    PYTHONPATH=python:research/NeuralMoments python research/NeuralMoments/r3_native_solver_replay_v1.py --request /path/to/request.json --initialization /path/to/faster-initialization.json --evaluator /path/to/evaluator.json --output reports/r3/native-replay.json

Run only after a real geometry bundle has been admitted. TRELLIS generation remains owned by the separate Work workflow.

## Remaining gap

The runner and binding contract are implemented, but no real native replay is claimed. A physically admitted TRELLIS/M69 case, masks, pin/contact and source-backed initial beta record are still required. The canonical full-rank profile is not asserted to be bitwise equivalent to the unreleased author Matlab implementation.
