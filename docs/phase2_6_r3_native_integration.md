## R3.1.3.2 — Native request integration contract

### Local build audit

The user successfully imported the compiled extension as top-level `_make_it_stand` and confirmed `solve_spin_it_faster_fullrank_quadric` exists. The preflight bug was a literal escaped newline accidentally committed into a Python dictionary, not a native build failure. The preflight has been rewritten and now tries both package and top-level import layouts. A source-parse test guards against recurrence.

### Request-to-replay integration

The native replay runner now exposes `validate_native_replay_inputs`, which composes the source-bound request envelope, Faster initialization record and evaluator configuration. It checks case/geometry/task identity, spinning-top mode, dual-monomial profile, payload capture, seed, iteration/restart/function-evaluation budgets and both tolerances. Evaluator identity is checked when supplied as an envelope alongside settings.

The helper returns the typed `SpinItFasterRequest` and `SpinEvaluatorConfig`; the existing runner then passes them to the existing pybind-backed native C++ function. No Python Newton implementation is introduced.

### Tests

Tests cover valid integration and rejection of geometry/task/evaluator identity mismatch, budget mismatch and missing evaluation payload capture. These are contract tests only; they do not execute a solver on synthetic geometry and do not claim a physical result.

### Remaining gap

User's local native extension is confirmed importable by direct command, but the rewritten preflight must still be run locally. Real request assembly and native solver execution remain pending a physically admitted TRELLIS case and reviewed source artifacts.
