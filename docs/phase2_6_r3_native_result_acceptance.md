# R3.1.4.2 — Native result integrity and acceptance

The native replay runner now validates the native result before creating a replay bundle. Required fields include status, diagnostic, total and per-restart function evaluation counts, restart count, events, trace, native hashes, final geometry verification, final ten raw moments, final quadric and final geometry hash.

The result gate rejects missing fields, non-finite numeric payloads, inconsistent restart accounting, non-ten-dimensional moment/quadric vectors, non-unit final quadric, and empty final geometry hash. Physical acceptance is reported only when the solver status is CONVERGED/SUCCESS and the native final geometry verifier returns true. Non-converged results remain diagnostic outputs with physically_accepted=false; they are never promoted to successful designs.

CI tests exercise acceptance and rejection paths with controlled software fixtures. They do not represent a real physical case. The user's local native extension preflight passed with process-local LD_PRELOAD of the system GCC libstdc++; this resolves the observed GLIBCXX_3.4.32 runtime conflict without replacing Conda libraries.

Remaining gap: a real TRELLIS case, reviewed geometry/task/pin/support, source-backed Faster initial quadrics, and independent raw-moment backend output are still required before R3.1 Exact Replay can be claimed.
