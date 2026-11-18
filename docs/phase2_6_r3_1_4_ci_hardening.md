## R3.1.4 CI hardening — source asset revalidation

The local native build preflight supplied by the user passed all checks. The compiled extension SHA-256 is `7ba14676a5e07df938482ff736acc02017e1c3b7ecf105ef42dde0817ec64c7b`; it imports as top-level `_make_it_stand`, exposes `solve_spin_it_faster_fullrank_quadric`, and the Python facade and CMake options are valid. The Conda/system libstdc++ conflict is resolved process-locally with `LD_PRELOAD`; no library replacement is required.

The R3.1.4 replay gate has been strengthened: request assembly now requires a source-backed pin/support artifact in addition to outer/design meshes and TaskSpec. Before native execution, the admission guard reopens and SHA-256 rechecks outer mesh, design mesh, pin/support and TaskSpec files. This closes the time-of-check/time-of-use gap where an asset could change after request assembly. The evaluator configuration is also rehashed and compared with the physical admission certificate.

Tests now include mutation-after-assembly rejection. These CI fixtures validate fail-closed behavior only; they are not physical admissions. The native replay remains blocked until Work provides a reviewed TRELLIS geometry/task bundle and a native geometry evidence certificate.
