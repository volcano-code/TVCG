# Noodles deterministic inference profile

The two workers each evaluated the 25 recorded seed-0 queries twice on CPUs 6 and 7. Each request rebuilt the native moving operand, ran the production physical-boundary audit, and built its surface index. Query 3 reproduced the archived source rejection in every pass. The M-only entry pays the full source-motion construction cost used by the joint entry.

| Entry | Process setup (s) | First-query JIT (s) | Cold 25 callbacks (s) | Warm 25 callbacks (s) | Warm median (s) | Peak process RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| M only | 10.40 | 2.35 | 78.42 | 74.10 | 2.322 | 1.377 |
| M and raw J | 9.24 | 1.94 | 76.58 | 74.07 | 2.355 | 1.378 |

| Warm pass, 25 requests | Source and physical audit (s) | Surface indexes (s) | Local readouts (s) |
|---|---:|---:|---:|
| M only | 8.70 | 0.05 | 65.35 |
| M and raw J | 8.67 | 0.04 | 65.35 |

The original native solver journal records **329.44 s** across these 25 joint callbacks on the same CPU IDs, including one rejection. Its warm median excluding the first callback was 14.607 s. Applying that exclusion to V2.1's warm joint replay gives 2.359 s. The archived native process peaked at 0.680 GiB. Its verified solve, including independent verification, took 354.16 s. These historical values were collected in a separate run.

The frozen material pack contains 11,250,366 positive nodes and 7,205 occupied blocks. Its preparation record separately lists 0.837 s for material preparation and 2.253 s for indexing. The frozen-pack profile starts from those assets.

Each worker started a fresh process with an empty Numba compiler cache. Both passes used the same frozen positive material index. The operating-system page cache was available. Cold callback totals include first-point JIT and exclude static setup. The JSON includes process startup, static setup, source, index, reductions, callback wall and CPU times, current RSS, 10 ms sampled per-query peaks, and cumulative kernel RSS.

CPU affinity was pinned on a shared host. The measured runs replayed fixed queries. They performed zero solver steps, line searches, fallbacks, and independent final verifications. Qualified solve time and learned-model speed remain unmeasured.

Across the joint warm callbacks, Linux reported 0.9% busy time on sibling CPU 22 and 1.2% on sibling CPU 23. The JSON includes raw scheduler ticks for every callback. A first complete run overlapped a chart diagnostic on CPU 23. The earlier output is retained in [the shared-host run](shared_host_first/summary.md).

Reproduce from the worktree root with `bash experiments/compact_pino_v21/profile_inference/reproduce.sh /path/to/output`. To rebuild the static pack from a registered geometry export, run `python -m experiments.compact_pino_v21.pack_geometry --native-root /path/to/native --indexes /path/to/indexes noodles`, then set `V21_NOODLES_PACK=/path/to/indexes/noodles` for the launcher. The JSON binds the inputs and executable sources by SHA-256. The archive export is documented in [the native reproduction instructions](../native/README.md).
