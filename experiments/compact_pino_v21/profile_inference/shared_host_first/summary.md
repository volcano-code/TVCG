# First Noodles profile on a shared SMT core

This precursor run overlapped a native chart diagnostic pinned to CPU 23, the hardware sibling of CPU 7, during the final joint warm replay. Its observed wall times are retained for transparency. The primary matched profile reran every request in both modes after releasing the sibling. During the precursor M-only worker the wrapper gained a solver-status diagnostic. The JSON records both wrapper hashes. All 24 accepted numerical outputs matched across the two precursor workers.

The two workers each evaluated the 25 recorded seed-0 queries twice on CPUs 6 and 7. Each request rebuilt the native moving operand, ran the production physical-boundary audit, and built its surface index. Query 3 reproduced the archived source rejection in every pass. The M-only entry pays the full source-motion construction cost used by the joint entry.

| Entry | Process setup (s) | First-query JIT (s) | Cold 25 callbacks (s) | Warm 25 callbacks (s) | Warm median (s) | Peak process RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| M only | 10.13 | 2.31 | 79.71 | 73.74 | 2.302 | 1.377 |
| M and raw J | 9.23 | 1.95 | 76.93 | 95.74 | 3.957 | 1.378 |

| Warm pass, 25 requests | Source and physical audit (s) | Surface indexes (s) | Local readouts (s) |
|---|---:|---:|---:|
| M only | 8.74 | 0.04 | 64.96 |
| M and raw J | 8.81 | 0.05 | 86.89 |

The original native solver journal records **329.44 s** across these 25 joint callbacks on the same CPU IDs, including one rejection. Its warm median excluding the first callback was 14.607 s. Applying that exclusion to V2.1's warm joint replay gives 3.963 s. The archived native process peaked at 0.680 GiB. Its verified solve, including independent verification, took 354.16 s. These historical values were collected in a separate run.

The frozen material pack contains 11,250,366 positive nodes and 7,205 occupied blocks. Its preparation record separately lists 0.837 s for material preparation and 2.253 s for indexing. The frozen-pack profile starts from those assets.

Each worker started a fresh process with an empty Numba compiler cache. Both passes used the same frozen positive material index. The operating-system page cache was available. Cold callback totals include first-point JIT and exclude static setup. The JSON includes process startup, static setup, source, index, reductions, callback wall and CPU times, current RSS, 10 ms sampled per-query peaks, and cumulative kernel RSS.

CPU affinity was pinned on a shared host. The measured runs replayed fixed queries. They performed zero solver steps, line searches, fallbacks, and independent final verifications. Qualified solve time and learned-model speed remain unmeasured.

Reproduce from the worktree root with `bash experiments/compact_pino_v21/profile_inference/reproduce.sh /path/to/output`. The launcher uses the registered archive and frozen material locations. Its environment variables allow relocation of those inputs. The JSON binds their contents and the executable sources by SHA-256.
