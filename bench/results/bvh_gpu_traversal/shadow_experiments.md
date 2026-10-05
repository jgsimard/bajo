# GPU shadow traversal experiments — 2026-10-05

No candidate retained. Near-first/shared callbacks improve camera shadows but regress secondary shadows 3.7–7.6% ordered and 12.0–14.3% unsorted. Early triangle rejection and 21/27-bit Morton keys are approximately neutral. Lower registers and higher theoretical occupancy did not translate into faster secondary traversal.

**Measurement:** RTX 5060 Ti; 524,288 rays, 108 instances, compressed BLASes. Three alternating pairs per mesh/experiment; five warmups, 21 batches of eight synchronized launches. Ordered includes keys, sort, gather and hit writes. Build, allocation, upload and readback excluded.

| Experiment | Dragon | Buddha | Bunny |
|---|---|---|---|
| Near-first, 32 threads | 2.1995 → 2.3665 (-7.6%) | 1.3798 → 1.4520 (-5.2%) | 2.2196 → 2.3104 (-4.1%) |
| Shared eager, 32 threads | 2.2043 → 2.3614 (-7.1%) | 1.3797 → 1.4492 (-5.0%) | 2.2189 → 2.3005 (-3.7%) |
| Shared eager, 128 threads | 2.2025 → 2.3645 (-7.4%) | 1.3790 → 1.4446 (-4.8%) | 2.2178 → 2.3006 (-3.7%) |
| Early triangle rejection | 2.2030 → 2.2063 (-0.1%) | 1.3837 → 1.3870 (-0.2%) | 2.2281 → 2.2317 (-0.2%) |
| 21-bit origin Morton | 2.2042 → 2.2068 (-0.1%) | 1.3845 → 1.3852 (-0.0%) | 2.2270 → 2.2298 (-0.1%) |
| 27-bit origin Morton | 2.2050 → 2.2053 (-0.0%) | 1.3799 → 1.3792 (+0.1%) | 2.2169 → 2.2171 (-0.0%) |
| Shared eager, unsorted 128 threads | 5.5633 → 6.3439 (-14.0%) | 4.6836 → 5.2450 (-12.0%) | 5.2552 → 6.0053 (-14.3%) |

**Validation:** All 378 paired visibility digests and closest-hit digests match. Near-first changes which occluder is returned; other candidates preserve any-hit digests. Raw results: `shadow_experiments_*_timings.json`; resources/manifest: `shadow_experiments_*.json`.
