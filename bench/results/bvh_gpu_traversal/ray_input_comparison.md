# GPU ordered traversal without ray gathering — 2026-10-05

Retained indexed ray reads, removing the gather kernel and 32-byte-per-ray buffer. Compressed secondary gains: 2.3–6.1% at 524K rays and 24.8–29.2% at 2M rays. Wide Dragon closest at 524K regresses 1.1% (6.0674 → 6.1338 ms). Small-queue gains are roughly 0.8–2.4%; unsorted controls are approximately unchanged. Normal workspace saves 16 MiB at 524K rays and 64 MiB at 2M; overlapping buffers lazily allocate a reusable copy.

**Measurement:** RTX 5060 Ti, 108 instances; compressed CWBVH8/max-leaf1 or BVH4/leaf2 BLAS, LBVH2/leaf1 TLAS. Three or five alternating pairs by case, five warmups, 21 batches of eight synchronized launches. Ordered/direct blocks: 32/128. Includes keys, sort, traversal and hit writes; build, allocation, upload and readback excluded. Large-batch gains do not predict small-queue or full-render gains.

| Rays | Mesh | Closest before → after | Time reduction | Shadow before → after | Time reduction |
|---:|---|---:|---:|---:|---:|
| 524,288 | dragon | 2.3701 → 2.3147 | 2.3% | 2.1314 → 2.0638 | 3.2% |
| 524,288 | buddha | 1.5832 → 1.4913 | 5.8% | 1.3225 → 1.2414 | 6.1% |
| 524,288 | bunny | 2.4338 → 2.3336 | 4.1% | 2.1244 → 2.0159 | 5.1% |
| 2,097,152 | dragon | 11.7464 → 8.8338 | 24.8% | 10.8151 → 8.0729 | 25.4% |
| 2,097,152 | buddha | 10.0480 → 7.1805 | 28.5% | 8.9138 → 6.3151 | 29.2% |
| 2,097,152 | bunny | 11.9308 → 8.7322 | 26.8% | 10.8893 → 7.8920 | 27.5% |

**Validation:** 27 GPU tests pass; memcheck reports zero errors, including aliased buffers. All 972 full-hit and visibility digests match across 486 paired cases. Raw results: `ray_input_final_*_timings.json`; diagnostics: `ray_input_*.json`.
