# GPU TLAS vector hit stores — 2026-10-05

Retained two aligned 128-bit hit stores in public TLAS launches, with scalar fallback for unaligned pointers. Ordered secondary gains are 0.1–2.1%; unsorted secondary shadows improve only about 0.1%. Example warm trace: 0.656 → 0.639 ms (2.6%), with a noisy 0.975 ms final sample.

**Measurement:** Five alternating pairs per mesh/layout/launch; 524,288 rays, 108 instances. Five warmups, 21 batches of eight synchronized launches; median of process medians. Ordered timings include keys, sort, gather and traversal (32-thread blocks); direct uses 128. Build, allocation, upload and readback excluded.

| BLAS | Mesh | Closest before → after | Time reduction | Shadow before → after | Time reduction |
|---|---|---:|---:|---:|---:|
| CWBVH8 | dragon | 2.3700 → 2.3576 | 0.5% | 2.1337 → 2.1219 | 0.6% |
| CWBVH8 | buddha | 1.5884 → 1.5756 | 0.8% | 1.3337 → 1.3166 | 1.3% |
| CWBVH8 | bunny | 2.4317 → 2.4224 | 0.4% | 2.1036 → 2.1015 | 0.1% |
| BVH4 | dragon | 6.1177 → 6.0664 | 0.8% | 2.1393 → 2.1226 | 0.8% |
| BVH4 | buddha | 3.3090 → 3.2700 | 1.2% | 1.3146 → 1.2867 | 2.1% |
| BVH4 | bunny | 5.3648 → 5.3157 | 0.9% | 1.9989 → 1.9749 | 1.2% |

**Validation:** 25 GPU tests pass; CUDA memcheck reports zero errors. All 720 full-hit digests and visibility digests match; example image hashes match. Shader register/stack counts unchanged, with no spills. Raw results: `hit_store_*_timings.json`; diagnostics: `hit_store_*.json`.
