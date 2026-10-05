# GPU BVH finalization — 2026-10-04

GPU: NVIDIA GeForce RTX 5060 Ti. Baseline: `6cfe28fa`; candidate:
device-side validation and compact-offset generation. At most two compiler
processes ran concurrently, each with an 8 GiB address-space limit. Compilation
was stopped during timing runs.

## Fresh builds

Measured with `bench/bvh/bench_gpu_blas_tlas_build.mojo`, compiled for `sm_120`.
Five alternating baseline/candidate process pairs, CPU affinity fixed to one
available core. Each process reports a median of 11 runs for scene cases or
seven runs for synthetic fixtures. The table reports the median of those five
process medians. Times include allocation, input uploads, and synchronization;
scene-file loading and compilation are outside the timed region.

| Case | Before ms | After ms | Time reduction |
|---|---:|---:|---:|
| Three BLASes, 417,934 triangles, CWBVH8 | 4.727 | 4.776 | -1.0% |
| TLAS LBVH2, 108 instances | 0.263 | 0.212 | 19.4% |
| TLAS LBVH2/leaf1, batch of 32 builds | 6.815 | 6.327 | 7.2% |
| Dragon H-PLOC CWBVH8 | 2.312 | 2.271 | 1.8% |
| Dragon H-PLOC wide4 | 1.178 | 1.126 | 4.4% |
| LBVH wide4, 1 × 16,384 triangles | 0.444 | 0.395 | 11.0% |
| LBVH wide4, 128 × 4 triangles | 0.204 | 0.158 | 22.5% |
| LBVH wide4, 16 × 1,024 triangles | 0.432 | 0.382 | 11.6% |
| LBVH wide4, mixed sizes | 0.366 | 0.321 | 12.3% |
| H-PLOC wide4, 1 × 16,384 triangles | 0.508 | 0.451 | 11.2% |
| H-PLOC wide4, 128 × 4 triangles | 0.245 | 0.191 | 22.0% |
| H-PLOC wide4, 16 × 1,024 triangles | 0.470 | 0.412 | 12.3% |
| H-PLOC wide4, mixed sizes | 0.406 | 0.350 | 13.8% |
| H-PLOC CWBVH8, 1 × 16,384 triangles | 0.527 | 0.468 | 11.2% |
| H-PLOC CWBVH8, 128 × 4 triangles | 0.271 | 0.196 | 27.7% |
| H-PLOC CWBVH8, 16 × 1,024 triangles | 0.517 | 0.452 | 12.6% |
| H-PLOC CWBVH8, mixed sizes | 0.462 | 0.393 | 14.9% |

The large three-BLAS result varied between passes, with an earlier pass measuring
4.743 → 4.718 ms. No reliable improvement is claimed for this case. Mixed sizes
are 1, 17, 257, and 4,096 triangles. Node counts, leaf counts, and final storage
sizes match the baseline for all 12 synthetic fixtures in all ten final runs.

## Isolated metadata finalization

Repeat finalization of an already completed H-PLOC wide4 ticket with four
triangles per segment. Baseline: `finish_synchronized()` followed by the old
`GpuCompactWideLayout` constructor. Candidate: `finalize_layout(ctx)`.
Both measurements end with `ctx.synchronize()` and include fresh metadata
allocations. Five warmups and 64 timed iterations per process, five alternating
process pairs; medians of 320 samples per version. Geometry construction,
compaction, and descriptor emission are excluded.

| Segments | Before ms | After ms | Time reduction |
|---:|---:|---:|---:|
| 1 | 0.0705 | 0.0171 | 75.7% |
| 16 | 0.0718 | 0.0172 | 76.0% |
| 128 | 0.0736 | 0.0175 | 76.2% |
| 4,099 | 0.1225 | 0.0349 | 71.5% |

Nsight on one H-PLOC build of 128 four-triangle BLASes shows device-to-host
copies falling from 6 to 1 and `cuStreamSynchronize` calls from 22 to 7.
The new finalizer reads a fixed 20-byte summary and leaves segment offsets on
the GPU. Kernel launches increase from 25 to 26. Exact-size output allocation
still requires the summary readback. Reusable build arenas and diagnostic
`finish_synchronized()` paths retain their existing host validation.

Validation: 38 GPU tests passed across finalization, empty segments, compact
layout, bounds BVHs, TLAS, and BLAS/TLAS traversal. New coverage includes tiled
prefix sums through 4,099 segments, first-failure reporting across tiles,
invalid build statuses/counts, encoding mismatches, and UInt32 overflow.
