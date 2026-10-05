# GPU BVH finalization — 2026-10-04

Device-side validation and compact offsets reduce synthetic build time 11.0–27.7% and TLAS time 19.4%. The large three-BLAS case has no reliable improvement (final pass: 1.0% slower).

**Measurement:** RTX 5060 Ti, sm_120; baseline `6cfe28fa`. Five alternating process pairs with fixed CPU affinity. Fresh builds include allocation, upload and synchronization; median of process medians (11 runs for scenes, seven for fixtures).

## Fresh builds

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

## Metadata finalization

Completed H-PLOC wide4 tickets, four triangles/segment; fresh metadata allocations and final synchronization included, geometry/compaction excluded. Five warmups, 64 samples per process, five pairs: median of 320 samples/version.

| Segments | Before ms | After ms | Time reduction |
|---:|---:|---:|---:|
| 1 | 0.0705 | 0.0171 | 75.7% |
| 16 | 0.0718 | 0.0172 | 76.0% |
| 128 | 0.0736 | 0.0175 | 76.2% |
| 4,099 | 0.1225 | 0.0349 | 71.5% |

**Validation:** 38 GPU tests pass; synthetic node/leaf counts and storage sizes match. On 128 four-triangle BLASes, device-to-host copies fall 6 → 1 and stream synchronizations 22 → 7; launches rise 25 → 26. Exact-size allocation still needs a 20-byte summary readback.
