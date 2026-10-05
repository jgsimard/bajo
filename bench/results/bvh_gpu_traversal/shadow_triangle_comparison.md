# GPU compressed shadow triangle tests — 2026-10-05

Retained determinant sign-bit normalization and removal of the redundant u upper-bound check for packed/indexed CWBVH8 shadows. Secondary gains: 1.0–1.3% packed, 0.7–1.2% indexed, 0.1–0.5% ordered TLAS, 0.2–0.3% unsorted TLAS. Short camera cases regress up to 1.5%; closest-hit controls are approximately unchanged.

**Measurement:** Five alternating pairs on RTX 5060 Ti; 524,288 rays, 108 instances for TLAS. Five warmups, 21 batches of eight synchronized launches. Standalone/direct blocks: 128; ordered: 32. Ordered includes keys, sort and gather; build, allocation, upload, readback and hashing excluded.

## Standalone secondary shadows (ms)

| Storage | Mesh | Maximum leaf | Before | After | Time reduction |
|---|---|---:|---:|---:|---:|
| Packed | dragon | 1 | 0.7623 | 0.7546 | 1.0% |
| Packed | dragon | 3 | 0.7382 | 0.7288 | 1.3% |
| Packed | buddha | 1 | 0.5841 | 0.5778 | 1.1% |
| Packed | buddha | 3 | 0.5554 | 0.5492 | 1.1% |
| Packed | bunny | 1 | 0.5768 | 0.5710 | 1.0% |
| Packed | bunny | 3 | 0.5423 | 0.5355 | 1.3% |
| Indexed | dragon | 1 | 0.8130 | 0.8073 | 0.7% |
| Indexed | dragon | 3 | 0.8109 | 0.8028 | 1.0% |
| Indexed | buddha | 1 | 0.6284 | 0.6230 | 0.9% |
| Indexed | buddha | 3 | 0.6186 | 0.6110 | 1.2% |
| Indexed | bunny | 1 | 0.6161 | 0.6102 | 1.0% |
| Indexed | bunny | 3 | 0.5941 | 0.5877 | 1.1% |

## TLAS secondary shadows (ms)

| Launch | Mesh | Before | After | Time reduction |
|---|---|---:|---:|---:|
| Ordered | dragon | 2.0951 | 2.0839 | 0.5% |
| Ordered | buddha | 1.3399 | 1.3333 | 0.5% |
| Ordered | bunny | 2.1437 | 2.1412 | 0.1% |
| Unsorted | dragon | 5.3463 | 5.3329 | 0.3% |
| Unsorted | buddha | 4.4997 | 4.4871 | 0.3% |
| Unsorted | bunny | 5.0415 | 5.0297 | 0.2% |

**Validation:** 33 GPU tests pass, including 6,291,456 mask comparisons; memcheck reports zero errors. All 1,080 full-hit and TLAS visibility digests match. Inspected ordered shader retains 95 registers, 368-byte stack and zero spills. Raw results: `shadow_triangle_*_final_timings.json`; diagnostics: `shadow_triangle_*.json`.
