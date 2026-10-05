# GPU CWBVH task masks — 2026-10-05

Retained predicated CWBVH slot/task masks, suppressing empty metadata without changing layout or task order. All 36 TLAS cases improve. Standalone Buddha coherent any-hit regresses 1.5% (leaf1) and 0.4% (leaf3).

**Measurement:** Five alternating pairs; 524,288 rays, 108 instances for TLAS. CWBVH8/max-leaf1 BLAS, LBVH2/leaf1 TLAS. Ordered/direct blocks: 32/128. Five warmups, 21 batches of eight synchronized launches; ordered includes keys, sort and gather. Build, allocation, upload and readback excluded.

## TLAS secondary rays (ms)

| Launch | Mesh | Trace | Before | After | Time reduction |
|---|---|---|---:|---:|---:|
| Ordered | dragon | closest | 2.4956 | 2.3700 | 5.0% |
| Ordered | dragon | any | 2.2035 | 2.1394 | 2.9% |
| Ordered | buddha | closest | 1.6722 | 1.5897 | 4.9% |
| Ordered | buddha | any | 1.3798 | 1.3393 | 2.9% |
| Ordered | bunny | closest | 2.5672 | 2.4316 | 5.3% |
| Ordered | bunny | any | 2.2179 | 2.1461 | 3.2% |
| Unsorted | dragon | closest | 5.0371 | 4.7386 | 5.9% |
| Unsorted | dragon | any | 5.5662 | 5.3440 | 4.0% |
| Unsorted | buddha | closest | 4.1808 | 3.9353 | 5.9% |
| Unsorted | buddha | any | 4.6842 | 4.5004 | 3.9% |
| Unsorted | bunny | closest | 4.6351 | 4.3590 | 6.0% |
| Unsorted | bunny | any | 5.2516 | 5.0430 | 4.0% |

## Standalone secondary rays (ms)

| Mesh | Maximum leaf | Closest-hit | Any-hit |
|---|---:|---|---|
| dragon | 1 | 1.0318 → 1.0172 (1.4%) | 0.7719 → 0.7629 (1.2%) |
| dragon | 3 | 1.0645 → 1.0265 (3.6%) | 0.7665 → 0.7372 (3.8%) |
| buddha | 1 | 0.8352 → 0.8241 (1.3%) | 0.5905 → 0.5842 (1.1%) |
| buddha | 3 | 0.8509 → 0.8209 (3.5%) | 0.5774 → 0.5554 (3.8%) |
| bunny | 1 | 0.7305 → 0.7202 (1.4%) | 0.5840 → 0.5777 (1.1%) |
| bunny | 3 | 0.7260 → 0.6992 (3.7%) | 0.5646 → 0.5425 (3.9%) |

**Validation:** 34 GPU tests pass; memcheck reports zero errors. All 720 full-hit digests and visibility digests match; decoder covers 1.62 million node/ray cases. Assembly reduces integer selects; no spills. Hardware counters unavailable. Raw results: `node_mask_*_final_timings.json`; diagnostics: `node_mask_*.json`.
