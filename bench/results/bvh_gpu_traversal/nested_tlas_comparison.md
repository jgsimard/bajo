# Near-first GPU TLAS callbacks — 2026-10-05

Retained near-first compressed TLAS callbacks and pop-time distance pruning for ordered closest-hit queries. Secondary closest improves 35.9–37.9% direct and 18.6–20.9% ordered; shadows are effectively flat. Example camera: 0.8926 → 0.6683 ms (25.1%). Coherent rays still favor direct launches.

**Measurement:** Five alternating pairs; 524,288 rays, 108 instances, CWBVH8/max-leaf1 BLAS and LBVH2/leaf1 TLAS. Direct/ordered blocks: 128/32. Five warmups, 21 batches of eight synchronized launches. Ordered includes sorting/gathering; build, allocation, upload and readback excluded.

| Path | Mesh | Rays | Before ms | After ms | Time reduction |
|---|---|---|---:|---:|---:|
| direct | dragon | coherent | 0.2145 | 0.1829 | 14.7% |
| direct | dragon | shuffled | 0.6476 | 0.5151 | 20.5% |
| direct | dragon | secondary | 7.9027 | 5.0367 | 36.3% |
| direct | buddha | coherent | 0.2270 | 0.1832 | 19.3% |
| direct | buddha | shuffled | 0.7413 | 0.5960 | 19.6% |
| direct | buddha | secondary | 6.7285 | 4.1814 | 37.9% |
| direct | bunny | coherent | 0.2453 | 0.1999 | 18.5% |
| direct | bunny | shuffled | 1.0556 | 0.8441 | 20.0% |
| direct | bunny | secondary | 7.2317 | 4.6329 | 35.9% |
| ordered | dragon | coherent | 0.4037 | 0.3639 | 9.9% |
| ordered | dragon | shuffled | 0.7969 | 0.6818 | 14.4% |
| ordered | dragon | secondary | 3.1567 | 2.4965 | 20.9% |
| ordered | buddha | coherent | 0.4294 | 0.3737 | 13.0% |
| ordered | buddha | shuffled | 0.8902 | 0.7697 | 13.5% |
| ordered | buddha | secondary | 2.0536 | 1.6725 | 18.6% |
| ordered | bunny | coherent | 0.4510 | 0.3945 | 12.5% |
| ordered | bunny | shuffled | 1.1478 | 0.9732 | 15.2% |
| ordered | bunny | secondary | 3.2021 | 2.5649 | 19.9% |

**Validation:** 23 GPU tests pass; memcheck reports zero errors. All 360 ray-batch hashes, camera hashes and per-backend images match. Ordered kernel registers: 111 → 80; theoretical resident threads/SM: 512 → 768; stack: 368 → 784 bytes, no spills. Raw results: `nested_tlas*.json`.
