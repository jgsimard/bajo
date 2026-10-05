# GPU ray ordering for incoherent TLAS traversal — 2026-10-05

Retained opt-in `GpuTlas.launch_ordered_rays` with octant/Morton keys, OneSweep and original-index hit writes. Secondary closest improves 55.7–69.5%, shadows 57.8–70.5%. Coherent and shuffled camera rays regress; use direct launches for them. These are traversal gains. Later passes: [near-first callbacks](nested_tlas_comparison.md), [indexed ray inputs](ray_input_comparison.md).

**Measurement:** Five alternating pairs on RTX 5060 Ti; 524,288 rays, 108 instances, H-PLOC CWBVH8/max-leaf1 BLAS and LBVH2/leaf1 TLAS. Direct/ordered blocks: 128/32. Five warmups, 21 batches of eight synchronized launches; median of process medians. Ordered includes keys, sort, gather and hit writes; build, workspace allocation, upload and readback excluded.

| Mesh | Rays | Trace | Direct ms | Ordered ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.2151 | 0.4034 | -87.5% |
| dragon | coherent | any | 0.1686 | 0.3441 | -104.0% |
| dragon | shuffled | closest | 0.6504 | 0.7968 | -22.5% |
| dragon | shuffled | any | 0.4184 | 0.5964 | -42.5% |
| dragon | secondary | closest | 7.9049 | 3.1577 | 60.1% |
| dragon | secondary | any | 5.5638 | 2.2019 | 60.4% |
| buddha | coherent | closest | 0.2267 | 0.4293 | -89.4% |
| buddha | coherent | any | 0.1555 | 0.3432 | -120.6% |
| buddha | shuffled | closest | 0.7411 | 0.8901 | -20.1% |
| buddha | shuffled | any | 0.4557 | 0.6202 | -36.1% |
| buddha | secondary | closest | 6.7314 | 2.0511 | 69.5% |
| buddha | secondary | any | 4.6841 | 1.3822 | 70.5% |
| bunny | coherent | closest | 0.2448 | 0.4510 | -84.2% |
| bunny | coherent | any | 0.1732 | 0.3623 | -109.2% |
| bunny | shuffled | closest | 1.0548 | 1.1480 | -8.8% |
| bunny | shuffled | any | 0.6784 | 0.8104 | -19.5% |
| bunny | secondary | closest | 7.2317 | 3.2034 | 55.7% |
| bunny | secondary | any | 5.2535 | 2.2160 | 57.8% |

**Validation:** All 18 full-hit cases match in every pair. Four new tests and eight BLAS/TLAS tests pass; new tests pass memcheck with zero errors. Workspace: 24.3 MiB at 524K rays in this historical pass. Raw results and ablations: `ray_ordering*.json`.
