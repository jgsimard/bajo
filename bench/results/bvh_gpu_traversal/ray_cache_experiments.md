# GPU ray-load cache experiments — 2026-10-05

Rejected all ray-load cache hints. Streaming loads improve wide Dragon secondary closest by 1.5% but regress Buddha/Bunny by 0.9–1.0%. Universal streaming regresses compressed secondary closest 1.7–8.0% and shadows 2.6–12.5%; read-only and L2-only hints offer no consistent gain.

**Measurement:** RTX 5060 Ti; 524,288 rays, 108 instances, TLAS2/leaf1 with compressed or wide BLASes. Three pairs for hint screening, five pairs for wide-only streaming confirmation. Five warmups, 21 batches of eight queries. Ordering, traversal and stores included; build, allocation, upload and readback excluded.

| Mesh | Secondary closest before ms | Candidate ms | Time reduction |
|---|---:|---:|---:|
| dragon | 6.1317 | 6.0407 | 1.5% |
| buddha | 3.2408 | 3.2704 | -0.9% |
| bunny | 5.2396 | 5.2935 | -1.0% |

**Validation:** All 612 full-hit and 612 visibility hashes match. Production unchanged. Raw results: `ray_cache_*_timings.json`; logs/resources: `.cache/bvh_ray_cache_2026_10_05`. No cache bottleneck established.
