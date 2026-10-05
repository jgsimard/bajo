# NexusBVH vs Bajo GPU BVH benchmark

2026-10-03 · RTX 5060 Ti · Mojo `1.2.0.dev2026100305 (4ad841b8)` · Nexus `dd6d7e9a017e`.

Dragon: 249,882 triangles, 589,824 camera closest-hit rays. Median of 11 synchronized runs.

Fastest Bajo build: indexed H-PLOC CWBVH8, 0.572 ms (0.911x Nexus). Fastest traversal: packed H-PLOC CWBVH8/max-leaf1, 0.171 ms (0.934x Nexus).

| Configuration | Build ms | Trace ms | MRay/s |
|---|---:|---:|---:|
| `H-PLOC-CWBVH8-n8-l4-m1` | 0.608 | 0.171 | 3443.5 |
| `H-PLOC-CWBVH8-n8-l4-m3` | 0.676 | 0.176 | 3343.0 |
| `H-PLOC-CWBVH8-n8-l4-m1-indexed` | 0.572 | 0.178 | 3308.2 |
| `LBVH-CWBVH8-n8-l4-m3` | 0.905 | 0.180 | 3282.9 |
| `NexusBVH-H-PLOC-CWBVH8` | 0.628 | 0.183 | 3215.9 |
| `H-PLOC-n2-l4` | 0.970 | 0.194 | 3041.8 |
| `H-PLOC-n2-l2` | 1.004 | 0.202 | 2922.6 |
| `LBVH-n2-l4` | 0.807 | 0.216 | 2733.1 |
| `LBVH-n2-l2` | 0.896 | 0.223 | 2644.2 |
| `LBVH-n4-l2` | 0.957 | 0.260 | 2265.5 |
| `H-PLOC-n4-l2` | 1.139 | 0.268 | 2198.4 |
| `LBVH-n4-l4` | 0.980 | 0.275 | 2141.8 |
| `H-PLOC-n4-l4` | 1.154 | 0.298 | 1976.8 |
| `H-PLOC-n8-l1` | 1.227 | 0.367 | 1607.5 |
| `H-PLOC-n8-l4` | 1.415 | 0.442 | 1335.2 |
| `H-PLOC-n8-l8` | 1.828 | 0.525 | 1123.8 |
| `LBVH-n8-l8` | 1.622 | 0.545 | 1081.9 |
| `LBVH-n8-l4` | 1.209 | 0.558 | 1057.4 |

Fresh build includes allocation, bounds, topology, conversion and synchronization; parsing, camera setup and initial upload excluded. Traversal includes launch/synchronization. Cached rebuilds are separate (0.491–0.623 ms for the three H-PLOC CWBVH8 variants).

All 17 Bajo configurations validate: one silhouette-edge hit difference, mean-distance delta 0.000325145 (allowed ≤ 0.01).

Full sweep, ranges, profiles and validation: [comparison.json](comparison.json). Reproduce with `pixi run bench_bvh_gpu_nexus_compare`.
