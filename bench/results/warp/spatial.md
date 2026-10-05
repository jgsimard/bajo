# Spatial bounds: Bajo vs Warp

2026-10-05 · Warp 1.17.0 · 256 boxes, four groups, 512 queries, seven warmed samples.

Bajo median/preorder vs Warp CPU SAH / GPU LBVH, one item per leaf. Ratios are Warp/Bajo; above 1 favors Bajo.

| Operation | CPU ratio | GPU ratio |
|---|---:|---:|
| Fresh build | 0.54 | 0.92 |
| Refit | 0.16 | 0.04 |
| Retained rebuild | 0.52 | 0.11 |
| AABB / global | 1.95 | 1.62 |
| RAY / global | 1.45 | 0.39 |
| SPHERE / global | 1.70 | 0.51 |
| CAPSULE / global | 0.70 | 0.12 |
| AABB / group | 4.45 | 1.84 |
| RAY / group | 3.12 | 0.63 |
| SPHERE / group | 4.09 | 0.97 |
| CAPSULE / group | 1.29 | 0.28 |

Queries time full per-query candidate matrices, including zeroing/output writes, launch and synchronization; download is excluded. GPU construction starts from host bounds and includes uploads and retained scratch. Updates use unchanged warmed inputs; GPU inputs are already resident. Bajo GPU updates include validation; status download is excluded. CPU/GPU rebuild algorithms differ: CPU median versus SAH, GPU X-order baseline versus LBVH.

All 16 query cases have identical complete candidate sets and match an independent exhaustive reference. Scope: one small bounds workload, not a library-wide speed ranking. [Raw samples and hashes](spatial.json).

Reproduce: `pixi run python bench/warp/compare_spatial.py --warp-path /tmp/bajo_warp_deps`.
