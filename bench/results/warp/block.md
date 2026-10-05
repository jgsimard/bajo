# Cooperative bounds: Bajo vs Warp

2026-10-05 · Warp 1.17.0 · 256 boxes, 512 queries, blocks of 32/64, seven warmed samples.

Ratios are Warp/Bajo; above 1 favors Bajo. Both CPU paths are serial fallbacks. GPU Bajo scans preorder leaves; Warp cooperatively traverses its LBVH.

| Operation | CPU ratio | GPU ratio |
|---|---:|---:|
| AABB / 32 lanes | 0.78 | 1.03 |
| Ray / 32 lanes | 0.39 | 0.31 |
| AABB / 64 lanes | 0.79 | 0.96 |
| Ray / 64 lanes | 0.37 | 0.24 |

All eight CPU/GPU count matrices match Warp and exhaustive reference exactly, including empty queries and partial batches. GPU timings include a separate output-zero operation, collective query launch, output writes and synchronization; input upload/download are excluded. Warp uses atomic output increments; Bajo writes each unique emitted item directly. Ray intervals are nonnegative and effectively unbounded. Scope: one small global-bounds workload; grouped cooperative queries are covered separately by Bajo regressions. [Raw samples and hashes](block.json).

Reproduce: `pixi run python bench/warp/compare_block.py --warp-path /tmp/bajo_warp_deps`.
