# Bajo vs NVIDIA Warp

2026-10-05 · AMD Ryzen 7 9700X 8-Core Processor · NVIDIA GeForce RTX 5060 Ti, 610.57.04 · Warp 1.17.0

Identical triangle soup and precomputed 512×288 camera rays; 11 warmed samples per case. Build includes fresh BVH allocation and bounds/packing; input upload and destruction are excluded. Queries include launch and synchronization, write distance or an any-hit flag, and exclude download. Meshes are centered and scaled to a largest extent of 2. CPU builds use library-default threading; CPU queries use one scalar ray per iteration (Bajo still uses SIMD within each BVH node).

Bajo: CPU median/SAH/LBVH/H-PLOC with node16/leaf16; GPU H-PLOC/CWBVH8 with leaf1/3. Warp: CPU SAH/median/cuBQL; GPU LBVH/SAH/cuBQL, each with leaf1/4. Fastest measured configuration for each phase; build and query winners may differ.

| Scene / backend | Bajo build ms | Warp build ms | Bajo closest ms | Warp closest ms | Bajo any ms | Warp any ms |
|---|---:|---:|---:|---:|---:|---:|
| bunny / cpu | 1.136 | 5.394 | 3.715 | 24.774 | 2.713 | 20.406 |
| bunny / cuda | 0.369 | 0.255 | 0.091 | 0.121 | 0.060 | 0.112 |
| dragon / cpu | 3.633 | 18.995 | 2.660 | 15.339 | 1.979 | 12.622 |
| dragon / cuda | 0.608 | 0.650 | 0.073 | 0.110 | 0.051 | 0.099 |

GPU stable key/value sorting: identical nonnegative 31-bit keys and 32-bit IDs, retained scratch, reset copies excluded. Patterns: mixed / duplicates / 24-bit.

| Pairs | Pattern | Bajo radix ms | Bajo OneSweep ms | Warp radix ms |
|---:|---|---:|---:|---:|
| 65,536 | mixed | 0.038 | 0.043 | 0.059 |
| 65,536 | duplicates | 0.036 | 0.041 | 0.048 |
| 65,536 | 24-bit | 0.038 | 0.043 | 0.056 |
| 1,048,576 | mixed | 0.132 | 0.132 | 0.193 |
| 1,048,576 | duplicates | 0.124 | 0.123 | 0.152 |
| 1,048,576 | 24-bit | 0.130 | 0.131 | 0.179 |

Every ray is checked against Warp CPU SAH/leaf4: ≤1 hit-mask disagreement per case (edge tolerance), distance tolerance `1e-4 + 1e-4 × |t|`. Every sort checks order, key/value association and stability. Observed maximum closest/any-hit mask disagreements: 0.

Scope: two static meshes, coherent camera rays, two sort sizes, one machine; no rendering, instancing, incoherent rays or dynamic updates. Warp uses triangle-soup indices here; its indexed-mesh memory advantage is not measured. CPU queries are not an all-core throughput comparison.

Full configurations, samples, input/source hashes, environment and validation: [comparison.json](comparison.json). Reproduce: `pixi run python bench/warp/compare.py --warp-path /tmp/bajo_warp_deps`.
