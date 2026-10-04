# NexusBVH vs Bajo GPU BVH benchmark

- Generated: `2026-10-03T21:03:14-04:00`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Mojo: `Mojo 1.2.0.dev2026100305 (4ad841b8)`
- Nexus checkout: `external/nexusbvh`
- Nexus revision: `dd6d7e9a017e`

## Summary

- Scene: Dragon OBJ, 249,882 triangles.
- Traversal: 1024x576 camera, 589,824 closest-hit rays.
- Timing: median of 11 synchronized runs; ranges show minimum to maximum.
- Fastest Bajo build: `H-PLOC-CWBVH8-n8-l4-m1-indexed` at 0.572 ms (0.911x Nexus build time).
- Fastest Bajo traversal: `H-PLOC-CWBVH8-n8-l4-m1` at 0.171 ms / 3443.5 MRay/s (0.934x Nexus traversal time).

## Build results

| Implementation | Configuration | Builder | Layout | Node width | Leaf width | Max leaf | Median ms | Min–max ms | Time / Nexus |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| bajo | `H-PLOC-CWBVH8-n8-l4-m1-indexed` | hploc | cwbvh8 | 8 | 4 | 1 | 0.572 | 0.568–0.702 | 0.911x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 8 | 4 | 1 | 0.608 | 0.596–0.633 | 0.968x |
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 8 | 1 | 1 | 0.628 | 0.608–1.103 | 1.000x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 8 | 4 | 3 | 0.676 | 0.660–0.882 | 1.076x |
| bajo | `LBVH-n2-l4` | lbvh | wide | 2 | 4 | 4 | 0.807 | 0.801–1.205 | 1.284x |
| bajo | `LBVH-n2-l2` | lbvh | wide | 2 | 2 | 2 | 0.896 | 0.889–1.333 | 1.426x |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 8 | 4 | 3 | 0.905 | 0.881–1.390 | 1.441x |
| bajo | `LBVH-n4-l2` | lbvh | wide | 4 | 2 | 2 | 0.957 | 0.946–1.195 | 1.523x |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 2 | 4 | 4 | 0.970 | 0.966–1.203 | 1.544x |
| bajo | `LBVH-n4-l4` | lbvh | wide | 4 | 4 | 4 | 0.980 | 0.958–1.498 | 1.560x |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 2 | 2 | 2 | 1.004 | 0.988–1.717 | 1.597x |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 4 | 2 | 2 | 1.139 | 1.107–1.547 | 1.813x |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 4 | 4 | 4 | 1.154 | 1.130–1.200 | 1.836x |
| bajo | `LBVH-n8-l4` | lbvh | wide | 8 | 4 | 4 | 1.209 | 1.174–1.645 | 1.924x |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 8 | 1 | 1 | 1.227 | 1.193–1.255 | 1.953x |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 8 | 4 | 4 | 1.415 | 1.364–2.418 | 2.252x |
| bajo | `LBVH-n8-l8` | lbvh | wide | 8 | 8 | 8 | 1.622 | 1.600–2.661 | 2.582x |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 8 | 8 | 8 | 1.828 | 1.799–2.865 | 2.909x |

## Cached geometry rebuilds

These reuse geometry bounds, scratch buffers, and segment offsets. They are separate from the fresh-build comparison above.

| Configuration | Median ms | Min–max ms |
|---|---:|---:|
| `H-PLOC-CWBVH8-n8-l4-m1` | 0.557 | 0.552–0.563 |
| `H-PLOC-CWBVH8-n8-l4-m1-indexed` | 0.491 | 0.485–0.502 |
| `H-PLOC-CWBVH8-n8-l4-m3` | 0.623 | 0.610–0.635 |

## Bajo build stages

Stage timings use separately synchronized host wall-clock measurements in cached rebuilds and include dispatch overhead. Geometry bounds and allocation are excluded from these stage totals and included in the fresh-build results above.

| Configuration | Morton ms | Sort ms | H-PLOC ms | Collapse ms | Pack ms | Instrumented ms |
|---|---:|---:|---:|---:|---:|---:|
| `H-PLOC-CWBVH8-n8-l4-m3` | 0.023 | 0.049 | 0.238 | 0.277 | 0.050 | 0.637 |
| `H-PLOC-CWBVH8-n8-l4-m1` | 0.026 | 0.049 | 0.230 | 0.218 | 0.055 | 0.577 |
| `H-PLOC-CWBVH8-n8-l4-m1-indexed` | 0.013 | 0.048 | 0.236 | 0.210 | 0.000 | 0.507 |

## Traversal results

| Implementation | Configuration | Builder | Layout | Median ms | MRay/s | Min–max ms | Time / Nexus | Hits |
|---|---|---|---|---:|---:|---:|---:|---:|
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 0.171 | 3443.5 | 0.168–0.177 | 0.934x | 71,598 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 0.176 | 3343.0 | 0.170–0.182 | 0.962x | 71,598 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1-indexed` | hploc | cwbvh8 | 0.178 | 3308.2 | 0.176–0.182 | 0.972x | 71,598 |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 0.180 | 3282.9 | 0.177–0.432 | 0.980x | 71,598 |
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 0.183 | 3215.9 | 0.180–0.189 | 1.000x | 71,599 |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 0.194 | 3041.8 | 0.191–0.198 | 1.057x | 71,598 |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 0.202 | 2922.6 | 0.197–0.207 | 1.100x | 71,598 |
| bajo | `LBVH-n2-l4` | lbvh | wide | 0.216 | 2733.1 | 0.207–0.220 | 1.177x | 71,598 |
| bajo | `LBVH-n2-l2` | lbvh | wide | 0.223 | 2644.2 | 0.217–0.232 | 1.216x | 71,598 |
| bajo | `LBVH-n4-l2` | lbvh | wide | 0.260 | 2265.5 | 0.250–0.645 | 1.420x | 71,598 |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 0.268 | 2198.4 | 0.262–0.272 | 1.463x | 71,598 |
| bajo | `LBVH-n4-l4` | lbvh | wide | 0.275 | 2141.8 | 0.271–0.277 | 1.501x | 71,598 |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 0.298 | 1976.8 | 0.291–0.554 | 1.627x | 71,598 |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 0.367 | 1607.5 | 0.360–0.372 | 2.001x | 71,598 |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 0.442 | 1335.2 | 0.424–0.589 | 2.409x | 71,598 |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 0.525 | 1123.8 | 0.523–0.530 | 2.862x | 71,598 |
| bajo | `LBVH-n8-l8` | lbvh | wide | 0.545 | 1081.9 | 0.540–0.549 | 2.972x | 71,598 |
| bajo | `LBVH-n8-l4` | lbvh | wide | 0.558 | 1057.4 | 0.440–1.110 | 3.041x | 71,598 |

## Traversal work

The instrumented kernels run after timing. Counts cover every camera ray and therefore do not perturb the headline traversal measurements.

| Implementation | Configuration | Nodes/ray | Leaf groups/ray | Triangles/ray | Maximum stack |
|---|---|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | 3.631 | 0.304 | 0.545 | 7 |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | 3.103 | 0.285 | 0.718 | 6 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | 2.971 | 0.280 | 0.730 | 6 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | 3.136 | 0.318 | 0.531 | 6 |

## Validation

Every Bajo row is compared with NexusBVH. A one-hit difference is accepted for a ray exactly on a silhouette edge; in that case the mean hit-distance difference must be at most 0.01.

| Bajo configuration | Hit-count delta | Mean-distance delta |
|---|---:|---:|
| `LBVH-n2-l2` | 1 | 0.000325145 |
| `LBVH-n2-l4` | 1 | 0.000325145 |
| `LBVH-n4-l2` | 1 | 0.000325145 |
| `LBVH-n4-l4` | 1 | 0.000325145 |
| `LBVH-n8-l4` | 1 | 0.000325145 |
| `LBVH-n8-l8` | 1 | 0.000325145 |
| `H-PLOC-n2-l2` | 1 | 0.000325145 |
| `H-PLOC-n2-l4` | 1 | 0.000325145 |
| `H-PLOC-n4-l2` | 1 | 0.000325145 |
| `H-PLOC-n4-l4` | 1 | 0.000325145 |
| `H-PLOC-n8-l4` | 1 | 0.000325145 |
| `H-PLOC-n8-l8` | 1 | 0.000325145 |
| `H-PLOC-n8-l1` | 1 | 0.000325145 |
| `LBVH-CWBVH8-n8-l4-m3` | 1 | 0.000325145 |
| `H-PLOC-CWBVH8-n8-l4-m3` | 1 | 0.000325145 |
| `H-PLOC-CWBVH8-n8-l4-m1` | 1 | 0.000325145 |
| `H-PLOC-CWBVH8-n8-l4-m1-indexed` | 1 | 0.000325145 |

## Methodology

Bajo's LBVH is Apetrei's 2014 agglomerative algorithm, which fuses binary topology construction and bounds propagation in one leaf-driven kernel. Bajo compares it with H-PLOC across ordinary-wide combinations; both builders then use Bajo's existing H-PLOC-derived wide collapse. H-PLOC includes an 8/1/1 row matching NexusBVH's one-triangle leaves. Bajo also measures LBVH and H-PLOC with CWBVH8; storage leaf width is 4 and maximum encoded leaf sizes are 3 and 1 where listed. NexusBVH uses its H-PLOC CWBVH8 builder and currently stores exactly one triangle per leaf. Both implementations trace the same generated camera rays with native packed CWBVH8 or ordinary-wide traversal.

OBJ parsing, camera setup, and initial host-to-device upload are outside the timed regions. Headline build timings include fresh construction, device allocation, segment-offset setup, geometry bounds, Morton generation/sort, topology, conversion, triangle representation, and final synchronization. Cached geometry rebuilds are reported separately for Bajo's reusable H-PLOC CWBVH8 arenas. Traversal timing includes kernel launch and synchronization. Different builder/layout rows do not imply equivalent hierarchy quality.
