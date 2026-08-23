# NexusBVH vs Bajo GPU BVH benchmark

Generated: `2026-08-23T14:55:54-04:00`  
GPU: `NVIDIA GeForce RTX 5060 Ti`  
Mojo: `Mojo 1.1.0.dev2026082305 (54b9e0e2)`  
Nexus checkout: `/tmp/nexusbvh-reference`  
Nexus revision: `dd6d7e9a017e`

## Summary

- Scene: Dragon OBJ, 249,882 triangles.
- Traversal: 1024x576 camera, 589,824 closest-hit rays.
- Timing: median of 11 synchronized runs; ranges show minimum to maximum.
- Fastest Bajo build: `LBVH-n2-l2` at 2.002 ms (3.158x Nexus build time).
- Fastest Bajo traversal: `LBVH-CWBVH8-n8-l4-m3` at 0.205 ms / 2872.4 MRay/s (1.129x Nexus traversal time).

## Build results

| Implementation | Configuration | Builder | Layout | Node width | Leaf width | Max leaf | Median ms | Min–max ms | Time / Nexus |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 8 | 1 | 1 | 0.634 | 0.623–0.938 | 1.000x |
| bajo | `LBVH-n2-l2` | lbvh | wide | 2 | 2 | 2 | 2.002 | 1.988–2.925 | 3.158x |
| bajo | `LBVH-n2-l4` | lbvh | wide | 2 | 4 | 4 | 2.108 | 2.080–2.921 | 3.327x |
| bajo | `LBVH-n4-l2` | lbvh | wide | 4 | 2 | 2 | 2.191 | 2.162–2.813 | 3.457x |
| bajo | `LBVH-n4-l4` | lbvh | wide | 4 | 4 | 4 | 2.405 | 2.316–3.255 | 3.795x |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 8 | 4 | 3 | 2.489 | 2.398–3.233 | 3.927x |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 2 | 2 | 2 | 2.597 | 2.579–3.560 | 4.098x |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 2 | 4 | 4 | 2.697 | 2.651–3.495 | 4.256x |
| bajo | `LBVH-n8-l4` | lbvh | wide | 8 | 4 | 4 | 2.793 | 2.773–3.709 | 4.407x |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 4 | 2 | 2 | 2.812 | 2.780–3.670 | 4.437x |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 8 | 1 | 1 | 2.970 | 2.936–3.840 | 4.686x |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 4 | 4 | 4 | 3.004 | 2.924–3.764 | 4.740x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 8 | 4 | 3 | 3.081 | 2.989–3.839 | 4.862x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 8 | 4 | 1 | 3.173 | 3.122–4.007 | 5.006x |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 8 | 4 | 4 | 3.383 | 3.299–4.244 | 5.339x |
| bajo | `LBVH-n8-l8` | lbvh | wide | 8 | 8 | 8 | 3.496 | 3.352–4.333 | 5.517x |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 8 | 8 | 8 | 4.069 | 3.952–5.118 | 6.420x |

## Traversal results

| Implementation | Configuration | Builder | Layout | Median ms | MRay/s | Min–max ms | Time / Nexus | Hits |
|---|---|---|---|---:|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 0.182 | 3241.9 | 0.176–0.186 | 1.000x | 71,599 |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 0.205 | 2872.4 | 0.202–0.215 | 1.129x | 71,598 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 0.206 | 2864.7 | 0.201–0.218 | 1.132x | 71,598 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 0.207 | 2850.7 | 0.199–0.212 | 1.137x | 71,598 |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 0.223 | 2646.4 | 0.219–0.224 | 1.225x | 71,598 |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 0.240 | 2453.5 | 0.237–0.247 | 1.321x | 71,598 |
| bajo | `LBVH-n2-l2` | lbvh | wide | 0.246 | 2394.9 | 0.236–0.247 | 1.354x | 71,598 |
| bajo | `LBVH-n2-l4` | lbvh | wide | 0.262 | 2247.5 | 0.258–0.270 | 1.442x | 71,598 |
| bajo | `LBVH-n4-l2` | lbvh | wide | 0.276 | 2134.5 | 0.270–0.281 | 1.519x | 71,598 |
| bajo | `LBVH-n4-l4` | lbvh | wide | 0.289 | 2039.8 | 0.281–0.644 | 1.589x | 71,598 |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 0.290 | 2030.5 | 0.284–0.294 | 1.597x | 71,598 |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 0.310 | 1901.6 | 0.301–0.659 | 1.705x | 71,598 |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 0.370 | 1592.0 | 0.361–0.373 | 2.036x | 71,598 |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 0.428 | 1376.5 | 0.418–0.788 | 2.355x | 71,598 |
| bajo | `LBVH-n8-l4` | lbvh | wide | 0.443 | 1332.6 | 0.439–0.447 | 2.433x | 71,598 |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 0.546 | 1079.5 | 0.541–0.549 | 3.003x | 71,598 |
| bajo | `LBVH-n8-l8` | lbvh | wide | 0.563 | 1047.3 | 0.553–0.910 | 3.096x | 71,598 |

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

## Methodology

Bajo covers LBVH and H-PLOC ordinary-wide combinations, an H-PLOC 8/1/1 row matching NexusBVH's one-triangle leaves, and LBVH/H-PLOC CWBVH8. Bajo CWBVH8 uses storage leaf width 4 and is measured with maximum encoded leaf sizes 3 and 1. NexusBVH uses its H-PLOC CWBVH8 builder and currently stores exactly one triangle per leaf. Both implementations trace the same generated camera rays with native packed CWBVH8 or ordinary-wide traversal.

OBJ parsing, camera setup, and initial host-to-device upload are outside the timed regions. Build timing includes the complete GPU build and synchronization. Traversal timing includes kernel launch and synchronization. Different builder/layout rows do not imply equivalent hierarchy quality.
