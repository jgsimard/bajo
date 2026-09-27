# NexusBVH vs Bajo GPU BVH benchmark

- Generated: `2026-09-27T10:42:50-04:00`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Mojo: `Mojo 1.2.0.dev2026092705 (3d1f3942)`
- Nexus checkout: `/home/jgs/dev/mojo/bajo/external/nexusbvh`
- Nexus revision: `dd6d7e9a017e`

## Summary

- Scene: Dragon OBJ, 249,882 triangles.
- Traversal: 1024x576 camera, 589,824 closest-hit rays.
- Timing: median of 11 synchronized runs; ranges show minimum to maximum.
- Fastest Bajo build: `H-PLOC-CWBVH8-n8-l4-m1` at 1.061 ms (1.740x Nexus build time).
- Fastest Bajo traversal: `H-PLOC-CWBVH8-n8-l4-m1` at 0.175 ms / 3364.1 MRay/s (0.947x Nexus traversal time).

## Build results

| Implementation | Configuration | Builder | Layout | Node width | Leaf width | Max leaf | Median ms | Min–max ms | Time / Nexus |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 8 | 1 | 1 | 0.610 | 0.604–0.628 | 1.000x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 8 | 4 | 1 | 1.061 | 1.053–1.455 | 1.740x |
| bajo | `LBVH-n2-l2` | lbvh | wide | 2 | 2 | 2 | 1.137 | 1.127–1.562 | 1.865x |
| bajo | `LBVH-n2-l4` | lbvh | wide | 2 | 4 | 4 | 1.167 | 1.157–1.581 | 1.914x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 8 | 4 | 3 | 1.170 | 1.161–1.314 | 1.918x |
| bajo | `LBVH-n4-l2` | lbvh | wide | 4 | 2 | 2 | 1.253 | 1.227–1.776 | 2.055x |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 2 | 2 | 2 | 1.331 | 1.322–1.812 | 2.182x |
| bajo | `LBVH-n4-l4` | lbvh | wide | 4 | 4 | 4 | 1.352 | 1.337–1.476 | 2.217x |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 8 | 4 | 3 | 1.383 | 1.359–1.738 | 2.268x |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 2 | 4 | 4 | 1.385 | 1.366–1.824 | 2.271x |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 4 | 2 | 2 | 1.473 | 1.445–1.593 | 2.415x |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 4 | 4 | 4 | 1.545 | 1.531–2.035 | 2.533x |
| bajo | `LBVH-n8-l4` | lbvh | wide | 8 | 4 | 4 | 1.576 | 1.547–2.023 | 2.584x |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 8 | 1 | 1 | 1.599 | 1.570–2.035 | 2.621x |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 8 | 4 | 4 | 1.755 | 1.736–2.316 | 2.878x |
| bajo | `LBVH-n8-l8` | lbvh | wide | 8 | 8 | 8 | 2.037 | 1.981–2.423 | 3.339x |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 8 | 8 | 8 | 2.202 | 2.169–2.643 | 3.610x |

## Bajo build stages

Stage timings use separately instrumented warm rebuilds; the synchronization barriers are excluded from the headline build results above.

| Configuration | Morton ms | Sort ms | H-PLOC ms | Collapse ms | Pack ms | Instrumented ms |
|---|---:|---:|---:|---:|---:|---:|
| `H-PLOC-CWBVH8-n8-l4-m3` | 0.019 | 0.129 | 0.295 | 0.636 | 0.104 | 1.183 |
| `H-PLOC-CWBVH8-n8-l4-m1` | 0.018 | 0.129 | 0.301 | 0.524 | 0.106 | 1.078 |

## Traversal results

| Implementation | Configuration | Builder | Layout | Median ms | MRay/s | Min–max ms | Time / Nexus | Hits |
|---|---|---|---|---:|---:|---:|---:|---:|
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 0.175 | 3364.1 | 0.167–0.314 | 0.947x | 71,598 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 0.178 | 3314.5 | 0.173–0.581 | 0.961x | 71,598 |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 0.182 | 3238.3 | 0.180–0.186 | 0.984x | 71,598 |
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 0.185 | 3185.2 | 0.181–0.187 | 1.000x | 71,599 |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 0.195 | 3027.3 | 0.193–0.330 | 1.052x | 71,598 |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 0.195 | 3026.1 | 0.192–0.199 | 1.053x | 71,598 |
| bajo | `LBVH-n2-l4` | lbvh | wide | 0.215 | 2748.6 | 0.209–0.351 | 1.159x | 71,598 |
| bajo | `LBVH-n2-l2` | lbvh | wide | 0.219 | 2690.4 | 0.215–0.221 | 1.184x | 71,598 |
| bajo | `LBVH-n4-l2` | lbvh | wide | 0.256 | 2300.2 | 0.254–0.260 | 1.385x | 71,598 |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 0.265 | 2229.6 | 0.262–0.713 | 1.429x | 71,598 |
| bajo | `LBVH-n4-l4` | lbvh | wide | 0.266 | 2217.4 | 0.259–0.668 | 1.436x | 71,598 |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 0.288 | 2046.5 | 0.284–0.296 | 1.556x | 71,598 |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 0.368 | 1604.0 | 0.364–0.512 | 1.986x | 71,598 |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 0.388 | 1520.5 | 0.380–0.391 | 2.095x | 71,598 |
| bajo | `LBVH-n8-l4` | lbvh | wide | 0.403 | 1462.5 | 0.398–0.408 | 2.178x | 71,598 |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 0.517 | 1141.3 | 0.513–0.519 | 2.791x | 71,598 |
| bajo | `LBVH-n8-l8` | lbvh | wide | 0.536 | 1100.9 | 0.532–0.540 | 2.893x | 71,598 |

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

## Methodology

Bajo's LBVH is Apetrei's 2014 agglomerative algorithm, which fuses binary topology construction and bounds propagation in one leaf-driven kernel. Bajo compares it with H-PLOC across ordinary-wide combinations; both builders then use Bajo's existing H-PLOC-derived wide collapse. H-PLOC includes an 8/1/1 row matching NexusBVH's one-triangle leaves. Bajo also measures LBVH and H-PLOC with CWBVH8; storage leaf width is 4 and maximum encoded leaf sizes are 3 and 1 where listed. NexusBVH uses its H-PLOC CWBVH8 builder and currently stores exactly one triangle per leaf. Both implementations trace the same generated camera rays with native packed CWBVH8 or ordinary-wide traversal.

OBJ parsing, camera setup, and initial host-to-device upload are outside the timed regions. H-PLOC CWBVH8 timings are warm rebuilds through a fixed-capacity arena: Morton generation/sort, H-PLOC, direct CWBVH8 conversion, triangle repacking, and synchronization are included; one-time allocation, invariant-offset upload, and cached triangle/root bounds are excluded. Other Bajo rows retain their allocation-owning build API. Traversal timing includes kernel launch and synchronization. Different builder/layout rows do not imply equivalent hierarchy quality.
