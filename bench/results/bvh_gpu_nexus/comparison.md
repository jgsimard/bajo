# NexusBVH vs Bajo GPU BVH benchmark

- Generated: `2026-08-24T17:55:48-04:00`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Mojo: `Mojo 1.1.0.dev2026082405 (3ecdb7b2)`
- Nexus checkout: `/home/jgs/dev/mojo/bajo/external/nexusbvh`
- Nexus revision: `dd6d7e9a017e`

## Summary

- Scene: Dragon OBJ, 249,882 triangles.
- Traversal: 1024x576 camera, 589,824 closest-hit rays.
- Timing: median of 11 synchronized runs; ranges show minimum to maximum.
- Fastest Bajo build: `H-PLOC-CWBVH8-n8-l4-m1` at 1.084 ms (1.777x Nexus build time).
- Fastest Bajo traversal: `H-PLOC-CWBVH8-n8-l4-m1` at 0.175 ms / 3364.3 MRay/s (0.956x Nexus traversal time).

## Build results

| Implementation | Configuration | Builder | Layout | Node width | Leaf width | Max leaf | Median ms | Min–max ms | Time / Nexus |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 8 | 1 | 1 | 0.610 | 0.603–0.619 | 1.000x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 8 | 4 | 1 | 1.084 | 1.044–1.542 | 1.777x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 8 | 4 | 3 | 1.183 | 1.155–1.486 | 1.939x |
| bajo | `LBVH-n2-l2` | lbvh | wide | 2 | 2 | 2 | 2.115 | 1.933–3.533 | 3.467x |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 2 | 2 | 2 | 2.318 | 2.293–3.226 | 3.799x |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 2 | 4 | 4 | 2.375 | 2.337–3.146 | 3.893x |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 8 | 4 | 3 | 2.398 | 2.195–2.962 | 3.930x |
| bajo | `LBVH-n4-l2` | lbvh | wide | 4 | 2 | 2 | 2.455 | 2.090–3.482 | 4.024x |
| bajo | `LBVH-n2-l4` | lbvh | wide | 2 | 4 | 4 | 2.506 | 2.009–2.738 | 4.107x |
| bajo | `LBVH-n4-l4` | lbvh | wide | 4 | 4 | 4 | 2.574 | 2.239–3.393 | 4.219x |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 4 | 2 | 2 | 2.609 | 2.444–3.484 | 4.276x |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 4 | 4 | 4 | 2.650 | 2.576–3.319 | 4.344x |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 8 | 1 | 1 | 2.951 | 2.576–3.579 | 4.837x |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 8 | 4 | 4 | 3.086 | 3.028–3.806 | 5.059x |
| bajo | `LBVH-n8-l4` | lbvh | wide | 8 | 4 | 4 | 3.298 | 2.642–4.209 | 5.406x |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 8 | 8 | 8 | 3.878 | 3.552–5.178 | 6.356x |
| bajo | `LBVH-n8-l8` | lbvh | wide | 8 | 8 | 8 | 3.878 | 3.384–4.596 | 6.357x |

## Traversal results

| Implementation | Configuration | Builder | Layout | Median ms | MRay/s | Min–max ms | Time / Nexus | Hits |
|---|---|---|---|---:|---:|---:|---:|---:|
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 0.175 | 3364.3 | 0.171–0.181 | 0.956x | 71,598 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 0.181 | 3265.1 | 0.177–0.802 | 0.985x | 71,598 |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 0.181 | 3256.6 | 0.180–0.185 | 0.988x | 71,598 |
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 0.183 | 3216.9 | 0.179–0.749 | 1.000x | 71,599 |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 0.193 | 3055.9 | 0.191–0.196 | 1.053x | 71,598 |
| bajo | `LBVH-n2-l2` | lbvh | wide | 0.220 | 2684.2 | 0.216–1.073 | 1.198x | 71,598 |
| bajo | `LBVH-n2-l4` | lbvh | wide | 0.235 | 2509.5 | 0.232–0.243 | 1.282x | 71,598 |
| bajo | `LBVH-n4-l2` | lbvh | wide | 0.244 | 2419.9 | 0.242–0.247 | 1.329x | 71,598 |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 0.256 | 2307.4 | 0.251–0.259 | 1.394x | 71,598 |
| bajo | `LBVH-n4-l4` | lbvh | wide | 0.263 | 2241.5 | 0.258–0.631 | 1.435x | 71,598 |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 0.285 | 2072.6 | 0.277–0.842 | 1.552x | 71,598 |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 0.318 | 1855.2 | 0.212–0.528 | 1.734x | 71,598 |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 0.342 | 1723.4 | 0.336–0.679 | 1.867x | 71,598 |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 0.391 | 1510.2 | 0.383–0.576 | 2.130x | 71,598 |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 0.459 | 1286.0 | 0.456–0.462 | 2.502x | 71,598 |
| bajo | `LBVH-n8-l8` | lbvh | wide | 0.483 | 1220.7 | 0.480–0.487 | 2.635x | 71,598 |
| bajo | `LBVH-n8-l4` | lbvh | wide | 0.589 | 1001.3 | 0.405–0.739 | 3.213x | 71,598 |

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
