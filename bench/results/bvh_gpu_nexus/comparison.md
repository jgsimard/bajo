# NexusBVH vs Bajo GPU BVH benchmark

- Generated: `2026-08-23T16:11:33-04:00`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Mojo: `Mojo 1.1.0.dev2026082305 (54b9e0e2)`
- Nexus checkout: `/tmp/nexusbvh-reference`
- Nexus revision: `dd6d7e9a017e`

## Summary

- Scene: Dragon OBJ, 249,882 triangles.
- Traversal: 1024x576 camera, 589,824 closest-hit rays.
- Timing: median of 11 synchronized runs; ranges show minimum to maximum.
- Fastest Bajo build: `H-PLOC-CWBVH8-n8-l4-m1` at 1.681 ms (2.642x Nexus build time).
- Fastest Bajo traversal: `H-PLOC-CWBVH8-n8-l4-m3` at 0.197 ms / 2999.5 MRay/s (1.081x Nexus traversal time).

## Build results

| Implementation | Configuration | Builder | Layout | Node width | Leaf width | Max leaf | Median ms | Min–max ms | Time / Nexus |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 8 | 1 | 1 | 0.636 | 0.623–1.077 | 1.000x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 8 | 4 | 1 | 1.681 | 1.520–1.932 | 2.642x |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 8 | 4 | 3 | 1.708 | 1.532–1.908 | 2.685x |
| bajo | `LBVH-n2-l2` | lbvh | wide | 2 | 2 | 2 | 2.104 | 1.997–2.614 | 3.306x |
| bajo | `LBVH-n2-l4` | lbvh | wide | 2 | 4 | 4 | 2.297 | 2.130–2.620 | 3.611x |
| bajo | `LBVH-n4-l2` | lbvh | wide | 4 | 2 | 2 | 2.414 | 2.201–2.709 | 3.795x |
| bajo | `LBVH-n4-l4` | lbvh | wide | 4 | 4 | 4 | 2.503 | 2.380–3.110 | 3.935x |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 2 | 2 | 2 | 2.513 | 2.457–2.999 | 3.950x |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 2 | 4 | 4 | 2.537 | 2.508–3.080 | 3.988x |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 8 | 4 | 3 | 2.548 | 2.446–3.013 | 4.005x |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 8 | 1 | 1 | 2.957 | 2.818–3.714 | 4.648x |
| bajo | `LBVH-n8-l4` | lbvh | wide | 8 | 4 | 4 | 3.040 | 2.771–3.349 | 4.778x |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 4 | 2 | 2 | 3.081 | 2.628–3.199 | 4.842x |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 4 | 4 | 4 | 3.094 | 2.785–3.495 | 4.862x |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 8 | 4 | 4 | 3.486 | 3.174–3.775 | 5.479x |
| bajo | `LBVH-n8-l8` | lbvh | wide | 8 | 8 | 8 | 3.763 | 3.416–4.061 | 5.915x |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 8 | 8 | 8 | 3.964 | 3.761–4.513 | 6.230x |

## Traversal results

| Implementation | Configuration | Builder | Layout | Median ms | MRay/s | Min–max ms | Time / Nexus | Hits |
|---|---|---|---|---:|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | hploc | cwbvh8 | 0.182 | 3242.2 | 0.177–0.195 | 1.000x | 71,599 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | hploc | cwbvh8 | 0.197 | 2999.5 | 0.193–0.355 | 1.081x | 71,598 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | hploc | cwbvh8 | 0.198 | 2978.4 | 0.191–0.204 | 1.089x | 71,598 |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | lbvh | cwbvh8 | 0.199 | 2960.6 | 0.189–0.595 | 1.095x | 71,598 |
| bajo | `H-PLOC-n2-l2` | hploc | wide | 0.222 | 2661.5 | 0.217–0.554 | 1.218x | 71,598 |
| bajo | `LBVH-n2-l2` | lbvh | wide | 0.240 | 2454.2 | 0.238–0.245 | 1.321x | 71,598 |
| bajo | `H-PLOC-n2-l4` | hploc | wide | 0.241 | 2451.6 | 0.238–0.655 | 1.323x | 71,598 |
| bajo | `LBVH-n2-l4` | lbvh | wide | 0.261 | 2256.7 | 0.256–0.366 | 1.437x | 71,598 |
| bajo | `LBVH-n4-l2` | lbvh | wide | 0.279 | 2111.5 | 0.277–0.598 | 1.536x | 71,598 |
| bajo | `LBVH-n4-l4` | lbvh | wide | 0.287 | 2051.7 | 0.277–0.683 | 1.580x | 71,598 |
| bajo | `H-PLOC-n4-l2` | hploc | wide | 0.291 | 2029.1 | 0.284–0.303 | 1.598x | 71,598 |
| bajo | `H-PLOC-n4-l4` | hploc | wide | 0.303 | 1946.7 | 0.300–0.708 | 1.665x | 71,598 |
| bajo | `H-PLOC-n8-l1` | hploc | wide | 0.373 | 1582.2 | 0.368–0.713 | 2.049x | 71,598 |
| bajo | `H-PLOC-n8-l4` | hploc | wide | 0.424 | 1389.9 | 0.419–0.833 | 2.333x | 71,598 |
| bajo | `LBVH-n8-l4` | lbvh | wide | 0.444 | 1327.8 | 0.434–0.794 | 2.442x | 71,598 |
| bajo | `H-PLOC-n8-l8` | hploc | wide | 0.545 | 1081.6 | 0.539–0.862 | 2.998x | 71,598 |
| bajo | `LBVH-n8-l8` | lbvh | wide | 0.564 | 1046.6 | 0.557–0.915 | 3.098x | 71,598 |

## Traversal work

The instrumented kernels run after timing. Counts cover every camera ray and therefore do not perturb the headline traversal measurements.

| Implementation | Configuration | Nodes/ray | Leaf groups/ray | Triangles/ray | Maximum stack |
|---|---|---:|---:|---:|---:|
| nexusbvh | `NexusBVH-H-PLOC-CWBVH8` | 3.631 | 0.304 | 0.545 | 7 |
| bajo | `LBVH-CWBVH8-n8-l4-m3` | 3.108 | 0.282 | 0.708 | 6 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m3` | 2.947 | 0.273 | 0.721 | 6 |
| bajo | `H-PLOC-CWBVH8-n8-l4-m1` | 3.112 | 0.313 | 0.528 | 6 |

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

OBJ parsing, camera setup, and initial host-to-device upload are outside the timed regions. H-PLOC CWBVH8 timings are warm rebuilds through a fixed-capacity arena: Morton generation/sort, H-PLOC, direct CWBVH8 conversion, triangle repacking, and synchronization are included; one-time allocation, invariant-offset upload, and cached triangle/root bounds are excluded. Other Bajo rows retain their allocation-owning build API. Traversal timing includes kernel launch and synchronization. Different builder/layout rows do not imply equivalent hierarchy quality.
