# Indexed mesh: Bajo vs Warp

2026-10-05 · Warp 1.17.0 · 64 disjoint outward tetrahedra, 256 faces, four groups, 256 queries, seven warmed samples.

Ratios are Warp/Bajo; above 1 favors Bajo. CPU median/SAH and GPU median-upload/LBVH, one face per leaf.

| Operation | CPU ratio | GPU ratio |
|---|---:|---:|
| Refit | 0.22 | 0.08 |
| closest / global | 0.01 | 0.01 |
| furthest / global | 0.02 | 0.04 |
| normal sign / global | 0.02 | 0.01 |
| parity sign / global | 0.02 | 0.01 |
| closest ray / global | 0.34 | 0.11 |
| ray count / global | 0.47 | 0.20 |
| any-hit ray / global | 0.46 | 0.14 |
| sphere / global | 1.93 | 0.71 |
| AABB / global | 3.20 | 1.25 |
| winding sign / global | 0.02 | 0.03 |
| closest ray / group | 0.44 | 0.20 |
| ray count / group | 0.56 | 0.30 |
| any-hit ray / group | 0.60 | 0.30 |
| AABB / group | 3.74 | 1.51 |

All 28 CPU/GPU query cases match Warp and an independent exhaustive Float64 geometry / analytic tetrahedron reference. Discrete answers and overlap multiplicities match exactly; floating fields use 5e-6 absolute/relative tolerance. Shared-edge/vertex nearest and furthest ties compare reconstructed positions, distances and velocities, allowing different face IDs.

Query timings include outputs, launch and synchronization; download and fixture upload are excluded. Refit uses unchanged warmed positions; GPU inputs are resident, Bajo validation is included and status download is excluded. The winding case uses separate ungrouped owners with Warp winding support enabled; other cases/refit use grouped owners without Warp winding support. Bajo sums exact solid angles while Warp uses its accelerated winding approximation. The raw record includes a separate correctness check of grouped Warp CPU winding; any failed configuration is excluded from timings. This small workload does not establish scaling or large-deformation performance. Cutoffs and query points avoid semantic boundaries. [Raw samples and hashes](mesh.json).

Reproduce: `pixi run python bench/warp/compare_mesh.py --warp-path /tmp/bajo_warp_deps`.
