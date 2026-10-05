# BVH capability parity with Warp

Reference: NVIDIA Warp 1.17 `Bvh`, `Mesh`, and geometry query built-ins.
Goal: correct CPU/GPU behavior and regression coverage before optimization.
Existing rendering BVHs remain supported; spatial-query APIs will share geometry
predicates and expose explicit ownership, query bounds, grouping, and updates.

| Capability | CPU | GPU | Evidence / remaining work |
|---|---|---|---|
| Build bounds/triangle BVHs | Existing | Existing | Existing BVH suites |
| Closest-hit rays | Existing | Existing | Warp comparison + BVH suites |
| Any-hit rays | Existing | Existing | Warp comparison + BVH suites |
| Enumerate ray-overlapping bounds | Tested | Tested | `test_spatial`: parallel rays, cutoffs, original IDs |
| AABB overlap iterator | Bounds/mesh tested | Bounds/mesh tested | Exhaustive bounds sets, grouped original mesh IDs + CPU/GPU parity across updates |
| Sphere overlap iterator | Bounds/mesh tested | Bounds/mesh tested | Exact box/triangle distance, corner false positives, tangency + CPU/GPU parity |
| Conservative capsule sweep iterator | Tested | Tested | Inclusive expanded-box slabs; zero-length and interval cases |
| Closest mesh point / barycentrics | Tested | Tested | Analytic projection regions, degenerate/skinny faces, extreme scales + CPU/GPU parity |
| Furthest mesh point | Tested | Tested | Vertex reference, inclusive cutoffs, deterministic face ties + CPU/GPU parity |
| Inside/outside: normal, parity, winding | Tested | Tested | Analytic closed tetrahedron grid, reversed orientation, boundary-hit retries + CPU/GPU parity; open surfaces have no universal inside |
| Count ray/triangle intersections | Tested | Tested | Inclusive intervals, shared-edge face counting, groups + CPU/GPU parity |
| Group-restricted queries | Bounds/mesh tested | Bounds/mesh tested | Original IDs, missing groups and initial/refitted/rebuilt trees |
| Indexed mesh + vertex velocities / evaluation | Tested | Tested | Shared vertex IDs, position/velocity/normal evaluation, missing velocities = zero |
| Bounds refit | Tested | Device-input tested | Retained buffers, validation before mutation, groups + post-update queries |
| Mesh deformation refit | Tested | Device-input tested | Fixed topology, retained addresses, failed-input atomicity; host refit after device reorder |
| In-place bounds rebuild | Tested | Device-input/graph tested | Retained buffers/scratch; device leaf reordering + captured update/query replay and invalid-input recovery |
| In-place mesh rebuild | Tested | Device-input/graph tested | Retained geometry/topology buffers, groups, invalid-input preservation and graph replay |
| Cooperative block queries | Serial fallback tested | Tested | Per-lane/per-batch validity, partial batches, multiple blocks, groups, updates, cutoff monotonicity and invalid launch guard |
| Mesh query/evaluation derivatives | Excluded | Excluded | Excluded from this task at the user's request |
| Cooperative mesh AABB queries | Tested | Tested | Bounds batches wrap mesh AABB iterators; original face IDs, partial batches, groups and updates |
| CPU/GPU forward API and result parity | Tested | Tested | 49 result fields across global/group/missing roots and updates; matched Warp/exhaustive forward queries |

“Existing” covers the listed operation, not identical Warp API signatures.
Forward operation coverage is complete for the scoped Warp 1.17 BVH/Mesh
query families; derivatives are excluded. Semantic differences are documented
in the [API contracts](spatial/README.md#differences-from-warp-117). Every completed row must cite
tests covering boundary, empty, invalid-input, grouping and update behavior where
applicable. Algorithm names and Python signatures need not match Warp.

Progress history: [PROGRESS.md](PROGRESS.md).
Performance tracking: [PERFORMANCE.md](PERFORMANCE.md).

New APIs: [spatial](spatial/README.md). New regression suites:
[CPU](../../test/bvh/cpu/test_spatial.mojo),
[GPU](../../test/bvh/gpu/test_spatial.mojo).

Indexed mesh regression suites: [CPU](../../test/bvh/cpu/test_spatial_mesh.mojo),
[GPU](../../test/bvh/gpu/test_spatial_mesh.mojo).

Device-input update/graph regressions: [GPU](../../test/bvh/gpu/test_spatial_device_updates.mojo).

Batch regressions: [CPU](../../test/bvh/cpu/test_spatial_batch.mojo),
[GPU](../../test/bvh/gpu/test_spatial_block.mojo).

Matched general bounds conformance: [report](../../bench/results/warp/spatial.md)
(16 CPU/GPU global/group cases; complete sets match Warp and exhaustive reference).

Matched indexed mesh conformance: [report](../../bench/results/warp/mesh.md)
(28 CPU/GPU cases; reconstructed points/velocities, signs, rays and overlap sets
match Warp and exhaustive/analytic references; grouped Warp CPU winding failure
is recorded and the winding comparison uses ungrouped owners).

Matched cooperative bounds conformance: [report](../../bench/results/warp/block.md)
(eight CPU/GPU AABB/ray cases at 32/64 lanes; complete counts match Warp and
exhaustive reference). Mesh AABB collective regressions are in the
[GPU mesh suite](../../test/bvh/gpu/test_spatial_mesh.mojo); serial mesh batches
are in the [CPU batch suite](../../test/bvh/cpu/test_spatial_batch.mojo).
