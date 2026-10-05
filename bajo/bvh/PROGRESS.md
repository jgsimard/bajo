# BVH capability progress

## 2026-10-05

- Inspected current public CPU/GPU BVH APIs and Warp 1.17 query families.
- Established the capability matrix; existing ray benchmarks do not establish
  general spatial-query parity.
- Confirmed CPU TLAS refit and retained GPU CWBVH rebuild storage already exist.
- Next: shared conservative bounds predicates and resumable overlap iterators,
  followed by indexed mesh queries and explicit CPU/GPU ownership/update APIs.
- Validation: no new spatial capability is marked complete yet.

### Shared bounds queries and ownership

- Added `SpatialBoundsBvh`, borrowed `BoundsQueryView` and resumable
  `BoundsQuery` under `bajo.bvh.spatial`; reused the existing median builder.
- Implemented inclusive AABB/sphere/ray candidates and conservative capsule
  sweeps, original item IDs, global/group roots, empty trees and invalid queries.
- Added fixed-count refit and rebuild with stable published buffer addresses;
  invalid updates are validated before mutation. Rebuild uses temporary scratch.
- Added `GpuSpatialBoundsBvh`: device queries share CPU predicates; synchronous
  host-driven construction and updates retain device allocations. Native
  GPU-resident updates and graph capture guarantees are not implemented yet.
- Verification: seven CPU tests pass; two GPU tests pass on RTX 5060 Ti. GPU
  compares complete candidate sets across four query types, three root choices
  and initial/refitted/rebuilt trees. Empty/invalid GPU queries and failed-update
  atomicity pass. CPU overlap sets also match exhaustive item checks.
- Commands: `pixi run mojo run -I . test/bvh/cpu/test_spatial.mojo`; GPU suite
  compiled with `--target-accelerator sm_120` and executed with GPU access.
- Next: indexed mesh ownership/evaluation, exact sphere/point queries, all-hit
  counting and sign classification on both backends; then native update and
  cooperative query contracts. The full parity goal remains open.

### Performance tracking

- Added a separate CPU/GPU performance matrix at `PERFORMANCE.md` as requested.
- Linked existing matched Warp 1.17 ray/build measurements with their workload
  scope. New general spatial APIs are explicitly unmeasured; existing optimized
  renderer results are not reused as spatial-query performance claims.
- New matched measurements will follow correctness validation and retain raw
  samples, input/source hashes, timing boundaries and query distributions.

### Indexed mesh queries and updates

- Added `SpatialMesh`, borrowed `MeshQueryView` and `GpuSpatialMesh` snapshots
  with shared vertex indices, original face IDs, group roots and velocities.
- Added exact sphere/triangle enumeration, closest/furthest points, barycentric
  position/velocity/normal evaluation, nearest/any rays and all-face hit counts.
- Added normal, parity and exact winding sign methods. Parity retries shared
  boundary hits; normal signs use indexed edge/vertex pseudonormals. Closed,
  consistently oriented surfaces are the documented classification contract.
- Added fixed-count mesh refit/rebuild and velocity updates with retained CPU
  and GPU buffers; host inputs are validated before mutation. GPU updates are
  still synchronous and host-driven.
- Replaced a host-only double `atan2` call found by actual GPU execution with a
  portable scalar function; checked 14,007 finite argument pairs against libm.
  Reworked projection to avoid Gram cancellation on skinny triangles.
- Verification: ten CPU mesh tests and three GPU mesh tests pass. CPU uses
  analytic triangle projection regions, a 1,000-point tetrahedron sign grid,
  reversed winding, a boundary-hit octahedron, degenerate/skinny geometry and
  1e-20/1e20 scales. GPU compares 17 outputs for 105 queries across global,
  group and missing roots in initial/refitted/rebuilt states, and independently
  checks extreme scales, invalid queries, evaluation and empty/failed updates.
- Commands: `pixi run mojo run -I . test/bvh/cpu/test_spatial_mesh.mojo`;
  `pixi run mojo build -I . --target-accelerator sm_120
  test/bvh/gpu/test_spatial_mesh.mojo -o /tmp/bajo_test_spatial_mesh_gpu_final`,
  then execute that binary with GPU access.
- New mesh performance remains unmeasured. Next: device-input update/graph
  contracts, cooperative queries and matched
  Warp measurements. Full parity remains open.

### Device-input updates, graph replay and collective batches

- Added allocation-free device-input bounds/mesh refit and rebuild, retained
  scratch/status buffers and mesh velocity updates. Kernels validate all input
  before mutation; wrong counts and invalid geometry leave published state intact.
- Device rebuild reorders original IDs by X centroid within each root, then
  refits the retained skeleton. This baseline prioritizes correctness; input and
  owner lifetimes and context serialization are explicit API requirements.
- Added reusable update graphs, including captured rebuild plus query replay,
  changed input contents, invalid-input preservation and successful repair/replay.
- Fixed host-input refit after device reorder to use current device topology.
  Device updates leave the host snapshot unchanged, as documented.
- Added `BoundsQueryBatch` serial fallback and `BlockBoundsQuery` collective
  candidate batches with separate lane/batch validity, partial/exhausted output
  preservation, monotonic cutoffs and a guard for incompatible launch shapes.
- Verification: six device-update/graph GPU tests and four collective GPU tests
  pass on RTX 5060 Ti; existing two bounds and three mesh GPU tests still pass.
  All 19 CPU spatial tests pass (7 bounds, 10 mesh, 2 batches); all 15 GPU
  spatial tests pass. Complete candidate sets match scalar CPU queries
  across four query kinds, two block sizes, independent blocks, groups and
  initial/refitted/rebuilt trees.
- New suites: `test/bvh/gpu/test_spatial_device_updates.mojo`,
  `test/bvh/gpu/test_spatial_block.mojo`, `test/bvh/cpu/test_spatial_batch.mojo`.
  GPU binaries were built with `--target-accelerator sm_120` and executed with
  GPU access; formatting and whitespace checks pass.
- Next: broaden independent Warp conformance and public result-field checks,
  then measure the shared spatial APIs in the performance matrix. Full parity
  remains open; these new baselines are still explicitly unmeasured.

### Scope correction and result-field checks

- Query/evaluation derivative support is excluded at the user's request. Removed
  the temporary derivative API, implementation and tests.
- Added a full result-field GPU check: 40 closest/furthest/ray, evaluation, winding
  and sign fields match CPU across 105 queries, global/group/missing roots and
  initial/refitted/rebuilt geometry. Integer and Boolean fields compare exactly.
- Next: matched forward-query conformance and performance against Warp.

### Matched bounds conformance and performance

- Added a reproducible Bajo/Warp bounds harness with identical Float32 fixtures:
  256 boxes, four groups and 512 queries. All 16 CPU/GPU global/group cases for
  AABB, ray, sphere and conservative capsule return identical complete candidate
  sets and match an independent exhaustive NumPy reference; per-item counts
  also detect duplicate enumeration.
- Recorded seven warmed samples for construction, refit, rebuild and queries;
  retained 44 timing series, hardware/version metadata, input and source hashes.
  Verified current source hashes against the recorded run; matrix refresh is automatic.
- Added `bench_warp_spatial_compare`; concise results live in
  `bench/results/warp/spatial.md`, and the performance matrix now tracks measured
  bounds rows separately from pending mesh/cooperative comparisons.
- Timing scopes include full candidate-matrix output writes, host-to-device
  construction uploads, and resident-input updates on unchanged warmed geometry.
  GPU update timings include Bajo validation but exclude status downloads.
- Results are mixed: AABB queries favor Bajo on this workload; several native
  GPU queries/updates favor Warp. These baseline measurements do not establish
  large-tree or deformation performance and do not replace rendering results.
- Next: matched forward mesh and cooperative-query conformance/performance,
  followed by a requirement-by-requirement completion audit. VJPs stay excluded.

### Forward mesh/cooperative comparisons and completion audit

- Added matched mesh queries: closest/furthest point, three sign methods,
  closest/any rays, intersection counts, exact sphere and AABB enumeration,
  plus grouped rays/counts/AABB. All 28 CPU/GPU cases match Warp and independent
  exhaustive Float64 geometry / analytic closed-tetrahedron references.
- The comparison exposed corrupted barycentric fields in Bajo's normal-sign
  result. Replaced temporary SIMD weight packing with scalar weights; added
  reconstruction/velocity regressions and expanded GPU result checks to 49 fields.
- Recorded a Warp 1.17 grouped CPU winding failure: all 64 interior probes were
  classified outside. The harness reproduces and retains the failure separately;
  accepted winding timings use ungrouped owners in both libraries.
- Added cooperative bounds comparisons at 32/64 lanes: all eight CPU/GPU count
  matrices match Warp and exhaustive reference, including empty queries and
  partial batches. Added serial/GPU mesh AABB batch coverage across groups and
  GPU refit/rebuild, using the existing shared bounds iterator.
- Added `bench_warp_mesh_compare` and `bench_warp_block_compare`; concise reports
  are `bench/results/warp/mesh.md` and `block.md`. All three forward comparison
  harnesses refresh their measured matrix rows after successful validation.
- Final validation: 21 CPU and 17 GPU spatial regression tests pass on Ryzen
  9700X / RTX 5060 Ti. All 52 matched forward cases pass; 120 timing series retain
  seven positive samples each. Verified input/version/hardware metadata and
  current source hashes; Python syntax and whitespace checks pass.
- Audit: the scoped Warp 1.17 forward BVH/Mesh operation families are covered,
  including retained updates/graphs, grouped queries and collective mesh AABB
  enumeration. Documented cutoff, degeneracy, tie, group-sentinel, default sign
  and winding semantics. VJPs/autodiff remain excluded as requested.
- The correctness/coverage goal is complete. Performance is a baseline, with
  substantial point/sign-query gaps; standalone construction/evaluation,
  velocity-update, mesh-collective and graph timings remain visibly unmeasured.
  The matrix/log support subsequent scaling and optimization work.
