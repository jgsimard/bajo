# General spatial BVH queries

Correctness-first bounds queries share a stackless representation on CPU/GPU.
Construction uses the CPU spatial builder; these APIs do not replace the
optimized rendering BVHs.

```mojo
from bajo.bvh.spatial import SpatialBoundsBvh

var tree = SpatialBoundsBvh(boxes, groups)  # groups optional
var query = tree.view().query_sphere(center, radius, tree.group_root(group_id))
var index = UInt32(0)
while query.next(index):
    # index refers to the original input item, not build order
    pass
```

`query_aabb`, `query_ray`, `query_sphere`, and `query_capsule` also accept the
global root (default 0). Missing groups and empty trees return no candidates.
Touching boundaries overlap. Sphere queries use exact box distance; capsule
queries conservatively intersect radius-expanded boxes. Capsule direction is
`end - start`, with initial parameter interval [0, 1]. Zero directions work.
Invalid query geometry returns no candidates. Candidate order is unspecified.
`next(index, max_dist)` can shrink ray/capsule cutoffs; later increases are
clamped. Cutoffs are ignored for box/sphere queries. Exhaustion preserves index.

Views/iterators borrow their owner. Discard them before `refit(boxes)` or
`rebuild(boxes)`; both require unchanged item count and group membership. Failed
validation leaves the tree unchanged. Refit preserves topology; rebuild may
change order but retains published buffers and uses temporary builder scratch.

GPU owner: `bajo.bvh.spatial.gpu.GpuSpatialBoundsBvh(ctx, host_tree^)`. Pass its
`bounds`/`nodes` as immutable device spans to a kernel, construct
`BoundsQueryView[ImmutAnyOrigin, frame]`, then use the same query methods.
The raw view requires valid preorder buffers produced by the owner.
Synchronous `refit(ctx, boxes)` uploads host bounds and refits the current device
topology; `rebuild(ctx, boxes)` runs the host builder and uploads its result. Serialize kernels and updates on the
owner context; raw device spans must not outlive their device buffers. Native
device-input updates and cooperative block iterators are described below.

Spatial query and update coverage lives in the
[CPU tests](../../../test/bvh/cpu/) and [GPU tests](../../../test/bvh/gpu/).

## Indexed mesh queries

`SpatialMesh(points, indices, groups?, velocities?)` snapshots finite indexed
vertices, triples of vertex IDs, optional per-face groups and per-vertex
velocities. Shared vertices retain their original IDs. Missing velocities
interpolate to zero; `set_velocities` updates or resets them in place.

```mojo
from bajo.bvh.spatial import SpatialMesh

var mesh = SpatialMesh(points, indices)
var view = mesh.view()
var result = view.query_point_no_sign(point, max_dist)
if result.result:
    var projected = view.eval_position(result.face, result.u, result.v)
```

Barycentrics `u`, `v`, `1-u-v` weight vertices A, B, C. Queries return original
face IDs. Closest/furthest ties choose the lowest face ID. Point cutoffs and ray
intervals are inclusive. `query_furthest_point_no_sign` returns a mesh vertex;
`query_aabb` returns face-bound candidates, while `query_sphere` tests exact
triangle distance. Degenerate faces remain point/segment distance geometry and
produce no ray hits. `query_ray`, `query_ray_anyhit` and
`query_ray_count_intersections` test triangles; counts include every intersected
face, even faces sharing an edge. This face count is distinct from crossing
parity used by `query_point_sign_parity`.

`query_point_sign_normal`, `query_point_sign_parity` and
`query_point_sign_winding_number` return sign -1 inside, +1 outside, 0 exactly on
the surface. These classifications assume closed, consistently oriented
surfaces; normal signs follow orientation. Parity retries rays that meet
boundary features and uses exact winding if all sampled directions are
ambiguous. Winding sums exact solid angles without an approximation hierarchy.
Normal signs use face normals, edge pseudonormals and angle-weighted vertex
pseudonormals; shared features are identified by vertex ID. `query_point` uses
the normal method. Invalid query geometry yields a miss. Evaluation methods
require valid face IDs and barycentrics provided by the caller.

`refit(points)` and `rebuild(points)` require fixed vertex/index/group counts,
validate before mutation and retain published buffers. The GPU owner is
`bajo.bvh.spatial.gpu_mesh.GpuSpatialMesh(ctx, host_mesh^)`; kernels construct
`MeshQueryView` from its points, indices, velocities, bounds and nodes spans.
Host-input mesh updates synchronize on return; device-input APIs are below.

## Device-input updates and graphs

Both GPU owners expose `refit_device(ctx, input)` and
`rebuild_device(ctx, input)`. Bounds input packs six Float32 values per original
item: min xyz then max xyz. Mesh input packs three Float32 positions per vertex.
Counts, indices and groups stay fixed. Kernels validate the whole input before
mutating geometry or topology. Invalid inputs leave the published state intact;
synchronous wrappers raise after checking device status. Device mesh velocities
can be updated/reset with `set_velocities_device(ctx, input)`.
`enqueue_velocities[block_size=0](ctx, input)` provides the asynchronous velocity
update; check its status with `check_update(ctx)`. Nonempty input contains three
finite Float32 values per vertex, including unreferenced vertices. Empty input
clears every velocity. Small inputs validate and publish in one block. Automatic
thread selection uses 1,024 threads for 1,025–4,095 components, and 256 otherwise;
explicit positive thread counts are preserved. Inputs of at least 4,096
components validate across blocks before a separate publication launch. The last validation block publishes status through an acquire/release
arrival chain and resets retained counters for the next serialized update.

Large resident velocity inputs (at least 65,536 components) use four adjacent
components per thread when input and destination are 16-byte aligned. Tail
components use scalar accesses; other buffers retain scalar processing.
Validation still completes before guarded publication. Explicit thread counts
remain unchanged. Rejected inputs preserve all velocities, and in-place updates
are supported.

`enqueue_update[rebuild, block_size](ctx, input)` performs no allocation, synchronization or
host geometry copy; call `check_update(ctx)` after completion to detect invalid
input. `create_update_graph[rebuild, block_size](ctx, input)` returns a reusable
[DeviceGraph](https://max.modular.com/stable/api/mojo/max/gpu/host/device_graph/DeviceGraph/).
Replay reads current input contents and uses retained scratch/geometry buffers.
Keep the owner and input alive until all operations/replays finish, serialize
queries and updates on their context, and check validation status after replay.
The default block size is 256; explicit powers of two from 32 through 1024 are
available. Bounds updates with at least 1,024 items validate across blocks, then
publish leaves and merge ancestors through retained parent links and
acquire/release child-arrival counters. Separate launches finish validation
before any bounds or leaf IDs can change. Direct updates and captured graphs
share this path, and host-input rebuilds refresh parent links in place. Smaller
bounds inputs and small mesh updates retain the single-block path. Mesh refits
with at least 1,024 faces validate across blocks, merge disjoint subtrees of at
most 64 leaves in shared memory, then merge remaining ancestors through retained
parent links and child-arrival counters. Both direct updates and captured refit
graphs use this path, including owners with winding support enabled. Winding
cache refits with at least 1,024 closed components also publish leaves and merge
ancestors in parallel. All scratch and parent/chunk metadata are retained by the
owner. Rejected input leaves published geometry and cache bounds unchanged.
Raw one-block update kernels still require their matching block size.

Device rebuild stably sorts original face/item IDs by X centroid within each
independent root, then refits the retained tree skeleton. It changes leaf
placement, preserves groups and retains addresses; host rebuild runs the CPU
spatial builder. Mesh rebuilds with at least 256 faces cache the original Float64
centroid sums as exact UInt64 ordering keys, sort 256-leaf tiles in shared memory,
and merge centroid/original-ID order across GPU blocks. Sorted IDs are published
before clearing arrival counters and running parallel geometry/bounds publication.
Direct updates and captured graphs use the same dependencies, including winding
cache updates. The owner retains sort scratch and tile descriptors; device
rebuild operations allocate no new storage. Bounds and smaller mesh rebuilds
retain their single-block sorting path.
The GPU bounds update benchmark `pixi run bench_bvh_gpu_spatial_bounds_updates`
compares direct submissions and graph replay with full CPU-reference hashes.
Device updates leave `host` as the last host-input snapshot, not a mirror of
current device geometry. Subsequent host-input refit remains valid after a
device reorder because it updates bounds using the current device topology.

## Query batches

`BoundsQuery.next_batch(output, cutoff)` is the CPU/GPU serial fallback. It
fills up to the span's capacity, returns the filled count and preserves unused
slots. Zero capacity consumes nothing. It accepts every bounds query kind and
retains the iterator's cutoff and grouping behavior.
CPU `next_batch[width]` tests a SIMD packet of preorder bounds before following
subtree escapes; the default width is 2, capped to the native Float64 SIMD width.
GPU serial batches and width 1 use scalar traversal. Cutoffs are rechecked on
each call, so partially consumed packets retain no stale decisions.

GPU kernels can use
`bajo.bvh.spatial.block_query.BlockBoundsQuery[origin, frame, block_size](query^)`.
Every lane calls `next(index, cutoff)` with identical query inputs/cutoffs.
Its return value reports that lane's result; `batch_valid` reports uniformly
whether any lane received a result. Process valid lanes, then call again in
**every** lane until `batch_valid` becomes false. A partial final batch remains
valid at batch level; unused/exhausted lanes preserve index. Candidate order is
unspecified. Launch exactly `(block_size, 1, 1)` threads per block; mismatched
shapes return empty without writing shared scratch. Blocks have independent
scratch, and there is no fixed global result capacity.

The collective tests consecutive preorder nodes in parallel and compacts
emitted original IDs in shared memory. Missed internal nodes can skip their
complete subtree before the next batch: the block takes the largest safe
escape among nodes already tested in that batch. This preserves partial batches
and per-lane validity while avoiding scans of distant subtrees. Power-of-two
blocks use collective reductions; other positive block sizes use shared scratch.

Mesh AABB queries return the same bounds iterator: call `next_batch` on
`mesh.view().query_aabb(box, root)` or wrap the iterator in `BlockBoundsQuery`
inside a GPU kernel to cooperatively enumerate original face IDs. Sphere mesh
queries also perform a triangle narrow phase and are not interchangeable with
bounds-only batches. `mesh.view().bvh` exposes the bounds view for conservative
ray/sphere/capsule candidate enumeration against face bounds.

## Differences from Warp 1.17

These APIs cover forward spatial operations, with Bajo contracts rather than
identical Python signatures. VJPs and autodiff are excluded.

| Case | Bajo | Warp |
|---|---|---|
| Closest/furthest distance cutoff | Inclusive | Strict |
| Degenerate point queries | Point/segment geometry | Skips degenerate or nearly degenerate faces |
| Negative sphere/capsule radius | Empty query | Clamps radius to zero |
| Missing group | Empty root | Root lookup returns -1, also the default global-root sentinel |
| Closest/furthest face ties | Lowest original face ID | Traversal-dependent |
| Default signed point query | Normal pseudonormal | Three axis-ray orientation votes |
| Winding sign | Absolute exact solid-angle sum | Oriented accelerated approximation |
| Normal feature epsilon | Default 1e-6 barycentric weight tolerance | Default 1e-3 with average-edge scaling |

Normal signs depend on orientation in both libraries. Surface classifications
and open/nonmanifold geometry do not have universal equivalent signs. The matched
benchmarks use valid inputs away from cutoff boundaries and document tie handling.
Warp's grouped CPU winding configuration failed the analytic fixture; the matched
winding comparison uses separate ungrouped owners in both libraries.

CPU nearest/furthest traversal specializes on SIMD width and orders child visits
by distance. Masked strided loads read each child pair; unused lanes never load
EMPTY_LANE. CPU/GPU `query_ray[width]` similarly visits nearer child bounds first and
prunes after each closer hit; the default is width 2, capped to native Float64
SIMD width. Ray counts use bounds packets of width 4, while any-hit and parity
use width 2. These defaults live in `policy.mojo`; explicit widths remain
available. Parity abandons ambiguous boundary-hit trials and retries another
direction. Hit counts, any-hit and parity share one implementation; their bounds
iterator uses SIMD lookahead on CPU and scalar traversal on GPU.
Exact winding shares packet accumulation, with retained CPU triangle packets
(default width 8), width 1 on GPU and a gather fallback for grouped roots.
`SpatialMesh[frame, storage_width]` controls the cache width. Geometry predicates
share scalar/SIMD implementations, with width 1 as the scalar specialization.
Reusable triangle projection, normals, ray tests and solid angles live in
`bajo.core.intersect`. Robust spatial ray tests specialize the existing core edge
predicate for Float64, exact-zero determinants and inclusive intervals.
`AABB.distance2[furthest]` supplies point traversal and sphere broad phase.

Bounds `refit[width]` validates before mutation and updates retained buffers
without allocation; mesh refit fuses face bounds with triangle packing. The
shared default in `policy.mojo` caps width 4 to the target native Float32 width.
Explicit widths remain available for workload-specific measurements.

Spatial construction selects median partitions below 16,384 input items and
balanced, stable Morton order for larger inputs, directly into preallocated
preorder buffers. All global/group roots borrow one temporary array of build
items; groups select subsets of original IDs. Median split axes specialize
at compile time, and bounds reductions are generic over SIMD width. Morton
keys use a shared spatial scale, finite Float64 midpoint arithmetic and stable
original-ID ties. Balanced emission preserves the existing refit schedule.


CPU `SpatialMesh` accepts optional `groups` and `velocities` spans. Pass cache
controls by name: `SpatialMesh(points, indices, cache_winding=False)` skips
closed-component metadata and triangle packing. The default remains `True`.
Exact winding queries still work through indexed Float64 accumulation; refit and
rebuild maintain geometry and bounds with either setting. A `GpuSpatialMesh`
requesting `enable_winding=True` builds missing host winding metadata before
publishing its device cache. Complete construction benchmarks match host and
device winding-cache settings to Warp's disabled/enabled modes.

When the global tree uses Morton order, every group preserves that same stable
order instead of recomputing and sorting group-local codes. Smaller inputs use
median partitions for both global and group trees. Both paths retain balanced
splits, exact node unions and original face IDs.

Forests with more than eight groups discover arbitrary UInt32 labels through a
temporary directory, then count and scatter items into disjoint slices of one
flat buffer. Group preparation takes expected O(items + groups) work and
O(items + groups) scratch instead of scanning every item for every group.
Labels and roots remain in first-appearance order. Up to eight groups retain
the simpler scan path without a label directory. The construction benchmark
`pixi run bench_bvh_cpu_spatial_groups` covers both paths and reports full forest
hashes for comparisons across revisions.
