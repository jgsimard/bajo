# General spatial BVH queries

Correctness-first bounds queries share a stackless representation on CPU/GPU.
Construction reuses the CPU median builder; these APIs do not replace the
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

Coverage and remaining capabilities: [matrix](../CAPABILITIES.md).

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

`enqueue_update[rebuild](ctx, input)` performs no allocation, synchronization or
host geometry copy; call `check_update(ctx)` after completion to detect invalid
input. `create_update_graph[rebuild](ctx, input)` returns a reusable
[DeviceGraph](https://max.modular.com/stable/api/mojo/max/gpu/host/device_graph/DeviceGraph/).
Replay reads current input contents and uses retained scratch/geometry buffers.
Keep the owner and input alive until all operations/replays finish, serialize
queries and updates on their context, and check validation status after replay.

Device rebuild stably sorts original face/item IDs by X centroid within each
independent root, then refits the retained tree skeleton. This is a deliberately
simple allocation-free baseline. It changes leaf placement, preserves groups
and retains addresses; host rebuild instead runs the CPU median builder.
Device updates leave `host` as the last host-input snapshot, not a mirror of
current device geometry. Subsequent host-input refit remains valid after a
device reorder because it updates bounds using the current device topology.

## Query batches

`BoundsQueryBatch(query)` is the CPU/GPU serial fallback. `next(output, cutoff)`
fills up to the span's capacity, returns the filled count and preserves unused
slots. Zero capacity consumes nothing. It accepts every bounds query kind and
retains the underlying iterator's cutoff and grouping behavior.

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

This collective baseline tests preorder leaf candidates in parallel and compacts
a block's results in shared memory. It scans the root's nodes rather than
pruning subtrees cooperatively; optimization awaits matched measurements.

Mesh AABB queries return the same bounds iterator: wrap
`mesh.view().query_aabb(box, root)` in `BoundsQueryBatch` or, inside a GPU kernel,
`BlockBoundsQuery` to cooperatively enumerate original face IDs. Sphere mesh
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
