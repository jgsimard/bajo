# GPU wavefront execution contract

The GPU path uses bounded field-major queues while the optimized CPU renderer
keeps its faster direct AoS queues. `PackedWavePathQueue` and
`PackedWaveShadeQueue` are the explicit boundary between those layouts.

## Buffers

For a queue capacity `N`:

| Buffer | Layout | Bytes per entry |
| --- | --- | ---: |
| Path IDs | `UInt32[N]` | 4 |
| Path fields | 12 `Float32[N]` planes | 48 |
| Shade path references | packed `UInt32[N]` | 4 |
| Shade surfaces | packed `UInt32[N]` | 4 |
| Shade fields | 4 `Float32[N]` planes | 16 |
| Shadow path IDs | `UInt32[N]` | 4 |
| Shadow fields | 11 `Float32[N]` planes | 44 |
| Sample radiance | 3 `Float32[N]` planes | 12 |

Path fields are origin, ray interval, direction, throughput, and previous BSDF
PDF. Delta status occupies the high bit of the path ID. Shade fields are normal
and distance. Front-face state occupies the high bit of the path reference; the
surface kind occupies the high nibble of the surface word.

The reusable arena owns two path queues for ping-pong execution, one tagged
shade queue, one shadow queue, a sample-radiance buffer, and five atomic
counters. The trace view exposes the tagged queue's three pointers; material
kind is carried by the surface word rather than duplicated queue state. Render
targets process complete-pixel path chunks and default to a 256K-path working
set; callers can override capacity without changing global path IDs, RNG
streams, or output order.

## Stage sequence

1. PATH primary generation writes the active path queue with a lean standalone
   kernel. NORMALS, AO, NEE, and MIS fuse primary generation into bounce-zero
   scene tracing, avoiding one launch and the initial path-queue round trip.
   The fused specialization stores source paths only when deferred metal or
   dielectric shading requires them. Both paths clear per-sample radiance.
2. Compile-time-specialized scene tracing processes active paths. Geometry
   absent from `GpuRtSceneKind` is erased at compile time. Hits
   fuse Lambertian BSDF sampling into hit routing, append non-Lambertian tagged
   material records, and compact AO/direct-light visibility work.
3. One any-hit shadow kernel consumes deferred visibility work.
4. When metal or dielectric work exists, one material-dispatch kernel consumes
   the tagged shade queue and invokes compile-time-specialized BSDF code before
   compacting surviving paths. Lambertian/emissive scenes skip this launch.
5. A tiny device kernel promotes `next` to `active` and clears output counters;
   the host only swaps its path-slot selector and advances the bounce. No
   counter readback or host synchronization is required between bounces.
6. A final kernel reduces one complete-pixel chunk into its disjoint range of
   device-resident pixels before the next chunk reuses the arena.

Trace uses grid-stride launches with 64-thread blocks; the independently tuned
shade and shadow stages use 128-thread blocks. The grid cap remains a
compile-time benchmark override, while the production default exposes enough
blocks to cover the working set. Direct-light selection uses a Walker-Vose
alias table built once with the scene and sampled in O(1).
Prepared scenes also classify their lights by surface-sampling primitive. NEE
and MIS select a host-side compile-time kernel specialization for homogeneous
sphere or triangle-based lights, erasing the per-hit primitive branch and its
kind-table read; mixed light sets retain generic device dispatch.
Alias probabilities occupy a dense prefix in the existing light-field buffer,
so random alias-column reads are contiguous while selected-light geometry and
emission remain in the existing array-of-structures records.

Queue ordering is intentionally unspecified. Every path retains its global
path ID. BSDF sampling uses the existing `bounce + 1` stage, while Russian
roulette uses the independent `0x40000000 | (bounce + 1)` domain. Both feed
`wavefront_rng_subsequence(path_id, stage)`. The generator remains
`std.random.Random` (Philox), so scheduling and atomic append order cannot
change either random stream.

`test_wavefront_contract_gpu.mojo` is an explicit diagnostic: paths are loaded
from field-major planes, atomically compacted into a ping-pong queue, tagged as
all three material kinds in one shade queue, and checked after download
together with their Philox subsequences.

## Renderer status

`render_gpu[integrator, node_width, leaf_width]` is the general entry
point. It dispatches by scene contents to compile-time-specialized sphere,
triangle, mixed-static, instance-only, or combined static-plus-instance
pipelines. It accepts backend-neutral `SceneData`, so GPU-only callers do not
construct or retain CPU BVHs. All geometry combinations are supported,
including signed sphere radii and transformed triangle BLAS/TLAS instances.

`SceneBuilder` is explicitly destroyed: callers must consume it with `finish()`.
Finalization validates geometry/material sidecars and constructs the light
distribution, producing read-only `SceneData`. `CpuScene` consumes that snapshot
and exposes only a read-only `scene_data()` view. The concrete device-resident
scene specializations upload independent snapshots containing GPU acceleration,
material, and light buffers. Build and finalize a new scene to observe authored
changes; there is deliberately no implicit dirty tracking, update, or refit API
yet.

A caller that needs both backends can prepare the GPU scene from `SceneData`
first, then move the same data into `CpuScene`; a GPU-only caller never needs a
`CpuScene` or imports CPU acceleration code through the neutral type layer.

`GpuRtScene[kind, sphere_format, triangle_format, tlas_format, blas_format]`
is the sole prepared-scene owner. `prepare_gpu_scene` builds exactly the device
resources selected by `kind`; absent geometry does not allocate dummy buffers
or require a separate owner type. `GpuRtBvhFormat` carries only the layout and
node/leaf widths needed for traversal. Builder methods are parameters of scene
preparation, so identically laid-out BVHs share one prepared scene and kernel
type regardless of how they were built. `enqueue_render_gpu` submits every
prepared scene through one API, while `render_gpu_configured` is the synchronous
explicit-format entry point.

Triangle-only rendering defaults to H-PLOC construction and native CWBVH8
traversal (`node8/leaf4`). Instance rendering defaults to TLAS2/leaf1 and
selects its BLAS specialization once on the host: H-PLOC+CWBVH8 when the
instance-weighted mean mesh size is at least 32 triangles, otherwise
LBVH+wide4 for micro-BLASes. This policy follows the measured crossover and
adds no runtime format branch to device traversal. Explicit callers retain the
full compile-time builder, node/leaf width, and compressed/wide overrides by
passing formats and build methods.

Mixed and combined scenes specialize sphere BVH, static-triangle, TLAS, and
BLAS widths independently. Their static triangles reuse the same wide/CWBVH
representation and 32-triangle host policy as triangle-only rendering. The
default TLAS is width2/leaf1; sphere widths continue to use the general
`node_width`/`leaf_width` parameters. TLAS construction remains LBVH by
default because it wins PATH, NEE, and MIS on the long 256-instance workload;
H-PLOC is available as a compile-time TLAS builder policy and is useful for
AO-heavy workloads. The default TLAS2 closest-hit kernel also selects a
dedicated array-free BVH2 traversal loop at compile time.

The GPU implements `PATH`, `NORMALS`, `AO`, `NEE`, and `MIS`, with Lambertian,
metal, dielectric, and emissive surfaces. CPU and GPU both import device-safe
BSDF, lighting, Philox-stage, sky, and Russian-roulette math from the shared RT
namespace; GPU code does not depend on the CPU renderer namespace.

`GpuRtRenderTarget` retains path/shade/shadow queues, camera data, and
device-resident pixels. `enqueue_render_gpu` is asynchronous for hot
submission. `render_gpu_configured` allocates, synchronizes, and downloads for an
explicit scene kind, format, and builder set; `render_gpu` additionally selects
the kind and size-sensitive defaults from backend-neutral `SceneData`.

## CPU/GPU performance comparison

Run `pixi run bench_rt_cpu_gpu` for an identical 1024x1024x8, depth-8 comparison
of PATH, AO, NEE, MIS, and the 64-light NEE guard. It reports CPU render and
host-output total time, GPU device-resident time, GPU time including host pixel
download, throughput, speedup, and cross-backend checksum deltas. Results are
recorded in `bench/results/rt_cpu_gpu/comparison.md`.

Run `pixi run bench_rt_gpu_materials` for shade-heavy metal and dielectric
sphere guardrails under PATH, NEE, and MIS. The scenes include an emissive
sphere so the benchmark also covers direct-light sampling.

Run `pixi run bench_rt_gpu_sphere_lights` for the 64-emissive-sphere NEE/MIS
scalability guard. It uses 64 spp so small light-record and selection changes
are measured over a longer, less noisy workload.
