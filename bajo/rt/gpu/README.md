# GPU ray tracing

Device-resident wavefront rendering for spheres, triangles and transformed
instances. Supports PATH, NORMALS, AO, NEE and MIS; Lambertian, metal,
dielectric, coated-diffuse and emissive materials; textures and image lighting.

## Usage

- `render_gpu[integrator](settings, camera, scene_data)` selects geometry and
  BVH defaults, renders synchronously and downloads pixels.
- `GpuRtPreparedRenderer[kind]` retains the scene and allocations across renders.
- For asynchronous rendering, use `prepare_gpu_scene[kind]`,
  `scene.make_render_target`, `enqueue_render_gpu[integrator]` and
  `download_gpu_pixels`. Update the camera with `update_gpu_camera`.
- Explicit formats and builders are available through `prepare_gpu_scene` and
  `render_gpu_configured`. The scene kind must match the geometry present.

Finalize `SceneBuilder` with `finish()` to obtain backend-neutral `SceneData`.
GPU preparation needs no CPU BVH. Prepared scenes own snapshots; prepare a new
scene after authored changes. There is no scene update/refit API yet.

## Execution

Each chunk contains complete pixels. Tracing fuses Lambertian shading and
queues other materials; deferred material/coated shading precedes shadow
tracing. Device counters advance ping-pong path queues without host readback
between bounces. Resolve writes the chunk into persistent device pixels.

Queues use field-major storage: 13 path, 13 shade, 11 shadow and 3 radiance
float planes, plus IDs and packed material/face flags. Coated shade records
retain UVs, boundary geometry and medium-adjusted throughput. Typed TileTensor
views in `queue_layout.mojo` define these field-major contracts and borrow the
existing buffers; material tables use record-major views. Queue order is
unspecified; global path IDs and sample offsets preserve all six sampler
sequences across scheduling, chunks and sample batches.

Geometry, BVH layout and integrator specialize kernels. Runtime material
presence controls deferred launches; material combinations do not enumerate
shader variants. Direct-light selection uses a prepared Walker–Vose alias table.

The scene target factory chooses capacity from geometry/material metadata.
A positive `path_capacity` overrides it; direct `GpuRtRenderTarget` construction
uses a 256K-path fallback. Larger queues trade GPU memory for throughput.

## BVH defaults

Automatic `render_gpu` selection and the viewer use CWBVH8 (node8/leaf4)
with H-PLOC for all triangles and instance BLASes, including small meshes.
TLAS uses H-PLOC, node2/leaf1; sphere BVHs use wide4, node4/leaf4.
One format set per scene kind reduces compilation by eliminating size-based
layout dispatch. Explicit formats/builders remain available through the
configured and prepared-scene APIs.

Near-first TLAS traversal orders closest-hit leaf tasks for wide4/leaf4 BLASes.
Large static-triangle MIS queues use a separate compressed-traversal function.
These policies are restricted to the workloads where measurements support them.

## Benchmarks and tests

```sh
pixi run bench_rt_gpu
pixi run bench_rt_gpu_mixed
pixi run bench_rt_cpu_gpu
```

Run GPU measurements sequentially after compilation. GPU RT coverage lives in
[test/rt/gpu](../../../test/rt/gpu/), with traversal tests in
[test/bvh/gpu](../../../test/bvh/gpu/).

The 2026-10-11 architecture trials are recorded in
[gpu_optimization_trials.csv](../../../bench/rt/gpu_optimization_trials.csv).
HPLOC TLAS construction reduced warm render time by about 5–6% on the Crown
and BMW quick workloads on an RTX 5060 Ti. Typed TileTensor queue/material
views are retained as an ABI foundation, without a claimed runtime gain.
Full timings, image comparisons and rejected prototypes are stored locally in
`bench/results/gpu_next10_20261011/` (ignored by Git).
