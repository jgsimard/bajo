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

Queues use field-major storage: 12 path, 6 shade, 11 shadow and 3 radiance
float planes, plus IDs and packed material/face flags. Coated shade records
retain UVs. Queue order is unspecified; global path IDs and sample offsets
preserve all six sampler sequences across scheduling, chunks and sample batches.

Geometry, BVH layout and integrator specialize kernels. Runtime material
presence controls deferred launches; material combinations do not enumerate
shader variants. Direct-light selection uses a prepared Walker–Vose alias table.

The scene target factory chooses capacity from geometry/material metadata.
A positive `path_capacity` overrides it; direct `GpuRtRenderTarget` construction
uses a 256K-path fallback. Larger queues trade GPU memory for throughput.

## BVH defaults

Automatic `render_gpu` selection uses H-PLOC for triangles and instance BLASes,
choosing CWBVH8 at 32 triangles and wide4 below that (instances use weighted
mean mesh size). TLAS defaults to LBVH, node2/leaf1; sphere BVHs use wide4.
Prepared-scene formats/builders can be selected explicitly.

Near-first TLAS traversal orders closest-hit leaf tasks for wide4/leaf4 BLASes.
Large static-triangle MIS queues use a separate compressed-traversal function.
These policies are restricted to the workloads where measurements support them.

## Performance tracking

RTX 5060 Ti, 1024×1024, 8 spp, depth 8; hot device-resident MIS times,
excluding scene preparation, allocation and pixel download. Three paired
trials; 21 hot frames for instances/overlap/combined, 9 for dragon workloads.
Latest retained measurements: 2026-10-04.

| Workload | Render time |
| --- | ---: |
| Instances | 9.82 ms |
| Overlapping instances | 10.95 ms |
| Combined | 22.63 ms |
| Dragon, compressed instance BLAS | 13.88 ms |
| Dragon, wide4 instance BLAS | 20.20 ms |
| Dragon, static CWBVH8 | 10.28 ms |

Static dragon improved from 54.13 ms through larger batches and the traversal
function boundary. Raw trials, profiles and validation evidence are kept locally
in [benchmark results](../../../bench/results/rt_gpu/).

## Benchmarks and tests

```sh
pixi run bench_rt_gpu
pixi run bench_rt_gpu_materials
pixi run bench_rt_gpu_tlas
pixi run bench_rt_gpu_instance_blas
```

Run GPU measurements sequentially after compilation. GPU RT coverage lives in
[test/rt/gpu](../../../test/rt/gpu/), with traversal tests in
[test/bvh/gpu](../../../test/bvh/gpu/).
