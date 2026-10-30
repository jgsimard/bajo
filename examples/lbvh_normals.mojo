"""Instanced normal render using H-PLOC CWBVH8 BLASes and an LBVH2 TLAS.

Camera rays use direct traversal; ray sorting is intended for incoherent rays.
"""

from max.gpu.host import DeviceContext
from std.math import max, round
from std.sys import has_accelerator
from std.time import perf_counter_ns

from bajo.core import AABB, Quat, Vec3f32, Point3f32
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.bvh.cpu import CpuTlas, build_cpu_triangle_blas_set
from bajo.bvh.gpu import (
    GpuBvhBuildMethod,
    GpuBvhLayout,
    build_gpu_triangle_blas_set,
    build_gpu_tlas,
)
from bajo.core.aabb import compute_bounds
from bajo.core.camera import Camera
from bajo.bvh import Hit, Instance
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.bvh.constants import MISS_PRIM
from bajo.bvh.gpu.utils import upload_list
from bajo.core.random import Rng
from examples.mesh_helpers import (
    normalized_instance_scale,
    bottom_centered_transform,
)
from bajo.rt.image_output import write_ppm_from_colors
from bajo.core.color import Color

comptime OBJ_PATHS = (
    "./assets/bunny/bunny.obj",
    "./assets/buddha/buddha.obj",
    "./assets/dragon/dragon.obj",
)
comptime CPU_OUTPUT_PATH = "./example_tlas_lbvh_normals_cpu.ppm"
comptime GPU_OUTPUT_PATH = "./example_tlas_lbvh_normals_gpu.ppm"
comptime WIDTH = 1280
comptime HEIGHT = 720
comptime GRID_X = 6
comptime GRID_Z = 6
comptime DEMO_BLAS_COUNT = len(OBJ_PATHS)
comptime BLAS_WIDTH_CPU = 16
comptime TLAS_WIDTH_CPU = 4
comptime BLAS_BUILD_METHOD_GPU = GpuBvhBuildMethod.HPLOC
comptime BLAS_LAYOUT_GPU = GpuBvhLayout.CWBVH8
comptime BLAS_WIDTH_GPU = 8
# CWBVH8's storage leaf width is four; construction uses one triangle per leaf.
comptime BLAS_LEAF_WIDTH_GPU = 4
comptime BLAS_MAX_LEAF_SIZE_GPU = 1
comptime TLAS_BUILD_METHOD_GPU = GpuBvhBuildMethod.LBVH
comptime TLAS_WIDTH_GPU = 2
comptime TLAS_LEAF_WIDTH_GPU = 1


def _make_instances(
    bounds_list: ImmSpan[AABB[.LOCAL], _],
) -> List[Instance]:
    var rng = Rng(123, 123)
    comptime TARGET_WORLD_EXTENT = Float32(1.60)

    comptime cell_spacing = TARGET_WORLD_EXTENT * Float32(5.2)
    comptime blas_spacing = TARGET_WORLD_EXTENT * Float32(2.0)

    var blas_count = len(bounds_list)
    var instances = List[Instance](capacity=GRID_X * GRID_Z * blas_count)
    for z in range(GRID_Z):
        for x in range(GRID_X):
            for b in range(blas_count):
                var idx = (z * GRID_X + x) * blas_count + b
                ref bounds = bounds_list[b]

                var cell_x = (
                    Float32(x) - Float32(GRID_X - 1) * 0.5
                ) * cell_spacing
                var cell_z = (
                    Float32(z) - Float32(GRID_Z - 1) * 0.5
                ) * cell_spacing
                var local_x = (
                    Float32(b) - Float32(blas_count - 1) * 0.5
                ) * blas_spacing

                var angle = rng.f32(-1, 1)
                var rotation = Quat.from_axis_angle(
                    Vec3f32[.LOCAL](0, 1, 0), angle
                )
                var variation = Float32(1.0) + Float32(idx % 3) * Float32(0.025)
                var scale = (
                    normalized_instance_scale(
                        bounds, TARGET_WORLD_EXTENT, variation
                    )
                    * 1.5
                )
                var bottom_center = Vec3f32[.WORLD](
                    cell_x + local_x, 0.0, cell_z
                )
                var transform = bottom_centered_transform(
                    bounds, rotation, scale, bottom_center
                )
                instances.append(
                    Instance(
                        transform,
                        UInt32(b),
                        bounds,
                        .TRIANGLE,
                    )
                )
    return instances^


def _make_camera[width: SIMDLength](tlas: CpuTlas[width, _]) -> Camera:
    var bounds = tlas.bounds()
    var center = bounds.centroid()
    var extent = bounds.extent()

    var scene_w = max(extent.x, extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    comptime CAMERA_DISTANCE_SCALE = Float32(0.3)
    comptime CAMERA_HEIGHT_SCALE = Float32(0.02)

    var eye = Point3f32[.WORLD](
        center.x,
        center.y + scene_w * CAMERA_HEIGHT_SCALE + extent.y * 0.35,
        center.z - scene_w * CAMERA_DISTANCE_SCALE,
    )

    var target = Point3f32[.WORLD](
        center.x,
        bounds._min.y + extent.y * 0.35,
        center.z,
    )
    return Camera(
        eye,
        target,
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        Float32(0.78),
    )


def write_ppm_normals_from_hits(
    path: String,
    width: Int,
    height: Int,
    hits: ImmSpan[Float32, _],
) raises:
    var pixel_count = width * height
    var pixels = List[Color](unsafe_uninit_length=pixel_count)
    var background = Color(18.0, 22.0, 30.0) / 255.0
    for i in range(pixel_count):
        var hit = Hit.load(hits, i)
        if hit.prim == MISS_PRIM or hit.inst == MISS_PRIM:
            pixels[i] = background
        else:
            pixels[i] = Color(
                hit.normal.x, hit.normal.y, hit.normal.z
            ) * 0.5 + Color(0.5)
    write_ppm_from_colors[gamma_correct=False](path, width, height, pixels)


def _print_bounds_by_blas(instances: ImmSpan[Instance, _], blas_count: Int):
    var bounds = List[AABB[.WORLD]](
        length=blas_count, fill=AABB[.WORLD].invalid()
    )
    var counts = List[Int](length=blas_count, fill=0)

    for inst in instances:
        var blas_idx = Int(inst.blas_idx)
        bounds[blas_idx].grow(inst.bounds)
        counts[blas_idx] += 1

    print("World instance bounds by BLAS:")
    for blas_idx in range(blas_count):
        var b = bounds[blas_idx]
        print(
            t"  BLAS {blas_idx} count={counts[blas_idx]} "
            t"min={round(b._min, 3)} max={round(b._max, 3)}"
        )


def print_hit_counts_by_blas(
    label: String,
    width: Int,
    height: Int,
    instances: ImmSpan[Instance, _],
    hits: ImmSpan[Float32, _],
    blas_count: Int,
):
    var hit_counts = List[Int](length=blas_count, fill=0)
    var total_hits = 0
    var pixel_count = width * height

    for i in range(pixel_count):
        var hit = Hit[.WORLD].load(hits, i)
        var inst = hit.inst

        if inst != MISS_PRIM:
            total_hits += 1

            var inst_idx = Int(inst)
            if inst_idx < len(instances):
                var blas_idx = Int(instances[inst_idx].blas_idx)
                if blas_idx < blas_count:
                    hit_counts[blas_idx] += 1

    print(t"{label} visible hit pixels by BLAS:")
    for blas_idx in range(blas_count):
        print(t"  BLAS {blas_idx}: {hit_counts[blas_idx]}")

    print(t"  total={total_hits}")


def render_cpu(
    tri_vertex_sets: ImmSpan[List[Point3f32[.LOCAL]], _],
    instances: ImmSpan[Instance, _],
    cpu_tlas: CpuTlas[TLAS_WIDTH_CPU, 1],
    camera: Camera,
) raises:
    var ray_count = WIDTH * HEIGHT

    print("\nBuilding CPU BLAS set...")
    var blas_t0 = perf_counter_ns()
    var cpu_blases = build_cpu_triangle_blas_set[BLAS_WIDTH_CPU, method=.HPLOC](
        tri_vertex_sets
    )
    var blas_t1 = perf_counter_ns()
    print(
        t"CPU BLAS set build: "
        t"total={round(ns_to_ms(Int(blas_t1 - blas_t0)), 3)} ms"
    )

    print("\nTracing TLAS on CPU...")
    var hits = List[Float32](unsafe_uninit_length=ray_count * Hit.STRIDE)

    var trace_t0 = perf_counter_ns()
    cpu_tlas.trace_camera[BLAS_WIDTH_CPU](
        cpu_blases,
        camera,
        WIDTH,
        HEIGHT,
        Span(hits),
    )
    var trace_t1 = perf_counter_ns()
    var trace_ns = Int(trace_t1 - trace_t0)
    print(
        t"CPU trace: {round(ns_to_ms(trace_ns), 3)} ms | "
        t"{round(ns_to_mrays_per_s(trace_ns, ray_count), 3)} Mrays/s"
    )
    print_hit_counts_by_blas(
        "CPU",
        WIDTH,
        HEIGHT,
        instances,
        hits,
        len(tri_vertex_sets),
    )

    print("\nWriting CPU normal PPM...")
    var write_t0 = perf_counter_ns()
    write_ppm_normals_from_hits(
        CPU_OUTPUT_PATH,
        WIDTH,
        HEIGHT,
        hits,
    )
    var write_t1 = perf_counter_ns()
    print(t"CPU write: {round(ns_to_ms(Int(write_t1 - write_t0)), 3)} ms")


def render_gpu(
    tri_vertex_sets: ImmSpan[List[Point3f32[.LOCAL]], _],
    instances: ImmSpan[Instance, _],
    camera: Camera,
) raises:
    var ray_count = WIDTH * HEIGHT

    with DeviceContext() as ctx:
        # Warm up GPU runtime / allocator / copy path.
        var warm_t0 = perf_counter_ns()
        var warm_h = ctx.enqueue_create_host_buffer[.float32](1024)
        var warm_d = ctx.enqueue_create_buffer[.float32](1024)
        warm_h.enqueue_copy_to(warm_d)
        ctx.synchronize()
        var warm_t1 = perf_counter_ns()
        print(
            t"\nGPU warmup time ="
            t"{round(ns_to_ms(Int(warm_t1 - warm_t0)), 3)} ms "
        )

        print("\nBuilding GPU BLAS set...")
        var blas_t0 = perf_counter_ns()
        var gpu_blases = build_gpu_triangle_blas_set[
            BLAS_WIDTH_GPU,
            BLAS_LEAF_WIDTH_GPU,
            BLAS_BUILD_METHOD_GPU,
            BLAS_LAYOUT_GPU,
            max_leaf_size=BLAS_MAX_LEAF_SIZE_GPU,
        ](ctx, tri_vertex_sets)
        ctx.synchronize()
        var blas_t1 = perf_counter_ns()

        print(
            t"GPU BLAS set build:"
            t" total={round(ns_to_ms(Int(blas_t1 - blas_t0)), 3)} ms"
        )

        print("\nBuilding GPU TLAS...")
        var tlas_t0 = perf_counter_ns()
        var gpu_tlas = build_gpu_tlas[
            .TRIANGLE,
            TLAS_WIDTH_GPU,
            BLAS_WIDTH_GPU,
            TLAS_LEAF_WIDTH_GPU,
            BLAS_LEAF_WIDTH_GPU,
            TLAS_BUILD_METHOD_GPU,
            BLAS_LAYOUT_GPU,
        ](ctx, instances)
        ctx.synchronize()
        var tlas_t1 = perf_counter_ns()

        print(
            t"GPU TLAS build:"
            t" total={round(ns_to_ms(Int(tlas_t1 - tlas_t0)), 3)} ms"
        )

        print("\nUploading camera params and tracing TLAS on GPU...")
        var setup_t0 = perf_counter_ns()
        var camera_params = camera.flatten()
        var d_camera_params = upload_list(ctx, camera_params)

        var d_hits = ctx.enqueue_create_buffer[.float32](ray_count * Hit.STRIDE)
        ctx.synchronize()
        var setup_t1 = perf_counter_ns()
        print(
            t"GPU setup/upload:"
            t" {round(ns_to_ms(Int(setup_t1 - setup_t0)), 3)} ms"
        )

        var trace_t0 = perf_counter_ns()
        gpu_tlas.launch_camera(
            ctx,
            gpu_blases,
            d_camera_params,
            d_hits,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()
        var trace_t1 = perf_counter_ns()
        var first_trace_ns = Int(trace_t1 - trace_t0)
        print(
            t"GPU first trace: {round(ns_to_ms(first_trace_ns), 3)} ms | "
            t"{round(ns_to_mrays_per_s(first_trace_ns, ray_count), 3)} Mrays/s"
        )

        var hot_trace_t0 = perf_counter_ns()
        gpu_tlas.launch_camera(
            ctx,
            gpu_blases,
            d_camera_params,
            d_hits,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()
        var hot_trace_ns = Int(perf_counter_ns() - hot_trace_t0)
        print(
            t"GPU hot trace: {round(ns_to_ms(hot_trace_ns), 3)} ms | "
            t"{round(ns_to_mrays_per_s(hot_trace_ns, ray_count), 3)} Mrays/s"
        )
        with d_hits.map_to_host() as h:
            var hit_span = h.as_span()
            print_hit_counts_by_blas(
                "GPU", WIDTH, HEIGHT, instances, hit_span, len(tri_vertex_sets)
            )
            print("\nWriting GPU normal PPM...")
            var write_t0 = perf_counter_ns()
            write_ppm_normals_from_hits(
                GPU_OUTPUT_PATH,
                WIDTH,
                HEIGHT,
                hit_span,
            )
            var write_t1 = perf_counter_ns()
            print(
                t"GPU write: {round(ns_to_ms(Int(write_t1 - write_t0)), 3)} ms"
            )
        ctx.synchronize()


def main() raises:
    print("multi-BLAS instanced TLAS normal render")
    comptime for index in range(DEMO_BLAS_COUNT):
        print(t"OBJ {index}: {OBJ_PATHS[index]}")
    print(t"Resolution: {WIDTH} x {HEIGHT}")
    print(t"Instances: {GRID_X * GRID_Z * DEMO_BLAS_COUNT}")
    print(t"Cells: {GRID_X} x {GRID_Z}")
    print(t"Instances per cell: {DEMO_BLAS_COUNT}")
    print(t"BLAS width CPU: {BLAS_WIDTH_CPU}")
    print(t"TLAS width CPU: {TLAS_WIDTH_CPU}")
    print(t"BLAS width GPU: {BLAS_WIDTH_GPU}")
    print(t"BLAS leaf width GPU: {BLAS_LEAF_WIDTH_GPU}")
    print(
        t"GPU BLAS policy: H-PLOC + CWBVH8, max leaf size"
        t" {BLAS_MAX_LEAF_SIZE_GPU}"
    )
    print(t"TLAS width GPU: {TLAS_WIDTH_GPU}")
    print(t"TLAS leaf width GPU: {TLAS_LEAF_WIDTH_GPU}")
    print("GPU TLAS policy: LBVH2, one instance per leaf")
    print(t"CPU output: {CPU_OUTPUT_PATH}")
    print(t"GPU output: {GPU_OUTPUT_PATH}")
    print("Scene layout: each cell = bunny | buddha | dragon")
    print("Backend: CPU then GPU")

    print("\nLoading and packing geometry...")
    var load_t0 = perf_counter_ns()
    var tri_vertex_sets = List[List[Point3f32[.LOCAL]]](
        capacity=DEMO_BLAS_COUNT
    )
    comptime for index in range(DEMO_BLAS_COUNT):
        tri_vertex_sets.append(pack_obj_triangles(OBJ_PATHS[index]))
    var load_t1 = perf_counter_ns()

    print(t"BLASes: {len(tri_vertex_sets)}")
    var blas_bounds = List[AABB[.LOCAL]](capacity=DEMO_BLAS_COUNT)
    var total_tri_count = 0
    for index in range(len(tri_vertex_sets)):
        ref vertices = tri_vertex_sets[index]
        blas_bounds.append(compute_bounds(vertices))
        var tri_count = len(vertices) // 3
        total_tri_count += tri_count
        print(t"Triangles {index}: {tri_count}")
    print(t"Total unique triangles: {total_tri_count}")
    print(t"Load time: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)} ms")

    print("\nBuilding Scene...")
    var cpu_t0 = perf_counter_ns()
    var instances = _make_instances(blas_bounds)
    _print_bounds_by_blas(instances, len(tri_vertex_sets))

    var cpu_tlas = CpuTlas[TLAS_WIDTH_CPU, 1](instances)
    var camera = _make_camera(cpu_tlas)

    var ray_count = WIDTH * HEIGHT
    var cpu_t1 = perf_counter_ns()
    print(
        t"Host TLAS/camera setup: instances={len(instances)} | "
        t"rays={ray_count} | "
        t"time={round(ns_to_ms(Int(cpu_t1 - cpu_t0)), 3)} ms"
    )

    print("\n=== CPU render ===")
    render_cpu(
        tri_vertex_sets,
        instances,
        cpu_tlas,
        camera,
    )
    print(t"Wrote {CPU_OUTPUT_PATH}")

    print("\n=== GPU render ===")
    comptime if has_accelerator():
        render_gpu(
            tri_vertex_sets,
            instances,
            camera,
        )
        print(t"Wrote {GPU_OUTPUT_PATH}")
    else:
        print("SKIP: no accelerator available")

    print("\nDone.")
