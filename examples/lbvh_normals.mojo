from std.io.file_descriptor import FileDescriptor
from max.gpu.host import DeviceBuffer, DeviceContext
from max.algorithm import parallelize
from std.math import max, round, clamp
from std.sys import has_accelerator
from std.time import perf_counter_ns

from bajo.core import (
    AABB,
    Quat,
    Affine3f32,
    Vec3f32,
    Point3f32,
    cross,
    normalize,
    Frame,
)
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.bvh.cpu import CpuBlasSet, CpuTlas, build_cpu_triangle_blas_set
from bajo.bvh.gpu import (
    build_gpu_triangle_blas_set,
    build_gpu_tlas,
)
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh import Camera, Hit, Instance
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.bvh.constants import MISS_PRIM
from bajo.bvh.gpu.utils import upload_list
from bajo.core.random import Rng

comptime OBJ_PATH_0 = "./assets/bunny/bunny.obj"
comptime OBJ_PATH_1 = "./assets/buddha/buddha.obj"
comptime OBJ_PATH_2 = "./assets/dragon/dragon.obj"
comptime CPU_OUTPUT_PATH = "./example_tlas_lbvh_normals_cpu.ppm"
comptime GPU_OUTPUT_PATH = "./example_tlas_lbvh_normals_gpu.ppm"
comptime WIDTH = 1280
comptime HEIGHT = 720
comptime GRID_X = 6
comptime GRID_Z = 6
comptime DEMO_BLAS_COUNT = 3
comptime BLAS_WIDTH_CPU = 16
comptime TLAS_WIDTH_CPU = 4
comptime BLAS_WIDTH_GPU = 8
comptime BLAS_LEAF_WIDTH_GPU = 4
comptime TLAS_WIDTH_GPU = 2


def _max_blas_extent[frame: Frame](bounds_list: List[AABB[frame]]) -> Float32:
    var out = Float32(0.0)
    for bounds in bounds_list:
        var extent = bounds.extent()
        var e = max(max(extent.x, extent.y), extent.z)
        if e > out:
            out = e

    if out < Float32(1.0e-6):
        out = Float32(1.0)

    return out


def _normalized_instance_scale[
    frame: Frame
](
    bounds: AABB[frame],
    target_extent: Float32,
    variation: Float32,
) -> Vec3f32[
    frame
]:
    var extent = bounds.extent()
    var local_extent = max(max(extent.x, extent.y), extent.z)
    if local_extent < Float32(1.0e-6):
        local_extent = Float32(1.0)

    var s = target_extent / local_extent * variation
    return Vec3f32[frame](s)


def _make_centered_transform(
    bounds: AABB[.LOCAL],
    rotation: Quat,
    scale: Vec3f32[.LOCAL],
    bottom_center: Vec3f32[.WORLD],
) -> Affine3f32[.LOCAL, .WORLD]:
    var transform = Affine3f32[
        .LOCAL, .WORLD
    ].from_rotation_scale_translation(
        rotation, scale, Vec3f32[.WORLD](0.0)
    )
    var c = bounds.centroid()
    var local_anchor = Vec3f32[.LOCAL](c.x, bounds._min.y, c.z)
    var anchor_delta = transform.vector(local_anchor)
    transform.tx = bottom_center.x - anchor_delta.x
    transform.ty = bottom_center.y - anchor_delta.y
    transform.tz = bottom_center.z - anchor_delta.z
    return transform^


def _make_instances(
    bounds_list: List[AABB[.LOCAL]],
) raises -> List[Instance]:
    var rng = Rng(123, 123)
    comptime TARGET_WORLD_EXTENT = Float32(1.60)

    var target_extent = TARGET_WORLD_EXTENT

    var cell_spacing = target_extent * Float32(5.2)
    var blas_spacing = target_extent * Float32(2.0)
    if cell_spacing < Float32(1.0):
        cell_spacing = Float32(1.0)
    if blas_spacing < Float32(0.35):
        blas_spacing = Float32(0.35)

    var blas_count = len(bounds_list)
    var instances = List[Instance](capacity=GRID_X * GRID_Z * blas_count)
    for z in range(GRID_Z):
        for x in range(GRID_X):
            for b in range(blas_count):
                var idx = (z * GRID_X + x) * blas_count + b
                var blas_idx = UInt32(b)
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
                    _normalized_instance_scale(bounds, target_extent, variation)
                    * 1.5
                )
                var bottom_center = Vec3f32[.WORLD](
                    cell_x + local_x, 0.0, cell_z
                )
                var transform = _make_centered_transform(
                    bounds, rotation, scale, bottom_center
                )
                instances.append(
                    Instance(
                        transform,
                        blas_idx,
                        bounds,
                        .TRIANGLE,
                    )
                )
    return instances^


def _make_camera[width: SIMDLength](tlas: CpuTlas[width]) -> Camera:
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


def _make_camera_params[width: SIMDLength](tlas: CpuTlas[width]) -> List[Float32]:
    return _make_camera(tlas).flatten()


def _unit_to_u8(x: Float32) -> UInt8:
    return UInt8(clamp(x, 0.0, 1.0) * 255.0)


def write_ppm_normals_from_hits(
    path: String,
    width: Int,
    height: Int,
    tri_vertex_sets: List[List[Point3f32[.LOCAL]]],
    instances: List[Instance],
    hits: ImmSpan[Float32, _],
) raises:
    var pixel_count = width * height
    var byte_count = pixel_count * 3

    with open(path, "w") as f:
        var fd = FileDescriptor(f)
        fd.write(t"P6\n{width} {height}\n255\n")
        var _bytes = List[UInt8](length=byte_count, fill=0)
        var out = _bytes.unsafe_ptr()

        for i in range(pixel_count):
            var hit = Hit.load(hits, i)
            var prim = hit.prim
            var inst = hit.inst
            if prim == MISS_PRIM or inst == MISS_PRIM:
                out[unsafe_offset=i * 3 + 0] = 18
                out[unsafe_offset=i * 3 + 1] = 22
                out[unsafe_offset=i * 3 + 2] = 30
            else:
                var blas_idx = Int(instances[Int(inst)].blas_idx)
                ref tri_vertices = tri_vertex_sets[blas_idx]
                var base = Int(prim) * 3
                ref v0 = tri_vertices[base + 0]
                ref v1 = tri_vertices[base + 1]
                ref v2 = tri_vertices[base + 2]

                var local_n = normalize(cross(v1 - v0, v2 - v0))
                var world_n = normalize(
                    instances[Int(inst)].transform.vector(local_n)
                )

                var r = world_n.x[0] * 0.5 + 0.5
                var g = world_n.y[0] * 0.5 + 0.5
                var b = world_n.z[0] * 0.5 + 0.5

                out[unsafe_offset=i * 3 + 0] = _unit_to_u8(r)
                out[unsafe_offset=i * 3 + 1] = _unit_to_u8(g)
                out[unsafe_offset=i * 3 + 2] = _unit_to_u8(b)

        fd.write_bytes(_bytes)


def _print_bounds_by_blas(instances: List[Instance]):
    var blas_count = 0

    for inst in instances:
        var idx = Int(inst.blas_idx)
        if idx + 1 > blas_count:
            blas_count = idx + 1

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
    width: Int,
    height: Int,
    instances: List[Instance],
    hits: DeviceBuffer[.float32],
) raises:
    var blas_count = 0

    for inst in instances:
        var idx = Int(inst.blas_idx)
        if idx + 1 > blas_count:
            blas_count = idx + 1

    var hit_counts = List[Int](length=blas_count, fill=0)
    var total_hits = 0
    var pixel_count = width * height

    with hits.map_to_host() as hu:
        var hit_span = Span(unsafe_ptr=hu.unsafe_ptr(), length=len(hu))
        for i in range(pixel_count):
            var hit = Hit[.WORLD].load(hit_span, i)
            var inst = hit.inst

            if inst != MISS_PRIM:
                total_hits += 1

                var inst_idx = Int(inst)
                if inst_idx < len(instances):
                    var blas_idx = Int(instances[inst_idx].blas_idx)
                    if blas_idx < blas_count:
                        hit_counts[blas_idx] += 1

    print("GPU visible hit pixels by BLAS:")
    for blas_idx in range(blas_count):
        print(t"  BLAS {blas_idx}: {hit_counts[blas_idx]}")

    print(t"  total={total_hits}")


def _build_cpu_triangle_blas_set[
    width: SIMDLength
](tri_vertex_sets: List[List[Point3f32[.LOCAL]]]) -> CpuBlasSet[
    .TRIANGLE, width
]:
    return build_cpu_triangle_blas_set[width](tri_vertex_sets)


def _trace_cpu_tlas_camera[
    tlas_width: SIMDLength,
    blas_width: SIMDLength,
](
    width: Int,
    height: Int,
    tlas: CpuTlas[tlas_width],
    cpu_blases: CpuBlasSet[.TRIANGLE, blas_width],
    camera: Camera,
    mut hits: List[Float32],
):
    def worker(py: Int) {imm, mut hits}:
        for px in range(width):
            var ray_idx = py * width + px
            var ray = camera.make_ray(px, py, width, height)
            var hit = tlas.trace_blases[
                blas_width, blas_width, .CLOSEST_HIT
            ](ray, cpu_blases)
            hit.store(hits, ray_idx)

    parallelize(worker, height, height)


def print_hit_counts_by_blas_host(
    label: String,
    width: Int,
    height: Int,
    instances: List[Instance],
    hits: List[Float32],
):
    var blas_count = 0

    for inst in instances:
        var idx = Int(inst.blas_idx)
        if idx + 1 > blas_count:
            blas_count = idx + 1

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
    tri_vertex_sets: List[List[Point3f32[.LOCAL]]],
    instances: List[Instance],
    cpu_tlas: CpuTlas[TLAS_WIDTH_CPU],
    camera: Camera,
) raises:
    var ray_count = WIDTH * HEIGHT

    print("\nBuilding CPU BLAS set...")
    var blas_t0 = perf_counter_ns()
    var cpu_blases = _build_cpu_triangle_blas_set[BLAS_WIDTH_CPU](
        tri_vertex_sets
    )
    var blas_t1 = perf_counter_ns()
    print(
        t"CPU BLAS set build: "
        t"total={round(ns_to_ms(Int(blas_t1 - blas_t0)), 3)} ms"
    )

    print("\nTracing TLAS on CPU...")
    var hits = List[Float32](length=ray_count * Hit.STRIDE, fill=0.0)

    var trace_t0 = perf_counter_ns()
    _trace_cpu_tlas_camera[TLAS_WIDTH_CPU, BLAS_WIDTH_CPU](
        WIDTH,
        HEIGHT,
        cpu_tlas,
        cpu_blases,
        camera,
        hits,
    )
    var trace_t1 = perf_counter_ns()
    var trace_ns = Int(trace_t1 - trace_t0)
    print(
        t"CPU trace: {round(ns_to_ms(trace_ns), 3)} ms | "
        t"{round(ns_to_mrays_per_s(trace_ns, ray_count), 3)} Mrays/s"
    )
    print_hit_counts_by_blas_host(
        "CPU",
        WIDTH,
        HEIGHT,
        instances,
        hits,
    )

    print("\nWriting CPU normal PPM...")
    var write_t0 = perf_counter_ns()
    write_ppm_normals_from_hits(
        CPU_OUTPUT_PATH,
        WIDTH,
        HEIGHT,
        tri_vertex_sets,
        instances,
        hits,
    )
    var write_t1 = perf_counter_ns()
    print(t"CPU write: {round(ns_to_ms(Int(write_t1 - write_t0)), 3)} ms")


def render_gpu(
    tri_vertex_sets: List[List[Point3f32[.LOCAL]]],
    instances: List[Instance],
    camera_params: List[Float32],
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
            .HPLOC,
            .CWBVH8,
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
            TLAS_WIDTH_GPU,
            BLAS_LEAF_WIDTH_GPU,
            .LBVH,
            .CWBVH8,
        ](ctx, instances)
        ctx.synchronize()
        var tlas_t1 = perf_counter_ns()

        print(
            t"GPU TLAS build:"
            t" total={round(ns_to_ms(Int(tlas_t1 - tlas_t0)), 3)} ms"
        )

        print("\nUploading camera params and tracing TLAS on GPU...")
        var setup_t0 = perf_counter_ns()
        var d_camera_params = upload_list(ctx, camera_params)

        var d_hits = ctx.enqueue_create_buffer[.float32](
            ray_count * Hit.STRIDE
        )
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
        print_hit_counts_by_blas(WIDTH, HEIGHT, instances, d_hits)

        print("\nWriting GPU normal PPM...")
        var write_t0 = perf_counter_ns()
        with d_hits.map_to_host() as h:
            var hit_span = Span(unsafe_ptr=h.unsafe_ptr(), length=len(h))
            write_ppm_normals_from_hits(
                GPU_OUTPUT_PATH,
                WIDTH,
                HEIGHT,
                tri_vertex_sets,
                instances,
                hit_span,
            )
        ctx.synchronize()
        var write_t1 = perf_counter_ns()
        print(t"GPU write: {round(ns_to_ms(Int(write_t1 - write_t0)), 3)} ms")


def main() raises:
    print("multi-BLAS instanced TLAS normal render")
    print(t"OBJ 0: {OBJ_PATH_0}")
    print(t"OBJ 1: {OBJ_PATH_1}")
    print(t"OBJ 2: {OBJ_PATH_2}")
    print(t"Resolution: {WIDTH} x {HEIGHT}")
    print(t"Instances: {GRID_X * GRID_Z * DEMO_BLAS_COUNT}")
    print(t"Cells: {GRID_X} x {GRID_Z}")
    print(t"Instances per cell: {DEMO_BLAS_COUNT}")
    print(t"BLAS width CPU: {BLAS_WIDTH_CPU}")
    print(t"TLAS width CPU: {TLAS_WIDTH_CPU}")
    print(t"BLAS width GPU: {BLAS_WIDTH_GPU}")
    print(t"BLAS leaf width GPU: {BLAS_LEAF_WIDTH_GPU}")
    print("GPU BLAS policy: H-PLOC + CWBVH8")
    print(t"TLAS width GPU: {TLAS_WIDTH_GPU}")
    print(t"CPU output: {CPU_OUTPUT_PATH}")
    print(t"GPU output: {GPU_OUTPUT_PATH}")
    print("Scene layout: each cell = bunny | buddha | dragon")
    print("Backend: CPU then GPU")

    print("\nLoading and packing geometry...")
    var load_t0 = perf_counter_ns()
    var tri_vertices_0 = pack_obj_triangles(OBJ_PATH_0)
    var tri_vertices_1 = pack_obj_triangles(OBJ_PATH_1)
    var tri_vertices_2 = pack_obj_triangles(OBJ_PATH_2)
    var load_t1 = perf_counter_ns()

    var tri_vertex_sets = List[List[Point3f32[.LOCAL]]](capacity=3)
    tri_vertex_sets.append(tri_vertices_0.copy())
    tri_vertex_sets.append(tri_vertices_1.copy())
    tri_vertex_sets.append(tri_vertices_2.copy())

    var blas_bounds = List[AABB[.LOCAL]](capacity=3)
    blas_bounds.append(compute_bounds(tri_vertices_0))
    blas_bounds.append(compute_bounds(tri_vertices_1))
    blas_bounds.append(compute_bounds(tri_vertices_2))

    var tri_count_0 = len(tri_vertices_0) / 3
    var tri_count_1 = len(tri_vertices_1) / 3
    var tri_count_2 = len(tri_vertices_2) / 3
    var total_tri_count = tri_count_0 + tri_count_1 + tri_count_2

    print(t"BLASes: {len(tri_vertex_sets)}")
    print(t"Triangles 0: {tri_count_0}")
    print(t"Triangles 1: {tri_count_1}")
    print(t"Triangles 2: {tri_count_2}")
    print(t"Total unique triangles: {total_tri_count}")
    print(t"Load time: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)} ms")

    print("\nBuilding Scene...")
    var cpu_t0 = perf_counter_ns()
    var instances = _make_instances(blas_bounds)
    var inst_count_0 = 0
    var inst_count_1 = 0
    var inst_count_2 = 0
    for inst in instances:
        if inst.blas_idx == UInt32(0):
            inst_count_0 += 1
        elif inst.blas_idx == UInt32(1):
            inst_count_1 += 1
        elif inst.blas_idx == UInt32(2):
            inst_count_2 += 1
    print(
        t"Instance counts by BLAS: "
        t"{inst_count_0}, {inst_count_1}, {inst_count_2}"
    )
    _print_bounds_by_blas(instances)

    var cpu_tlas = CpuTlas[TLAS_WIDTH_CPU](instances)
    var camera = _make_camera(cpu_tlas)
    var camera_params = camera.flatten()

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
            camera_params,
        )
        print(t"Wrote {GPU_OUTPUT_PATH}")
    else:
        print("SKIP: no accelerator available")

    print("\nDone.")
