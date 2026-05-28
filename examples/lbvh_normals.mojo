from std.io.file_descriptor import FileDescriptor
from std.gpu import DeviceBuffer, DeviceContext, HostBuffer
from std.math import max, round, clamp
from std.sys import has_accelerator
from std.time import perf_counter_ns

from bajo.core.aabb import AABB
from bajo.core.quat import Quat
from bajo.core.transform import Affine3f32
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core.vec import Vec3f32, cross, normalize
from bajo.bvh.cpu.tlas import Tlas
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.gpu.tlas import GpuTriangleTlas
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBlasSetBuilder
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Instance
from bajo.obj.pack import pack_obj_triangles
from bajo.bvh.constants import Primitive, MISS_PRIM, TRACE
from bajo.bvh.camera import Camera
from bajo.bvh.gpu.utils import upload_list
from bajo.core.random import Rng

comptime OBJ_PATH_0 = "./assets/bunny/bunny.obj"
comptime OBJ_PATH_1 = "./assets/buddha/buddha.obj"
comptime OBJ_PATH_2 = "./assets/dragon/dragon.obj"
comptime DEFAULT_OUTPUT_PATH = "./example_tlas_lbvh_normals.ppm"
comptime WIDTH = 1280
comptime HEIGHT = 720
comptime GRID_X = 6
comptime GRID_Z = 6
comptime DEMO_BLAS_COUNT = 3
comptime BLAS_WIDTH = 2
comptime TLAS_WIDTH = 2
comptime TINT_BY_BLAS = True


def _max_blas_extent(bounds_list: List[AABB]) -> Float32:
    var out = Float32(0.0)
    for bounds in bounds_list:
        var extent = bounds.extent()
        var e = max(max(extent.x, extent.y), extent.z)
        if e > out:
            out = e

    if out < Float32(1.0e-6):
        out = Float32(1.0)

    return out


def _normalized_instance_scale(
    bounds: AABB,
    target_extent: Float32,
    variation: Float32,
) -> Vec3f32:
    var extent = bounds.extent()
    var local_extent = max(max(extent.x, extent.y), extent.z)
    if local_extent < Float32(1.0e-6):
        local_extent = Float32(1.0)

    var s = target_extent / local_extent * variation
    return Vec3f32(s)


def _make_centered_transform(
    bounds: AABB,
    rotation: Quat,
    scale: Vec3f32,
    bottom_center: Vec3f32,
) -> Affine3f32:
    var transform = Affine3f32.from_rotation_scale_translation(
        rotation, scale, Vec3f32(0.0)
    )
    var c = bounds.centroid()
    var local_anchor = Vec3f32(c.x, bounds._min.y, c.z)
    var anchor_delta = transform.vector(local_anchor)
    transform.tx = bottom_center.x - anchor_delta.x
    transform.ty = bottom_center.y - anchor_delta.y
    transform.tz = bottom_center.z - anchor_delta.z
    return transform^


def _make_instances(bounds_list: List[AABB]) raises -> List[Instance]:
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
                var rotation = Quat.from_axis_angle(Vec3f32(0, 1, 0), angle)
                var variation = Float32(1.0) + Float32(idx % 3) * Float32(0.025)
                var scale = (
                    _normalized_instance_scale(bounds, target_extent, variation)
                    * 1.5
                )
                var bottom_center = Vec3f32(cell_x + local_x, 0.0, cell_z)
                var transform = _make_centered_transform(
                    bounds, rotation, scale, bottom_center
                )
                var inv_transform = transform.inverse().inv.copy()
                instances.append(
                    Instance(
                        transform,
                        inv_transform,
                        blas_idx,
                        bounds,
                        Primitive.TRIANGLE,
                    )
                )
    return instances^


def _make_camera(tlas: Tlas[TLAS_WIDTH]) -> Camera:
    var bounds = tlas.bounds()
    var center = bounds.centroid()
    var extent = bounds.extent()

    var scene_w = max(extent.x, extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    comptime CAMERA_DISTANCE_SCALE = Float32(0.3)
    comptime CAMERA_HEIGHT_SCALE = Float32(0.02)

    var eye = Vec3f32(
        center.x,
        center.y + scene_w * CAMERA_HEIGHT_SCALE + extent.y * 0.35,
        center.z - scene_w * CAMERA_DISTANCE_SCALE,
    )

    var target = Vec3f32(
        center.x,
        bounds._min.y + extent.y * 0.35,
        center.z,
    )
    return Camera(
        eye,
        target,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.78),
    )


def _make_camera_params(tlas: Tlas[TLAS_WIDTH]) -> List[Float32]:
    return _make_camera(tlas).flatten()


def _unit_to_u8(x: Float32) -> UInt8:
    return UInt8(clamp(x, 0.0, 1.0) * 255.0)


def _blas_tint(blas_idx: Int) -> Vec3f32:
    var m = blas_idx % 3
    if m == 0:
        return Vec3f32(1.0, 0.25, 0.25)
    if m == 1:
        return Vec3f32(0.25, 1.0, 0.35)
    return Vec3f32(0.25, 0.45, 1.0)


def write_ppm_normals_from_hits(
    path: String,
    width: Int,
    height: Int,
    tri_vertex_sets: List[List[Vec3f32]],
    instances: List[Instance],
    hits_f32: DeviceBuffer[DType.float32],
    hits_u32: DeviceBuffer[DType.uint32],
) raises:
    var pixel_count = width * height
    var byte_count = pixel_count * 3

    with open(path, "w") as f:
        var fd = FileDescriptor(f)
        fd.write(t"P6\n{width} {height}\n255\n")
        var _bytes = List[UInt8](length=byte_count, fill=0)
        var out = _bytes.unsafe_ptr()

        with hits_f32.map_to_host() as hf, hits_u32.map_to_host() as hu:
            for i in range(pixel_count):
                var prim = UInt32(hu[i * 2 + 0])
                var inst = UInt32(hu[i * 2 + 1])
                if prim == MISS_PRIM or inst == MISS_PRIM:
                    out[i * 3 + 0] = UInt8(18)
                    out[i * 3 + 1] = UInt8(22)
                    out[i * 3 + 2] = UInt8(30)
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

                    comptime if TINT_BY_BLAS:
                        var tint = _blas_tint(blas_idx)
                        r = r * 0.75 + tint.x * 0.25
                        g = g * 0.75 + tint.y * 0.25
                        b = b * 0.75 + tint.z * 0.25

                    out[i * 3 + 0] = _unit_to_u8(r)
                    out[i * 3 + 1] = _unit_to_u8(g)
                    out[i * 3 + 2] = _unit_to_u8(b)

        fd.write_bytes(_bytes)


def _print_bounds_by_blas(instances: List[Instance]):
    var blas_count = 0

    for inst in instances:
        var idx = Int(inst.blas_idx)
        if idx + 1 > blas_count:
            blas_count = idx + 1

    var bounds = List[AABB](length=blas_count, fill=AABB.invalid())
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
    hits_u32: DeviceBuffer[DType.uint32],
) raises:
    var blas_count = 0

    for inst in instances:
        var idx = Int(inst.blas_idx)
        if idx + 1 > blas_count:
            blas_count = idx + 1

    var hit_counts = List[Int](length=blas_count, fill=0)
    var total_hits = 0
    var pixel_count = width * height

    with hits_u32.map_to_host() as hu:
        for i in range(pixel_count):
            var inst = UInt32(hu[i * 2 + 1])

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


def main() raises:
    print("GPU multi-BLAS instanced TLAS normal render")
    print(t"OBJ 0: {OBJ_PATH_0}")
    print(t"OBJ 1: {OBJ_PATH_1}")
    print(t"OBJ 2: {OBJ_PATH_2}")
    print(t"Resolution: {WIDTH} x {HEIGHT}")
    print(t"Instances: {GRID_X * GRID_Z * DEMO_BLAS_COUNT}")
    print(t"Cells: {GRID_X} x {GRID_Z}")
    print(t"Instances per cell: {DEMO_BLAS_COUNT}")
    print(t"BLAS width: {BLAS_WIDTH}")
    print(t"TLAS width: {TLAS_WIDTH}")
    print(t"Output: {DEFAULT_OUTPUT_PATH}")
    print("Scene layout: each cell = bunny | buddha | dragon")
    print(t"Tint by BLAS: {TINT_BY_BLAS}")

    print("\nLoading and packing geometry...")
    var load_t0 = perf_counter_ns()
    var tri_vertices_0 = pack_obj_triangles(OBJ_PATH_0)
    var tri_vertices_1 = pack_obj_triangles(OBJ_PATH_1)
    var tri_vertices_2 = pack_obj_triangles(OBJ_PATH_2)
    var load_t1 = perf_counter_ns()

    var tri_vertex_sets = List[List[Vec3f32]](capacity=3)
    tri_vertex_sets.append(tri_vertices_0.copy())
    tri_vertex_sets.append(tri_vertices_1.copy())
    tri_vertex_sets.append(tri_vertices_2.copy())

    var blas_bounds = List[AABB](capacity=3)
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

    var cpu_tlas = Tlas[TLAS_WIDTH](instances)
    var camera = _make_camera(cpu_tlas)
    var camera_params = camera.flatten()

    var ray_count = WIDTH * HEIGHT
    var cpu_t1 = perf_counter_ns()
    print(
        t"CPU TLAS: instances={cpu_tlas.inst_count} | "
        t"rays={ray_count} | "
        t"time={round(ns_to_ms(Int(cpu_t1 - cpu_t0)), 3)} ms"
    )

    comptime if not has_accelerator():
        raise "No accelerator available; skipped GPU render."

    with DeviceContext() as ctx:
        # Warm up GPU runtime / allocator / copy path.
        var warm_t0 = perf_counter_ns()
        var warm_h = ctx.enqueue_create_host_buffer[DType.float32](1024)
        var warm_d = ctx.enqueue_create_buffer[DType.float32](1024)
        warm_h.enqueue_copy_to(warm_d)
        ctx.synchronize()
        var warm_t1 = perf_counter_ns()
        print(
            t"\nGPU warmup time ="
            t"{round(ns_to_ms(Int(warm_t1 - warm_t0)), 3)} ms "
        )

        print("\nBuilding GPU BLAS set...")
        var blas_t0 = perf_counter_ns()
        var blas_builder = GpuTriangleBlasSetBuilder[BLAS_WIDTH]()
        for i in range(len(tri_vertex_sets)):
            _ = blas_builder.add(tri_vertex_sets[i])
        var gpu_blases = blas_builder.build(ctx)
        ctx.synchronize()
        var blas_t1 = perf_counter_ns()

        print(
            t"BLAS set build:"
            t" total={round(ns_to_ms(Int(blas_t1 - blas_t0)), 3)} ms"
        )

        print("\nBuilding GPU TLAS...")
        var tlas_t0 = perf_counter_ns()
        var gpu_tlas = GpuTriangleTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        ctx.synchronize()
        var tlas_t1 = perf_counter_ns()

        print(
            t"TLAS build:"
            t" total={round(ns_to_ms(Int(tlas_t1 - tlas_t0)), 3)} ms |"
        )

        print("\nUploading camera params and tracing TLAS on GPU...")
        var setup_t0 = perf_counter_ns()
        var d_camera_params = upload_list(ctx, camera_params)

        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)
        ctx.synchronize()
        var setup_t1 = perf_counter_ns()
        print(
            t"Setup/upload: {round(ns_to_ms(Int(setup_t1 - setup_t0)), 3)} ms"
        )

        var trace_t0 = perf_counter_ns()
        gpu_tlas.launch_camera(
            ctx,
            gpu_blases,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()
        var trace_t1 = perf_counter_ns()
        var trace_ns = Int(trace_t1 - trace_t0)
        print(
            t"Trace: {round(ns_to_ms(trace_ns), 3)} ms | "
            t"{round(ns_to_mrays_per_s(trace_ns, ray_count), 3)} Mrays/s"
        )
        print_hit_counts_by_blas(WIDTH, HEIGHT, instances, d_hits_u32)

        print("\nWriting normal PPM...")
        var write_t0 = perf_counter_ns()
        write_ppm_normals_from_hits(
            DEFAULT_OUTPUT_PATH,
            WIDTH,
            HEIGHT,
            tri_vertex_sets,
            instances,
            d_hits_f32,
            d_hits_u32,
        )
        ctx.synchronize()
        var write_t1 = perf_counter_ns()
        print(t"Write: {round(ns_to_ms(Int(write_t1 - write_t0)), 3)} ms")

    print("\nDone.")
    print(t"Wrote {DEFAULT_OUTPUT_PATH}")
