from std.math import cos, max, round, sin, clamp
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext
from std.io.file_descriptor import FileDescriptor

from bajo.core.aabb import AABB
from bajo.bvh.cpu.tlas import Tlas
from bajo.bvh.gpu.tlas import GpuTlas
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.bvh.host_utils import (
    append_camera_params,
    compute_bounds,
    copy_list_to_device,
)
from bajo.bvh.types import Instance
from bajo.core.mat import Mat44f32, inverse, transform_vector
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core.vec import Vec3f32, cross, normalize
from bajo.obj.pack import pack_obj_triangles


comptime DEFAULT_OBJ_PATH = "./assets/buddha/buddha.obj"
comptime DEFAULT_OUTPUT_PATH = "./example_tlas_lbvh_normals.ppm"
comptime WIDTH = 1280
comptime HEIGHT = 720
comptime GRID_X = 25
comptime GRID_Z = 25
comptime BLAS_WIDTH = 2
comptime TLAS_WIDTH = 2
comptime MISS_PRIM = UInt32(0xFFFFFFFF)


def _trs_y(
    tx: Float32, ty: Float32, tz: Float32, angle: Float32, s: Float32
) -> Mat44f32:
    var c = cos(angle)
    var sn = sin(angle)
    # fmt: off
    return Mat44f32(
        s * c,   0.0, s * sn,  tx,
        0.0,       s,    0.0,  ty,
        -s * sn, 0.0,  s * c,  tz,
        0.0,     0.0,    0.0, 1.0,
    )
    # fmt: on


def _make_instances(bounds: AABB) raises -> List[Instance]:
    var extent = bounds.extent()
    var base_extent = max(max(extent.x, extent.y), extent.z)

    comptime BASE_SCALE = Float32(8.0)

    var spacing = base_extent * BASE_SCALE * 1.35
    if spacing < 1.0:
        spacing = 1.0

    var instances = List[Instance](capacity=GRID_X * GRID_Z)
    for z in range(GRID_Z):
        for x in range(GRID_X):
            var idx = z * GRID_X + x
            var tx = (Float32(x - GRID_X - 1) / 2) * spacing
            var tz = (Float32(z - GRID_Z - 1) / 2) * spacing
            var angle = Float32(idx) * 1.0
            var scale = BASE_SCALE + Float32(idx % 5) * 0.075
            var transform = _trs_y(tx, 0.0, tz, angle, scale)
            var inv_transform = inverse(transform)
            instances.append(Instance(transform, inv_transform, 0, bounds))
    return instances^


def _instances_bounds(instances: List[Instance]) -> AABB:
    var out = AABB.invalid()
    for instance in instances:
        out.grow(instance.bounds)
    return out


def _make_camera_params(tlas: Tlas[TLAS_WIDTH]) -> List[Float32]:
    var bounds = tlas.bounds()
    var center = bounds.centroid()
    var extent = bounds.extent()

    var scene_w = max(extent.x, extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var eye = center + Vec3f32(
        scene_w * 0.18,
        extent.y * 0.22,
        -scene_w * 0.25,
    )

    var target = center + Vec3f32(
        0.0,
        extent.y * 0.18,
        scene_w * 0.10,
    )
    var fov = Float32(0.75)

    var params = List[Float32](capacity=12)
    append_camera_params(
        params,
        eye,
        target,
        Vec3f32(0.0, 1.0, 0.0),
        fov,
    )
    return params^


@always_inline
def _unit_to_u8(x: Float32) -> UInt8:
    return UInt8(clamp(x, 0.0, 1.0) * 255.0)


def write_ppm_normals_from_hits(
    path: String,
    width: Int,
    height: Int,
    tri_vertices: List[Vec3f32],
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

        with hits_f32.map_to_host() as hf:
            with hits_u32.map_to_host() as hu:
                var j = 0
                for i in range(pixel_count):
                    var prim = UInt32(hu[i * 2 + 0])
                    var inst = UInt32(hu[i * 2 + 1])
                    var t = hf[i * 3 + 0]

                    if prim == MISS_PRIM or inst == MISS_PRIM or t >= 1.0e20:
                        out[j + 0] = UInt8(18)
                        out[j + 1] = UInt8(22)
                        out[j + 2] = UInt8(30)
                    else:
                        var base = Int(prim) * 3
                        ref v0 = tri_vertices[base + 0]
                        ref v1 = tri_vertices[base + 1]
                        ref v2 = tri_vertices[base + 2]

                        var local_n = normalize(cross(v1 - v0, v2 - v0))
                        var world_n = normalize(
                            transform_vector(
                                instances[Int(inst)].transform,
                                local_n,
                            )
                        )

                        out[j + 0] = _unit_to_u8(world_n.x[0] * 0.5 + 0.5)
                        out[j + 1] = _unit_to_u8(world_n.y[0] * 0.5 + 0.5)
                        out[j + 2] = _unit_to_u8(world_n.z[0] * 0.5 + 0.5)

                    j += 3

        fd.write_bytes(_bytes)


def main() raises:
    print("GPU instanced TLAS normal render")
    print(t"OBJ: {DEFAULT_OBJ_PATH}")
    print(t"Resolution: {WIDTH} x {HEIGHT}")
    print(t"Instances: {GRID_X * GRID_Z}")
    print(t"BLAS width: {BLAS_WIDTH}")
    print(t"TLAS width: {TLAS_WIDTH}")
    print(t"Output: {DEFAULT_OUTPUT_PATH}")

    print("\nLoading and packing geometry...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = len(tri_vertices) / 3
    var blas_bounds = compute_bounds(tri_vertices)

    print(t"Triangles: {tri_count}")
    print(t"Load time: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)} ms")

    print("\nBuilding CPU TLAS layout for instance bounds + camera...")
    var cpu_t0 = perf_counter_ns()
    var instances = _make_instances(blas_bounds)
    var cpu_tlas = Tlas[TLAS_WIDTH](instances)
    var tlas_bounds = _instances_bounds(instances)
    var camera_params = _make_camera_params(cpu_tlas)
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
        print("\nBuilding GPU BLAS...")
        var blas_t0 = perf_counter_ns()
        var gpu_blas = GpuTriangleBvh[BLAS_WIDTH](ctx, tri_vertices)
        ctx.synchronize()
        var blas_t1 = perf_counter_ns()

        var blas_validation = gpu_blas.tree.validate(blas_bounds)
        print(
            t"BLAS build:"
            t" total={round(ns_to_ms(Int(blas_t1 - blas_t0)), 3)} ms |"
            t" collapse={round(ns_to_ms(gpu_blas.tree.collapse_ns), 3)} ms"
            t" | pack={round(ns_to_ms(gpu_blas.leaf_pack_ns), 3)} ms"
        )
        print(
            t"BLAS validation: sorted={blas_validation.sorted_ok} | "
            t"values={blas_validation.values_ok} | "
            t"topology={blas_validation.topology_ok} | "
            t"bounds={blas_validation.bounds_ok} | "
            t"root={blas_validation.root_idx}"
        )

        print("\nBuilding GPU TLAS...")
        var tlas_t0 = perf_counter_ns()
        var gpu_tlas = GpuTlas[TLAS_WIDTH](ctx, instances)
        ctx.synchronize()
        var tlas_t1 = perf_counter_ns()

        var tlas_validation = gpu_tlas.tree.validate(tlas_bounds)
        print(
            t"TLAS build:"
            t" total={round(ns_to_ms(Int(tlas_t1 - tlas_t0)), 3)} ms |"
            t" collapse={round(ns_to_ms(gpu_tlas.tree.collapse_ns), 3)} ms"
        )
        print(
            t"TLAS validation: sorted={tlas_validation.sorted_ok} | "
            t"values={tlas_validation.values_ok} | "
            t"topology={tlas_validation.topology_ok} | "
            t"bounds={tlas_validation.bounds_ok} | "
            t"root={tlas_validation.root_idx}"
        )

        print("\nUploading camera params and tracing TLAS on GPU...")
        var setup_t0 = perf_counter_ns()
        var d_camera_params = copy_list_to_device[DType.float32](
            ctx,
            camera_params,
        )
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)
        ctx.synchronize()
        var setup_t1 = perf_counter_ns()
        print(
            t"Setup/upload: {round(ns_to_ms(Int(setup_t1 - setup_t0)), 3)} ms"
        )

        var trace_t0 = perf_counter_ns()
        gpu_tlas.launch_camera_primary["triangle", BLAS_WIDTH](
            ctx,
            gpu_blas.tree.wide_bounds,
            gpu_blas.tree.wide_data,
            gpu_blas.tree.wide_counts,
            gpu_blas.leaf_vertices,
            gpu_blas.leaf_prims,
            gpu_blas.tree.root_idx,
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

        print("\nWriting normal PPM...")
        var write_t0 = perf_counter_ns()
        write_ppm_normals_from_hits(
            DEFAULT_OUTPUT_PATH,
            WIDTH,
            HEIGHT,
            tri_vertices,
            instances,
            d_hits_f32,
            d_hits_u32,
        )
        ctx.synchronize()
        var write_t1 = perf_counter_ns()
        print(t"Write: {round(ns_to_ms(Int(write_t1 - write_t0)), 3)} ms")

    print("\nDone.")
    print(t"Wrote {DEFAULT_OUTPUT_PATH}")
