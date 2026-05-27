from std.io.file_descriptor import FileDescriptor
from std.gpu import DeviceBuffer, DeviceContext, HostBuffer
from std.math import max, round, clamp
from std.sys import has_accelerator
from std.time import perf_counter_ns

from bajo.core.aabb import AABB
from bajo.core.quat import Quat
from bajo.core.transform import Affine3
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core.vec import Vec3f32, cross, normalize
from bajo.bvh.cpu.tlas import Tlas
from bajo.bvh.gpu.tlas import GpuTlas
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Instance
from bajo.obj.pack import pack_obj_triangles
from bajo.bvh.constants import Primitive
from bajo.bvh.camera import Camera
from bajo.bvh.gpu.utils import upload_vertices, upload_list

comptime DEFAULT_OBJ_PATH = "./assets/buddha/buddha.obj"
comptime DEFAULT_OUTPUT_PATH = "./example_tlas_lbvh_normals.ppm"
comptime WIDTH = 1280
comptime HEIGHT = 720
comptime GRID_X = 25
comptime GRID_Z = 25
comptime BLAS_WIDTH = 2
comptime TLAS_WIDTH = 2
comptime MISS_PRIM = UInt32(0xFFFFFFFF)


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
            var rotation = Quat.from_axis_angle(Vec3f32(0, 1, 0), angle)
            var scale = Vec3f32(BASE_SCALE + Float32(idx % 5) * 0.075)
            var translation = Vec3f32(tx, 0.0, tz)
            var transform = Affine3.from_rotation_scale_translation(
                rotation, scale, translation
            )
            var inv_transform = transform.inverse().inv.copy()
            instances.append(
                Instance(
                    transform,
                    inv_transform,
                    0,
                    bounds,
                    Primitive.TRIANGLE,
                )
            )
    return instances^


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
    var camera = Camera(
        eye,
        target,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.75),
    )
    return camera.flatten()


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

        with hits_f32.map_to_host() as hf, hits_u32.map_to_host() as hu:
            for i in range(pixel_count):
                var prim = UInt32(hu[i * 2 + 0])
                var inst = UInt32(hu[i * 2 + 1])
                if prim == MISS_PRIM or inst == MISS_PRIM:
                    out[i * 3 + 0] = UInt8(18)
                    out[i * 3 + 1] = UInt8(22)
                    out[i * 3 + 2] = UInt8(30)
                else:
                    var base = Int(prim) * 3
                    ref v0 = tri_vertices[base + 0]
                    ref v1 = tri_vertices[base + 1]
                    ref v2 = tri_vertices[base + 2]

                    var local_n = normalize(cross(v1 - v0, v2 - v0))
                    var world_n = normalize(
                        instances[Int(inst)].transform.transform_vector(
                            local_n,
                        )
                    )

                    out[i * 3 + 0] = _unit_to_u8(world_n.x[0] * 0.5 + 0.5)
                    out[i * 3 + 1] = _unit_to_u8(world_n.y[0] * 0.5 + 0.5)
                    out[i * 3 + 2] = _unit_to_u8(world_n.z[0] * 0.5 + 0.5)

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

        print("\nUploading vertices...")
        var up_t0 = perf_counter_ns()
        var vertices = upload_vertices(ctx, tri_vertices)
        ctx.synchronize()
        var up_t1 = perf_counter_ns()
        print(t"Upload time ={round(ns_to_ms(Int(up_t1 - up_t0)), 3)} ms ")

        print("\nBuilding GPU BLAS...")
        var blas_t0 = perf_counter_ns()

        var gpu_blas = GpuTriangleBvh[BLAS_WIDTH](
            ctx, vertices, len(tri_vertices) / 3
        )
        ctx.synchronize()
        var blas_t1 = perf_counter_ns()

        print(
            t"BLAS build:"
            t" total={round(ns_to_ms(Int(blas_t1 - blas_t0)), 3)} ms |"
            t" collapse={round(ns_to_ms(gpu_blas.timings.collapse_ns), 3)} ms"
            t" | pack={round(ns_to_ms(gpu_blas.leaf_pack_ns), 3)} ms"
        )

        print("\nBuilding GPU TLAS...")
        var tlas_t0 = perf_counter_ns()
        var gpu_tlas = GpuTlas[TLAS_WIDTH](ctx, instances)
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
        gpu_tlas.launch_camera["triangle", BLAS_WIDTH](
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
