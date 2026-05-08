from std.math import cos, max, round, sin
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext
from std.io.file_descriptor import FileDescriptor

from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.cpu.tlas import BvhInstance, Tlas

from bajo.core.bvh.host_utils import (
    append_camera_params,
    compute_bounds,
    copy_list_to_device,
)
from bajo.core.bvh.gpu.lbvh import GpuLBVH
from bajo.core.bvh.gpu.tlas import GpuTlasLayout
from bajo.core.bvh.gpu.tlas_traverse import (
    launch_tlas_lbvh_camera_primary,
    launch_shade_tlas_normals,
)
from bajo.core.mat import Mat44f32, inverse
from bajo.core.utils import pack_obj_triangles, ns_to_ms, ns_to_mrays_per_s
from bajo.core.vec import Vec3f32, length


comptime DEFAULT_OBJ_PATH = "./assets/buddha/buddha.obj"
comptime DEFAULT_OUTPUT_PATH = "./example_tlas_lbvh_normals.ppm"
comptime WIDTH = 1280
comptime HEIGHT = 720
comptime GRID_X = 25
comptime GRID_Z = 25


def _trs_y(
    tx: Float32, ty: Float32, tz: Float32, angle: Float32, s: Float32
) -> Mat44f32:
    var c = Float32(cos(Float64(angle)))
    var sn = Float32(sin(Float64(angle)))
    # fmt: off
    return Mat44f32(
        s * c,   0.0, s * sn,  tx,
        0.0,       s,    0.0,  ty,
        -s * sn, 0.0,  s * c,  tz,
        0.0,     0.0,    0.0, 1.0,
    )
    # fmt: off


def _make_instances(
    mut blas: BinaryBvh, bmin: Vec3f32, bmax: Vec3f32
) raises -> List[BvhInstance]:
    var extent = bmax - bmin
    var spacing = max(max(extent.x(), extent.y()), extent.z()) * 2.25
    if spacing < 1.0:
        spacing = 1.0

    var instances = List[BvhInstance](capacity=GRID_X * GRID_Z)
    for z in range(GRID_Z):
        for x in range(GRID_X):
            var idx = z * GRID_X + x
            var tx = Float32(x - GRID_X / 2) * spacing
            var tz = Float32(z - GRID_Z / 2) * spacing
            var angle = Float32(idx) * 0.35
            var scale = Float32(3.85) + Float32(idx % 5) * 0.075
            var transform = _trs_y(tx, 0.0, tz, angle, scale)
            var inv_transform = inverse(transform)
            instances.append(
                BvhInstance.from_blas(transform, inv_transform, 0, blas)
            )
    return instances^


def _make_camera_params(tlas: Tlas) -> List[Float32]:
    ref root = tlas.tlas_nodes[0]
    var center = (root.aabb._min + root.aabb._max) * 0.5
    var extent = root.aabb._max - root.aabb._min

    var scene_w = max(extent.x(), extent.z())
    if scene_w < 1.0:
        scene_w = 1.0

    # Lower and closer, like a crowd-level shot.
    var eye = center + Vec3f32(
        scene_w * 0.18,
        extent.y() * 0.22,
        -scene_w * 0.55,
    )

    # Look slightly into the group, around upper-body / head height.
    var target = center + Vec3f32(
        0.0,
        extent.y() * 0.18,
        scene_w * 0.10,
    )

    var params = List[Float32](capacity=12)
    append_camera_params(
        params,
        eye,
        target,
        Vec3f32(0.0, 1.0, 0.0),
    )
    return params^


def write_ppm_from_packed_rgb(
    path: String,
    width: Int,
    height: Int,
    rgb: DeviceBuffer[DType.uint32],
) raises:
    var pixel_count = width * height
    var byte_count = pixel_count * 3

    with open(path, "w") as f:
        var fd = FileDescriptor(f)

        fd.write(t"P6\n{width} {height}\n255\n")
        var _bytes = List[UInt8](length=byte_count, fill=0)
        var out = _bytes.unsafe_ptr()

        with rgb.map_to_host() as pixels:
            var j = 0
            for i in range(pixel_count):
                # packed is 0x00RRGGBB
                var packed = UInt32(pixels[i])
                out[j] = UInt8((packed >> 16) & UInt32(255))
                out[j + 1] = UInt8((packed >> 8) & UInt32(255))
                out[j + 2] = UInt8(packed & UInt32(255))

                j += 3

        fd.write_bytes(_bytes)


def main() raises:
    print("GPU instanced TLAS/LBVH normal render")
    print(t"OBJ: {DEFAULT_OBJ_PATH}")
    print(t"Resolution: {WIDTH} x {HEIGHT}")
    print(t"Instances: {GRID_X * GRID_Z}")
    print(t"Output: {DEFAULT_OUTPUT_PATH}")

    print("\nLoading and packing geometry...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = len(tri_vertices) / 3
    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()

    print(t"Triangles: {tri_count}")
    print(t"Load time: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)} ms")

    print("\nBuilding CPU BLAS/TLAS oracle layout...")
    var cpu_t0 = perf_counter_ns()
    var cpu_blas = BinaryBvh(tri_vertices.unsafe_ptr(), UInt32(tri_count))
    cpu_blas.build["sah", False]()
    var instances = _make_instances(cpu_blas, bmin, bmax)
    var cpu_tlas = Tlas(instances)
    cpu_tlas.build()
    var camera_params = _make_camera_params(cpu_tlas)
    var cpu_t1 = perf_counter_ns()
    print(
        t"CPU TLAS: nodes={cpu_tlas.nodes_used} |"
        t" instances={cpu_tlas.inst_count} |"
        t" time={round(ns_to_ms(Int(cpu_t1 - cpu_t0)), 3)} ms"
    )

    var ray_count = WIDTH * HEIGHT

    with DeviceContext() as ctx:
        print("\nBuilding GPU BLAS...")
        var gpu_blas = GpuLBVH(ctx, tri_vertices)
        var build_t0 = perf_counter_ns()
        var build_result = gpu_blas.build_from_triangles(ctx, tri_vertices)
        var build_t = build_result[0]
        var validation = build_result[1]
        var build_t1 = perf_counter_ns()

        print(
            t"Build: morton={round(ns_to_ms(build_t.morton_ns), 3)} ms |"
            t" sort={round(ns_to_ms(build_t.sort_ns), 3)} ms |"
            t" topo={round(ns_to_ms(build_t.topology_ns), 3)} ms |"
            t" refit={round(ns_to_ms(build_t.refit_ns), 3)} ms |"
            t" total={round(ns_to_ms(Int(build_t1 - build_t0)), 3)} ms"
        )
        print(
            t"Validation: sorted={validation.sorted_ok} |"
            t" values={validation.values_ok} |"
            t" topology={validation.topology_ok} |"
            t" bounds={validation.bounds_ok} | root={validation.root_idx}"
        )

        print("\nUploading TLAS and camera...")
        var upload_t0 = perf_counter_ns()
        var gpu_tlas = GpuTlasLayout(ctx, cpu_tlas)
        var d_camera_params = copy_list_to_device(ctx, camera_params)
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)
        var d_rgb = ctx.enqueue_create_buffer[DType.uint32](ray_count)
        ctx.synchronize()
        var upload_t1 = perf_counter_ns()
        print(
            t"Upload/setup: {round(ns_to_ms(Int(upload_t1 - upload_t0)), 3)} ms"
        )

        print("\nTracing TLAS camera rays on GPU...")
        var trace_t0 = perf_counter_ns()
        launch_tlas_lbvh_camera_primary(
            ctx,
            gpu_blas,
            gpu_tlas,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            WIDTH,
            HEIGHT,
            1,
        )
        ctx.synchronize()
        var trace_t1 = perf_counter_ns()
        var trace_ns = Int(trace_t1 - trace_t0)
        print(
            t"Trace: {round(ns_to_ms(trace_ns), 3)} ms | "
            t"{round(ns_to_mrays_per_s(trace_ns, ray_count), 3)} Mrays/s"
        )

        print("\nShading normals on GPU...")
        var shade_t0 = perf_counter_ns()
        launch_shade_tlas_normals(
            ctx,
            gpu_blas,
            gpu_tlas,
            d_hits_u32,
            d_rgb,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()
        var shade_t1 = perf_counter_ns()
        print(t"Shade: {round(ns_to_ms(Int(shade_t1 - shade_t0)), 3)} ms")

        print("\nWriting PPM...")
        var write_t0 = perf_counter_ns()
        write_ppm_from_packed_rgb(DEFAULT_OUTPUT_PATH, WIDTH, HEIGHT, d_rgb)
        ctx.synchronize()
        var write_t1 = perf_counter_ns()
        print(t"Write: {round(ns_to_ms(Int(write_t1 - write_t0)), 3)} ms")

    print("\nDone.")
    print(t"Wrote {DEFAULT_OUTPUT_PATH}")
