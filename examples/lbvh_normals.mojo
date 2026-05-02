from std.math import clamp, round
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from bajo.core.vec import Vec3f32, cross, normalize
from bajo.core.utils import pack_obj_triangles, ns_to_ms
from bajo.core.bvh import compute_bounds, copy_list_to_device
from bajo.core.bvh.gpu.kernels import (
    compute_centroid_bounds,
    generate_camera_params,
    append_camera_params,
    TRACE_PRIMARY_FULL,
)
from bajo.core.bvh.gpu.lbvh import GpuLBVH

comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
# comptime DEFAULT_OBJ_PATH = "./assets/powerplant/powerplant.obj"
comptime DEFAULT_OUTPUT_PATH = "./normal_render.ppm"
comptime WIDTH = 800
comptime HEIGHT = 600
comptime GPU_MISS_PRIM = UInt32.MAX
comptime GPU_INF_T = Float32.MAX


@always_inline
def _triangle_normal(
    tri_vertices: List[Vec3f32],
    prim_idx: UInt32,
) -> Vec3f32:
    var base = Int(prim_idx) * 3
    ref v0 = tri_vertices[base]
    ref v1 = tri_vertices[base + 1]
    ref v2 = tri_vertices[base + 2]
    return normalize(cross(v1 - v0, v2 - v0))


@always_inline
def _to_u8(x: Float32) -> Int:
    var y = clamp(x, 0.0, 1.0)
    return Int(round(Float64(y * 255.0)))


@always_inline
def _normal_to_rgb(n: Vec3f32) -> Tuple[Int, Int, Int]:
    var r = _to_u8(n.x() * 0.5 + 0.5)
    var g = _to_u8(n.y() * 0.5 + 0.5)
    var b = _to_u8(n.z() * 0.5 + 0.5)
    return (r, g, b)


def write_ppm(
    path: String,
    width: Int,
    height: Int,
    pixels: List[Int],
) raises:
    with open(path, "w") as f:
        str = String(t"P3\n{width} {height}\n255\n")
        for y in range(height):
            for x in range(width):
                var i = (y * width + x) * 3
                str += String(
                    t"{pixels[i + 0]} {pixels[i + 1]} {pixels[i + 2]}\n"
                )
        f.write(str)


def create_camera() -> List[Float32]:
    var params = List[Float32](capacity=12)
    var target = Vec3f32(0.0, 0.5, 0.0)
    var origin = Vec3f32(0.0, 1.0, 2.0)
    var up_hint = Vec3f32(0.0, 1.0, 0.0)
    var forward = normalize(target - origin)
    var right = normalize(cross(forward, up_hint))
    var up = normalize(cross(right, forward))

    params.append(origin.x())
    params.append(origin.y())
    params.append(origin.z())
    params.append(forward.x())
    params.append(forward.y())
    params.append(forward.z())
    params.append(right.x())
    params.append(right.y())
    params.append(right.z())
    params.append(up.x())
    params.append(up.y())
    params.append(up.z())
    return params^


def main() raises:
    print("GPU LBVH normal render example")
    print(t"OBJ: {DEFAULT_OBJ_PATH}")
    print(t"Resolution: {WIDTH} x {HEIGHT}")
    print(t"Output: {DEFAULT_OUTPUT_PATH}")

    print("\nLoading geometry...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var centroid_bounds = compute_centroid_bounds(tri_vertices)
    var cmin = centroid_bounds[0].copy()
    var cmax = centroid_bounds[1].copy()
    var camera_params = create_camera()

    var tri_count = len(tri_vertices) // 3
    var ray_count = WIDTH * HEIGHT

    print(t"Triangles: {tri_count}")
    print(t"Camera params floats: {len(camera_params)}")
    print(t"Load time: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)} ms")

    with DeviceContext() as ctx:
        print("\nBuilding GPU LBVH...")
        var lbvh = GpuLBVH(ctx, tri_vertices)
        var norm = normalize(cmax - cmin)
        var build_t = lbvh.build(ctx, cmin, norm)
        var validation = lbvh.validate(bmin, bmax)

        print(
            t"Build: morton={round(ns_to_ms(build_t.morton_ns), 3)} ms |"
            t" sort={round(ns_to_ms(build_t.sort_ns), 3)} ms |"
            t" topo={round(ns_to_ms(build_t.topology_ns), 3)} ms |"
            t" refit={round(ns_to_ms(build_t.refit_ns), 3)} ms |"
            t" total={round(ns_to_ms(build_t.total_ns), 3)} ms"
        )
        print(
            t"Validation: sorted={validation.sorted_ok} |"
            t" values={validation.values_ok} |"
            t" topology={validation.topology_ok} |"
            t" bounds={validation.bounds_ok} | root={validation.refit_root_idx}"
        )

        var d_camera_params = copy_list_to_device(ctx, camera_params)
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

        print("\nTracing camera rays...")
        var trace_t0 = perf_counter_ns()
        lbvh.launch_camera_trace[TRACE_PRIMARY_FULL](
            ctx,
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

        print(t"Trace time: {round(ns_to_ms(Int(trace_t1 - trace_t0)), 3)} ms")

        print("\nShading normals on CPU and writing PPM...")
        var pixels = List[Int](capacity=WIDTH * HEIGHT * 3)
        for _ in range(WIDTH * HEIGHT * 3):
            pixels.append(0)

        with d_hits_u32.map_to_host() as prims:
            for i in range(WIDTH * HEIGHT):
                var prim = UInt32(prims[i])
                if prim == GPU_MISS_PRIM:
                    continue

                var n = _triangle_normal(tri_vertices, prim)
                var rgb = _normal_to_rgb(n)
                var base = i * 3
                pixels[base + 0] = rgb[0]
                pixels[base + 1] = rgb[1]
                pixels[base + 2] = rgb[2]
        ctx.synchronize()

        write_ppm(DEFAULT_OUTPUT_PATH, WIDTH, HEIGHT, pixels)

    print("\nDone.")
    print(t"Wrote normal image to {DEFAULT_OUTPUT_PATH}")
