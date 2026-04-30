from std.benchmark import keep
from std.math import abs, min, max, round, sqrt
from std.time import perf_counter_ns
from std.memory import UnsafePointer
from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx, block_dim, DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.obj import read_obj, triangulated_indices
from bajo.core.vec import Vec3f32, vmin, vmax, cross, dot, length, normalize
from bajo.core.bvh.tinybvh import BVH, BVHGPU, Ray, WideBVH
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s


comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3
comptime TRAVERSAL_REPEATS = 8
comptime GPU_BLOCK_SIZE = 128
comptime GPU_STACK_SIZE = 64


def pack_obj_triangles(path: String) raises -> List[Vec3f32]:
    var mesh = read_obj(path)
    var idx = triangulated_indices(mesh)

    var out = List[Vec3f32](capacity=len(idx))

    for i in range(len(idx)):
        var p = Int(idx[i].p)

        # OBJ vertex indices refer to vertices, but mesh.positions is flat xyz.
        var base = p * 3

        out.append(
            Vec3f32(
                mesh.positions[base + 0],
                mesh.positions[base + 1],
                mesh.positions[base + 2],
            )
        )

    return out^


def compute_bounds(verts: List[Vec3f32]) -> Tuple[Vec3f32, Vec3f32]:
    var bmin = Vec3f32(1.0e30, 1.0e30, 1.0e30)
    var bmax = Vec3f32(-1.0e30, -1.0e30, -1.0e30)

    for i in range(len(verts)):
        bmin = vmin(bmin, verts[i])
        bmax = vmax(bmax, verts[i])

    return (bmin^, bmax^)


def append_camera_rays(
    mut rays: List[Ray],
    origin: Vec3f32,
    target: Vec3f32,
    up_hint: Vec3f32,
    width: Int,
    height: Int,
):
    var forward = normalize(target - origin)
    var right = normalize(cross(forward, up_hint))
    var up = normalize(cross(right, forward))

    var aspect = Float32(width) / Float32(height)
    var fov_scale = Float32(0.75)

    for y in range(height):
        for x in range(width):
            var sx = ((Float32(x) + 0.5) / Float32(width)) * 2.0 - 1.0
            var sy = 1.0 - ((Float32(y) + 0.5) / Float32(height)) * 2.0
            var dir = normalize(
                forward
                + right * (sx * aspect * fov_scale)
                + up * (sy * fov_scale)
            )
            rays.append(Ray(origin, dir))


def generate_primary_rays(
    bounds_min: Vec3f32,
    bounds_max: Vec3f32,
    width: Int,
    height: Int,
    views: Int,
) -> List[Ray]:
    var rays = List[Ray](capacity=width * height * views)

    var center = (bounds_min + bounds_max) * 0.5
    var extent = bounds_max - bounds_min
    var radius = length(extent) * 0.5
    if radius < 1.0:
        radius = 1.0
    var dist = radius * 2.8

    if views >= 1:
        append_camera_rays(
            rays,
            center + Vec3f32(0.0, 0.0, -dist),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            width,
            height,
        )

    if views >= 2:
        append_camera_rays(
            rays,
            center + Vec3f32(-dist, 0.0, 0.0),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            width,
            height,
        )

    if views >= 3:
        append_camera_rays(
            rays,
            center + Vec3f32(0.0, dist, 0.0),
            center,
            Vec3f32(0.0, 0.0, 1.0),
            width,
            height,
        )

    return rays^


@always_inline
def hit_t_for_checksum(t: Float32) -> Float64:
    if t < 1.0e20:
        return Float64(t)
    return 0.0


def trace_bvh_primary(bvh: BVH, rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    var hit_count = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        bvh.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1

    keep(checksum)
    keep(hit_count)
    return checksum


def trace_bvh_shadow(bvh: BVH, rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if bvh.is_occluded(ray):
            occluded += 1

    keep(occluded)
    return occluded


def trace_wide_primary[
    width: Int
](wide: WideBVH[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    var hit_count = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        wide.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1

    keep(checksum)
    keep(hit_count)
    return checksum


def trace_wide_shadow[width: Int](wide: WideBVH[width], rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if wide.is_occluded(ray):
            occluded += 1

    keep(occluded)
    return occluded


def trace_gpu_primary(gpu: BVHGPU, rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    var hit_count = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        gpu.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1

    keep(checksum)
    keep(hit_count)
    return checksum


def trace_gpu_shadow(gpu: BVHGPU, rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if gpu.is_occluded(ray):
            occluded += 1

    keep(occluded)
    return occluded


def print_build_bvh_result(
    name: String,
    ns: Int,
    nodes_used: UInt32,
    quality: Float32,
):
    var ms = round(ns_to_ms(ns), 3)
    var q = round(Float64(quality), 3)
    print(t"{name} | {ms} ms | nodes: {nodes_used} | quality: {q}")


def print_build_layout_result(
    name: String,
    ns: Int,
    nodes: Int,
    extra_label: String,
    extra_count: Int,
):
    var ms = round(ns_to_ms(ns), 3)
    print(t"{name} | {ms} ms | nodes: {nodes} | {extra_label}: {extra_count}")


def print_gpu_layout_result(
    ns: Int,
    nodes: Int,
    prims: Int,
    root_is_leaf: Bool,
):
    var ms = round(ns_to_ms(ns), 3)
    print(
        t"gpu layout       | {ms} ms | nodes: {nodes} | prims: {prims} | root"
        t" leaf: {root_is_leaf}"
    )


def print_traversal_result(
    name: String,
    best_ns: Int,
    ray_count: Int,
    checksum: Float64,
):
    var ms = round(ns_to_ms(best_ns), 3)
    var mrays = round(ns_to_mrays_per_s(best_ns, ray_count), 3)
    var csum = round(checksum, 3)
    print(t"{name} | {ms} ms | {mrays} MRays/s | checksum: {csum}")


def print_shadow_result(
    name: String,
    best_ns: Int,
    ray_count: Int,
    occluded: Int,
):
    var ms = round(ns_to_ms(best_ns), 3)
    var mrays = round(ns_to_mrays_per_s(best_ns, ray_count), 3)
    print(t"{name} | {ms} ms | {mrays} MRays/s | occluded: {occluded}")


def print_primary_validation(
    name: String,
    reference_checksum: Float64,
    checksum: Float64,
):
    var diff = round(abs(checksum - reference_checksum), 3)
    if diff <= 0.001:
        print(t"{name} primary validation: OK | diff: {diff}")
    else:
        print(t"{name} primary validation: MISMATCH | diff: {diff}")


def print_shadow_validation(
    name: String,
    reference_occluded: Int,
    occluded: Int,
):
    if occluded == reference_occluded:
        print(t"{name} shadow validation:  OK | occluded: {occluded}")
    else:
        print(
            t"{name} shadow validation:  MISMATCH | ref: {reference_occluded} |"
            t" got: {occluded}"
        )


def bench_bvh_primary(name: String, bvh: BVH, rays: List[Ray], repeats: Int):
    # Warmup.
    var checksum = trace_bvh_primary(bvh, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_bvh_primary(bvh, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_bvh_shadow(name: String, bvh: BVH, rays: List[Ray], repeats: Int):
    # Warmup.
    var occluded = trace_bvh_shadow(bvh, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        occluded = trace_bvh_shadow(bvh, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_shadow_result(name, best_ns, len(rays), occluded)


def bench_wide_primary[
    width: Int
](name: String, wide: WideBVH[width], rays: List[Ray], repeats: Int):
    # Warmup.
    var checksum = trace_wide_primary[width](wide, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_wide_primary[width](wide, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_wide_shadow[
    width: Int
](name: String, wide: WideBVH[width], rays: List[Ray], repeats: Int):
    # Warmup.
    var occluded = trace_wide_shadow[width](wide, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        occluded = trace_wide_shadow[width](wide, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_shadow_result(name, best_ns, len(rays), occluded)


def bench_gpu_primary(name: String, gpu: BVHGPU, rays: List[Ray], repeats: Int):
    # Warmup.
    var checksum = trace_gpu_primary(gpu, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_gpu_primary(gpu, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_gpu_shadow(name: String, gpu: BVHGPU, rays: List[Ray], repeats: Int):
    # Warmup.
    var occluded = trace_gpu_shadow(gpu, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        occluded = trace_gpu_shadow(gpu, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_shadow_result(name, best_ns, len(rays), occluded)


# -----------------------------------------------------------------------------
# GPU primary traversal, first version.
#
# TinyBVH's OpenCL path stores BVH_GPU as four float4 rows per node:
#   lmin + left, lmax + right, rmin + triCount, rmax + firstTri.
# Mojo's DeviceBuffer API is scalar-typed, so this benchmark flattens that same
# Aila-Laine layout into two GPU buffers:
#   node_bounds: 12 Float32 values per node: lmin, lmax, rmin, rmax.
#   node_meta:    4 UInt32 values per node: left, right, triCount, firstTri.
# -----------------------------------------------------------------------------


@always_inline
def _axis_t_near(o: Float32, rd: Float32, mn: Float32, mx: Float32) -> Float32:
    var t0 = (mn - o) * rd
    var t1 = (mx - o) * rd
    return min(t0, t1)


@always_inline
def _axis_t_far(o: Float32, rd: Float32, mn: Float32, mx: Float32) -> Float32:
    var t0 = (mn - o) * rd
    var t1 = (mx - o) * rd
    return max(t0, t1)


@always_inline
def _intersect_aabb_flat(
    ox: Float32,
    oy: Float32,
    oz: Float32,
    rdx: Float32,
    rdy: Float32,
    rdz: Float32,
    bminx: Float32,
    bminy: Float32,
    bminz: Float32,
    bmaxx: Float32,
    bmaxy: Float32,
    bmaxz: Float32,
    t_max: Float32,
) -> Tuple[Bool, Float32]:
    var tx1 = _axis_t_near(ox, rdx, bminx, bmaxx)
    var tx2 = _axis_t_far(ox, rdx, bminx, bmaxx)
    var ty1 = _axis_t_near(oy, rdy, bminy, bmaxy)
    var ty2 = _axis_t_far(oy, rdy, bminy, bmaxy)
    var tz1 = _axis_t_near(oz, rdz, bminz, bmaxz)
    var tz2 = _axis_t_far(oz, rdz, bminz, bmaxz)

    var tmin = max(max(tx1, ty1), max(tz1, Float32(0.0)))
    var tmax = min(min(tx2, ty2), min(tz2, t_max))
    return (tmin <= tmax, tmin)


@always_inline
def _intersect_tri_flat(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    prim_idx: UInt32,
    ox: Float32,
    oy: Float32,
    oz: Float32,
    dx: Float32,
    dy: Float32,
    dz: Float32,
    t_max: Float32,
) -> Tuple[Bool, Float32, Float32, Float32]:
    var base = Int(prim_idx) * 9
    var v0x = vertices[base + 0]
    var v0y = vertices[base + 1]
    var v0z = vertices[base + 2]
    var v1x = vertices[base + 3]
    var v1y = vertices[base + 4]
    var v1z = vertices[base + 5]
    var v2x = vertices[base + 6]
    var v2y = vertices[base + 7]
    var v2z = vertices[base + 8]

    var e1x = v1x - v0x
    var e1y = v1y - v0y
    var e1z = v1z - v0z
    var e2x = v2x - v0x
    var e2y = v2y - v0y
    var e2z = v2z - v0z

    var px = dy * e2z - dz * e2y
    var py = dz * e2x - dx * e2z
    var pz = dx * e2y - dy * e2x
    var det = e1x * px + e1y * py + e1z * pz

    if det > -1e-12 and det < 1e-12:
        return (False, Float32(1e30), Float32(0.0), Float32(0.0))

    var inv_det = Float32(1.0) / det
    var tx = ox - v0x
    var ty = oy - v0y
    var tz = oz - v0z

    var u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return (False, Float32(1e30), Float32(0.0), Float32(0.0))

    var qx = ty * e1z - tz * e1y
    var qy = tz * e1x - tx * e1z
    var qz = tx * e1y - ty * e1x

    var v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return (False, Float32(1e30), Float32(0.0), Float32(0.0))

    var t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    if t > 1e-4 and t < t_max:
        return (True, t, u, v)

    return (False, Float32(1e30), Float32(0.0), Float32(0.0))


def trace_bvh_gpu_primary_kernel(
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    prim_indices: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rays: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var ray_base = ray_idx * 10
    var ox = rays[ray_base + 0]
    var oy = rays[ray_base + 1]
    var oz = rays[ray_base + 2]
    var dx = rays[ray_base + 3]
    var dy = rays[ray_base + 4]
    var dz = rays[ray_base + 5]
    var rdx = rays[ray_base + 6]
    var rdy = rays[ray_base + 7]
    var rdz = rays[ray_base + 8]
    var best_t = rays[ray_base + 9]
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = UInt32(0xFFFFFFFF)

    var stack = InlineArray[UInt32, GPU_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var node_idx = UInt32(0)

    while True:
        var meta_base = Int(node_idx) * 4
        var bounds_base = Int(node_idx) * 12

        var left = node_meta[meta_base + 0]
        var right = node_meta[meta_base + 1]
        var tri_count = node_meta[meta_base + 2]
        var first_tri = node_meta[meta_base + 3]

        if tri_count > 0:
            for j in range(Int(tri_count)):
                var prim_idx = prim_indices[Int(first_tri) + j]
                var tri_hit = _intersect_tri_flat(
                    vertices,
                    prim_idx,
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    best_t,
                )
                if tri_hit[0]:
                    best_t = tri_hit[1]
                    best_u = tri_hit[2]
                    best_v = tri_hit[3]
                    best_prim = prim_idx

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            continue

        var left_hit = _intersect_aabb_flat(
            ox,
            oy,
            oz,
            rdx,
            rdy,
            rdz,
            node_bounds[bounds_base + 0],
            node_bounds[bounds_base + 1],
            node_bounds[bounds_base + 2],
            node_bounds[bounds_base + 3],
            node_bounds[bounds_base + 4],
            node_bounds[bounds_base + 5],
            best_t,
        )
        var right_hit = _intersect_aabb_flat(
            ox,
            oy,
            oz,
            rdx,
            rdy,
            rdz,
            node_bounds[bounds_base + 6],
            node_bounds[bounds_base + 7],
            node_bounds[bounds_base + 8],
            node_bounds[bounds_base + 9],
            node_bounds[bounds_base + 10],
            node_bounds[bounds_base + 11],
            best_t,
        )

        var hit_left = left_hit[0]
        var hit_right = right_hit[0]
        var dist_left = left_hit[1]
        var dist_right = right_hit[1]

        if not hit_left and not hit_right:
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
        elif hit_left and not hit_right:
            node_idx = left
        elif not hit_left and hit_right:
            node_idx = right
        else:
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left
            if stack_ptr < GPU_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1
            node_idx = near

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = best_t
    hits_f32[hit_base + 1] = best_u
    hits_f32[hit_base + 2] = best_v
    hits_u32[ray_idx] = best_prim


def trace_bvh_gpu_shadow_kernel(
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    prim_indices: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rays: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    occluded_out: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var ray_base = ray_idx * 10
    var ox = rays[ray_base + 0]
    var oy = rays[ray_base + 1]
    var oz = rays[ray_base + 2]
    var dx = rays[ray_base + 3]
    var dy = rays[ray_base + 4]
    var dz = rays[ray_base + 5]
    var rdx = rays[ray_base + 6]
    var rdy = rays[ray_base + 7]
    var rdz = rays[ray_base + 8]
    var t_max = rays[ray_base + 9]

    var stack = InlineArray[UInt32, GPU_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var node_idx = UInt32(0)

    while True:
        var meta_base = Int(node_idx) * 4
        var bounds_base = Int(node_idx) * 12

        var left = node_meta[meta_base + 0]
        var right = node_meta[meta_base + 1]
        var tri_count = node_meta[meta_base + 2]
        var first_tri = node_meta[meta_base + 3]

        if tri_count > 0:
            for j in range(Int(tri_count)):
                var prim_idx = prim_indices[Int(first_tri) + j]
                var tri_hit = _intersect_tri_flat(
                    vertices,
                    prim_idx,
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    t_max,
                )
                if tri_hit[0]:
                    occluded_out[ray_idx] = UInt32(1)
                    return

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            continue

        var left_hit = _intersect_aabb_flat(
            ox,
            oy,
            oz,
            rdx,
            rdy,
            rdz,
            node_bounds[bounds_base + 0],
            node_bounds[bounds_base + 1],
            node_bounds[bounds_base + 2],
            node_bounds[bounds_base + 3],
            node_bounds[bounds_base + 4],
            node_bounds[bounds_base + 5],
            t_max,
        )
        var right_hit = _intersect_aabb_flat(
            ox,
            oy,
            oz,
            rdx,
            rdy,
            rdz,
            node_bounds[bounds_base + 6],
            node_bounds[bounds_base + 7],
            node_bounds[bounds_base + 8],
            node_bounds[bounds_base + 9],
            node_bounds[bounds_base + 10],
            node_bounds[bounds_base + 11],
            t_max,
        )

        var hit_left = left_hit[0]
        var hit_right = right_hit[0]
        var dist_left = left_hit[1]
        var dist_right = right_hit[1]

        if not hit_left and not hit_right:
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
        elif hit_left and not hit_right:
            node_idx = left
        elif not hit_left and hit_right:
            node_idx = right
        else:
            # For shadow rays, traversal order only affects time-to-first-hit.
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left
            if stack_ptr < GPU_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1
            node_idx = near

    occluded_out[ray_idx] = UInt32(0)


def flatten_gpu_node_bounds(gpu: BVHGPU) -> List[Float32]:
    var out = List[Float32](capacity=len(gpu.nodes) * 12)
    for i in range(len(gpu.nodes)):
        ref n = gpu.nodes[i]
        out.append(n.lmin.x())
        out.append(n.lmin.y())
        out.append(n.lmin.z())
        out.append(n.lmax.x())
        out.append(n.lmax.y())
        out.append(n.lmax.z())
        out.append(n.rmin.x())
        out.append(n.rmin.y())
        out.append(n.rmin.z())
        out.append(n.rmax.x())
        out.append(n.rmax.y())
        out.append(n.rmax.z())
    return out^


def flatten_gpu_node_meta(gpu: BVHGPU) -> List[UInt32]:
    var out = List[UInt32](capacity=len(gpu.nodes) * 4)
    for i in range(len(gpu.nodes)):
        ref n = gpu.nodes[i]
        out.append(n.left)
        out.append(n.right)
        out.append(n.triCount)
        out.append(n.firstTri)
    return out^


def flatten_vertices(verts: List[Vec3f32]) -> List[Float32]:
    var out = List[Float32](capacity=len(verts) * 3)
    for i in range(len(verts)):
        out.append(verts[i].x())
        out.append(verts[i].y())
        out.append(verts[i].z())
    return out^


def flatten_rays(rays: List[Ray]) -> List[Float32]:
    var out = List[Float32](capacity=len(rays) * 10)
    for i in range(len(rays)):
        ref r = rays[i]
        out.append(r.O.x())
        out.append(r.O.y())
        out.append(r.O.z())
        out.append(r.D.x())
        out.append(r.D.y())
        out.append(r.D.z())
        out.append(r.rD.x())
        out.append(r.rD.y())
        out.append(r.rD.z())
        out.append(r.hit.t)
    return out^


def copy_f32_list_to_device(
    mut ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](len(values))
    with buf.map_to_host() as h:
        for i in range(len(values)):
            h[i] = values[i]
    return buf^


def copy_u32_list_to_device(
    mut ctx: DeviceContext, values: List[UInt32]
) raises -> DeviceBuffer[DType.uint32]:
    var buf = ctx.enqueue_create_buffer[DType.uint32](len(values))
    with buf.map_to_host() as h:
        for i in range(len(values)):
            h[i] = values[i]
    return buf^


def trace_gpu_primary_device(
    gpu: BVHGPU,
    tri_vertices: List[Vec3f32],
    rays: List[Ray],
    repeats: Int,
) raises -> Tuple[Float64, Int, Int, Int, Int, Int]:
    # Static scene data: upload once and keep resident for the whole measured batch.
    var node_bounds = flatten_gpu_node_bounds(gpu)
    var node_meta = flatten_gpu_node_meta(gpu)
    var vertices = flatten_vertices(tri_vertices)
    var rays_flat = flatten_rays(rays)

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_node_bounds = copy_f32_list_to_device(ctx, node_bounds)
        var d_node_meta = copy_u32_list_to_device(ctx, node_meta)
        var d_prims = copy_u32_list_to_device(ctx, gpu.prim_indices)
        var d_vertices = copy_f32_list_to_device(ctx, vertices)
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        # Per-frame buffers. Allocate once; each frame only rewrites rays and reads hits.
        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))
        ctx.synchronize()

        var blocks = (len(rays) + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE

        # Warmup: upload rays once and run the kernel once.
        with d_rays.map_to_host() as h:
            for i in range(len(rays_flat)):
                h[i] = rays_flat[i]
        ctx.synchronize()
        ctx.enqueue_function[
            trace_bvh_gpu_primary_kernel, trace_bvh_gpu_primary_kernel
        ](
            d_node_bounds.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_prims.unsafe_ptr(),
            d_vertices.unsafe_ptr(),
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            len(rays),
            grid_dim=blocks,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()

        var best_frame_ns = Int(9223372036854775807)
        var best_ray_upload_ns = Int(9223372036854775807)
        var best_kernel_ns = Int(9223372036854775807)
        var best_download_ns = Int(9223372036854775807)
        var checksum = Float64(0.0)
        var hit_count = 0

        for _ in range(repeats):
            var frame_t0 = perf_counter_ns()

            var upload_t0 = perf_counter_ns()
            with d_rays.map_to_host() as h:
                for i in range(len(rays_flat)):
                    h[i] = rays_flat[i]
            ctx.synchronize()
            var upload_t1 = perf_counter_ns()
            var upload_ns = Int(upload_t1 - upload_t0)
            if upload_ns < best_ray_upload_ns:
                best_ray_upload_ns = upload_ns

            var k0 = perf_counter_ns()
            ctx.enqueue_function[
                trace_bvh_gpu_primary_kernel, trace_bvh_gpu_primary_kernel
            ](
                d_node_bounds.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_prims.unsafe_ptr(),
                d_vertices.unsafe_ptr(),
                d_rays.unsafe_ptr(),
                d_hits_f32.unsafe_ptr(),
                d_hits_u32.unsafe_ptr(),
                len(rays),
                grid_dim=blocks,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var k1 = perf_counter_ns()
            var kernel_ns = Int(k1 - k0)
            if kernel_ns < best_kernel_ns:
                best_kernel_ns = kernel_ns

            var download_t0 = perf_counter_ns()
            checksum = Float64(0.0)
            hit_count = 0
            with d_hits_f32.map_to_host() as h:
                for i in range(len(rays)):
                    var t = h[i * 3]
                    if t < 1.0e20:
                        checksum += Float64(t)
                        hit_count += 1
            ctx.synchronize()
            var download_t1 = perf_counter_ns()
            var download_ns = Int(download_t1 - download_t0)
            if download_ns < best_download_ns:
                best_download_ns = download_ns

            var frame_t1 = perf_counter_ns()
            var frame_ns = Int(frame_t1 - frame_t0)
            if frame_ns < best_frame_ns:
                best_frame_ns = frame_ns

        keep(hit_count)
        keep(checksum)
        keep(len(node_bounds))
        keep(len(node_meta))
        return (
            checksum,
            Int(static_t1 - static_t0),
            best_ray_upload_ns,
            best_kernel_ns,
            best_download_ns,
            best_frame_ns,
        )


def trace_gpu_shadow_device(
    gpu: BVHGPU,
    tri_vertices: List[Vec3f32],
    rays: List[Ray],
    repeats: Int,
) raises -> Tuple[Int, Int, Int, Int, Int, Int]:
    # Static scene data: upload once and keep resident for the whole measured batch.
    var node_bounds = flatten_gpu_node_bounds(gpu)
    var node_meta = flatten_gpu_node_meta(gpu)
    var vertices = flatten_vertices(tri_vertices)
    var rays_flat = flatten_rays(rays)

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_node_bounds = copy_f32_list_to_device(ctx, node_bounds)
        var d_node_meta = copy_u32_list_to_device(ctx, node_meta)
        var d_prims = copy_u32_list_to_device(ctx, gpu.prim_indices)
        var d_vertices = copy_f32_list_to_device(ctx, vertices)
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        # Per-frame buffers. Allocate once; each frame only rewrites rays and reads occlusion flags.
        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_occluded = ctx.enqueue_create_buffer[DType.uint32](len(rays))
        ctx.synchronize()

        var blocks = (len(rays) + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE

        # Warmup: upload rays once and run the kernel once.
        with d_rays.map_to_host() as h:
            for i in range(len(rays_flat)):
                h[i] = rays_flat[i]
        ctx.synchronize()
        ctx.enqueue_function[
            trace_bvh_gpu_shadow_kernel, trace_bvh_gpu_shadow_kernel
        ](
            d_node_bounds.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_prims.unsafe_ptr(),
            d_vertices.unsafe_ptr(),
            d_rays.unsafe_ptr(),
            d_occluded.unsafe_ptr(),
            len(rays),
            grid_dim=blocks,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()

        var best_frame_ns = Int(9223372036854775807)
        var best_ray_upload_ns = Int(9223372036854775807)
        var best_kernel_ns = Int(9223372036854775807)
        var best_download_ns = Int(9223372036854775807)
        var occluded = 0

        for _ in range(repeats):
            var frame_t0 = perf_counter_ns()

            var upload_t0 = perf_counter_ns()
            with d_rays.map_to_host() as h:
                for i in range(len(rays_flat)):
                    h[i] = rays_flat[i]
            ctx.synchronize()
            var upload_t1 = perf_counter_ns()
            var upload_ns = Int(upload_t1 - upload_t0)
            if upload_ns < best_ray_upload_ns:
                best_ray_upload_ns = upload_ns

            var k0 = perf_counter_ns()
            ctx.enqueue_function[
                trace_bvh_gpu_shadow_kernel, trace_bvh_gpu_shadow_kernel
            ](
                d_node_bounds.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_prims.unsafe_ptr(),
                d_vertices.unsafe_ptr(),
                d_rays.unsafe_ptr(),
                d_occluded.unsafe_ptr(),
                len(rays),
                grid_dim=blocks,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var k1 = perf_counter_ns()
            var kernel_ns = Int(k1 - k0)
            if kernel_ns < best_kernel_ns:
                best_kernel_ns = kernel_ns

            var download_t0 = perf_counter_ns()
            occluded = 0
            with d_occluded.map_to_host() as h:
                for i in range(len(rays)):
                    if h[i] != UInt32(0):
                        occluded += 1
            ctx.synchronize()
            var download_t1 = perf_counter_ns()
            var download_ns = Int(download_t1 - download_t0)
            if download_ns < best_download_ns:
                best_download_ns = download_ns

            var frame_t1 = perf_counter_ns()
            var frame_ns = Int(frame_t1 - frame_t0)
            if frame_ns < best_frame_ns:
                best_frame_ns = frame_ns

        keep(occluded)
        keep(len(node_bounds))
        keep(len(node_meta))
        return (
            occluded,
            Int(static_t1 - static_t0),
            best_ray_upload_ns,
            best_kernel_ns,
            best_download_ns,
            best_frame_ns,
        )


def main() raises:
    print("TinyBVH Mojo bunny speedtest")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Image rays: {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x {PRIMARY_VIEWS}")
    print(t"Traversal repeats: {TRAVERSAL_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = UInt32(len(tri_vertices) // 3)
    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var load_ms = round(ns_to_ms(Int(load_t1 - load_t0)), 3)

    print(t"Packed vertices: {len(tri_vertices)}")
    print(t"Triangles: {tri_count}")
    print(t"Load+pack ms: {load_ms}")
    print_vec3_rounded("Bounds min:", bmin)
    print_vec3_rounded("Bounds max:", bmax)

    print("\nGenerating rays...")
    var rays = generate_primary_rays(
        bmin, bmax, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    print(t"Rays: {len(rays)}")

    print("\nBuild")
    print("-----")

    var t0 = perf_counter_ns()
    var bvh_median = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_median.build["median", False]()
    var t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary median ST",
        Int(t1 - t0),
        bvh_median.nodes_used,
        bvh_median.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah.build["sah", False]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary sah ST   ",
        Int(t1 - t0),
        bvh_sah.nodes_used,
        bvh_sah.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah_mt = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah_mt.build["sah", True]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary sah MT   ",
        Int(t1 - t0),
        bvh_sah_mt.nodes_used,
        bvh_sah_mt.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_lbvh = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_lbvh.build["lbvh", False]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary lbvh ST  ",
        Int(t1 - t0),
        bvh_lbvh.nodes_used,
        bvh_lbvh.tree_quality(),
    )

    t0 = perf_counter_ns()
    var wide4 = WideBVH[4](bvh_sah)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide4 collapse  ",
        Int(t1 - t0),
        len(wide4.nodes),
        "leaves",
        len(wide4.leaves),
    )

    t0 = perf_counter_ns()
    var wide8 = WideBVH[8](bvh_sah)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide8 collapse  ",
        Int(t1 - t0),
        len(wide8.nodes),
        "leaves",
        len(wide8.leaves),
    )

    t0 = perf_counter_ns()
    var wide8_lbvh = WideBVH[8](bvh_lbvh)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide8 lbvh     ",
        Int(t1 - t0),
        len(wide8_lbvh.nodes),
        "leaves",
        len(wide8_lbvh.leaves),
    )

    t0 = perf_counter_ns()
    var gpu = BVHGPU(bvh_sah)
    t1 = perf_counter_ns()
    var gpu_root_is_leaf = False
    if len(gpu.nodes) > 0:
        gpu_root_is_leaf = gpu.nodes[0].is_leaf()
    print_gpu_layout_result(
        Int(t1 - t0), len(gpu.nodes), len(gpu.prim_indices), gpu_root_is_leaf
    )

    t0 = perf_counter_ns()
    var gpu_lbvh = BVHGPU(bvh_lbvh)
    t1 = perf_counter_ns()
    var gpu_lbvh_root_is_leaf = False
    if len(gpu_lbvh.nodes) > 0:
        gpu_lbvh_root_is_leaf = gpu_lbvh.nodes[0].is_leaf()
    print_gpu_layout_result(
        Int(t1 - t0),
        len(gpu_lbvh.nodes),
        len(gpu_lbvh.prim_indices),
        gpu_lbvh_root_is_leaf,
    )

    print("\nValidation")
    print("----------")
    var ref_checksum = trace_bvh_primary(bvh_sah, rays)
    var ref_occluded = trace_bvh_shadow(bvh_sah, rays)
    print_primary_validation(
        "binary median", ref_checksum, trace_bvh_primary(bvh_median, rays)
    )
    print_primary_validation(
        "binary sah MT", ref_checksum, trace_bvh_primary(bvh_sah_mt, rays)
    )
    print_primary_validation(
        "binary lbvh", ref_checksum, trace_bvh_primary(bvh_lbvh, rays)
    )
    print_primary_validation(
        "wide4", ref_checksum, trace_wide_primary[4](wide4, rays)
    )
    print_primary_validation(
        "wide8", ref_checksum, trace_wide_primary[8](wide8, rays)
    )
    print_primary_validation(
        "wide8 lbvh", ref_checksum, trace_wide_primary[8](wide8_lbvh, rays)
    )
    print_primary_validation(
        "gpu layout CPU", ref_checksum, trace_gpu_primary(gpu, rays)
    )
    print_primary_validation(
        "gpu lbvh CPU", ref_checksum, trace_gpu_primary(gpu_lbvh, rays)
    )
    print_shadow_validation(
        "binary median", ref_occluded, trace_bvh_shadow(bvh_median, rays)
    )
    print_shadow_validation(
        "binary sah MT", ref_occluded, trace_bvh_shadow(bvh_sah_mt, rays)
    )
    print_shadow_validation(
        "binary lbvh", ref_occluded, trace_bvh_shadow(bvh_lbvh, rays)
    )
    print_shadow_validation(
        "wide4", ref_occluded, trace_wide_shadow[4](wide4, rays)
    )
    print_shadow_validation(
        "wide8", ref_occluded, trace_wide_shadow[8](wide8, rays)
    )
    print_shadow_validation(
        "wide8 lbvh", ref_occluded, trace_wide_shadow[8](wide8_lbvh, rays)
    )
    print_shadow_validation(
        "gpu layout CPU", ref_occluded, trace_gpu_shadow(gpu, rays)
    )
    print_shadow_validation(
        "gpu lbvh CPU", ref_occluded, trace_gpu_shadow(gpu_lbvh, rays)
    )

    print("\nPrimary traversal")
    print("-----------------")
    bench_bvh_primary("binary median ", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah ST ", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah MT ", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary lbvh   ", bvh_lbvh, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[4]("wide4         ", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[8]("wide8         ", wide8, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[8]("wide8 lbvh    ", wide8_lbvh, rays, TRAVERSAL_REPEATS)
    bench_gpu_primary("gpu layout CPU", gpu, rays, TRAVERSAL_REPEATS)
    bench_gpu_primary("gpu lbvh CPU  ", gpu_lbvh, rays, TRAVERSAL_REPEATS)

    print("\nShadow traversal")
    print("----------------")
    bench_bvh_shadow("binary median ", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah ST ", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah MT ", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary lbvh   ", bvh_lbvh, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[4]("wide4         ", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[8]("wide8         ", wide8, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[8]("wide8 lbvh    ", wide8_lbvh, rays, TRAVERSAL_REPEATS)
    bench_gpu_shadow("gpu layout CPU", gpu, rays, TRAVERSAL_REPEATS)
    bench_gpu_shadow("gpu lbvh CPU  ", gpu_lbvh, rays, TRAVERSAL_REPEATS)

    print("\nGPU primary traversal")
    print("---------------------")
    comptime if has_accelerator():
        var gpu_result = trace_gpu_primary_device(
            gpu, tri_vertices, rays, TRAVERSAL_REPEATS
        )
        var gpu_checksum = gpu_result[0]
        var static_upload_ns = gpu_result[1]
        var ray_upload_ns = gpu_result[2]
        var kernel_ns = gpu_result[3]
        var download_ns = gpu_result[4]
        var frame_ns = gpu_result[5]

        var static_upload_ms = round(ns_to_ms(static_upload_ns), 3)
        var ray_upload_ms = round(ns_to_ms(ray_upload_ns), 3)
        var kernel_ms = round(ns_to_ms(kernel_ns), 3)
        var download_ms = round(ns_to_ms(download_ns), 3)
        var frame_ms = round(ns_to_ms(frame_ns), 3)
        var kernel_mrays = round(ns_to_mrays_per_s(kernel_ns, len(rays)), 3)
        var frame_mrays = round(ns_to_mrays_per_s(frame_ns, len(rays)), 3)
        var diff = round(abs(gpu_checksum - ref_checksum), 3)

        print(t"gpu layout GPU primary validation: diff {diff}")
        print(t"gpu static upload once: {static_upload_ms} ms")
        print(t"gpu frame ray upload:  {ray_upload_ms} ms")
        print(t"gpu frame kernel:      {kernel_ms} ms | {kernel_mrays} MRays/s")
        print(t"gpu frame download:    {download_ms} ms")
        print(
            t"gpu frame total:       {frame_ms} ms | {frame_mrays} MRays/s |"
            t" checksum: {round(gpu_checksum, 3)}"
        )

        print("\nGPU shadow traversal")
        print("--------------------")
        var gpu_shadow_result = trace_gpu_shadow_device(
            gpu, tri_vertices, rays, TRAVERSAL_REPEATS
        )
        var gpu_occluded = gpu_shadow_result[0]
        var shadow_static_upload_ns = gpu_shadow_result[1]
        var shadow_ray_upload_ns = gpu_shadow_result[2]
        var shadow_kernel_ns = gpu_shadow_result[3]
        var shadow_download_ns = gpu_shadow_result[4]
        var shadow_frame_ns = gpu_shadow_result[5]

        var shadow_static_upload_ms = round(
            ns_to_ms(shadow_static_upload_ns), 3
        )
        var shadow_ray_upload_ms = round(ns_to_ms(shadow_ray_upload_ns), 3)
        var shadow_kernel_ms = round(ns_to_ms(shadow_kernel_ns), 3)
        var shadow_download_ms = round(ns_to_ms(shadow_download_ns), 3)
        var shadow_frame_ms = round(ns_to_ms(shadow_frame_ns), 3)
        var shadow_kernel_mrays = round(
            ns_to_mrays_per_s(shadow_kernel_ns, len(rays)), 3
        )
        var shadow_frame_mrays = round(
            ns_to_mrays_per_s(shadow_frame_ns, len(rays)), 3
        )

        if gpu_occluded == ref_occluded:
            print(
                t"gpu layout GPU shadow validation: OK | occluded:"
                t" {gpu_occluded}"
            )
        else:
            print(
                t"gpu layout GPU shadow validation: MISMATCH | ref:"
                t" {ref_occluded} | got: {gpu_occluded}"
            )
        print(t"gpu static upload once: {shadow_static_upload_ms} ms")
        print(t"gpu frame ray upload:  {shadow_ray_upload_ms} ms")
        print(
            t"gpu frame kernel:      {shadow_kernel_ms} ms |"
            t" {shadow_kernel_mrays} MRays/s"
        )
        print(t"gpu frame download:    {shadow_download_ms} ms")
        print(
            t"gpu frame total:       {shadow_frame_ms} ms |"
            t" {shadow_frame_mrays} MRays/s | occluded: {gpu_occluded}"
        )
    else:
        print("No compatible GPU found; skipped Mojo GPU kernel.")

    # Keep external vertex buffer alive until the end: BVH stores an UnsafePointer to it.
    keep(len(tri_vertices))
    keep(len(rays))
