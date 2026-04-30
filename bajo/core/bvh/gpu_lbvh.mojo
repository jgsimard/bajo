from std.benchmark import keep
from std.bit import count_leading_zeros
from std.atomic import Atomic
from std.math import abs, min, max, round, sqrt
from std.memory import UnsafePointer
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, block_dim, DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.obj import read_obj, triangulated_indices
from bajo.core.morton import morton3
from bajo.core.vec import Vec3f32, vmin, vmax, cross, length, normalize
from bajo.core.bvh import flatten_vertices, copy_list_to_device
from bajo.core.bvh.cpu_bvh import BVH, Ray
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace
from bajo.core.utils import (
    ns_to_mrays_per_s,
    print_vec3_rounded,
    pack_obj_triangles,
)

comptime GPU_TRAVERSAL_STACK_SIZE = 64
comptime GPU_REDUCE_THREADS = 4096
comptime LBVH_LEAF_FLAG = UInt32(0x80000000)
comptime LBVH_INDEX_MASK = UInt32(0x7FFFFFFF)
comptime LBVH_SENTINEL = UInt32(0xFFFFFFFF)


def compute_centroid_bounds(verts: List[Vec3f32]) -> Tuple[Vec3f32, Vec3f32]:
    var bmin = Vec3f32(1.0e30, 1.0e30, 1.0e30)
    var bmax = Vec3f32(-1.0e30, -1.0e30, -1.0e30)

    for i in range(len(verts) // 3):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]
        var tri_min = vmin(vmin(v0, v1), v2)
        var tri_max = vmax(vmax(v0, v1), v2)
        var c = (tri_min + tri_max) * 0.5
        bmin = vmin(bmin, c)
        bmax = vmax(bmax, c)

    return (bmin^, bmax^)


# -----------------------------------------------------------------------------
# GPU LBVH topology build benchmark.
#
# This is the first topology-only milestone after Morton generation + sort:
#   sorted Morton keys + sorted leaf ids
#       -> Karras-style internal-node topology
#       -> parent pointers + encoded child pointers
#
# Bounds/refit are intentionally NOT built here yet. This file only proves the
# hierarchy topology generated from sorted Morton codes.
# -----------------------------------------------------------------------------


@always_inline
def _common_prefix_gpu(
    keys: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    i: Int,
    j: Int,
    n: Int,
) -> Int:
    if j < 0 or j >= n:
        return -1

    var a = UInt32(keys[i])
    var b = UInt32(keys[j])

    if a != b:
        return Int(count_leading_zeros(a ^ b))

    # Tie-break equal Morton codes with the sorted leaf index. This makes the
    # prefix order total and keeps degenerate duplicate-code cases deterministic.
    var x = UInt32(i) ^ UInt32(j)
    if x == 0:
        return 64
    return 32 + Int(count_leading_zeros(x))


def compute_morton_codes_kernel(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    keys: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    values: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tri_count: Int,
    cmin_x: Float32,
    cmin_y: Float32,
    cmin_z: Float32,
    inv_extent_x: Float32,
    inv_extent_y: Float32,
    inv_extent_z: Float32,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= tri_count:
        return

    var base = i * 9
    var v0x = vertices[base + 0]
    var v0y = vertices[base + 1]
    var v0z = vertices[base + 2]
    var v1x = vertices[base + 3]
    var v1y = vertices[base + 4]
    var v1z = vertices[base + 5]
    var v2x = vertices[base + 6]
    var v2y = vertices[base + 7]
    var v2z = vertices[base + 8]

    var bmin_x = min(min(v0x, v1x), v2x)
    var bmin_y = min(min(v0y, v1y), v2y)
    var bmin_z = min(min(v0z, v1z), v2z)
    var bmax_x = max(max(v0x, v1x), v2x)
    var bmax_y = max(max(v0y, v1y), v2y)
    var bmax_z = max(max(v0z, v1z), v2z)

    var cx = ((bmin_x + bmax_x) * 0.5 - cmin_x) * inv_extent_x
    var cy = ((bmin_y + bmax_y) * 0.5 - cmin_y) * inv_extent_y
    var cz = ((bmin_z + bmax_z) * 0.5 - cmin_z) * inv_extent_z

    var vx = SIMD[DType.float32, 1](cx)
    var vy = SIMD[DType.float32, 1](cy)
    var vz = SIMD[DType.float32, 1](cz)
    var code = morton3[1](vx, vy, vz)

    keys[i] = code[0]
    values[i] = UInt32(i)


def init_lbvh_topology_kernel(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_parent: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    internal_count: Int,
    leaf_count: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)

    if i < internal_count:
        var base = i * 4
        node_meta[base + 0] = LBVH_SENTINEL  # parent
        node_meta[base + 1] = 0  # left child, encoded
        node_meta[base + 2] = 0  # right child, encoded
        node_meta[base + 3] = 0  # fence/debug: rightmost leaf in range

    if i < leaf_count:
        leaf_parent[i] = LBVH_SENTINEL


def build_lbvh_topology_kernel(
    sorted_keys: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_parent: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_count: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var internal_count = leaf_count - 1
    if i >= internal_count:
        return

    # Determine direction of the range for this internal node.
    var d_next = _common_prefix_gpu(sorted_keys, i, i + 1, leaf_count)
    var d_prev = _common_prefix_gpu(sorted_keys, i, i - 1, leaf_count)
    var d = 1
    if d_next < d_prev:
        d = -1

    # Minimum prefix outside the range.
    var delta_min = _common_prefix_gpu(sorted_keys, i, i - d, leaf_count)

    # Find an upper bound on the range length.
    var lmax = 2
    while (
        _common_prefix_gpu(sorted_keys, i, i + lmax * d, leaf_count) > delta_min
    ):
        lmax <<= 1
        if lmax > leaf_count * 2:
            break

    # Binary search for exact range length.
    var l = 0
    var t = lmax >> 1
    while t > 0:
        if (
            _common_prefix_gpu(sorted_keys, i, i + (l + t) * d, leaf_count)
            > delta_min
        ):
            l += t
        t >>= 1

    var j = i + l * d
    var first = min(i, j)
    var last = max(i, j)

    # Find split inside [first, last].
    var node_prefix = _common_prefix_gpu(sorted_keys, first, last, leaf_count)
    var split = first
    var step = last - first
    while step > 1:
        step = (step + 1) >> 1
        var new_split = split + step
        if new_split < last:
            var split_prefix = _common_prefix_gpu(
                sorted_keys, first, new_split, leaf_count
            )
            if split_prefix > node_prefix:
                split = new_split

    var left_encoded: UInt32
    var right_encoded: UInt32

    if split == first:
        left_encoded = UInt32(split) | LBVH_LEAF_FLAG
        if split >= 0 and split < leaf_count:
            leaf_parent[split] = UInt32(i)
    else:
        left_encoded = UInt32(split)
        if split >= 0 and split < internal_count:
            node_meta[split * 4 + 0] = UInt32(i)

    var right_child = split + 1
    if right_child == last:
        right_encoded = UInt32(right_child) | LBVH_LEAF_FLAG
        if right_child >= 0 and right_child < leaf_count:
            leaf_parent[right_child] = UInt32(i)
    else:
        right_encoded = UInt32(right_child)
        if right_child >= 0 and right_child < internal_count:
            node_meta[right_child * 4 + 0] = UInt32(i)

    var base = i * 4
    node_meta[base + 1] = left_encoded
    node_meta[base + 2] = right_encoded
    node_meta[base + 3] = UInt32(last)


# -----------------------------------------------------------------------------
# GPU LBVH bounds refit.
#
# One thread starts from each sorted leaf. It writes its leaf bounds into the
# parent child slot, then uses an atomic flag to let the second arriving child
# merge and propagate the internal-node bounds upward.
# -----------------------------------------------------------------------------


def init_lbvh_bounds_kernel(
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    node_flags: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    internal_count: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= internal_count:
        return

    var b = i * 12
    node_bounds[b + 0] = Float32(1.0e30)
    node_bounds[b + 1] = Float32(1.0e30)
    node_bounds[b + 2] = Float32(1.0e30)
    node_bounds[b + 3] = Float32(-1.0e30)
    node_bounds[b + 4] = Float32(-1.0e30)
    node_bounds[b + 5] = Float32(-1.0e30)
    node_bounds[b + 6] = Float32(1.0e30)
    node_bounds[b + 7] = Float32(1.0e30)
    node_bounds[b + 8] = Float32(1.0e30)
    node_bounds[b + 9] = Float32(-1.0e30)
    node_bounds[b + 10] = Float32(-1.0e30)
    node_bounds[b + 11] = Float32(-1.0e30)
    node_flags[i] = UInt32(0)


@always_inline
def _write_child_bounds(
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    parent: UInt32,
    write_left: Bool,
    mnx: Float32,
    mny: Float32,
    mnz: Float32,
    mxx: Float32,
    mxy: Float32,
    mxz: Float32,
):
    var b = Int(parent) * 12
    if write_left:
        node_bounds[b + 0] = mnx
        node_bounds[b + 1] = mny
        node_bounds[b + 2] = mnz
        node_bounds[b + 3] = mxx
        node_bounds[b + 4] = mxy
        node_bounds[b + 5] = mxz
    else:
        node_bounds[b + 6] = mnx
        node_bounds[b + 7] = mny
        node_bounds[b + 8] = mnz
        node_bounds[b + 9] = mxx
        node_bounds[b + 10] = mxy
        node_bounds[b + 11] = mxz


@always_inline
def _load_and_union_node_bounds(
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    parent: UInt32,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32]:
    var b = Int(parent) * 12
    var mnx = min(node_bounds[b + 0], node_bounds[b + 6])
    var mny = min(node_bounds[b + 1], node_bounds[b + 7])
    var mnz = min(node_bounds[b + 2], node_bounds[b + 8])
    var mxx = max(node_bounds[b + 3], node_bounds[b + 9])
    var mxy = max(node_bounds[b + 4], node_bounds[b + 10])
    var mxz = max(node_bounds[b + 5], node_bounds[b + 11])
    return (mnx, mny, mnz, mxx, mxy, mxz)


def refit_lbvh_bounds_kernel(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_parent: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    node_flags: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_count: Int,
):
    var leaf_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if leaf_idx >= leaf_count:
        return

    var prim_idx = UInt32(sorted_prim_ids[leaf_idx])
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

    var mnx = min(min(v0x, v1x), v2x)
    var mny = min(min(v0y, v1y), v2y)
    var mnz = min(min(v0z, v1z), v2z)
    var mxx = max(max(v0x, v1x), v2x)
    var mxy = max(max(v0y, v1y), v2y)
    var mxz = max(max(v0z, v1z), v2z)

    var current_encoded = UInt32(leaf_idx) | LBVH_LEAF_FLAG
    var parent = UInt32(leaf_parent[leaf_idx])

    while parent != LBVH_SENTINEL:
        var meta_base = Int(parent) * 4
        var left = UInt32(node_meta[meta_base + 1])
        var right = UInt32(node_meta[meta_base + 2])

        var is_left = current_encoded == left
        var is_right = current_encoded == right
        if not is_left and not is_right:
            break

        _write_child_bounds(
            node_bounds, parent, is_left, mnx, mny, mnz, mxx, mxy, mxz
        )

        var old = Atomic.fetch_add(node_flags + Int(parent), UInt32(1))
        if old == UInt32(0):
            break

        var merged = _load_and_union_node_bounds(node_bounds, parent)
        mnx = merged[0]
        mny = merged[1]
        mnz = merged[2]
        mxx = merged[3]
        mxy = merged[4]
        mxz = merged[5]

        current_encoded = parent
        parent = UInt32(node_meta[Int(current_encoded) * 4 + 0])


# -----------------------------------------------------------------------------
# Direct GPU LBVH primary traversal.
#
# This traverses the internal-node-only LBVH layout directly:
#   node_meta:   parent, left child, right child, fence
#   node_bounds: left AABB + right AABB per internal node
#   values:      sorted primitive ids, one implicit leaf per triangle
#
# Child pointers use LBVH_LEAF_FLAG in the high bit to distinguish leaf vs
# internal nodes, following the same encoding used by the topology kernel.
# -----------------------------------------------------------------------------


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


def trace_lbvh_gpu_primary_kernel(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rays: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
    root_idx: UInt32,
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

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        if (current & LBVH_LEAF_FLAG) != 0:
            var leaf_idx = current & LBVH_INDEX_MASK
            var prim_idx = UInt32(sorted_prim_ids[Int(leaf_idx)])
            var tri_hit = _intersect_tri_flat(
                vertices, prim_idx, ox, oy, oz, dx, dy, dz, best_t
            )
            if tri_hit[0]:
                best_t = tri_hit[1]
                best_u = tri_hit[2]
                best_v = tri_hit[3]
                best_prim = prim_idx

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            current = stack[stack_ptr]
            continue

        var node_idx = current & LBVH_INDEX_MASK
        var meta_base = Int(node_idx) * 4
        var bounds_base = Int(node_idx) * 12
        var left = UInt32(node_meta[meta_base + 1])
        var right = UInt32(node_meta[meta_base + 2])

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
            current = stack[stack_ptr]
        elif hit_left and not hit_right:
            current = left
        elif not hit_left and hit_right:
            current = right
        else:
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left
            if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1
            current = near

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = best_t
    hits_f32[hit_base + 1] = best_u
    hits_f32[hit_base + 2] = best_v
    hits_u32[ray_idx] = best_prim


def run_gpu_lbvh_direct_traversal_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
    rays: List[Ray],
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Float64,
    Bool,
    Bool,
    Bool,
    Float64,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)
    var rays_flat = flatten_rays(rays)

    var extent = centroid_max - centroid_min
    var inv_x = Float32(0.0)
    var inv_y = Float32(0.0)
    var inv_z = Float32(0.0)
    if extent.x() > 1.0e-20:
        inv_x = 1.0 / extent.x()
    if extent.y() > 1.0e-20:
        inv_y = 1.0 / extent.y()
    if extent.z() > 1.0e-20:
        inv_z = 1.0 / extent.z()

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_vertices = copy_list_to_device(ctx, vertices)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_count * 12
        )
        var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
            internal_count
        )
        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_rays = (len(rays) + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE

        # Build and refit the LBVH once, keeping all buffers resident for traversal.
        var b0 = perf_counter_ns()
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_keys.unsafe_ptr(),
            d_values.unsafe_ptr(),
            tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            inv_x,
            inv_y,
            inv_z,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var m1 = perf_counter_ns()
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.synchronize()
        var s1 = perf_counter_ns()
        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            internal_count,
            tri_count,
            grid_dim=blocks_init,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            d_keys.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            internal_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()

        var sorted_validation = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        var topo_validation = validate_topology(
            d_node_meta, d_leaf_parent, tri_count
        )
        var refit_validation = validate_refit_bounds(
            d_node_bounds,
            d_node_flags,
            d_node_meta,
            tri_count,
            scene_min,
            scene_max,
        )
        var root_idx = refit_validation[2]

        # Warmup traversal.
        with d_rays.map_to_host() as h:
            for i in range(len(rays_flat)):
                h[i] = rays_flat[i]
        ctx.synchronize()
        ctx.enqueue_function[
            trace_lbvh_gpu_primary_kernel, trace_lbvh_gpu_primary_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            len(rays),
            root_idx,
            grid_dim=blocks_rays,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()

        var best_ray_upload_ns = Int(9223372036854775807)
        var best_traversal_ns = Int(9223372036854775807)
        var best_download_ns = Int(9223372036854775807)
        var best_frame_ns = Int(9223372036854775807)
        var checksum = Float64(0.0)
        var hit_count = 0

        for _ in range(repeats):
            var frame0 = perf_counter_ns()

            var u0 = perf_counter_ns()
            with d_rays.map_to_host() as h:
                for i in range(len(rays_flat)):
                    h[i] = rays_flat[i]
            ctx.synchronize()
            var u1 = perf_counter_ns()
            var upload_ns = Int(u1 - u0)
            if upload_ns < best_ray_upload_ns:
                best_ray_upload_ns = upload_ns

            var k0 = perf_counter_ns()
            ctx.enqueue_function[
                trace_lbvh_gpu_primary_kernel, trace_lbvh_gpu_primary_kernel
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_rays.unsafe_ptr(),
                d_hits_f32.unsafe_ptr(),
                d_hits_u32.unsafe_ptr(),
                len(rays),
                root_idx,
                grid_dim=blocks_rays,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var k1 = perf_counter_ns()
            var kernel_ns = Int(k1 - k0)
            if kernel_ns < best_traversal_ns:
                best_traversal_ns = kernel_ns

            var d0 = perf_counter_ns()
            checksum = Float64(0.0)
            hit_count = 0
            with d_hits_f32.map_to_host() as h:
                for i in range(len(rays)):
                    var t = h[i * 3]
                    if t < 1.0e20:
                        checksum += Float64(t)
                        hit_count += 1
            ctx.synchronize()
            var d1 = perf_counter_ns()
            var download_ns = Int(d1 - d0)
            if download_ns < best_download_ns:
                best_download_ns = download_ns

            var frame1 = perf_counter_ns()
            var frame_ns = Int(frame1 - frame0)
            if frame_ns < best_frame_ns:
                best_frame_ns = frame_ns

        var diff = abs(checksum - reference_checksum)
        var combined_checksum = (
            sorted_validation[6]
            + topo_validation[3]
            + refit_validation[3]
            + UInt64(hit_count)
        )
        keep(combined_checksum)
        keep(checksum)

        return (
            Int(static_t1 - static_t0),
            Int(m1 - b0),
            Int(s1 - m1),
            Int(t1 - s1),
            Int(r1 - t1),
            best_ray_upload_ns,
            best_traversal_ns,
            best_download_ns,
            best_frame_ns,
            checksum,
            sorted_validation[0] and sorted_validation[1],
            topo_validation[0],
            refit_validation[0],
            diff,
            root_idx,
            combined_checksum,
        )


# -----------------------------------------------------------------------------
# GPU-generated primary rays.
#
# This variant removes the per-frame ray upload. Instead, a tiny camera parameter
# buffer is uploaded once:
#   per view: origin.xyz, forward.xyz, right.xyz, up.xyz
# Each GPU thread maps ray_idx -> (view, x, y), generates the same pinhole ray as
# generate_primary_rays(), then traverses the LBVH directly.
# -----------------------------------------------------------------------------


def append_camera_params(
    mut params: List[Float32],
    origin: Vec3f32,
    target: Vec3f32,
    up_hint: Vec3f32,
):
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


def generate_camera_params(
    bounds_min: Vec3f32,
    bounds_max: Vec3f32,
    views: Int,
) -> List[Float32]:
    var params = List[Float32](capacity=views * 12)

    var center = (bounds_min + bounds_max) * 0.5
    var extent = bounds_max - bounds_min
    var radius = length(extent) * 0.5
    if radius < 1.0:
        radius = 1.0
    var dist = radius * 2.8

    if views >= 1:
        append_camera_params(
            params,
            center + Vec3f32(0.0, 0.0, -dist),
            center,
            Vec3f32(0.0, 1.0, 0.0),
        )

    if views >= 2:
        append_camera_params(
            params,
            center + Vec3f32(-dist, 0.0, 0.0),
            center,
            Vec3f32(0.0, 1.0, 0.0),
        )

    if views >= 3:
        append_camera_params(
            params,
            center + Vec3f32(0.0, dist, 0.0),
            center,
            Vec3f32(0.0, 0.0, 1.0),
        )

    return params^


@always_inline
def _normalize3(
    x: Float32,
    y: Float32,
    z: Float32,
) -> Tuple[Float32, Float32, Float32]:
    var len2 = x * x + y * y + z * z
    if len2 <= 1.0e-20:
        return (Float32(0.0), Float32(0.0), Float32(0.0))
    var inv_len = Float32(1.0) / sqrt(len2)
    return (x * inv_len, y * inv_len, z * inv_len)


def trace_lbvh_gpu_primary_camera_kernel(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    camera_params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
    views: Int,
    root_idx: UInt32,
):
    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx // pixels_per_view
    if view_idx >= views:
        return
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx // width

    var cam_base = view_idx * 12
    var ox = camera_params[cam_base + 0]
    var oy = camera_params[cam_base + 1]
    var oz = camera_params[cam_base + 2]
    var fx = camera_params[cam_base + 3]
    var fy = camera_params[cam_base + 4]
    var fz = camera_params[cam_base + 5]
    var rx = camera_params[cam_base + 6]
    var ry = camera_params[cam_base + 7]
    var rz = camera_params[cam_base + 8]
    var ux = camera_params[cam_base + 9]
    var uy = camera_params[cam_base + 10]
    var uz = camera_params[cam_base + 11]

    var aspect = Float32(width) / Float32(height)
    var fov_scale = Float32(0.75)
    var sx = ((Float32(px_i) + 0.5) / Float32(width)) * 2.0 - 1.0
    var sy = 1.0 - ((Float32(py_i) + 0.5) / Float32(height)) * 2.0

    var dir_x = fx + rx * (sx * aspect * fov_scale) + ux * (sy * fov_scale)
    var dir_y = fy + ry * (sx * aspect * fov_scale) + uy * (sy * fov_scale)
    var dir_z = fz + rz * (sx * aspect * fov_scale) + uz * (sy * fov_scale)
    var nd = _normalize3(dir_x, dir_y, dir_z)
    var dx = nd[0]
    var dy = nd[1]
    var dz = nd[2]
    var rdx = Float32(1.0) / dx
    var rdy = Float32(1.0) / dy
    var rdz = Float32(1.0) / dz

    var best_t = Float32(3.4028234663852886e38)
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = UInt32(0xFFFFFFFF)

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        if (current & LBVH_LEAF_FLAG) != 0:
            var leaf_idx = current & LBVH_INDEX_MASK
            var prim_idx = UInt32(sorted_prim_ids[Int(leaf_idx)])
            var tri_hit = _intersect_tri_flat(
                vertices, prim_idx, ox, oy, oz, dx, dy, dz, best_t
            )
            if tri_hit[0]:
                best_t = tri_hit[1]
                best_u = tri_hit[2]
                best_v = tri_hit[3]
                best_prim = prim_idx

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            current = stack[stack_ptr]
            continue

        var node_idx = current & LBVH_INDEX_MASK
        var meta_base = Int(node_idx) * 4
        var bounds_base = Int(node_idx) * 12
        var left = UInt32(node_meta[meta_base + 1])
        var right = UInt32(node_meta[meta_base + 2])

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
            current = stack[stack_ptr]
        elif hit_left and not hit_right:
            current = left
        elif not hit_left and hit_right:
            current = right
        else:
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left
            if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1
            current = near

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = best_t
    hits_f32[hit_base + 1] = best_u
    hits_f32[hit_base + 2] = best_v
    hits_u32[ray_idx] = best_prim


# -----------------------------------------------------------------------------
# GPU-generated rays with on-device reductions.
# -----------------------------------------------------------------------------


def trace_lbvh_gpu_primary_camera_t_kernel(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    camera_params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hit_t: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
    views: Int,
    root_idx: UInt32,
):
    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx // pixels_per_view
    if view_idx >= views:
        hit_t[ray_idx] = Float32(3.4028234663852886e38)
        return
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx // width

    var cam_base = view_idx * 12
    var ox = camera_params[cam_base + 0]
    var oy = camera_params[cam_base + 1]
    var oz = camera_params[cam_base + 2]
    var fx = camera_params[cam_base + 3]
    var fy = camera_params[cam_base + 4]
    var fz = camera_params[cam_base + 5]
    var rx = camera_params[cam_base + 6]
    var ry = camera_params[cam_base + 7]
    var rz = camera_params[cam_base + 8]
    var ux = camera_params[cam_base + 9]
    var uy = camera_params[cam_base + 10]
    var uz = camera_params[cam_base + 11]

    var aspect = Float32(width) / Float32(height)
    var fov_scale = Float32(0.75)
    var sx = ((Float32(px_i) + 0.5) / Float32(width)) * 2.0 - 1.0
    var sy = 1.0 - ((Float32(py_i) + 0.5) / Float32(height)) * 2.0
    var dir_x = fx + rx * (sx * aspect * fov_scale) + ux * (sy * fov_scale)
    var dir_y = fy + ry * (sx * aspect * fov_scale) + uy * (sy * fov_scale)
    var dir_z = fz + rz * (sx * aspect * fov_scale) + uz * (sy * fov_scale)
    var nd = _normalize3(dir_x, dir_y, dir_z)
    var dx = nd[0]
    var dy = nd[1]
    var dz = nd[2]
    var rdx = Float32(1.0) / dx
    var rdy = Float32(1.0) / dy
    var rdz = Float32(1.0) / dz

    var best_t = Float32(3.4028234663852886e38)
    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        if (current & LBVH_LEAF_FLAG) != 0:
            var leaf_idx = current & LBVH_INDEX_MASK
            var prim_idx = UInt32(sorted_prim_ids[Int(leaf_idx)])
            var tri_hit = _intersect_tri_flat(
                vertices, prim_idx, ox, oy, oz, dx, dy, dz, best_t
            )
            if tri_hit[0]:
                best_t = tri_hit[1]
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            current = stack[stack_ptr]
            continue

        var node_idx = current & LBVH_INDEX_MASK
        var meta_base = Int(node_idx) * 4
        var bounds_base = Int(node_idx) * 12
        var left = UInt32(node_meta[meta_base + 1])
        var right = UInt32(node_meta[meta_base + 2])
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
            current = stack[stack_ptr]
        elif hit_left and not hit_right:
            current = left
        elif not hit_left and hit_right:
            current = right
        else:
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left
            if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1
            current = near
    hit_t[ray_idx] = best_t


def trace_lbvh_gpu_shadow_camera_kernel(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    camera_params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    occluded: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
    views: Int,
    root_idx: UInt32,
):
    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx // pixels_per_view
    if view_idx >= views:
        occluded[ray_idx] = UInt32(0)
        return
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx // width

    var cam_base = view_idx * 12
    var ox = camera_params[cam_base + 0]
    var oy = camera_params[cam_base + 1]
    var oz = camera_params[cam_base + 2]
    var fx = camera_params[cam_base + 3]
    var fy = camera_params[cam_base + 4]
    var fz = camera_params[cam_base + 5]
    var rx = camera_params[cam_base + 6]
    var ry = camera_params[cam_base + 7]
    var rz = camera_params[cam_base + 8]
    var ux = camera_params[cam_base + 9]
    var uy = camera_params[cam_base + 10]
    var uz = camera_params[cam_base + 11]

    var aspect = Float32(width) / Float32(height)
    var fov_scale = Float32(0.75)
    var sx = ((Float32(px_i) + 0.5) / Float32(width)) * 2.0 - 1.0
    var sy = 1.0 - ((Float32(py_i) + 0.5) / Float32(height)) * 2.0
    var dir_x = fx + rx * (sx * aspect * fov_scale) + ux * (sy * fov_scale)
    var dir_y = fy + ry * (sx * aspect * fov_scale) + uy * (sy * fov_scale)
    var dir_z = fz + rz * (sx * aspect * fov_scale) + uz * (sy * fov_scale)
    var nd = _normalize3(dir_x, dir_y, dir_z)
    var dx = nd[0]
    var dy = nd[1]
    var dz = nd[2]
    var rdx = Float32(1.0) / dx
    var rdy = Float32(1.0) / dy
    var rdz = Float32(1.0) / dz
    var t_max = Float32(3.4028234663852886e38)

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx
    while True:
        if (current & LBVH_LEAF_FLAG) != 0:
            var leaf_idx = current & LBVH_INDEX_MASK
            var prim_idx = UInt32(sorted_prim_ids[Int(leaf_idx)])
            var tri_hit = _intersect_tri_flat(
                vertices, prim_idx, ox, oy, oz, dx, dy, dz, t_max
            )
            if tri_hit[0]:
                occluded[ray_idx] = UInt32(1)
                return
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            current = stack[stack_ptr]
            continue

        var node_idx = current & LBVH_INDEX_MASK
        var meta_base = Int(node_idx) * 4
        var bounds_base = Int(node_idx) * 12
        var left = UInt32(node_meta[meta_base + 1])
        var right = UInt32(node_meta[meta_base + 2])
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
            current = stack[stack_ptr]
        elif hit_left and not hit_right:
            current = left
        elif not hit_left and hit_right:
            current = right
        else:
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left
            if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1
            current = near
    occluded[ray_idx] = UInt32(0)


def reduce_hit_t_kernel(
    hit_t: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    partial_sums: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    partial_counts: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
    partial_count: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= partial_count:
        return
    var sum = Float64(0.0)
    var count = UInt32(0)
    var j = i
    while j < ray_count:
        var t = hit_t[j]
        if t < 1.0e20:
            sum += Float64(t)
            count += 1
        j += partial_count
    partial_sums[i] = sum
    partial_counts[i] = count


def reduce_u32_flags_kernel(
    flags: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    partial_counts: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
    partial_count: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= partial_count:
        return
    var count = UInt32(0)
    var j = i
    while j < ray_count:
        count += UInt32(flags[j])
        j += partial_count
    partial_counts[i] = count
