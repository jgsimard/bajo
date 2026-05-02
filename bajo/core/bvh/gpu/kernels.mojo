from std.bit import count_leading_zeros
from std.atomic import Atomic
from std.math import abs, min, max, round, sqrt
from std.gpu import thread_idx, block_idx, block_dim, global_idx, DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.core.bvh.types import RayFlat, Hit
from bajo.core.bvh.gpu.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    LBVH_SENTINEL,
)
from bajo.core.intersect import intersect_ray_tri, intersect_ray_aabb
from bajo.core.morton import morton3
from bajo.core.vec import Vec3f32, vmin, vmax, cross, length, normalize
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace

comptime GPU_TRAVERSAL_STACK_SIZE = 64
comptime GPU_REDUCE_THREADS = 4096

comptime LBVH_NODE_META_STRIDE = 4
comptime LBVH_NODE_PARENT = 0
comptime LBVH_NODE_LEFT = 1
comptime LBVH_NODE_RIGHT = 2
comptime LBVH_NODE_FENCE = 3

comptime LBVH_NODE_BOUNDS_STRIDE = 12
comptime LBVH_BOUNDS_LEFT = 0
comptime LBVH_BOUNDS_RIGHT = 6

comptime CAMERA_PARAM_STRIDE = 12
comptime CAMERA_ORIGIN = 0
comptime CAMERA_FORWARD = 3
comptime CAMERA_RIGHT = 6
comptime CAMERA_UP = 9

comptime TRACE_PRIMARY_FULL = "primary_full"
comptime TRACE_PRIMARY_T = "primary_t"
comptime TRACE_SHADOW = "shadow"

comptime _gpu_inf_t = Float32(3.4028234663852886e38)
comptime _gpu_tri_miss_t = Float32.MAX
comptime _gpu_miss_prim = UInt32(0xFFFFFFFF)


@always_inline
def _node_meta_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * LBVH_NODE_META_STRIDE


@always_inline
def _node_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * LBVH_NODE_BOUNDS_STRIDE


@always_inline
def _node_parent_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_PARENT


@always_inline
def _node_left_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_LEFT


@always_inline
def _node_right_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_RIGHT


@always_inline
def _node_fence_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_FENCE


@always_inline
def _node_left(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_left_index(node_idx)])


@always_inline
def _node_right(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_right_index(node_idx)])


def compute_centroid_bounds(verts: List[Vec3f32]) -> Tuple[Vec3f32, Vec3f32]:
    var bmin = Vec3f32(Float32.MAX)
    var bmax = Vec3f32(Float32.MIN)

    for i in range(len(verts) / 3):
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

    keys[i] = morton3(cx, cy, cz)
    values[i] = UInt32(i)


def init_lbvh_topology_kernel(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_parent: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    internal_count: Int,
    leaf_count: Int,
):
    var i = global_idx.x

    if i < internal_count:
        var base = i * LBVH_NODE_META_STRIDE
        node_meta[base + LBVH_NODE_PARENT] = LBVH_SENTINEL  # parent
        node_meta[base + LBVH_NODE_LEFT] = 0  # left child, encoded
        node_meta[base + LBVH_NODE_RIGHT] = 0  # right child, encoded
        # fence/debug: rightmost leaf in range
        node_meta[base + LBVH_NODE_FENCE] = 0

    if i < leaf_count:
        leaf_parent[i] = LBVH_SENTINEL


def build_lbvh_topology_kernel(
    sorted_keys: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_parent: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    leaf_count: Int,
):
    var i = global_idx.x
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
            node_meta[_node_parent_index(UInt32(split))] = UInt32(i)

    var right_child = split + 1
    if right_child == last:
        right_encoded = UInt32(right_child) | LBVH_LEAF_FLAG
        if right_child >= 0 and right_child < leaf_count:
            leaf_parent[right_child] = UInt32(i)
    else:
        right_encoded = UInt32(right_child)
        if right_child >= 0 and right_child < internal_count:
            node_meta[_node_parent_index(UInt32(right_child))] = UInt32(i)

    var base = i * LBVH_NODE_META_STRIDE
    node_meta[base + LBVH_NODE_LEFT] = left_encoded
    node_meta[base + LBVH_NODE_RIGHT] = right_encoded
    node_meta[base + LBVH_NODE_FENCE] = UInt32(last)


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
    var i = global_idx.x
    if i >= internal_count:
        return

    var b = i * LBVH_NODE_BOUNDS_STRIDE
    node_bounds[b + 0] = Float32.MAX
    node_bounds[b + 1] = Float32.MAX
    node_bounds[b + 2] = Float32.MAX
    node_bounds[b + 3] = Float32.MIN
    node_bounds[b + 4] = Float32.MIN
    node_bounds[b + 5] = Float32.MIN
    node_bounds[b + 6] = Float32.MAX
    node_bounds[b + 7] = Float32.MAX
    node_bounds[b + 8] = Float32.MAX
    node_bounds[b + 9] = Float32.MIN
    node_bounds[b + 10] = Float32.MIN
    node_bounds[b + 11] = Float32.MIN
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
    var b = _node_bounds_base(parent)
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
    var b = _node_bounds_base(parent)
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
    var leaf_idx = global_idx.x
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
        var left = _node_left(node_meta, parent)
        var right = _node_right(node_meta, parent)

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
        parent = UInt32(node_meta[_node_parent_index(current_encoded)])


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
@always_inline
def _load_buffer_ray(
    rays: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ray_idx: Int,
) -> RayFlat:
    var ray_base = ray_idx * 10
    return RayFlat(
        rays[ray_base + 0],
        rays[ray_base + 1],
        rays[ray_base + 2],
        rays[ray_base + 3],
        rays[ray_base + 4],
        rays[ray_base + 5],
        rays[ray_base + 6],
        rays[ray_base + 7],
        rays[ray_base + 8],
        rays[ray_base + 9],
    )


@always_inline
def _intersect_child_bounds[
    child_bounds_offset: Int
](
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    node_idx: UInt32,
    ray: RayFlat,
    t_max: Float32,
) -> Tuple[Bool, Float32]:
    var b = _node_bounds_base(node_idx) + child_bounds_offset
    return intersect_ray_aabb(
        ray.ox,
        ray.oy,
        ray.oz,
        ray.rdx,
        ray.rdy,
        ray.rdz,
        node_bounds[b + 0],
        node_bounds[b + 1],
        node_bounds[b + 2],
        node_bounds[b + 3],
        node_bounds[b + 4],
        node_bounds[b + 5],
        t_max,
    )


@always_inline
def _trace_lbvh_ray[
    mode: String
](
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ray: RayFlat,
    root_idx: UInt32,
) -> Hit:
    comptime assert mode in [
        TRACE_PRIMARY_FULL,
        TRACE_PRIMARY_T,
        TRACE_SHADOW,
    ], "unknown GPU LBVH trace mode"

    var best_t = ray.t_max
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        if (current & LBVH_LEAF_FLAG) != 0:
            var leaf_idx = current & LBVH_INDEX_MASK
            var prim_idx = UInt32(sorted_prim_ids[Int(leaf_idx)])

            var tri_hit = intersect_ray_tri(
                vertices,
                prim_idx,
                ray.ox,
                ray.oy,
                ray.oz,
                ray.dx,
                ray.dy,
                ray.dz,
                best_t,
            )

            if tri_hit.mask[0]:
                comptime if mode == TRACE_SHADOW:
                    return Hit(
                        Float32(0.0),
                        Float32(0.0),
                        Float32(0.0),
                        prim_idx,
                        UInt32(1),
                    )
                else:
                    best_t = tri_hit.t
                    comptime if mode == TRACE_PRIMARY_FULL:
                        best_u = tri_hit.u
                        best_v = tri_hit.v
                        best_prim = prim_idx

            if stack_ptr == 0:
                break

            stack_ptr -= 1
            current = stack[stack_ptr]
            continue

        var node_idx = current & LBVH_INDEX_MASK
        var left = _node_left(node_meta, node_idx)
        var right = _node_right(node_meta, node_idx)

        var left_hit = _intersect_child_bounds[LBVH_BOUNDS_LEFT](
            node_bounds,
            node_idx,
            ray,
            best_t,
        )

        var right_hit = _intersect_child_bounds[LBVH_BOUNDS_RIGHT](
            node_bounds,
            node_idx,
            ray,
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

    return Hit(best_t, best_u, best_v, best_prim, UInt32(0))


@always_inline
def _write_primary_full_result(
    hits_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_idx: Int,
    hit: Hit,
):
    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v
    hits_u32[ray_idx] = hit.prim


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

    var ray = _load_buffer_ray(rays, ray_idx)
    var hit = _trace_lbvh_ray[TRACE_PRIMARY_FULL](
        vertices,
        sorted_prim_ids,
        node_meta,
        node_bounds,
        ray,
        root_idx,
    )
    _write_primary_full_result(hits_f32, hits_u32, ray_idx, hit)


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


@always_inline
def _make_camera_ray(
    camera_params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ray_idx: Int,
    width: Int,
    height: Int,
) -> RayFlat:
    var pixels_per_view = width * height
    var view_idx = ray_idx // pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx // width

    var cam_base = view_idx * CAMERA_PARAM_STRIDE

    var ox = camera_params[cam_base + CAMERA_ORIGIN + 0]
    var oy = camera_params[cam_base + CAMERA_ORIGIN + 1]
    var oz = camera_params[cam_base + CAMERA_ORIGIN + 2]

    var fx = camera_params[cam_base + CAMERA_FORWARD + 0]
    var fy = camera_params[cam_base + CAMERA_FORWARD + 1]
    var fz = camera_params[cam_base + CAMERA_FORWARD + 2]

    var rx = camera_params[cam_base + CAMERA_RIGHT + 0]
    var ry = camera_params[cam_base + CAMERA_RIGHT + 1]
    var rz = camera_params[cam_base + CAMERA_RIGHT + 2]

    var ux = camera_params[cam_base + CAMERA_UP + 0]
    var uy = camera_params[cam_base + CAMERA_UP + 1]
    var uz = camera_params[cam_base + CAMERA_UP + 2]

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

    return RayFlat(
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        Float32(1.0) / dx,
        Float32(1.0) / dy,
        Float32(1.0) / dz,
        _gpu_inf_t,
    )


@always_inline
def _write_camera_miss_result[
    mode: String
](
    out_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    out_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_idx: Int,
):
    comptime if mode == TRACE_PRIMARY_FULL:
        var hit_base = ray_idx * 3
        out_f32[hit_base + 0] = _gpu_inf_t
        out_f32[hit_base + 1] = Float32(0.0)
        out_f32[hit_base + 2] = Float32(0.0)
        out_u32[ray_idx] = _gpu_miss_prim
    elif mode == TRACE_PRIMARY_T:
        out_f32[ray_idx] = _gpu_inf_t
    else:
        out_u32[ray_idx] = UInt32(0)


@always_inline
def _write_camera_result[
    mode: String
](
    out_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    out_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_idx: Int,
    hit: Hit,
):
    comptime if mode == TRACE_PRIMARY_FULL:
        _write_primary_full_result(out_f32, out_u32, ray_idx, hit)
    elif mode == TRACE_PRIMARY_T:
        out_f32[ray_idx] = hit.t
    else:
        out_u32[ray_idx] = hit.occluded


# -----------------------------------------------------------------------------
# GPU-generated rays with on-device reductions.
# -----------------------------------------------------------------------------
def trace_lbvh_gpu_camera_kernel[
    mode: String
](
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    camera_params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    # Generic outputs:
    # - primary_full: out_f32 = hits_f32, out_u32 = hits_u32
    # - primary_t:    out_f32 = hit_t,    out_u32 unused
    # - shadow:       out_f32 unused,     out_u32 = occluded
    out_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    out_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
    views: Int,
    root_idx: UInt32,
):
    comptime assert mode in [
        TRACE_PRIMARY_FULL,
        TRACE_PRIMARY_T,
        TRACE_SHADOW,
    ], "unknown GPU LBVH camera trace mode"

    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx // pixels_per_view

    if view_idx >= views:
        _write_camera_miss_result[mode](out_f32, out_u32, ray_idx)
        return

    var ray = _make_camera_ray(camera_params, ray_idx, width, height)
    var hit = _trace_lbvh_ray[mode](
        vertices,
        sorted_prim_ids,
        node_meta,
        node_bounds,
        ray,
        root_idx,
    )
    _write_camera_result[mode](out_f32, out_u32, ray_idx, hit)


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
    var sum = 0.0
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
