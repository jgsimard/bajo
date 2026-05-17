from std.bit import count_leading_zeros
from std.atomic import Atomic
from std.math import min, max, sqrt
from std.gpu import global_idx

from bajo.core.bvh.types import RayFlat, Hit
from bajo.core.bvh.gpu.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    LBVH_SENTINEL,
)
from bajo.core.intersect import (
    intersect_ray_tri,
    intersect_ray_aabb,
    RayAabbHit,
)
from bajo.core.morton import morton3
from bajo.core.vec import Vec3f32, normalize, vmin, vmax

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
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_left_index(node_idx)])


@always_inline
def _node_right(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_right_index(node_idx)])


# -----------------------------------------------------------------------------
# GPU LBVH topology build benchmark.
# -----------------------------------------------------------------------------
@always_inline
def _common_prefix_gpu(
    keys: UnsafePointer[UInt32, MutAnyOrigin],
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


def init_lbvh_topology_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
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
    sorted_keys: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
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
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_flags: UnsafePointer[UInt32, MutAnyOrigin],
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
def _write_primary_full_result(
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_idx: Int,
    hit: Hit,
):
    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v
    hits_u32[ray_idx] = hit.prim


# -----------------------------------------------------------------------------
# GPU-generated primary rays.
# -----------------------------------------------------------------------------
@always_inline
def _make_camera_ray(
    camera_params: UnsafePointer[Float32, MutAnyOrigin],
    ray_idx: Int,
    width: Int,
    height: Int,
) -> RayFlat:
    var pixels_per_view = width * height
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx / width

    var base = view_idx * CAMERA_PARAM_STRIDE

    var o = Vec3f32.load(camera_params, base + CAMERA_ORIGIN)
    var f = Vec3f32.load(camera_params, base + CAMERA_FORWARD)
    var r = Vec3f32.load(camera_params, base + CAMERA_RIGHT)
    var u = Vec3f32.load(camera_params, base + CAMERA_UP)

    var aspect = Float32(width) / Float32(height)
    var fov_scale = Float32(0.75)

    var sx = ((Float32(px_i) + 0.5) / Float32(width)) * 2.0 - 1.0
    var sy = 1.0 - ((Float32(py_i) + 0.5) / Float32(height)) * 2.0

    var dir = f + r * (sx * aspect * fov_scale) + u * (sy * fov_scale)

    var nd = normalize(dir)

    return RayFlat(o, nd, 1.0 / nd, _gpu_inf_t)


@always_inline
def _write_camera_miss_result[
    mode: String
](
    out_f32: UnsafePointer[Float32, MutAnyOrigin],
    out_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_idx: Int,
):
    comptime if mode == TRACE_PRIMARY_FULL:
        var hit_base = ray_idx * 3
        out_f32[hit_base + 0] = _gpu_inf_t
        out_f32[hit_base + 1] = 0.0
        out_f32[hit_base + 2] = 0.0
        out_u32[ray_idx] = _gpu_miss_prim
    elif mode == TRACE_PRIMARY_T:
        out_f32[ray_idx] = _gpu_inf_t
    else:
        out_u32[ray_idx] = 0


@always_inline
def _write_camera_result[
    mode: String
](
    out_f32: UnsafePointer[Float32, MutAnyOrigin],
    out_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_idx: Int,
    hit: Hit,
):
    comptime if mode == TRACE_PRIMARY_FULL:
        _write_primary_full_result(out_f32, out_u32, ray_idx, hit)
    elif mode == TRACE_PRIMARY_T:
        out_f32[ray_idx] = hit.t
    else:
        out_u32[ray_idx] = hit.occluded
