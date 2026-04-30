from std.benchmark import keep
from std.bit import count_leading_zeros
from std.atomic import Atomic
from std.math import abs, min, max, round
from std.memory import UnsafePointer
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, block_dim, DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.obj import read_obj, triangulated_indices
from bajo.core.morton import morton3
from bajo.core.vec import Vec3f32, vmin, vmax
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace


comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime GPU_BLOCK_SIZE = 128
comptime BENCH_REPEATS = 8
comptime LBVH_LEAF_FLAG = UInt32(0x80000000)
comptime LBVH_INDEX_MASK = UInt32(0x7FFFFFFF)
comptime LBVH_SENTINEL = UInt32(0xFFFFFFFF)


@always_inline
def ns_to_ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


def print_vec3_rounded(name: String, v: Vec3f32):
    var x = round(Float64(v.x()), 3)
    var y = round(Float64(v.y()), 3)
    var z = round(Float64(v.z()), 3)
    print(t"{name} ({x}, {y}, {z})")


def pack_obj_triangles(path: String) raises -> List[Vec3f32]:
    var mesh = read_obj(path)
    var idx = triangulated_indices(mesh)

    var out = List[Vec3f32](capacity=len(idx))

    for i in range(len(idx)):
        var p = Int(idx[i].p)
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


def flatten_vertices(verts: List[Vec3f32]) -> List[Float32]:
    var out = List[Float32](capacity=len(verts) * 3)
    for i in range(len(verts)):
        out.append(verts[i].x())
        out.append(verts[i].y())
        out.append(verts[i].z())
    return out^


def copy_f32_list_to_device(
    mut ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](len(values))
    with buf.map_to_host() as h:
        for i in range(len(values)):
            h[i] = values[i]
    return buf^


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
def _clz32(v: UInt32) -> Int:
    if v == 0:
        return 32
    return Int(count_leading_zeros(v))


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
        return _clz32(a ^ b)

    # Tie-break equal Morton codes with the sorted leaf index. This makes the
    # prefix order total and keeps degenerate duplicate-code cases deterministic.
    var x = UInt32(i) ^ UInt32(j)
    if x == 0:
        return 64
    return 32 + _clz32(x)


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


def validate_sorted_keys(
    keys: DeviceBuffer[DType.uint32],
    values: DeviceBuffer[DType.uint32],
    size: Int,
) raises -> Tuple[Bool, Bool, Int, Int, UInt32, UInt32, UInt64]:
    var keys_sorted = True
    var values_valid = True
    var first_bad_key = -1
    var first_bad_value = -1
    var first_code = UInt32(0)
    var last_code = UInt32(0)
    var checksum = UInt64(0)

    with keys.map_to_host() as k:
        if size > 0:
            first_code = k[0]
            last_code = k[size - 1]

        for i in range(1, size):
            if k[i - 1] > k[i]:
                keys_sorted = False
                if first_bad_key == -1:
                    first_bad_key = i

        # Lightweight checksum so the readback cannot be optimized away.
        for i in range(0, size, 1024):
            checksum += UInt64(k[i])

    with values.map_to_host() as v:
        for i in range(0, size, 1024):
            if v[i] >= UInt32(size):
                values_valid = False
                if first_bad_value == -1:
                    first_bad_value = i
            checksum += UInt64(v[i])

    keep(checksum)
    return (
        keys_sorted,
        values_valid,
        first_bad_key,
        first_bad_value,
        first_code,
        last_code,
        checksum,
    )


def print_values_window(
    label: String,
    values: DeviceBuffer[DType.uint32],
    center: Int,
    size: Int,
) raises:
    if center < 0:
        return

    var lo = center - 4
    if lo < 0:
        lo = 0
    var hi = center + 5
    if hi > size:
        hi = size

    print(t"{label} value window around {center}:")
    with values.map_to_host() as v:
        for i in range(lo, hi):
            print(t"  v[{i}] = {v[i]}")


def validate_topology(
    node_meta: DeviceBuffer[DType.uint32],
    leaf_parent: DeviceBuffer[DType.uint32],
    leaf_count: Int,
) raises -> Tuple[Bool, Int, UInt32, UInt64]:
    var ok = True
    var root_count = 0
    var root_idx = UInt32(0xFFFFFFFF)
    var checksum = UInt64(0)
    var internal_count = leaf_count - 1

    with node_meta.map_to_host() as m:
        for i in range(internal_count):
            var base = i * 4
            var parent = UInt32(m[base + 0])
            var left = UInt32(m[base + 1])
            var right = UInt32(m[base + 2])
            var fence = UInt32(m[base + 3])

            checksum += UInt64(parent)
            checksum += UInt64(left)
            checksum += UInt64(right)
            checksum += UInt64(fence)

            if parent == LBVH_SENTINEL:
                root_count += 1
                root_idx = UInt32(i)
            elif parent >= UInt32(internal_count):
                ok = False

            var left_is_leaf = (left & LBVH_LEAF_FLAG) != 0
            var left_idx = left & LBVH_INDEX_MASK
            if left_is_leaf:
                if left_idx >= UInt32(leaf_count):
                    ok = False
            else:
                if left_idx >= UInt32(internal_count):
                    ok = False
                elif UInt32(m[Int(left_idx) * 4 + 0]) != UInt32(i):
                    ok = False

            var right_is_leaf = (right & LBVH_LEAF_FLAG) != 0
            var right_idx = right & LBVH_INDEX_MASK
            if right_is_leaf:
                if right_idx >= UInt32(leaf_count):
                    ok = False
            else:
                if right_idx >= UInt32(internal_count):
                    ok = False
                elif UInt32(m[Int(right_idx) * 4 + 0]) != UInt32(i):
                    ok = False

    with leaf_parent.map_to_host() as p:
        for i in range(leaf_count):
            var parent = UInt32(p[i])
            checksum += UInt64(parent)
            if parent == LBVH_SENTINEL or parent >= UInt32(internal_count):
                ok = False

    if root_count != 1:
        ok = False

    keep(checksum)
    return (ok, root_count, root_idx, checksum)


def validate_refit_bounds(
    node_bounds: DeviceBuffer[DType.float32],
    node_flags: DeviceBuffer[DType.uint32],
    node_meta: DeviceBuffer[DType.uint32],
    leaf_count: Int,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
) raises -> Tuple[Bool, Float64, UInt32, UInt64]:
    var ok = True
    var internal_count = leaf_count - 1
    var root_idx = UInt32(0xFFFFFFFF)
    var checksum = UInt64(0)

    with node_meta.map_to_host() as m:
        for i in range(internal_count):
            var parent = UInt32(m[i * 4 + 0])
            if parent == LBVH_SENTINEL:
                root_idx = UInt32(i)

    with node_flags.map_to_host() as f:
        for i in range(internal_count):
            var flag = UInt32(f[i])
            checksum += UInt64(flag)
            if flag != UInt32(2):
                ok = False

    var diff = Float64(1.0e30)
    if root_idx != UInt32(0xFFFFFFFF):
        with node_bounds.map_to_host() as b:
            var rb = Int(root_idx) * 12
            var mnx = min(Float32(b[rb + 0]), Float32(b[rb + 6]))
            var mny = min(Float32(b[rb + 1]), Float32(b[rb + 7]))
            var mnz = min(Float32(b[rb + 2]), Float32(b[rb + 8]))
            var mxx = max(Float32(b[rb + 3]), Float32(b[rb + 9]))
            var mxy = max(Float32(b[rb + 4]), Float32(b[rb + 10]))
            var mxz = max(Float32(b[rb + 5]), Float32(b[rb + 11]))

            checksum += UInt64(abs(Float64(mnx)) * 1000.0)
            checksum += UInt64(abs(Float64(mny)) * 1000.0)
            checksum += UInt64(abs(Float64(mnz)) * 1000.0)
            checksum += UInt64(abs(Float64(mxx)) * 1000.0)
            checksum += UInt64(abs(Float64(mxy)) * 1000.0)
            checksum += UInt64(abs(Float64(mxz)) * 1000.0)

            diff = max(
                max(
                    max(
                        abs(Float64(mnx - scene_min.x())),
                        abs(Float64(mny - scene_min.y())),
                    ),
                    max(
                        abs(Float64(mnz - scene_min.z())),
                        abs(Float64(mxx - scene_max.x())),
                    ),
                ),
                max(
                    abs(Float64(mxy - scene_max.y())),
                    abs(Float64(mxz - scene_max.z())),
                ),
            )
    else:
        ok = False

    if diff > 1.0e-4:
        ok = False

    keep(checksum)
    return (ok, diff, root_idx, checksum)


def run_gpu_lbvh_topology_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    repeats: Int,
) raises -> Tuple[
    Int,  # static upload ns
    Int,  # best morton ns
    Int,  # best sort ns
    Int,  # best topology ns
    Bool,  # keys sorted immediately after morton generation
    Bool,  # sampled values valid immediately after morton generation
    Int,  # first bad key index immediately after morton generation
    Int,  # first bad sampled value index immediately after morton generation
    UInt32,
    UInt32,
    UInt64,
    Bool,  # keys sorted immediately after sort
    Bool,  # sampled values valid immediately after sort
    Int,  # first bad key index immediately after sort
    Int,  # first bad sampled value index immediately after sort
    UInt32,
    UInt32,
    UInt64,
    Bool,  # keys sorted after topology
    Bool,  # sampled values valid after topology
    Int,  # first bad key index after topology
    Int,  # first bad sampled value index after topology
    UInt32,
    UInt32,
    UInt64,
    Bool,  # topology valid
    Int,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)

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
        var d_vertices = copy_f32_list_to_device(ctx, vertices)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE

        # Warmup.
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
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
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

        var best_morton_ns = Int(9223372036854775807)
        var best_sort_ns = Int(9223372036854775807)
        var best_topology_ns = Int(9223372036854775807)
        var best_total_ns = Int(9223372036854775807)

        for _ in range(repeats):
            var total_t0 = perf_counter_ns()

            var m0 = perf_counter_ns()
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
            var morton_ns = Int(m1 - m0)
            if morton_ns < best_morton_ns:
                best_morton_ns = morton_ns

            var s0 = perf_counter_ns()
            device_radix_sort_pairs[DType.uint32, DType.uint32](
                ctx, workspace, d_keys, d_values, tri_count
            )
            ctx.synchronize()
            var s1 = perf_counter_ns()
            var sort_ns = Int(s1 - s0)
            if sort_ns < best_sort_ns:
                best_sort_ns = sort_ns

            var t0 = perf_counter_ns()
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
            var topology_ns = Int(t1 - t0)
            if topology_ns < best_topology_ns:
                best_topology_ns = topology_ns

            var total_t1 = perf_counter_ns()
            var total_ns = Int(total_t1 - total_t0)
            if total_ns < best_total_ns:
                best_total_ns = total_ns

        # Validation pass: regenerate + sort, validate the sorted buffers,
        # then run topology and validate both topology and whether keys survived.
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
        var sorted_after_morton = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        if not sorted_after_morton[1]:
            print_values_window(
                "after morton", d_values, sorted_after_morton[3], tri_count
            )

        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.synchronize()
        var sorted_after_sort = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        if not sorted_after_sort[1]:
            print_values_window(
                "after sort", d_values, sorted_after_sort[3], tri_count
            )

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
        var sorted_after_topology = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        if not sorted_after_topology[1]:
            print_values_window(
                "after topology", d_values, sorted_after_topology[3], tri_count
            )
        var topo_validation = validate_topology(
            d_node_meta, d_leaf_parent, tri_count
        )

        return (
            Int(static_t1 - static_t0),
            best_morton_ns,
            best_sort_ns,
            best_topology_ns,
            sorted_after_morton[0],
            sorted_after_morton[1],
            sorted_after_morton[2],
            sorted_after_morton[3],
            sorted_after_morton[4],
            sorted_after_morton[5],
            sorted_after_morton[6],
            sorted_after_sort[0],
            sorted_after_sort[1],
            sorted_after_sort[2],
            sorted_after_sort[3],
            sorted_after_sort[4],
            sorted_after_sort[5],
            sorted_after_sort[6],
            sorted_after_topology[0],
            sorted_after_topology[1],
            sorted_after_topology[2],
            sorted_after_topology[3],
            sorted_after_topology[4],
            sorted_after_topology[5],
            sorted_after_topology[6],
            topo_validation[0],
            topo_validation[1],
            topo_validation[2],
            topo_validation[3],
        )


def run_gpu_lbvh_refit_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
    repeats: Int,
) raises -> Tuple[
    Int,
    Int,
    Int,
    Int,
    Int,
    Bool,
    Bool,
    Bool,
    Int,
    UInt32,
    Bool,
    Float64,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)

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
        var d_vertices = copy_f32_list_to_device(ctx, vertices)
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
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE

        var best_morton_ns = Int(9223372036854775807)
        var best_sort_ns = Int(9223372036854775807)
        var best_topology_ns = Int(9223372036854775807)
        var best_refit_ns = Int(9223372036854775807)

        for _ in range(repeats):
            var m0 = perf_counter_ns()
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
            var morton_ns = Int(m1 - m0)
            if morton_ns < best_morton_ns:
                best_morton_ns = morton_ns

            var s0 = perf_counter_ns()
            device_radix_sort_pairs[DType.uint32, DType.uint32](
                ctx, workspace, d_keys, d_values, tri_count
            )
            ctx.synchronize()
            var s1 = perf_counter_ns()
            var sort_ns = Int(s1 - s0)
            if sort_ns < best_sort_ns:
                best_sort_ns = sort_ns

            var t0 = perf_counter_ns()
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
            var topology_ns = Int(t1 - t0)
            if topology_ns < best_topology_ns:
                best_topology_ns = topology_ns

            var r0 = perf_counter_ns()
            ctx.enqueue_function[
                init_lbvh_bounds_kernel, init_lbvh_bounds_kernel
            ](
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
            var refit_ns = Int(r1 - r0)
            if refit_ns < best_refit_ns:
                best_refit_ns = refit_ns

        # Validation pass using the same buffers.
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
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
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
        var checksum = (
            sorted_validation[6] + topo_validation[3] + refit_validation[3]
        )
        keep(checksum)

        return (
            Int(static_t1 - static_t0),
            best_morton_ns,
            best_sort_ns,
            best_topology_ns,
            best_refit_ns,
            sorted_validation[0],
            sorted_validation[1],
            topo_validation[0],
            topo_validation[1],
            topo_validation[2],
            refit_validation[0],
            refit_validation[1],
            refit_validation[2],
            checksum,
        )


def main() raises:
    print("GPU LBVH refit benchmark")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Repeats: {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = len(tri_vertices) // 3
    var bounds = compute_bounds(tri_vertices)
    var centroid_bounds = compute_centroid_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var cmin = centroid_bounds[0].copy()
    var cmax = centroid_bounds[1].copy()

    print(t"Packed vertices: {len(tri_vertices)}")
    print(t"Triangles: {tri_count}")
    print(t"Internal nodes: {tri_count - 1}")
    print(t"Load+pack ms: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)}")
    print_vec3_rounded("Bounds min:", bmin)
    print_vec3_rounded("Bounds max:", bmax)
    print_vec3_rounded("Centroid min:", cmin)
    print_vec3_rounded("Centroid max:", cmax)

    print("\nGPU LBVH refit")
    print("---------------")
    comptime if has_accelerator():
        var res = run_gpu_lbvh_refit_benchmark(
            tri_vertices, cmin, cmax, bmin, bmax, BENCH_REPEATS
        )
        var static_upload_ns = res[0]
        var morton_ns = res[1]
        var sort_ns = res[2]
        var topology_ns = res[3]
        var refit_ns = res[4]
        var keys_sorted = res[5]
        var values_valid = res[6]
        var topology_ok = res[7]
        var root_count = res[8]
        var topology_root = res[9]
        var refit_ok = res[10]
        var root_diff = res[11]
        var refit_root = res[12]
        var checksum = res[13]
        var build_ns = morton_ns + sort_ns + topology_ns + refit_ns

        print(
            t"static upload + workspace:"
            t" {round(ns_to_ms(static_upload_ns), 3)} ms"
        )
        print(t"morton generation:        {round(ns_to_ms(morton_ns), 3)} ms")
        print(t"radix sort pairs:         {round(ns_to_ms(sort_ns), 3)} ms")
        print(t"topology build:           {round(ns_to_ms(topology_ns), 3)} ms")
        print(t"bounds refit:             {round(ns_to_ms(refit_ns), 3)} ms")
        print(t"lbvh build total:         {round(ns_to_ms(build_ns), 3)} ms")
        print(
            t"sorted ids valid:         keys={keys_sorted} |"
            t" values={values_valid}"
        )
        print(
            t"topology valid:           {topology_ok} | root_count:"
            t" {root_count} | root: {topology_root}"
        )
        print(
            t"bounds valid:             {refit_ok} | root: {refit_root} | root"
            t" diff: {round(root_diff, 6)}"
        )
        print(t"checksum:                 {checksum}")
    else:
        print("No compatible GPU found; skipped GPU LBVH refit benchmark.")

    keep(len(tri_vertices))
