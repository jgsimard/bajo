from bajo.core.bvh.gpu.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    LBVH_SENTINEL,
)
from bajo.core.bvh.gpu.utils import (
    SortedKeysValidation,
    TopologyValidation,
    RefitBoundsValidation,
)


def validate_sorted_keys(
    keys: DeviceBuffer[DType.uint32],
    values: DeviceBuffer[DType.uint32],
    size: Int,
) raises -> SortedKeysValidation:
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

    return SortedKeysValidation(
        keys_sorted,
        values_valid,
        first_bad_key,
        first_bad_value,
        first_code,
        last_code,
        checksum,
    )


def validate_topology(
    node_meta: DeviceBuffer[DType.uint32],
    leaf_parent: DeviceBuffer[DType.uint32],
    leaf_count: Int,
) raises -> TopologyValidation:
    var ok = True
    var root_count = UInt32(0)
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

    return TopologyValidation(ok, root_count, root_idx, checksum)


def validate_refit_bounds(
    node_bounds: DeviceBuffer[DType.float32],
    node_flags: DeviceBuffer[DType.uint32],
    node_meta: DeviceBuffer[DType.uint32],
    leaf_count: Int,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
) raises -> RefitBoundsValidation:
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

    return RefitBoundsValidation(ok, diff, root_idx, checksum)
