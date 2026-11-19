from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer

from bajo.bvh.constants import (
    LBVH_SENTINEL,
)
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.bvh.gpu.utils import (
    SortedKeysValidation,
    TopologyValidation,
    RefitBoundsValidation,
)
from bajo.core import AABB


def validate_sorted_keys(
    keys: DeviceBuffer[.uint32],
    values: DeviceBuffer[.uint32],
    size: Int,
) raises -> SortedKeysValidation:
    if size < 0 or size > len(keys) or size > len(values):
        raise Error("sort validation size is outside the input buffers")
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

    var seen = List[Bool](length=size, fill=False)
    with values.map_to_host() as v:
        for i in range(size):
            var value = Int(v[i])
            if value >= size:
                values_valid = False
                if first_bad_value == -1:
                    first_bad_value = i
            elif seen[value]:
                values_valid = False
                if first_bad_value == -1:
                    first_bad_value = i
            else:
                seen[value] = True
            if i % 1024 == 0:
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
    node_meta: DeviceBuffer[.uint32],
    leaf_parent: DeviceBuffer[.uint32],
    leaf_count: Int,
) raises -> TopologyValidation:
    var ok = True
    var root_count = UInt32(0)
    var root_idx = UInt32(0xFFFFFFFF)
    var checksum = UInt64(0)
    var internal_count = leaf_count - 1

    with node_meta.map_to_host() as m:
        var records = TileTensor(m.as_span(), row_major(internal_count, 4))
        for i in range(internal_count):
            var parent = UInt32(records[i, 0])
            var left = UInt32(records[i, 1])
            var right = UInt32(records[i, 2])
            var fence = UInt32(records[i, 3])

            checksum += UInt64(parent)
            checksum += UInt64(left)
            checksum += UInt64(right)
            checksum += UInt64(fence)

            if parent == LBVH_SENTINEL:
                root_count += 1
                root_idx = UInt32(i)
            elif parent >= UInt32(internal_count):
                ok = False

            var left_is_leaf = is_leaf_ref(left)
            var left_idx = decode_ref_index(left)
            if left_is_leaf:
                if left_idx >= UInt32(leaf_count):
                    ok = False
            else:
                if left_idx >= UInt32(internal_count):
                    ok = False
                elif UInt32(records[Int(left_idx), 0]) != UInt32(i):
                    ok = False

            var right_is_leaf = is_leaf_ref(right)
            var right_idx = decode_ref_index(right)
            if right_is_leaf:
                if right_idx >= UInt32(leaf_count):
                    ok = False
            else:
                if right_idx >= UInt32(internal_count):
                    ok = False
                elif UInt32(records[Int(right_idx), 0]) != UInt32(i):
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
    node_bounds: DeviceBuffer[.float32],
    node_flags: DeviceBuffer[.uint32],
    node_meta: DeviceBuffer[.uint32],
    leaf_count: Int,
    scene_bounds: AABB,
) raises -> RefitBoundsValidation:
    var ok = True
    var internal_count = leaf_count - 1
    var root_idx = UInt32(0xFFFFFFFF)
    var checksum = UInt64(0)

    with node_meta.map_to_host() as m:
        var records = TileTensor(m.as_span(), row_major(internal_count, 4))
        for i in range(internal_count):
            var parent = UInt32(records[i, 0])
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
            var bounds = TileTensor(
                b.as_span()[: internal_count * 12],
                row_major(internal_count, 2, 6),
            )
            var root = bounds[Int(root_idx), :, :]
            var mnx = min(root[0, 0], root[1, 0])
            var mny = min(root[0, 1], root[1, 1])
            var mnz = min(root[0, 2], root[1, 2])
            var mxx = max(root[0, 3], root[1, 3])
            var mxy = max(root[0, 4], root[1, 4])
            var mxz = max(root[0, 5], root[1, 5])

            checksum += UInt64(abs(Float64(mnx)) * 1000.0)
            checksum += UInt64(abs(Float64(mny)) * 1000.0)
            checksum += UInt64(abs(Float64(mnz)) * 1000.0)
            checksum += UInt64(abs(Float64(mxx)) * 1000.0)
            checksum += UInt64(abs(Float64(mxy)) * 1000.0)
            checksum += UInt64(abs(Float64(mxz)) * 1000.0)

            diff = max(
                max(
                    max(
                        abs(Float64(mnx - scene_bounds._min.x)),
                        abs(Float64(mny - scene_bounds._min.y)),
                    ),
                    max(
                        abs(Float64(mnz - scene_bounds._min.z)),
                        abs(Float64(mxx - scene_bounds._max.x)),
                    ),
                ),
                max(
                    abs(Float64(mxy - scene_bounds._max.y)),
                    abs(Float64(mxz - scene_bounds._max.z)),
                ),
            )
    else:
        ok = False

    if diff > 1.0e-4:
        ok = False

    return RefitBoundsValidation(ok, diff, root_idx, checksum)
