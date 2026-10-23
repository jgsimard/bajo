"""Bounded refinement of colliding Morton codes.

Each tile refines equal-code runs using their local centroid bounds. Runs
crossing tiles are refined independently. Original coarse codes stay sorted,
segment boundaries stay intact, and no device allocations are required.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier
from std.memory import stack_allocation
from std.math import min
from bajo.bvh.constants import MORTON_REFINEMENT_MIN_RUN
from bajo.core import AABB, Point3f32
from bajo.core.morton import morton3
from .binary_layout import _segment_for_item


def refine_morton_tiles_kernel[
    tile_size: Int, segmented: Bool
](
    leaf_bounds: ImmSpan[Float32, ImmutAnyOrigin],
    coarse_codes: ImmSpan[UInt32, ImmutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_ids: MutSpan[UInt32, MutAnyOrigin],
):
    comptime assert tile_size == 128 or tile_size == 256
    var lane = Int(thread_idx.x)
    var first = Int(block_idx.x) * tile_size
    var count = min(tile_size, len(leaf_ids) - first)
    var coarse = stack_allocation[
        tile_size, UInt64, address_space=AddressSpace.SHARED
    ]()
    var keys = stack_allocation[
        tile_size, UInt64, address_space=AddressSpace.SHARED
    ]()
    var ids = stack_allocation[
        tile_size, UInt32, address_space=AddressSpace.SHARED
    ]()
    var centers = stack_allocation[
        tile_size * 3, Float32, address_space=AddressSpace.SHARED
    ]()
    var code = UInt64.MAX
    var item = UInt32(0)
    if lane < count:
        code = UInt64(coarse_codes.unsafe_get(first + lane))
        comptime if segmented:
            code |= (
                UInt64(_segment_for_item(segment_offsets, first + lane)) << 32
            )
        item = leaf_ids.unsafe_get(first + lane)
    coarse[unsafe_offset=lane] = code
    keys[unsafe_offset=lane] = (UInt64(lane) << 40) | UInt64(
        lane
    ) if lane < count else UInt64.MAX
    ids[unsafe_offset=lane] = item
    barrier()
    var head = lane < count
    if lane > 0 and head:
        head = code != coarse[unsafe_offset=lane - 1]
    var end = lane + 1
    if head:
        while end < count and coarse[unsafe_offset=end] == code:
            end += 1
    var eligible = head and end - lane >= MORTON_REFINEMENT_MIN_RUN
    if block.sum[block_size=tile_size, broadcast=True](UInt32(eligible)) == 0:
        return
    if lane < count:
        var box = AABB[.WORLD].load6(leaf_bounds, Int(item) * AABB.STRIDE)
        var center = box.centroid()
        comptime for axis in range(3):
            centers[unsafe_offset=axis * tile_size + lane] = center[axis]
    barrier()
    if eligible:
        var bounds = AABB[.WORLD].invalid()
        for i in range(lane, end):
            bounds.grow(
                Point3f32[.WORLD](
                    centers[unsafe_offset=i],
                    centers[unsafe_offset=tile_size + i],
                    centers[unsafe_offset=2 * tile_size + i],
                )
            )
        var inverse = bounds.extent().safe_inv()
        for i in range(lane, end):
            var center = Point3f32[.WORLD](
                centers[unsafe_offset=i],
                centers[unsafe_offset=tile_size + i],
                centers[unsafe_offset=2 * tile_size + i],
            )
            var c = (center - bounds._min) * inverse
            var local = morton3(c.x, c.y, c.z)
            keys[unsafe_offset=i] = (
                (UInt64(lane) << 40) | (UInt64(local) << 8) | UInt64(i)
            )
    barrier()
    # Stable ties use the original tile lane, including identical centroids.
    var size = 2
    while size <= tile_size:
        var stride = size >> 1
        while stride > 0:
            var other = lane ^ stride
            var own_key = keys[unsafe_offset=lane]
            var other_key = keys[unsafe_offset=other]
            var own_id = ids[unsafe_offset=lane]
            var other_id = ids[unsafe_offset=other]
            var ascending = (lane & size) == 0
            var take_other = (other_key < own_key) if (
                (lane & stride) == 0
            ) == ascending else (other_key > own_key)
            barrier()
            keys[unsafe_offset=lane] = other_key if take_other else own_key
            ids[unsafe_offset=lane] = other_id if take_other else own_id
            barrier()
            stride >>= 1
        size <<= 1
    if lane < count:
        leaf_ids.unsafe_get(first + lane) = ids[unsafe_offset=lane]
