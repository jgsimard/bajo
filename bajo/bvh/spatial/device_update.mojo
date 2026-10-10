"""Allocation-free device-input updates for the preorder spatial forest.

Bounds and small mesh updates use one block; larger mesh rebuilds sort tiles
and merge their exact orders across blocks before refitting. Updates retain
every published address, need no host geometry transfer and support capture. Rebuild sorts leaf IDs by X centroid within each independent
root, then refits the retained balanced skeleton. Group membership stays fixed.
"""

from max.gpu import global_idx, thread_idx, block_idx, block_dim, grid_dim
from max.gpu.primitives import block
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.atomic import Atomic, Ordering
from std.bit import log2_floor, pop_count
from std.sys.info import is_nvidia_gpu
from std.math import min, max
from std.utils.numerics import isfinite
from bajo.bvh.constants import EMPTY_LANE

comptime UPDATE_BLOCK_SIZE = 256


@no_inline
def refit_forest[
    block_size: Int
](
    items: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var lane = Int(thread_idx.x)
    var count = len(nodes) // 2
    for node in range(lane, count, block_size):
        var item = nodes[node * 2 + 1]
        if item != EMPTY_LANE:
            comptime for axis in range(6):
                bounds[node * 6 + axis] = items[Int(item) * 6 + axis]
    barrier()
    # Each lane first merges complete subtrees in its contiguous node slice.
    # Reverse preorder makes children ready before their parent. No lane reads
    # another slice until the barrier; only slice-crossing ancestors remain.
    var chunk = (count + block_size - 1) // block_size
    # Tiny slices do not amortize the extra merge pass and barrier.
    var sliced = count > block_size * 4
    var begin = min(lane * chunk, count) if sliced else lane
    var end = min(begin + chunk, count) if sliced else count
    var step = 1 if sliced else block_size
    if sliced:
        for node in range(end - 1, begin - 1, -1):
            if (
                nodes[node * 2 + 1] == EMPTY_LANE
                and Int(nodes[node * 2]) <= end
            ):
                var left = node + 1
                var right = Int(nodes[left * 2])
                comptime for axis in range(3):
                    bounds[node * 6 + axis] = min(
                        bounds[left * 6 + axis], bounds[right * 6 + axis]
                    )
                    bounds[node * 6 + axis + 3] = max(
                        bounds[left * 6 + axis + 3],
                        bounds[right * 6 + axis + 3],
                    )
        barrier()
    # The retained median skeleton has ceil(log2(leaf_count)) height. Its
    # preorder span is 2*leaf_count-1, so span ranges identify ready levels
    # without scratch allocation or a host dependency. Rebuild retains it.
    var previous = UInt64(1)
    var limit = UInt64(3)
    while previous < UInt64(count):
        var node = begin
        while node < end:
            var escape = Int(nodes[node * 2])
            if sliced and escape <= end:
                # This entire subtree was already merged by this lane.
                node = escape
                continue
            var span = UInt64(nodes[node * 2]) - UInt64(node)
            if span > previous and span <= limit:
                var left = node + 1
                var right = Int(nodes[left * 2])
                comptime for axis in range(3):
                    bounds[node * 6 + axis] = min(
                        bounds[left * 6 + axis], bounds[right * 6 + axis]
                    )
                    bounds[node * 6 + axis + 3] = max(
                        bounds[left * 6 + axis + 3],
                        bounds[right * 6 + axis + 3],
                    )
            node += step
        barrier()
        previous = limit
        limit = limit * 2 + 1


@always_inline
def _item_less[
    cached_keys: Bool = False
](items: ImmSpan[Float32, ImmutAnyOrigin], a: UInt32, b: UInt32) -> Bool:
    # Padding sorts last. Float64 centroids preserve ordering at extreme scales.
    if a == EMPTY_LANE:
        return False
    if b == EMPTY_LANE:
        return True
    comptime if cached_keys:
        var keys = items.unsafe_ptr().unsafe_bitcast[UInt64]()
        var base = len(items) // 8 * 3
        var ka = keys.unsafe_load(base + Int(a))
        var kb = keys.unsafe_load(base + Int(b))
        return ka < kb or (ka == kb and a < b)
    var ca = Float64(items[Int(a) * 6]) + Float64(items[Int(a) * 6 + 3])
    var cb = Float64(items[Int(b) * 6]) + Float64(items[Int(b) * 6 + 3])
    return ca < cb or (ca == cb and a < b)


@no_inline
def reorder_forest[
    block_size: Int, cached_keys: Bool = False
](
    items: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    order: MutSpan[UInt32, MutAnyOrigin],
):
    var lane = Int(thread_idx.x)
    var root = 0
    while root < len(nodes) // 2:
        var end = Int(nodes[root * 2])
        var count = (end - root + 1) // 2
        var padded = 1
        while padded < count:
            padded *= 2
        # A full binary subtree has 2*count-1 nodes, enough retained scratch
        # for next-power-of-two padding, including singleton group roots.
        var leaf_base = UInt32(0)
        var perfect = (count & (count - 1)) == 0
        var height = Int(log2_floor(UInt32(count)))
        if perfect:
            # Preorder leaf i is 2*i + height - popcount(i) from the root.
            for i in range(lane, count, block_size):
                var node = root + 2 * i + height - Int(pop_count(UInt32(i)))
                order[root + i] = nodes[node * 2 + 1]
        else:
            for begin in range(root, end, block_size):
                var node = begin + lane
                var item = EMPTY_LANE
                if node < end:
                    item = nodes[node * 2 + 1]
                var valid = UInt32(item != EMPTY_LANE)
                var rank = block.prefix_sum[
                    block_size=block_size, exclusive=True
                ](valid)
                if valid != 0:
                    order[root + Int(leaf_base + rank)] = item
                leaf_base += block.sum[block_size=block_size, broadcast=True](
                    valid
                )
        for i in range(count + lane, padded, block_size):
            order[root + i] = EMPTY_LANE
        barrier()
        # Retained rebuilds often preserve centroid order. Check current inputs,
        # including ID tie breaking, before paying for the sorting network.
        var inverted = False
        for i in range(lane + 1, count, block_size):
            inverted |= _item_less[cached_keys](
                items, order[root + i], order[root + i - 1]
            )
        if (
            block.sum[block_size=block_size, broadcast=True](UInt32(inverted))
            == 0
        ):
            root = end
            continue
        var size = 2
        while size <= padded:
            var stride = size // 2
            while stride > 0:
                for i in range(lane, padded, block_size):
                    var partner = i ^ stride
                    if partner > i:
                        var a = order[root + i]
                        var b = order[root + partner]
                        var ascending = (i & size) == 0
                        if _item_less[cached_keys](
                            items, b, a
                        ) if ascending else _item_less[cached_keys](
                            items, a, b
                        ):
                            order[root + i] = b
                            order[root + partner] = a
                barrier()
                stride //= 2
            size *= 2
        leaf_base = 0
        if perfect:
            for i in range(lane, count, block_size):
                var node = root + 2 * i + height - Int(pop_count(UInt32(i)))
                nodes[node * 2 + 1] = order[root + i]
        else:
            for begin in range(root, end, block_size):
                var node = begin + lane
                var valid = UInt32(0)
                if node < end:
                    valid = UInt32(nodes[node * 2 + 1] != EMPTY_LANE)
                var rank = block.prefix_sum[
                    block_size=block_size, exclusive=True
                ](valid)
                if valid != 0:
                    nodes[node * 2 + 1] = order[root + Int(leaf_base + rank)]
                leaf_base += block.sum[block_size=block_size, broadcast=True](
                    valid
                )
        barrier()
        root = end


def reorder_mesh_kernel[
    block_size: Int
](
    items: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    order: MutSpan[UInt32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    # Prior validation covers all geometry before any published leaf ID changes.
    if status[0] == 0:
        reorder_forest[block_size, True](items, nodes, order)
    # Reordering shares scratch with parent-arrival counters. Clear it before
    # the following multi-block merge; preserve rejected-input status.
    var lane = Int(thread_idx.x)
    for i in range(lane, len(order), block_size):
        order[i] = 0


def bounds_update_kernel[
    rebuild: Bool, block_size: Int = UPDATE_BLOCK_SIZE
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    items: MutSpan[Float32, MutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    order: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    comptime assert (
        block_size >= 32
        and block_size <= 1024
        and (block_size & (block_size - 1)) == 0
    )
    if (
        Int(block_dim.x) != block_size
        or Int(block_dim.y) != 1
        or Int(block_dim.z) != 1
        or Int(grid_dim.x) != 1
        or Int(grid_dim.y) != 1
        or Int(grid_dim.z) != 1
    ):
        if global_idx.x == 0 and global_idx.y == 0 and global_idx.z == 0:
            status[0] = 1
        return
    var lane = Int(thread_idx.x)
    if lane == 0:
        status[0] = 1
    var invalid = False
    for item in range(lane, len(input) // 6, block_size):
        comptime for axis in range(3):
            var lo = input[item * 6 + axis]
            var hi = input[item * 6 + axis + 3]
            invalid |= not isfinite(lo) or not isfinite(hi) or lo > hi
    if block.sum[block_size=block_size, broadcast=True](UInt32(invalid)) != 0:
        return
    # Validation completes before any geometry or topology buffer is mutated.
    for i in range(lane, len(input), block_size):
        items[i] = input[i]
    barrier()
    comptime if rebuild:
        reorder_forest[block_size](items, nodes, order)
    refit_forest[block_size](items, bounds, nodes)
    if lane == 0:
        status[0] = 0


def mesh_update_kernel[
    rebuild: Bool, block_size: Int = UPDATE_BLOCK_SIZE
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    points: MutSpan[Float32, MutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    items: MutSpan[Float32, MutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    order: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    comptime assert (
        block_size >= 32
        and block_size <= 1024
        and (block_size & (block_size - 1)) == 0
    )
    if (
        Int(block_dim.x) != block_size
        or Int(block_dim.y) != 1
        or Int(block_dim.z) != 1
        or Int(grid_dim.x) != 1
        or Int(grid_dim.y) != 1
        or Int(grid_dim.z) != 1
    ):
        if global_idx.x == 0 and global_idx.y == 0 and global_idx.z == 0:
            status[0] = 1
        return
    var lane = Int(thread_idx.x)
    if lane == 0:
        status[0] = 1
    var invalid = False
    for i in range(lane, len(input), block_size):
        invalid |= not isfinite(input[i])
    if block.sum[block_size=block_size, broadcast=True](UInt32(invalid)) != 0:
        return
    for face in range(lane, len(indices) // 3, block_size):
        comptime for axis in range(3):
            var lo = input[Int(indices[face * 3]) * 3 + axis]
            var hi = lo
            comptime for corner in range(1, 3):
                var value = input[Int(indices[face * 3 + corner]) * 3 + axis]
                lo = min(lo, value)
                hi = max(hi, value)
            items[face * 6 + axis] = lo
            items[face * 6 + axis + 3] = hi
    for i in range(lane, len(input), block_size):
        points[i] = input[i]
    barrier()
    comptime if rebuild:
        reorder_forest[block_size](items, nodes, order)
    refit_forest[block_size](items, bounds, nodes)
    if lane == 0:
        status[0] = 0


def velocity_update_kernel[
    checked: Bool, block_size: Int = UPDATE_BLOCK_SIZE
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    velocities: MutSpan[Float32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    comptime assert (
        block_size >= 32
        and block_size <= 1024
        and (block_size & (block_size - 1)) == 0
    )
    if (
        Int(block_dim.x) != block_size
        or Int(block_dim.y) != 1
        or Int(block_dim.z) != 1
        or Int(grid_dim.x) != 1
        or Int(grid_dim.y) != 1
        or Int(grid_dim.z) != 1
    ):
        if global_idx.x == 0 and global_idx.y == 0 and global_idx.z == 0:
            status[0] = 1
        return
    var lane = Int(thread_idx.x)
    if lane == 0:
        status[0] = 1
    var invalid = False
    for i in range(lane, len(input), block_size):
        invalid |= not isfinite(input.unsafe_ptr().unsafe_load(i))
    if block.sum[block_size=block_size, broadcast=True](UInt32(invalid)) != 0:
        return
    for i in range(lane, len(velocities), block_size):
        velocities.unsafe_ptr().unsafe_store(
            i,
            input.unsafe_ptr().unsafe_load(i) if len(input)
            > 0 else Float32(0.0),
        )
    barrier()
    if lane == 0:
        status[0] = 0


def validate_velocity_blocks[
    block_size: Int, width: SIMDLength = 1
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    counters: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    """Validate width adjacent components per lane, then publish one status.

    For width>1, input must be aligned to width*4 bytes. Every tail component
    is checked separately. Only the last arrival publishes status; no block spins.
    """
    comptime assert width <= 8
    comptime AtomicU32 = Atomic[
        UInt32, scope="device" if is_nvidia_gpu() else ""
    ]
    var i = Int(global_idx.x) * width
    var invalid = False
    if i + width <= len(input):
        var values = input.unsafe_ptr().unsafe_load[
            width=width, alignment=width * 4
        ](i)
        comptime for component in range(width):
            invalid |= not isfinite(values[component])
    else:
        comptime for component in range(width):
            if i + component < len(input):
                invalid |= not isfinite(
                    input.unsafe_ptr().unsafe_load(i + component)
                )
    var rejected = block.sum[block_size=block_size, broadcast=True](
        UInt32(invalid)
    )
    if thread_idx.x == 0:
        var arrivals = counters.unsafe_ptr()
        var rejection = counters.unsafe_ptr().unsafe_offset(1)
        if rejected != 0:
            AtomicU32.store[ordering=Ordering.RELAXED](rejection, UInt32(1))
        var previous = AtomicU32.fetch_add[ordering=Ordering.ACQUIRE_RELEASE](
            arrivals, UInt32(1)
        )
        if previous == UInt32(grid_dim.x) - 1:
            # The acquire/release arrival chain observes every rejection store.
            # Only the last block publishes status; the next launch copies.
            # Reset counters after all arrivals for the next serialized update.
            status[0] = AtomicU32.load[ordering=Ordering.RELAXED](rejection)
            AtomicU32.store[ordering=Ordering.RELAXED](rejection, UInt32(0))
            AtomicU32.store[ordering=Ordering.RELEASE](arrivals, UInt32(0))


def publish_velocity_blocks[
    block_size: Int, width: SIMDLength = 1
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    velocities: MutSpan[Float32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    """Guarded copy after validation; both spans need width*4-byte alignment."""
    comptime assert width <= 8
    if status.unsafe_ptr().unsafe_load(0) != 0:
        return
    var i = Int(global_idx.x) * width
    if i + width <= len(velocities):
        velocities.unsafe_ptr().unsafe_store[width=width, alignment=width * 4](
            i,
            input.unsafe_ptr().unsafe_load[width=width, alignment=width * 4](i),
        )
    else:
        comptime for component in range(width):
            if i + component < len(velocities):
                velocities.unsafe_ptr().unsafe_store(
                    i + component, input.unsafe_ptr().unsafe_load(i + component)
                )


def clear_velocity_kernel(
    velocities: MutSpan[Float32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < len(velocities):
        velocities.unsafe_ptr().unsafe_store(i, Float32(0.0))
    if i == 0:
        status[0] = 0


@always_inline
def _rebuild_leaf(
    nodes: ImmSpan[UInt32, ImmutAnyOrigin], root: Int, count: Int, ordinal: Int
) -> Int:
    if (count & (count - 1)) == 0:
        return (
            root
            + 2 * ordinal
            + Int(log2_floor(UInt32(count)))
            - Int(pop_count(UInt32(ordinal)))
        )
    var node = root
    var rank = ordinal
    while nodes[node * 2 + 1] == EMPTY_LANE:
        var left = node + 1
        var right = Int(nodes[left * 2])
        var left_count = (right - left + 1) // 2
        if rank < left_count:
            node = left
        else:
            rank -= left_count
            node = right
    return node


def rebuild_tile_sort_kernel(
    items: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    tiles: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    if status[0] != 0:
        return
    # Each descriptor names one independent 256-leaf tile in a retained root.
    var tile = Int(block_idx.x) * 3
    var root = Int(tiles[tile])
    var count = Int(tiles[tile + 1])
    var begin = Int(tiles[tile + 2])
    var lane = Int(thread_idx.x)
    var ids = stack_allocation[256, UInt32, address_space=AddressSpace.SHARED]()
    var keys = stack_allocation[
        256, UInt64, address_space=AddressSpace.SHARED
    ]()
    var item = EMPTY_LANE
    var key = UInt64.MAX
    if begin + lane < count:
        item = nodes[_rebuild_leaf(nodes, root, count, begin + lane) * 2 + 1]
        key = (
            items.unsafe_ptr()
            .unsafe_bitcast[UInt64]()
            .unsafe_load(len(items) // 8 * 3 + Int(item))
        )
    ids[unsafe_offset=lane] = item
    keys[unsafe_offset=lane] = key
    barrier()
    var size = 2
    while size <= 256:
        var stride = size // 2
        while stride > 0:
            var partner = lane ^ stride
            if partner > lane:
                var a = ids[unsafe_offset=lane]
                var b = ids[unsafe_offset=partner]
                var ka = keys[unsafe_offset=lane]
                var kb = keys[unsafe_offset=partner]
                var ascending = (lane & size) == 0
                var swap = (kb < ka or (kb == ka and b < a)) if ascending else (
                    ka < kb or (ka == kb and a < b)
                )
                if swap:
                    ids[unsafe_offset=lane] = b
                    ids[unsafe_offset=partner] = a
                    keys[unsafe_offset=lane] = kb
                    keys[unsafe_offset=partner] = ka
            barrier()
            stride //= 2
        size *= 2
    if begin + lane < count:
        output[root + begin + lane] = ids[unsafe_offset=lane]


def rebuild_merge_order_kernel(
    items: ImmSpan[Float32, ImmutAnyOrigin],
    tiles: ImmSpan[UInt32, ImmutAnyOrigin],
    source: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
    merge_width: UInt32,
):
    if status[0] != 0:
        return
    var tile = Int(block_idx.x) * 3
    var root = Int(tiles[tile])
    var count = Int(tiles[tile + 1])
    var i = Int(tiles[tile + 2]) + Int(thread_idx.x)
    if i >= count:
        return
    var width = Int(merge_width)
    # A leaf ranks itself in the other sorted half. Original-ID ties make
    # the order strict, so every destination is unique without atomics.
    var pair = i // (2 * width) * (2 * width)
    var half = i // width * width
    var other = pair + width if half == pair else pair
    var end = min(other + width, count)
    var lo = min(other, count)
    var hi = end
    var item = source[root + i]
    while lo < hi:
        var mid = (lo + hi) // 2
        if _item_less[True](items, source[root + mid], item):
            lo = mid + 1
        else:
            hi = mid
    output[root + pair + (i - half) + (lo - min(other, count))] = item


def publish_rebuild_order_kernel(
    tiles: ImmSpan[UInt32, ImmutAnyOrigin],
    source: ImmSpan[UInt32, ImmutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    if status[0] != 0:
        return
    var tile = Int(block_idx.x) * 3
    var root = Int(tiles[tile])
    var count = Int(tiles[tile + 1])
    var i = Int(tiles[tile + 2]) + Int(thread_idx.x)
    if i < count:
        var node = _rebuild_leaf(nodes, root, count, i)
        nodes[node * 2 + 1] = source[root + i]


def clear_rebuild_order_kernel(order: MutSpan[UInt32, MutAnyOrigin]):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < len(order):
        order[i] = 0
