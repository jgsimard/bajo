"""Validated refit across GPU blocks with retained parent links and counters.

Separate validation and publication launches keep rejected input transactional.
Each mesh block merges a complete subtree in shared memory. A following kernel
publishes subtree roots through acquire/release child-arrival counters. The
validation launch precedes all publication, preserving rejected input.
"""
from std.atomic import Atomic, Ordering
from std.math import min, max
from std.sys.info import is_nvidia_gpu
from std.utils.numerics import isfinite
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation, bitcast
from bajo.bvh.constants import EMPTY_LANE


def forest_parents(nodes: ImmSpan[UInt32, _]) -> List[UInt32]:
    var parents = List[UInt32](length=len(nodes) // 2, fill=EMPTY_LANE)
    for node in range(len(parents)):
        if nodes[node * 2 + 1] == EMPTY_LANE:
            var left = node + 1
            var right = Int(nodes[left * 2])
            parents[left] = UInt32(node)
            parents[right] = UInt32(node)
    return parents^


def forest_parents_kernel(
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    parents: MutSpan[UInt32, MutAnyOrigin],
):
    # Each non-root has exactly one parent, so child writes are disjoint.
    # Thread zero follows forest escapes to initialize the independent roots.
    var node = Int(global_idx.x)
    if node < len(parents) and nodes.unsafe_get(node * 2 + 1) == EMPTY_LANE:
        var left = node + 1
        var right = Int(nodes.unsafe_get(left * 2))
        parents.unsafe_get(left) = UInt32(node)
        parents.unsafe_get(right) = UInt32(node)
    if node == 0:
        var root = 0
        while root < len(parents):
            parents.unsafe_get(root) = EMPTY_LANE
            root = Int(nodes.unsafe_get(root * 2))


def forest_chunks(nodes: ImmSpan[UInt32, _]) -> List[UInt32]:
    """Disjoint complete subtrees with at most 64 leaves."""
    var roots = List[UInt32]()
    var node = 0
    while node < len(nodes) // 2:
        var end = Int(nodes[node * 2])
        if end - node <= 127:
            roots.append(UInt32(node))
            node = end
        else:
            node += 1
    return roots^


def clear_refit_kernel(
    counters: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < len(counters):
        counters.unsafe_get(i) = 0
    if i == 0:
        status.unsafe_get(0) = 0


@always_inline
def _reject(status: MutSpan[UInt32, MutAnyOrigin]):
    comptime AtomicU32 = Atomic[
        UInt32, scope="device" if is_nvidia_gpu() else ""
    ]
    AtomicU32.store[ordering=Ordering.RELAXED](status.unsafe_ptr(), UInt32(1))


def validate_mesh_refit_kernel[
    cache_keys: Bool = False
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    items: MutSpan[Float32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < len(input) and not isfinite(input.unsafe_get(i)):
        _reject(status)
    if i < len(indices) // 3:
        comptime for axis in range(3):
            var lo = input.unsafe_get(Int(indices.unsafe_get(i * 3)) * 3 + axis)
            var hi = lo
            comptime for corner in range(1, 3):
                var value = input.unsafe_get(
                    Int(indices.unsafe_get(i * 3 + corner)) * 3 + axis
                )
                lo = min(lo, value)
                hi = max(hi, value)
            items.unsafe_get(i * 6 + axis) = lo
            items.unsafe_get(i * 6 + axis + 3) = hi
        comptime if cache_keys:
            # Finite Float32 endpoints have finite Float64 sums. Preserve the
            # original sum exactly, then encode its numeric ordering as UInt64.
            var center = Float64(items.unsafe_get(i * 6)) + Float64(
                items.unsafe_get(i * 6 + 3)
            )
            var bits = bitcast[.uint64](center)
            if center == 0.0:
                bits = 0  # Both signed zeros tie in the original comparator.
            comptime sign = UInt64(0x8000000000000000)
            var key = ~bits if (bits & sign) != 0 else bits ^ sign
            items.unsafe_ptr().unsafe_bitcast[UInt64]().unsafe_store(
                len(indices) + i, key
            )


def validate_bounds_refit_kernel(
    input: ImmSpan[Float32, ImmutAnyOrigin],
    items: MutSpan[Float32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < len(input) // 6:
        comptime for axis in range(3):
            var lo = input.unsafe_get(i * 6 + axis)
            var hi = input.unsafe_get(i * 6 + axis + 3)
            if not isfinite(lo) or not isfinite(hi) or lo > hi:
                _reject(status)
            items.unsafe_get(i * 6 + axis) = lo
            items.unsafe_get(i * 6 + axis + 3) = hi


@always_inline
def _publish_leaf(
    node: Int,
    items: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    parents: ImmSpan[UInt32, ImmutAnyOrigin],
    counters: MutSpan[UInt32, MutAnyOrigin],
):
    var item = nodes.unsafe_get(node * 2 + 1)
    if item == EMPTY_LANE:
        return
    comptime for axis in range(6):
        bounds.unsafe_get(node * 6 + axis) = items.unsafe_get(
            Int(item) * 6 + axis
        )
    _publish_parent(node, bounds, nodes, parents, counters)


@always_inline
def _publish_parent(
    node: Int,
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    parents: ImmSpan[UInt32, ImmutAnyOrigin],
    counters: MutSpan[UInt32, MutAnyOrigin],
):
    comptime AtomicU32 = Atomic[
        UInt32, scope="device" if is_nvidia_gpu() else ""
    ]
    var parent = parents.unsafe_get(node)
    while parent != EMPTY_LANE:
        var previous = AtomicU32.fetch_add[ordering=Ordering.ACQUIRE_RELEASE](
            counters.unsafe_ptr().unsafe_offset(Int(parent)), UInt32(1)
        )
        if previous == 0:
            return
        var left = Int(parent) + 1
        var right = Int(nodes.unsafe_get(left * 2))
        comptime for axis in range(3):
            bounds.unsafe_get(Int(parent) * 6 + axis) = min(
                bounds.unsafe_get(left * 6 + axis),
                bounds.unsafe_get(right * 6 + axis),
            )
            bounds.unsafe_get(Int(parent) * 6 + axis + 3) = max(
                bounds.unsafe_get(left * 6 + axis + 3),
                bounds.unsafe_get(right * 6 + axis + 3),
            )
        parent = parents.unsafe_get(Int(parent))


def publish_bounds_refit_kernel(
    items: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    parents: ImmSpan[UInt32, ImmutAnyOrigin],
    counters: MutSpan[UInt32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    if status.unsafe_get(0) != 0:
        return
    var i = Int(global_idx.x)
    if i < len(parents):
        _publish_leaf(i, items, bounds, nodes, parents, counters)


def refit_mesh_chunks_kernel[
    block_size: Int
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    points: MutSpan[Float32, MutAnyOrigin],
    items: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
    chunks: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var scratch = stack_allocation[
        768, Float32, address_space=AddressSpace.SHARED
    ]()
    var valid = status.unsafe_get(0) == 0
    var lane = Int(thread_idx.x)
    var root = Int(chunks.unsafe_get(Int(block_idx.x)))
    var end = Int(nodes.unsafe_get(root * 2))
    var i = Int(global_idx.x)
    while i < len(input):
        if valid:
            points.unsafe_get(i) = input.unsafe_get(i)
        i += len(chunks) * block_size
    for node in range(root + lane, end, block_size):
        var item = nodes.unsafe_get(node * 2 + 1)
        if valid and item != EMPTY_LANE:
            comptime for axis in range(6):
                scratch[
                    unsafe_offset=axis * 128 + node - root
                ] = items.unsafe_get(Int(item) * 6 + axis)
    barrier()
    comptime for level in range(6):
        comptime previous = (1 << (level + 1)) - 1
        comptime limit = (1 << (level + 2)) - 1
        for node in range(root + lane, end, block_size):
            var span = Int(nodes.unsafe_get(node * 2)) - node
            if valid and span > previous and span <= limit:
                var left = node + 1
                var right = Int(nodes.unsafe_get(left * 2))
                comptime for axis in range(3):
                    scratch[unsafe_offset=axis * 128 + node - root] = min(
                        scratch[unsafe_offset=axis * 128 + left - root],
                        scratch[unsafe_offset=axis * 128 + right - root],
                    )
                    scratch[unsafe_offset=(axis + 3) * 128 + node - root] = max(
                        scratch[unsafe_offset=(axis + 3) * 128 + left - root],
                        scratch[unsafe_offset=(axis + 3) * 128 + right - root],
                    )
        barrier()

    if valid:
        for node in range(root + lane, end, block_size):
            comptime for axis in range(6):
                bounds.unsafe_get(node * 6 + axis) = scratch[
                    unsafe_offset=axis * 128 + node - root
                ]


def publish_mesh_chunk_parents_kernel(
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    parents: ImmSpan[UInt32, ImmutAnyOrigin],
    counters: MutSpan[UInt32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
    chunks: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var i = Int(global_idx.x)
    if status.unsafe_get(0) == 0 and i < len(chunks):
        _publish_parent(
            Int(chunks.unsafe_get(i)), bounds, nodes, parents, counters
        )
