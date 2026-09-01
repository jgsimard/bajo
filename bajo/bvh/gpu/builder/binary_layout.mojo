from std.math import max, ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import global_idx

from bajo.core import AABB, SegmentOffsets
from bajo.bvh.constants import (
    BinaryBvhNode,
    REDUCED_BOUNDS_STRIDE,
    BOUNDS_REDUCE_CHUNK,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.utils import _device_span, upload_list
from bajo.bvh.tagged_ref import is_leaf_ref, decode_ref_index
from bajo.sort.gpu.radix_sort import RadixSortWorkspace


def _node_meta_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * BinaryBvhNode.META_STRIDE


def _node_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * BinaryBvhNode.BOUNDS_STRIDE


def _node_parent_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BinaryBvhNode.PARENT


def _node_left_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BinaryBvhNode.LEFT


def _node_right_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BinaryBvhNode.RIGHT


def _node_left(node_meta: ImmSpan[UInt32, _], node_idx: UInt32) -> UInt32:
    var base = _node_meta_base(node_idx)
    debug_assert["safe", _use_compiler_assume=True](
        base >= 0 and base <= len(node_meta) - BinaryBvhNode.META_STRIDE,
        "binary node metadata is outside the input span",
    )
    return node_meta.unsafe_get(base + BinaryBvhNode.LEFT)


def _node_right(node_meta: ImmSpan[UInt32, _], node_idx: UInt32) -> UInt32:
    var base = _node_meta_base(node_idx)
    debug_assert["safe", _use_compiler_assume=True](
        base >= 0 and base <= len(node_meta) - BinaryBvhNode.META_STRIDE,
        "binary node metadata is outside the input span",
    )
    return node_meta.unsafe_get(base + BinaryBvhNode.RIGHT)


# Raw-pointer overloads remain for GPU kernels that have not yet crossed a
# length-carrying ABI boundary. They are removed as those kernels migrate.
def _node_left(node_meta: ImmPointer[UInt32, _], node_idx: UInt32) -> UInt32:
    return node_meta[unsafe_offset=_node_left_index(node_idx)]


def _node_right(node_meta: ImmPointer[UInt32, _], node_idx: UInt32) -> UInt32:
    return node_meta[unsafe_offset=_node_right_index(node_idx)]


def _encoded_bounds(
    encoded: UInt32,
    leaf_bounds: ImmSpan[Float32, _],
    leaf_ids: ImmSpan[UInt32, _],
    node_bounds: ImmSpan[Float32, _],
) -> AABB[.WORLD]:
    """Load bounds for either encoded leaf or internal binary topology."""
    if is_leaf_ref(encoded):
        var sorted_leaf_idx = decode_ref_index(encoded)
        debug_assert["safe", _use_compiler_assume=True](
            Int(sorted_leaf_idx) < len(leaf_ids),
            "encoded leaf index is outside the leaf-id span",
        )
        var item_idx = UInt32(leaf_ids.unsafe_get(Int(sorted_leaf_idx)))
        return AABB[.WORLD].load6(leaf_bounds, Int(item_idx) * AABB.STRIDE)
    return _load_and_union_node_bounds(node_bounds, decode_ref_index(encoded))


def _write_child_bounds(
    node_bounds: MutSpan[Float32, _],
    parent: UInt32,
    write_left: Bool,
    bounds: AABB,
):
    var b = _node_bounds_base(parent)
    if not write_left:
        b += 6
    bounds.store6(node_bounds, b)


def _load_and_union_node_bounds(
    node_bounds: ImmSpan[Float32, _], parent: UInt32
) -> AABB[.WORLD]:
    var b = _node_bounds_base(parent)
    var b1 = AABB[.WORLD].load6(node_bounds, b)
    var b2 = AABB[.WORLD].load6(node_bounds, b + 6)
    return AABB.merge(b1, b2)


def init_empty_bounds_kernel(bounds: MutSpan[Float32, MutAnyOrigin]):
    var node_count = len(bounds) / BinaryBvhNode.BOUNDS_STRIDE
    var i = global_idx.x
    if i >= node_count:
        return

    var b = i * BinaryBvhNode.BOUNDS_STRIDE
    var invalid = AABB[.WORLD].invalid()
    invalid.store6(bounds, b)
    invalid.store6(bounds, b + AABB.STRIDE)


def _segment_for_item(offsets: ImmSpan[UInt32, _], item_idx: Int) -> Int:
    var low = 0
    var high = len(offsets) - 1
    while low + 1 < high:
        var mid = (low + high) // 2
        if item_idx < Int(offsets.unsafe_get(mid)):
            high = mid
        else:
            low = mid
    return low


def compute_segment_bounds_partials_kernel(
    leaf_bounds: ImmSpan[Float32, ImmutAnyOrigin],
    leaf_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    partial_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    out_partials: MutSpan[Float32, MutAnyOrigin],
):
    var partial_idx = global_idx.x
    var partial_count = Int(
        partial_segment_offsets.unsafe_get(len(partial_segment_offsets) - 1)
    )
    if partial_idx >= partial_count:
        return

    var segment_idx = _segment_for_item(partial_segment_offsets, partial_idx)
    var segment_partial_begin = Int(
        partial_segment_offsets.unsafe_get(segment_idx)
    )
    var segment_leaf_begin = Int(leaf_segment_offsets.unsafe_get(segment_idx))
    var segment_leaf_end = Int(leaf_segment_offsets.unsafe_get(segment_idx + 1))
    var first = (
        segment_leaf_begin
        + (partial_idx - segment_partial_begin) * BOUNDS_REDUCE_CHUNK
    )
    var last = min(first + BOUNDS_REDUCE_CHUNK, segment_leaf_end)
    var bounds = AABB[.WORLD].invalid()
    var centroid_bounds = AABB[.WORLD].invalid()

    for leaf_idx in range(first, last):
        var b = leaf_idx * AABB.STRIDE
        var aabb = AABB[.WORLD].load6(leaf_bounds, b)

        bounds.grow(aabb)
        centroid_bounds.grow(aabb.centroid())

    var out = partial_idx * REDUCED_BOUNDS_STRIDE
    bounds.store6(out_partials, out)
    centroid_bounds.store6(out_partials, out + AABB.STRIDE)


def reduce_segment_bounds_partials_kernel(
    in_partials: ImmSpan[Float32, ImmutAnyOrigin],
    in_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    out_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    out_partials: MutSpan[Float32, MutAnyOrigin],
):
    var out_idx = global_idx.x
    var out_count = Int(
        out_segment_offsets.unsafe_get(len(out_segment_offsets) - 1)
    )
    if out_idx >= out_count:
        return

    var segment_idx = _segment_for_item(out_segment_offsets, out_idx)
    var segment_out_begin = Int(out_segment_offsets.unsafe_get(segment_idx))
    var segment_in_begin = Int(in_segment_offsets.unsafe_get(segment_idx))
    var segment_in_end = Int(in_segment_offsets.unsafe_get(segment_idx + 1))
    var first = (
        segment_in_begin + (out_idx - segment_out_begin) * BOUNDS_REDUCE_CHUNK
    )
    var last = min(first + BOUNDS_REDUCE_CHUNK, segment_in_end)
    var bounds = AABB[.WORLD].invalid()
    var centroid_bounds = AABB[.WORLD].invalid()

    for i in range(first, last):
        var b = i * REDUCED_BOUNDS_STRIDE

        var partial_bounds = AABB[.WORLD].load6(in_partials, b)
        var partial_centroid_bounds = AABB[.WORLD].load6(
            in_partials, b + AABB.STRIDE
        )

        bounds.grow(partial_bounds)
        centroid_bounds.grow(partial_centroid_bounds)

    var out = out_idx * REDUCED_BOUNDS_STRIDE
    bounds.store6(out_partials, out)
    centroid_bounds.store6(out_partials, out + AABB.STRIDE)


def _reduced_segment_offsets(segments: SegmentOffsets) -> SegmentOffsets:
    var counts = List[Int](capacity=segments.segment_count())
    for segment_idx in range(segments.segment_count()):
        counts.append(
            max(
                ceildiv(Int(segments.count(segment_idx)), BOUNDS_REDUCE_CHUNK),
                1,
            )
        )
    return SegmentOffsets.from_counts(counts^)


def _internal_segment_offsets(segments: SegmentOffsets) -> SegmentOffsets:
    var counts = List[Int](capacity=segments.segment_count())
    for segment_idx in range(segments.segment_count()):
        counts.append(max(Int(segments.count(segment_idx)) - 1, 0))
    return SegmentOffsets.from_counts(counts^)


struct GpuBinaryTopologyWorkspace(Copyable):
    """Transient Morton, parent, and refit state."""

    var morton_keys: DeviceBuffer[.uint32]
    var sort_keys: DeviceBuffer[.uint64]
    var leaf_parent: DeviceBuffer[.uint32]
    var node_flags: DeviceBuffer[.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_capacity: Int,
        internal_capacity: Int,
    ) raises:
        self.morton_keys = ctx.enqueue_create_buffer[.uint32](leaf_capacity)
        self.sort_keys = ctx.enqueue_create_buffer[.uint64](leaf_capacity)
        self.leaf_parent = ctx.enqueue_create_buffer[.uint32](leaf_capacity)
        self.node_flags = ctx.enqueue_create_buffer[.uint32](
            max(internal_capacity, 1)
        )


struct GpuBinaryBuildWorkspace:
    """Reusable scratch for one fixed segmented leaf workload."""

    var leaf_capacity: Int
    var internal_capacity: Int
    var segments: SegmentOffsets
    var segment_offsets: DeviceBuffer[.uint32]
    var bounds_scratch_a: DeviceBuffer[.float32]
    var bounds_scratch_b: DeviceBuffer[.float32]
    var sort: RadixSortWorkspace[.uint64, .uint32]
    var topology: Optional[GpuBinaryTopologyWorkspace]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        segments: SegmentOffsets,
    ) raises:
        var leaf_capacity = segments.item_count()
        debug_assert["safe", _use_compiler_assume=True](
            leaf_capacity > 0, "binary workspace capacity must be positive"
        )
        debug_assert["safe", _use_compiler_assume=True](
            segments.segment_count() > 0,
            "binary workspace requires at least one segment",
        )
        var partial_segments = _reduced_segment_offsets(segments)
        var internal_segments = _internal_segment_offsets(segments)
        self.leaf_capacity = leaf_capacity
        self.internal_capacity = internal_segments.item_count()
        self.segments = segments.copy()
        self.segment_offsets = upload_list(ctx, self.segments.offsets)
        self.bounds_scratch_a = ctx.enqueue_create_buffer[.float32](
            max(partial_segments.item_count(), 1) * REDUCED_BOUNDS_STRIDE
        )
        self.bounds_scratch_b = ctx.enqueue_create_buffer[.float32](
            max(partial_segments.item_count(), 1) * REDUCED_BOUNDS_STRIDE
        )
        self.sort = RadixSortWorkspace[.uint64, .uint32](ctx, leaf_capacity)
        self.topology = Optional[GpuBinaryTopologyWorkspace]()

    def ensure_topology(mut self, mut ctx: DeviceContext) raises:
        if not self.topology:
            self.topology = Optional[GpuBinaryTopologyWorkspace](
                GpuBinaryTopologyWorkspace(
                    ctx, self.leaf_capacity, self.internal_capacity
                )
            )


struct GpuBinaryBoundsBvh:
    """Compact binary topology consumed by quality and wide collapse."""

    var leaf_count: Int
    var internal_count: Int
    var segments: SegmentOffsets
    var segment_offsets: DeviceBuffer[.uint32]
    var internal_segments: SegmentOffsets
    var internal_segment_offsets: DeviceBuffer[.uint32]
    var roots: DeviceBuffer[.uint32]

    var bounds_device: DeviceBuffer[.float32]
    """Per segment: six root-bound then six centroid-bound values."""

    var leaf_bounds: DeviceBuffer[.float32]
    var leaf_payloads: DeviceBuffer[.uint32]

    var leaf_ids: DeviceBuffer[.uint32]

    # Binary node layout:
    #   node_meta   : parent, left encoded child, right encoded child, fence/range end
    #   leaf_parent : parent internal node for each sorted leaf
    #   node_bounds : two child AABBs per internal node, 12 floats total
    #   node_flags  : refit synchronization flags
    var node_meta: DeviceBuffer[.uint32]
    var node_bounds: DeviceBuffer[.float32]
    var node_leaf_counts: DeviceBuffer[.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[.float32],
        leaf_payloads: DeviceBuffer[.uint32],
        mut workspace: GpuBinaryBuildWorkspace,
    ) raises:
        self.leaf_count = len(leaf_payloads)
        debug_assert["safe", _use_compiler_assume=True](
            self.leaf_count > 0, "passed empty input."
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(leaf_bounds) == self.leaf_count * AABB.STRIDE,
            "leaf bounds buffer has the wrong length",
        )
        debug_assert["safe", _use_compiler_assume=True](
            workspace.leaf_capacity == self.leaf_count,
            "binary workspace capacity must match the input leaf count",
        )
        self.segments = workspace.segments.copy()
        self.segment_offsets = workspace.segment_offsets.copy()
        self.internal_segments = _internal_segment_offsets(self.segments)
        self.internal_segment_offsets = upload_list(
            ctx, self.internal_segments.offsets
        )
        self.roots = ctx.enqueue_create_buffer[.uint32](
            self.segments.segment_count()
        )
        self.internal_count = self.internal_segments.item_count()
        debug_assert["safe", _use_compiler_assume=True](
            workspace.internal_capacity == self.internal_count,
            "binary workspace internal capacity must match the segments",
        )

        var n_leaf = self.leaf_count
        var n_internal = max(self.internal_count, 1)

        self.leaf_bounds = leaf_bounds
        self.leaf_payloads = leaf_payloads

        self.bounds_device = ctx.enqueue_create_buffer[.float32](
            self.segments.segment_count() * REDUCED_BOUNDS_STRIDE
        )

        var partial_segments = _reduced_segment_offsets(self.segments)
        var partial_offsets = upload_list(ctx, partial_segments.offsets)

        var reduce_grid = ceildiv(
            partial_segments.item_count(),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.enqueue_function[compute_segment_bounds_partials_kernel](
            _device_span[mut=False](self.leaf_bounds),
            _device_span[mut=False](self.segment_offsets),
            _device_span[mut=False](partial_offsets),
            _device_span[mut=True](workspace.bounds_scratch_a),
            grid_dim=reduce_grid,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        var in_buf = workspace.bounds_scratch_a.copy()
        var out_buf = workspace.bounds_scratch_b.copy()

        while partial_segments.item_count() > partial_segments.segment_count():
            var next_segments = _reduced_segment_offsets(partial_segments)
            var next_offsets = upload_list(ctx, next_segments.offsets)
            var grid = ceildiv(
                next_segments.item_count(),
                GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            ctx.enqueue_function[reduce_segment_bounds_partials_kernel](
                _device_span[mut=False](in_buf),
                _device_span[mut=False](partial_offsets),
                _device_span[mut=False](next_offsets),
                _device_span[mut=True](out_buf),
                grid_dim=grid,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            swap(in_buf, out_buf)
            partial_segments = next_segments^
            partial_offsets = next_offsets^
        in_buf.enqueue_copy_to(self.bounds_device)

        self.leaf_ids = ctx.enqueue_create_buffer[.uint32](n_leaf)

        self.node_meta = ctx.enqueue_create_buffer[.uint32](
            n_internal * BinaryBvhNode.META_STRIDE
        )
        self.node_leaf_counts = ctx.enqueue_create_buffer[.uint32](n_internal)
        self.node_bounds = ctx.enqueue_create_buffer[.float32](
            n_internal * BinaryBvhNode.BOUNDS_STRIDE
        )
        workspace.ensure_topology(ctx)

    def blocks_leaves(self) -> Int:
        return ceildiv(self.leaf_count, GPU_BOUNDS_BVH_BLOCK_SIZE)

    def blocks_internal(self) -> Int:
        return ceildiv(max(self.internal_count, 1), GPU_BOUNDS_BVH_BLOCK_SIZE)

    def root_bounds(self, segment_idx: Int = 0) raises -> AABB[.WORLD]:
        with self.bounds_device.map_to_host() as h:
            return AABB[.WORLD].load6(
                Span(unsafe_ptr=h.unsafe_ptr(), length=len(h)),
                segment_idx * REDUCED_BOUNDS_STRIDE,
            )

    def centroid_bounds(self, segment_idx: Int = 0) raises -> AABB[.WORLD]:
        with self.bounds_device.map_to_host() as h:
            return AABB[.WORLD].load6(
                Span(unsafe_ptr=h.unsafe_ptr(), length=len(h)),
                segment_idx * REDUCED_BOUNDS_STRIDE + AABB.STRIDE,
            )
