from std.atomic import Atomic, Ordering, fence
from std.bit import pop_count
from std.gpu import WARP_SIZE, global_idx, lane_id, thread_idx
from std.gpu.primitives import warp
from std.math import abs, ceildiv, max, min
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import syncwarp

from bajo.core.morton import morton_key_delta
from bajo.bvh.constants import (
    BinaryBvhNode,
    LBVH_SENTINEL,
)
from bajo.bvh.tagged_ref import (
    decode_ref_index,
    encode_internal_ref,
    encode_leaf_ref,
    is_leaf_ref,
)
from bajo.bvh.gpu.builder.binary_layout import (
    _node_parent_index,
    _segment_for_item,
)
from bajo.bvh.gpu.builder.hploc_layout import (
    HPLOC_STATUS_OK,
    HPLOC_STATUS_NO_PROGRESS,
    HPLOC_STATUS_INVALID_RESULT,
    HPLOC_MERGING_THRESHOLD,
    HPLOC_SEARCH_RADIUS,
    _hploc_load_bounds,
    _hploc_store_bounds,
)
from bajo.bvh.gpu.builder.hploc_wave import (
    hploc_wave_ballot,
    hploc_wave_first_lane,
    hploc_wave_rank,
)
from bajo.bvh.gpu.utils import _device_span, upload_list
from bajo.core import AABB, Frame, Point3f32, SegmentOffsets


comptime HPLOC_MULTI_WAVE_BLOCK_SIZE = 256


@always_inline
def _hploc_encode_offset(thread_index: Int, neighbor_index: Int) -> UInt32:
    """Encode the paper tie order: near first, then right before left."""

    var signed_offset = neighbor_index - thread_index
    var distance = abs(signed_offset) - 1
    return UInt32(distance * 2 + (1 if signed_offset < 0 else 0))


@always_inline
def _hploc_decode_offset(thread_index: Int, encoded: UInt32) -> Int:
    var distance = Int(encoded >> 1) + 1
    return thread_index - distance if (encoded & UInt32(1)) != 0 else (
        thread_index + distance
    )


@always_inline
def _hploc_pack_distance_offset(
    area: Float32, encoded_offset: UInt32
) -> UInt64:
    # Positive IEEE-754 bit patterns preserve numerical ordering. HIPRT packs
    # both fields into 32 bits and quantizes the low area bits; keep a 64-bit
    # key so production selection remains exact with the literature oracle.
    var area_bits = bitcast[DType.uint32](area)
    return (UInt64(area_bits) << 32) | UInt64(encoded_offset)


@always_inline
def _hploc_cluster_bounds(
    encoded: UInt32,
    leaf_bounds: ImmSpan[Float32, _],
    sorted_leaf_ids: ImmSpan[UInt32, _],
    scratch_bounds: ImmSpan[Float32, _],
) -> AABB[Frame.WORLD]:
    var cluster_idx = decode_ref_index(encoded)
    if is_leaf_ref(encoded):
        var leaf_id = sorted_leaf_ids.unsafe_get(Int(cluster_idx))
        return AABB[Frame.WORLD].load6(leaf_bounds, Int(leaf_id) * AABB.STRIDE)
    return _hploc_load_bounds(scratch_bounds, cluster_idx)


@always_inline
def _hploc_cluster_leaf_count(
    encoded: UInt32,
    node_leaf_counts: ImmSpan[UInt32, _],
) -> UInt32:
    if is_leaf_ref(encoded):
        return UInt32(1)
    return node_leaf_counts.unsafe_get(Int(decode_ref_index(encoded)))


@always_inline
def _hploc_set_child_parent(
    encoded: UInt32,
    parent: UInt32,
    node_meta: MutSpan[UInt32, _],
    leaf_parent: MutSpan[UInt32, _],
):
    var child_idx = decode_ref_index(encoded)
    if is_leaf_ref(encoded):
        leaf_parent.unsafe_get(Int(child_idx)) = parent
    else:
        node_meta.unsafe_get(_node_parent_index(child_idx)) = parent


@always_inline
def _hploc_delta(
    sorted_morton_codes: ImmSpan[UInt32, _], a: Int, b: Int
) -> UInt64:
    return morton_key_delta(
        sorted_morton_codes.unsafe_get(a),
        UInt32(a),
        sorted_morton_codes.unsafe_get(b),
        UInt32(b),
    )


@always_inline
def _hploc_find_parent_id(
    sorted_morton_codes: ImmSpan[UInt32, _],
    left: Int,
    right: Int,
    segment_begin: Int,
    segment_end: Int,
) -> Int:
    if left == segment_begin or (
        right + 1 != segment_end
        and _hploc_delta(sorted_morton_codes, right, right + 1)
        < _hploc_delta(sorted_morton_codes, left - 1, left)
    ):
        return right
    return left - 1


@always_inline
def _hploc_atomic_exchange(
    ptr: MutPointer[UInt32, _], desired: UInt32
) -> UInt32:
    """Atomic exchange expressed through the current std.atomic CAS API."""

    var expected = Atomic.load[ordering=Ordering.ACQUIRE](ptr)
    while not Atomic.compare_exchange[
        success_ordering=Ordering.ACQUIRE_RELEASE,
        failure_ordering=Ordering.ACQUIRE,
    ](ptr, expected, desired):
        pass
    return expected


def init_hploc_multi_wave_kernel(
    sorted_leaf_ids: ImmSpan[UInt32, ImmutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    parent_slots: MutSpan[UInt32, MutAnyOrigin],
    cluster_indices: MutSpan[UInt32, MutAnyOrigin],
    leaf_parent: MutSpan[UInt32, MutAnyOrigin],
    node_counter: MutSpan[UInt32, MutAnyOrigin],
    root: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    var sorted_pos = global_idx.x
    var leaf_count = len(sorted_leaf_ids)
    if sorted_pos < leaf_count:
        parent_slots.unsafe_get(sorted_pos) = LBVH_SENTINEL
        leaf_parent.unsafe_get(sorted_pos) = LBVH_SENTINEL
        cluster_indices.unsafe_get(sorted_pos) = encode_leaf_ref(
            UInt32(sorted_pos)
        )

    var segment_count = len(segment_offsets) - 1
    if sorted_pos < segment_count:
        var begin = segment_offsets.unsafe_get(sorted_pos)
        var end = segment_offsets.unsafe_get(sorted_pos + 1)
        node_counter.unsafe_get(sorted_pos) = UInt32(0)
        root.unsafe_get(sorted_pos) = (
            encode_leaf_ref(begin) if end - begin == 1 else LBVH_SENTINEL
        )
        status.unsafe_get(sorted_pos) = UInt32(HPLOC_STATUS_OK)


def build_hploc_multi_wave_kernel[
    block_size: Int,
    search_radius: Int,
    merging_threshold: Int,
](
    sorted_morton_codes: ImmSpan[UInt32, ImmutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    internal_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_bounds: ImmSpan[Float32, ImmutAnyOrigin],
    sorted_leaf_ids: ImmSpan[UInt32, ImmutAnyOrigin],
    parent_slots: MutSpan[UInt32, MutAnyOrigin],
    cluster_indices: MutSpan[UInt32, MutAnyOrigin],
    scratch_bounds: MutSpan[Float32, MutAnyOrigin],
    node_meta: MutSpan[UInt32, MutAnyOrigin],
    leaf_parent: MutSpan[UInt32, MutAnyOrigin],
    node_bounds: MutSpan[Float32, MutAnyOrigin],
    node_flags: MutSpan[UInt32, MutAnyOrigin],
    node_leaf_counts: MutSpan[UInt32, MutAnyOrigin],
    node_counter: MutSpan[UInt32, MutAnyOrigin],
    root: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    """Literature H-PLOC outer and inner loops for arbitrary leaf counts."""

    comptime assert block_size % WARP_SIZE == 0
    var boxes_cache = stack_allocation[
        block_size * AABB.STRIDE,
        Float32,
        address_space=AddressSpace.SHARED,
    ]()
    var distance_offsets = stack_allocation[
        block_size, UInt64, address_space=AddressSpace.SHARED
    ]()
    var node_indices_cache = stack_allocation[
        block_size, UInt32, address_space=AddressSpace.SHARED
    ]()

    def load_cached_bounds(slot: Int) {imm} -> AABB[Frame.WORLD]:
        var base = slot * AABB.STRIDE
        return AABB[Frame.WORLD](
            Point3f32[Frame.WORLD](
                boxes_cache[unsafe_offset=base + 0],
                boxes_cache[unsafe_offset=base + 1],
                boxes_cache[unsafe_offset=base + 2],
            ),
            Point3f32[Frame.WORLD](
                boxes_cache[unsafe_offset=base + 3],
                boxes_cache[unsafe_offset=base + 4],
                boxes_cache[unsafe_offset=base + 5],
            ),
        )

    def store_cached_bounds(slot: Int, bounds: AABB[Frame.WORLD]) {imm}:
        var base = slot * AABB.STRIDE
        boxes_cache[unsafe_offset=base + 0] = bounds._min.x
        boxes_cache[unsafe_offset=base + 1] = bounds._min.y
        boxes_cache[unsafe_offset=base + 2] = bounds._min.z
        boxes_cache[unsafe_offset=base + 3] = bounds._max.x
        boxes_cache[unsafe_offset=base + 4] = bounds._max.y
        boxes_cache[unsafe_offset=base + 5] = bounds._max.z

    var leaf_count = len(sorted_morton_codes)
    if leaf_count == 1:
        return

    var idx = global_idx.x
    var lane = lane_id()
    var shared_lane = Int(thread_idx.x)
    var shared_wave_base = shared_lane - lane
    var left = idx
    var right = idx
    var split = 0
    var lane_active = idx < leaf_count
    var segment_idx = 0
    var segment_begin = 0
    var segment_end = 0
    var segment_leaf_count = 0
    if lane_active:
        segment_idx = _segment_for_item(segment_offsets, idx)
        segment_begin = Int(segment_offsets.unsafe_get(segment_idx))
        segment_end = Int(segment_offsets.unsafe_get(segment_idx + 1))
        segment_leaf_count = segment_end - segment_begin
    while hploc_wave_ballot(lane_active) != 0:
        if lane_active:
            var previous = LBVH_SENTINEL
            var parent_id = _hploc_find_parent_id(
                sorted_morton_codes,
                left,
                right,
                segment_begin,
                segment_end,
            )
            if parent_id == right:
                previous = _hploc_atomic_exchange(
                    parent_slots.unsafe_ptr().unsafe_offset(right),
                    UInt32(left),
                )
                if previous != LBVH_SENTINEL:
                    split = right + 1
                    right = Int(previous)
            else:
                previous = _hploc_atomic_exchange(
                    parent_slots.unsafe_ptr().unsafe_offset(left - 1),
                    UInt32(right),
                )
                if previous != LBVH_SENTINEL:
                    split = left
                    left = Int(previous)

            if previous == LBVH_SENTINEL:
                lane_active = False

        var range_size = right - left + 1
        var final = lane_active and range_size == segment_leaf_count
        var task_mask = hploc_wave_ballot(
            lane_active and (range_size > merging_threshold or final)
        )

        while task_mask != 0:
            var task_lane = hploc_wave_first_lane(task_mask)
            var task_left = Int(
                warp.shuffle_idx(UInt32(left), UInt32(task_lane))
            )
            var task_right = Int(
                warp.shuffle_idx(UInt32(right), UInt32(task_lane))
            )
            var task_split = Int(
                warp.shuffle_idx(UInt32(split), UInt32(task_lane))
            )
            var task_final = Bool(
                warp.shuffle_idx(UInt32(final), UInt32(task_lane))
            )
            var task_segment = Int(
                warp.shuffle_idx(UInt32(segment_idx), UInt32(task_lane))
            )

            # The two child lists are packed at their Morton range starts and
            # contain at most merging_threshold valid entries each.
            var cluster_idx = LBVH_SENTINEL
            var left_span = min(task_split - task_left, merging_threshold)
            var load_left = lane < left_span
            if load_left:
                cluster_idx = cluster_indices.unsafe_get(task_left + lane)
            var num_left = Int(
                pop_count(
                    hploc_wave_ballot(
                        load_left and cluster_idx != LBVH_SENTINEL
                    )
                )
            )

            var right_index = lane - num_left
            var right_span = min(task_right + 1 - task_split, merging_threshold)
            var load_right = right_index >= 0 and right_index < right_span
            if load_right:
                cluster_idx = cluster_indices.unsafe_get(
                    task_split + right_index
                )
            elif lane >= num_left:
                cluster_idx = LBVH_SENTINEL
            var num_right = Int(
                pop_count(
                    hploc_wave_ballot(
                        load_right and cluster_idx != LBVH_SENTINEL
                    )
                )
            )
            var cluster_count = num_left + num_right
            var previous_count = cluster_count
            var threshold = 1 if task_final else merging_threshold

            if lane < cluster_count:
                node_indices_cache[unsafe_offset=shared_lane] = cluster_idx
                store_cached_bounds(
                    shared_lane,
                    _hploc_cluster_bounds(
                        cluster_idx,
                        leaf_bounds,
                        sorted_leaf_ids,
                        scratch_bounds,
                    ),
                )
            else:
                node_indices_cache[unsafe_offset=shared_lane] = LBVH_SENTINEL
                store_cached_bounds(shared_lane, AABB[Frame.WORLD].invalid())
            syncwarp()

            while cluster_count > threshold:
                var active = lane < cluster_count
                var own_bounds = AABB[Frame.WORLD].invalid()
                if active:
                    cluster_idx = node_indices_cache[unsafe_offset=shared_lane]
                    own_bounds = load_cached_bounds(shared_lane)

                distance_offsets[unsafe_offset=shared_lane] = UInt64.MAX
                syncwarp()

                # HIPRT-style pair ownership: lane i evaluates only j > i,
                # then atomically publishes the same packed area to both
                # endpoint minima. Each candidate pair is tested once.
                var own_minimum = UInt64.MAX
                if active:
                    var last_neighbor = min(
                        lane + search_radius, cluster_count - 1
                    )
                    for neighbor_lane in range(lane + 1, last_neighbor + 1):
                        var neighbor_bounds = load_cached_bounds(
                            shared_wave_base + neighbor_lane
                        )
                        var merged = AABB[Frame.WORLD].merge(
                            own_bounds, neighbor_bounds
                        )
                        var area = merged.surface_area()[0]
                        var right_offset = _hploc_encode_offset(
                            lane, neighbor_lane
                        )
                        own_minimum = min(
                            own_minimum,
                            _hploc_pack_distance_offset(area, right_offset),
                        )
                        var left_offset = _hploc_encode_offset(
                            neighbor_lane, lane
                        )
                        Atomic.min[ordering=Ordering.RELAXED](
                            distance_offsets.unsafe_offset(
                                shared_wave_base + neighbor_lane
                            ),
                            _hploc_pack_distance_offset(area, left_offset),
                        )
                    Atomic.min[ordering=Ordering.RELAXED](
                        distance_offsets.unsafe_offset(shared_lane),
                        own_minimum,
                    )

                syncwarp()

                var nearest = WARP_SIZE
                var neighbor_nearest = WARP_SIZE
                if active:
                    nearest = _hploc_decode_offset(
                        lane,
                        UInt32(distance_offsets[unsafe_offset=shared_lane]),
                    )
                    neighbor_nearest = _hploc_decode_offset(
                        nearest,
                        UInt32(
                            distance_offsets[
                                unsafe_offset=(shared_wave_base + nearest)
                            ]
                        ),
                    )
                var mutual = active and lane == neighbor_nearest
                var merge = mutual and lane < nearest
                var merge_mask = hploc_wave_ballot(merge)
                var merge_count = Int(pop_count(merge_mask))

                if merge_count == 0:
                    status.unsafe_get(task_segment) = UInt32(
                        HPLOC_STATUS_NO_PROGRESS
                    )
                    return

                var allocation_base = UInt32(0)
                if lane == 0:
                    allocation_base = Atomic.fetch_add[
                        ordering=Ordering.RELAXED
                    ](
                        node_counter.unsafe_ptr().unsafe_offset(task_segment),
                        UInt32(merge_count),
                    )
                allocation_base = warp.shuffle_idx(allocation_base, 0)

                var output_idx = LBVH_SENTINEL
                var output_bounds = own_bounds
                if merge:
                    var neighbor_idx = node_indices_cache[
                        unsafe_offset=shared_wave_base + nearest
                    ]
                    var neighbor_bounds = load_cached_bounds(
                        shared_wave_base + nearest
                    )
                    output_bounds.grow(neighbor_bounds)
                    var node_idx = (
                        internal_segment_offsets.unsafe_get(task_segment)
                        + allocation_base
                        + UInt32(hploc_wave_rank(merge_mask, lane))
                    )
                    _hploc_store_bounds(scratch_bounds, node_idx, output_bounds)

                    var meta = Int(node_idx) * BinaryBvhNode.META_STRIDE
                    node_meta.unsafe_get(
                        meta + BinaryBvhNode.PARENT
                    ) = LBVH_SENTINEL
                    node_meta.unsafe_get(
                        meta + BinaryBvhNode.LEFT
                    ) = cluster_idx
                    node_meta.unsafe_get(
                        meta + BinaryBvhNode.RIGHT
                    ) = neighbor_idx
                    node_meta.unsafe_get(
                        meta + BinaryBvhNode.FENCE
                    ) = LBVH_SENTINEL

                    var bounds_base = (
                        Int(node_idx) * BinaryBvhNode.BOUNDS_STRIDE
                    )
                    own_bounds.store6(node_bounds, bounds_base)
                    neighbor_bounds.store6(
                        node_bounds, bounds_base + AABB.STRIDE
                    )
                    node_flags.unsafe_get(Int(node_idx)) = UInt32(2)
                    node_leaf_counts.unsafe_get(
                        Int(node_idx)
                    ) = _hploc_cluster_leaf_count(
                        cluster_idx, node_leaf_counts
                    ) + _hploc_cluster_leaf_count(
                        neighbor_idx, node_leaf_counts
                    )
                    _hploc_set_child_parent(
                        cluster_idx, node_idx, node_meta, leaf_parent
                    )
                    _hploc_set_child_parent(
                        neighbor_idx, node_idx, node_meta, leaf_parent
                    )
                    output_idx = encode_internal_ref(node_idx)
                elif active and not mutual:
                    output_idx = cluster_idx

                var keep = output_idx != LBVH_SENTINEL
                var keep_mask = hploc_wave_ballot(keep)
                var compacted_lane = Int(hploc_wave_rank(keep_mask, lane))
                syncwarp()
                if keep:
                    var compacted_shared_lane = (
                        shared_wave_base + compacted_lane
                    )
                    node_indices_cache[
                        unsafe_offset=compacted_shared_lane
                    ] = output_idx
                    store_cached_bounds(compacted_shared_lane, output_bounds)
                syncwarp()

                cluster_count = Int(pop_count(keep_mask))
                cluster_idx = (
                    node_indices_cache[unsafe_offset=shared_lane] if lane
                    < cluster_count else LBVH_SENTINEL
                )

            if lane < previous_count:
                cluster_indices.unsafe_get(task_left + lane) = cluster_idx

            # Equivalent to the paper implementation's __threadfence before
            # a task owner advertises this completed child at its next parent.
            fence[ordering=Ordering.SEQUENTIAL]()

            if task_final and lane == 0:
                root.unsafe_get(task_segment) = cluster_idx

            task_mask &= task_mask - UInt64(1)


def finalize_hploc_multi_wave_kernel(
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    internal_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_counter: ImmSpan[UInt32, ImmutAnyOrigin],
    root: ImmSpan[UInt32, ImmutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    var segment_idx = global_idx.x
    if segment_idx >= len(segment_offsets) - 1 or status.unsafe_get(
        segment_idx
    ) != UInt32(HPLOC_STATUS_OK):
        return
    var leaf_count = segment_offsets.unsafe_get(
        segment_idx + 1
    ) - segment_offsets.unsafe_get(segment_idx)
    var internal_count = internal_segment_offsets.unsafe_get(
        segment_idx + 1
    ) - internal_segment_offsets.unsafe_get(segment_idx)
    if (
        node_counter.unsafe_get(segment_idx) != internal_count
        or root.unsafe_get(segment_idx) == LBVH_SENTINEL
    ):
        if leaf_count == 0 and internal_count == 0:
            return
        status.unsafe_get(segment_idx) = UInt32(HPLOC_STATUS_INVALID_RESULT)


struct GpuHplocBuildState[
    search_radius: Int = HPLOC_SEARCH_RADIUS,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
]:
    """Scratch and completion state for a direct production-layout build."""

    var leaf_count: Int
    var internal_count: Int
    var segments: SegmentOffsets
    var internal_segments: SegmentOffsets
    var segment_offsets: DeviceBuffer[DType.uint32]
    var internal_segment_offsets: DeviceBuffer[DType.uint32]
    var scratch_bounds: DeviceBuffer[DType.float32]
    var parent_slots: DeviceBuffer[DType.uint32]
    var cluster_indices: DeviceBuffer[DType.uint32]
    var node_counter: DeviceBuffer[DType.uint32]
    var root: DeviceBuffer[DType.uint32]
    var status: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        sorted_morton_codes: DeviceBuffer[DType.uint32],
        sorted_leaf_ids: DeviceBuffer[DType.uint32],
        node_meta: DeviceBuffer[DType.uint32],
        leaf_parent: DeviceBuffer[DType.uint32],
        node_bounds: DeviceBuffer[DType.float32],
        node_flags: DeviceBuffer[DType.uint32],
        node_leaf_counts: DeviceBuffer[DType.uint32],
    ) raises:
        var segments = SegmentOffsets.single(len(sorted_leaf_ids))
        var internal_segments = SegmentOffsets.single(
            max(len(sorted_leaf_ids) - 1, 0)
        )
        var segment_offsets = upload_list(ctx, segments.offsets)
        var internal_segment_offsets = upload_list(
            ctx, internal_segments.offsets
        )
        self = Self(
            ctx,
            leaf_bounds,
            sorted_morton_codes,
            sorted_leaf_ids,
            segments,
            segment_offsets,
            internal_segments,
            internal_segment_offsets,
            node_meta,
            leaf_parent,
            node_bounds,
            node_flags,
            node_leaf_counts,
        )

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        sorted_morton_codes: DeviceBuffer[DType.uint32],
        sorted_leaf_ids: DeviceBuffer[DType.uint32],
        segments: SegmentOffsets,
        segment_offsets: DeviceBuffer[DType.uint32],
        internal_segments: SegmentOffsets,
        internal_segment_offsets: DeviceBuffer[DType.uint32],
        node_meta: DeviceBuffer[DType.uint32],
        leaf_parent: DeviceBuffer[DType.uint32],
        node_bounds: DeviceBuffer[DType.float32],
        node_flags: DeviceBuffer[DType.uint32],
        node_leaf_counts: DeviceBuffer[DType.uint32],
    ) raises:
        self.leaf_count = len(sorted_leaf_ids)
        self.segments = segments.copy()
        self.internal_segments = internal_segments.copy()
        self.segment_offsets = segment_offsets
        self.internal_segment_offsets = internal_segment_offsets
        self.internal_count = self.internal_segments.item_count()
        if self.leaf_count <= 0:
            raise "multi-wave H-PLOC requires at least one leaf"
        if (
            len(sorted_morton_codes) != self.leaf_count
            or len(leaf_bounds) != self.leaf_count * AABB.STRIDE
        ):
            raise "multi-wave H-PLOC input lengths do not match"
        if (
            self.segments.segment_count() <= 0
            or self.segments.item_count() != self.leaf_count
            or self.internal_segments.segment_count()
            != self.segments.segment_count()
            or len(self.segment_offsets) != self.segments.segment_count() + 1
            or len(self.internal_segment_offsets)
            != self.segments.segment_count() + 1
        ):
            raise "multi-wave H-PLOC segment layout does not match input"
        if (
            Self.search_radius <= 0
            or Self.merging_threshold <= 0
            or Self.merging_threshold > WARP_SIZE / 2
        ):
            raise "H-PLOC threshold must be in 1..WARP_SIZE/2"

        var internal_capacity = max(self.internal_count, 1)
        if (
            len(node_meta) < internal_capacity * BinaryBvhNode.META_STRIDE
            or len(leaf_parent) < self.leaf_count
            or len(node_bounds)
            < internal_capacity * BinaryBvhNode.BOUNDS_STRIDE
            or len(node_flags) < internal_capacity
            or len(node_leaf_counts) < internal_capacity
        ):
            raise "multi-wave H-PLOC production output is too short"

        self.scratch_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_capacity * AABB.STRIDE
        )
        self.parent_slots = ctx.enqueue_create_buffer[DType.uint32](
            self.leaf_count
        )
        self.cluster_indices = ctx.enqueue_create_buffer[DType.uint32](
            self.leaf_count
        )
        self.root = ctx.enqueue_create_buffer[DType.uint32](
            self.segments.segment_count()
        )
        self.status = ctx.enqueue_create_buffer[DType.uint32](
            self.segments.segment_count()
        )
        self.node_counter = ctx.enqueue_create_buffer[DType.uint32](
            self.segments.segment_count()
        )

        var init_blocks = ceildiv(
            max(self.leaf_count, self.segments.segment_count()),
            HPLOC_MULTI_WAVE_BLOCK_SIZE,
        )
        ctx.enqueue_function[init_hploc_multi_wave_kernel](
            _device_span[mut=False](sorted_leaf_ids),
            _device_span[mut=False](self.segment_offsets),
            _device_span[mut=True](self.parent_slots),
            _device_span[mut=True](self.cluster_indices),
            _device_span[mut=True](leaf_parent),
            _device_span[mut=True](self.node_counter),
            _device_span[mut=True](self.root),
            _device_span[mut=True](self.status),
            grid_dim=init_blocks,
            block_dim=HPLOC_MULTI_WAVE_BLOCK_SIZE,
        )
        var build_blocks = ceildiv(self.leaf_count, HPLOC_MULTI_WAVE_BLOCK_SIZE)
        ctx.enqueue_function[
            build_hploc_multi_wave_kernel[
                HPLOC_MULTI_WAVE_BLOCK_SIZE,
                Self.search_radius,
                Self.merging_threshold,
            ]
        ](
            _device_span[mut=False](sorted_morton_codes),
            _device_span[mut=False](self.segment_offsets),
            _device_span[mut=False](self.internal_segment_offsets),
            _device_span[mut=False](leaf_bounds),
            _device_span[mut=False](sorted_leaf_ids),
            _device_span[mut=True](self.parent_slots),
            _device_span[mut=True](self.cluster_indices),
            _device_span[mut=True](self.scratch_bounds),
            _device_span[mut=True](node_meta),
            _device_span[mut=True](leaf_parent),
            _device_span[mut=True](node_bounds),
            _device_span[mut=True](node_flags),
            _device_span[mut=True](node_leaf_counts),
            _device_span[mut=True](self.node_counter),
            _device_span[mut=True](self.root),
            _device_span[mut=True](self.status),
            grid_dim=build_blocks,
            block_dim=HPLOC_MULTI_WAVE_BLOCK_SIZE,
        )
        ctx.enqueue_function[finalize_hploc_multi_wave_kernel](
            _device_span[mut=False](self.segment_offsets),
            _device_span[mut=False](self.internal_segment_offsets),
            _device_span[mut=False](self.node_counter),
            _device_span[mut=False](self.root),
            _device_span[mut=True](self.status),
            grid_dim=ceildiv(
                self.segments.segment_count(), HPLOC_MULTI_WAVE_BLOCK_SIZE
            ),
            block_dim=HPLOC_MULTI_WAVE_BLOCK_SIZE,
        )

    def result_status(self) raises -> UInt32:
        with self.status.map_to_host() as host:
            for segment_idx in range(len(host)):
                if host[segment_idx] != UInt32(HPLOC_STATUS_OK):
                    return host[segment_idx]
            return UInt32(HPLOC_STATUS_OK)

    def result_root(self, segment_idx: Int = 0) raises -> UInt32:
        with self.root.map_to_host() as host:
            return host[segment_idx]

    def result_node_count(self) raises -> UInt32:
        with self.node_counter.map_to_host() as host:
            var total = UInt32(0)
            for segment_idx in range(len(host)):
                total += host[segment_idx]
            return total


struct GpuHplocMultiWaveBvh[
    search_radius: Int = HPLOC_SEARCH_RADIUS,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
]:
    """Standalone direct-layout H-PLOC builder used by correctness tests."""

    var leaf_count: Int
    var leaf_bounds: DeviceBuffer[DType.float32]
    var sorted_morton_codes: DeviceBuffer[DType.uint32]
    var sorted_leaf_ids: DeviceBuffer[DType.uint32]
    var node_meta: DeviceBuffer[DType.uint32]
    var leaf_parent: DeviceBuffer[DType.uint32]
    var node_bounds: DeviceBuffer[DType.float32]
    var node_flags: DeviceBuffer[DType.uint32]
    var node_leaf_counts: DeviceBuffer[DType.uint32]
    var state: GpuHplocBuildState[Self.search_radius, Self.merging_threshold]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        sorted_morton_codes: DeviceBuffer[DType.uint32],
        sorted_leaf_ids: DeviceBuffer[DType.uint32],
    ) raises:
        self.leaf_count = len(sorted_leaf_ids)
        self.leaf_bounds = leaf_bounds
        self.sorted_morton_codes = sorted_morton_codes
        self.sorted_leaf_ids = sorted_leaf_ids

        var internal_capacity = max(self.leaf_count - 1, 1)
        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_capacity * BinaryBvhNode.META_STRIDE
        )
        self.leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
            self.leaf_count
        )
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_capacity * BinaryBvhNode.BOUNDS_STRIDE
        )
        self.node_flags = ctx.enqueue_create_buffer[DType.uint32](
            internal_capacity
        )
        self.node_leaf_counts = ctx.enqueue_create_buffer[DType.uint32](
            internal_capacity
        )
        self.state = GpuHplocBuildState[
            Self.search_radius, Self.merging_threshold
        ](
            ctx,
            self.leaf_bounds.copy(),
            self.sorted_morton_codes.copy(),
            self.sorted_leaf_ids.copy(),
            self.node_meta.copy(),
            self.leaf_parent.copy(),
            self.node_bounds.copy(),
            self.node_flags.copy(),
            self.node_leaf_counts.copy(),
        )

    def result_status(self) raises -> UInt32:
        return self.state.result_status()

    def result_root(self) raises -> UInt32:
        return self.state.result_root()

    def result_node_count(self) raises -> UInt32:
        return self.state.result_node_count()
