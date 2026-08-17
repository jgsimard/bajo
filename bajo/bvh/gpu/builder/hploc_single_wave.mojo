from std.bit import pop_count
from std.gpu import WARP_SIZE, lane_id
from std.gpu.primitives import warp
from max.gpu import barrier
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import LBVH_SENTINEL
from bajo.bvh.gpu.builder.hploc_wave import (
    hploc_wave_ballot,
    hploc_wave_rank,
)
from bajo.bvh.gpu.builder.hploc_layout import (
    HPLOC_NODE_META_STRIDE,
    HPLOC_NODE_PARENT,
    HPLOC_NODE_LEFT,
    HPLOC_NODE_RIGHT,
    HPLOC_NODE_LEAF_ID,
    HPLOC_STATUS_OK,
    HPLOC_STATUS_NO_PROGRESS,
    HPLOC_STATUS_INVALID_RESULT,
    HPLOC_MERGING_THRESHOLD,
    HPLOC_SEARCH_RADIUS,
    _hploc_meta_base,
    _hploc_load_bounds,
    _hploc_store_bounds,
)
from bajo.bvh.gpu.builder.lbvh import _lbvh_find_range, _lbvh_find_split
from bajo.bvh.gpu.utils import _device_span
from bajo.core import AABB, Frame


comptime HPLOC_GUIDE_STRIDE = 3
comptime HPLOC_GUIDE_FIRST = 0
comptime HPLOC_GUIDE_LAST = 1
comptime HPLOC_GUIDE_SPLIT = 2


def init_hploc_single_wave_kernel[
    block_size: Int,
](
    leaf_bounds: ImmSpan[Float32, ImmutAnyOrigin],
    sorted_morton_codes: ImmSpan[UInt32, ImmutAnyOrigin],
    sorted_leaf_ids: ImmSpan[UInt32, ImmutAnyOrigin],
    node_bounds: MutSpan[Float32, MutAnyOrigin],
    node_meta: MutSpan[UInt32, MutAnyOrigin],
    guide_tasks: MutSpan[UInt32, MutAnyOrigin],
    cluster_indices: MutSpan[UInt32, MutAnyOrigin],
    cluster_counts: MutSpan[UInt32, MutAnyOrigin],
    node_counter: MutSpan[UInt32, MutAnyOrigin],
    root: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    comptime assert block_size == WARP_SIZE
    var lane = lane_id()
    var leaf_count = len(sorted_leaf_ids)

    if lane < leaf_count:
        var leaf_id = sorted_leaf_ids.unsafe_get(lane)
        var bounds = AABB[Frame.WORLD].load6(
            leaf_bounds, Int(leaf_id) * AABB.STRIDE
        )
        _hploc_store_bounds(node_bounds, leaf_id, bounds)

        var meta = _hploc_meta_base(leaf_id)
        node_meta.unsafe_get(meta + HPLOC_NODE_PARENT) = LBVH_SENTINEL
        node_meta.unsafe_get(meta + HPLOC_NODE_LEFT) = LBVH_SENTINEL
        node_meta.unsafe_get(meta + HPLOC_NODE_RIGHT) = LBVH_SENTINEL
        node_meta.unsafe_get(meta + HPLOC_NODE_LEAF_ID) = leaf_id
        cluster_indices.unsafe_get(lane) = leaf_id
        cluster_counts.unsafe_get(lane) = UInt32(1)

    if lane < leaf_count - 1:
        var first, last = _lbvh_find_range(sorted_morton_codes, lane)
        var split = _lbvh_find_split(sorted_morton_codes, first, last)
        var task = lane * HPLOC_GUIDE_STRIDE
        guide_tasks.unsafe_get(task + HPLOC_GUIDE_FIRST) = UInt32(first)
        guide_tasks.unsafe_get(task + HPLOC_GUIDE_LAST) = UInt32(last)
        guide_tasks.unsafe_get(task + HPLOC_GUIDE_SPLIT) = UInt32(split)

    if lane == 0:
        node_counter.unsafe_get(0) = UInt32(leaf_count)
        root.unsafe_get(0) = (
            sorted_leaf_ids.unsafe_get(0) if leaf_count == 1 else LBVH_SENTINEL
        )
        status.unsafe_get(0) = UInt32(HPLOC_STATUS_OK)


def build_hploc_single_wave_kernel[
    block_size: Int,
](
    guide_tasks: ImmSpan[UInt32, ImmutAnyOrigin],
    cluster_indices: MutSpan[UInt32, MutAnyOrigin],
    cluster_counts: MutSpan[UInt32, MutAnyOrigin],
    nearest_scratch: MutSpan[UInt32, MutAnyOrigin],
    compact_scratch: MutSpan[UInt32, MutAnyOrigin],
    allocation_base: MutSpan[UInt32, MutAnyOrigin],
    node_bounds: MutSpan[Float32, MutAnyOrigin],
    node_meta: MutSpan[UInt32, MutAnyOrigin],
    node_counter: MutSpan[UInt32, MutAnyOrigin],
    root: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
    leaf_count_i32: Int32,
    search_radius_i32: Int32,
    merging_threshold_i32: Int32,
):
    """Correctness-first H-PLOC prototype for one physical wave.

    Guide ranges are processed bottom-up by span. Every lane participates in
    every subgroup primitive; lanes outside the current cluster use sentinels.
    """

    comptime assert block_size == WARP_SIZE
    var lane = lane_id()
    var leaf_count = Int(leaf_count_i32)
    var search_radius = Int(search_radius_i32)
    var merging_threshold = Int(merging_threshold_i32)

    if leaf_count == 1:
        return

    for span in range(2, leaf_count + 1):
        for task_idx in range(leaf_count - 1):
            var task = task_idx * HPLOC_GUIDE_STRIDE
            var first = Int(guide_tasks.unsafe_get(task + HPLOC_GUIDE_FIRST))
            var last = Int(guide_tasks.unsafe_get(task + HPLOC_GUIDE_LAST))
            if last - first + 1 != span:
                continue

            var split = Int(guide_tasks.unsafe_get(task + HPLOC_GUIDE_SPLIT))
            var left_count = Int(cluster_counts.unsafe_get(first))
            var right_count = Int(cluster_counts.unsafe_get(split + 1))
            var cluster_count = left_count + right_count

            # Stage the two child lists before packing them. Directly packing
            # in cluster_indices can overwrite a right-child source.
            if lane < cluster_count:
                var node_idx = UInt32(0)
                if lane < left_count:
                    node_idx = cluster_indices.unsafe_get(first + lane)
                else:
                    node_idx = cluster_indices.unsafe_get(
                        split + 1 + lane - left_count
                    )
                compact_scratch.unsafe_get(lane) = node_idx
            barrier()

            var active = lane < cluster_count
            var cluster_idx = compact_scratch.unsafe_get(
                lane
            ) if active else LBVH_SENTINEL
            barrier()

            var threshold = merging_threshold
            if first == 0 and last == leaf_count - 1:
                threshold = 1

            while cluster_count > threshold:
                # Keep the current cluster list visible to neighbor lanes.
                if active:
                    compact_scratch.unsafe_get(lane) = cluster_idx
                barrier()

                var best_area = Float32.MAX
                var nearest = UInt32(WARP_SIZE)
                if active:
                    var own_bounds = _hploc_load_bounds(
                        node_bounds, cluster_idx
                    )
                    for radius in range(1, search_radius + 1):
                        var right = lane + radius
                        if right < cluster_count:
                            var candidate = compact_scratch.unsafe_get(right)
                            var merged = AABB[Frame.WORLD].merge(
                                own_bounds,
                                _hploc_load_bounds(node_bounds, candidate),
                            )
                            var area = merged.surface_area()[0]
                            if area < best_area:
                                best_area = area
                                nearest = UInt32(right)

                        var left = lane - radius
                        if left >= 0:
                            var candidate = compact_scratch.unsafe_get(left)
                            var merged = AABB[Frame.WORLD].merge(
                                own_bounds,
                                _hploc_load_bounds(node_bounds, candidate),
                            )
                            var area = merged.surface_area()[0]
                            if area < best_area:
                                best_area = area
                                nearest = UInt32(left)

                nearest_scratch.unsafe_get(lane) = nearest
                barrier()

                var mutual = False
                if active and nearest < UInt32(cluster_count):
                    mutual = nearest_scratch.unsafe_get(Int(nearest)) == UInt32(
                        lane
                    )
                var merge = mutual and UInt32(lane) < nearest
                var merge_mask = hploc_wave_ballot(merge)
                var merge_count = Int(pop_count(merge_mask))

                if merge_count == 0:
                    if lane == 0:
                        status.unsafe_get(0) = UInt32(HPLOC_STATUS_NO_PROGRESS)
                    return

                if lane == 0:
                    allocation_base.unsafe_get(0) = node_counter.unsafe_get(0)
                    node_counter.unsafe_get(0) += UInt32(merge_count)
                barrier()

                if merge:
                    var node_idx = allocation_base.unsafe_get(0) + UInt32(
                        hploc_wave_rank(merge_mask, lane)
                    )
                    var right_idx = compact_scratch.unsafe_get(Int(nearest))
                    var merged_bounds = AABB[Frame.WORLD].merge(
                        _hploc_load_bounds(node_bounds, cluster_idx),
                        _hploc_load_bounds(node_bounds, right_idx),
                    )
                    _hploc_store_bounds(node_bounds, node_idx, merged_bounds)

                    var meta = _hploc_meta_base(node_idx)
                    node_meta.unsafe_get(
                        meta + HPLOC_NODE_PARENT
                    ) = LBVH_SENTINEL
                    node_meta.unsafe_get(meta + HPLOC_NODE_LEFT) = cluster_idx
                    node_meta.unsafe_get(meta + HPLOC_NODE_RIGHT) = right_idx
                    node_meta.unsafe_get(
                        meta + HPLOC_NODE_LEAF_ID
                    ) = LBVH_SENTINEL
                    node_meta.unsafe_get(
                        _hploc_meta_base(cluster_idx) + HPLOC_NODE_PARENT
                    ) = node_idx
                    node_meta.unsafe_get(
                        _hploc_meta_base(right_idx) + HPLOC_NODE_PARENT
                    ) = node_idx
                    cluster_idx = node_idx
                barrier()

                # Same stable ballot/scan compaction pattern as OneSweep:
                # lower list position survives and replaced pairs stay ordered.
                var keep = active and (merge or not mutual)
                var compact_offset = warp.prefix_sum[exclusive=True](
                    UInt32(keep)
                )
                if keep:
                    compact_scratch.unsafe_get(
                        Int(compact_offset)
                    ) = cluster_idx
                barrier()

                cluster_count -= merge_count
                active = lane < cluster_count
                cluster_idx = compact_scratch.unsafe_get(
                    lane
                ) if active else LBVH_SENTINEL
                barrier()

            if active:
                cluster_indices.unsafe_get(first + lane) = cluster_idx
            if lane == 0:
                cluster_counts.unsafe_get(first) = UInt32(cluster_count)
            barrier()

    if lane == 0:
        if (
            node_counter.unsafe_get(0) == UInt32(leaf_count * 2 - 1)
            and cluster_counts.unsafe_get(0) == 1
        ):
            root.unsafe_get(0) = cluster_indices.unsafe_get(0)
        else:
            status.unsafe_get(0) = UInt32(HPLOC_STATUS_INVALID_RESULT)


struct GpuHplocSingleWaveBvh:
    """Standalone one-wave H-PLOC BVH2 used only as a correctness gate."""

    var leaf_count: Int
    var leaf_bounds: DeviceBuffer[DType.float32]
    var sorted_morton_codes: DeviceBuffer[DType.uint32]
    var sorted_leaf_ids: DeviceBuffer[DType.uint32]

    var node_bounds: DeviceBuffer[DType.float32]
    var node_meta: DeviceBuffer[DType.uint32]
    var guide_tasks: DeviceBuffer[DType.uint32]
    var cluster_indices: DeviceBuffer[DType.uint32]
    var cluster_counts: DeviceBuffer[DType.uint32]
    var nearest_scratch: DeviceBuffer[DType.uint32]
    var compact_scratch: DeviceBuffer[DType.uint32]
    var allocation_base: DeviceBuffer[DType.uint32]
    var node_counter: DeviceBuffer[DType.uint32]
    var root: DeviceBuffer[DType.uint32]
    var status: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        sorted_morton_codes: DeviceBuffer[DType.uint32],
        sorted_leaf_ids: DeviceBuffer[DType.uint32],
        search_radius: Int = HPLOC_SEARCH_RADIUS,
        merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
    ) raises:
        self.leaf_count = len(sorted_leaf_ids)
        if self.leaf_count <= 0 or self.leaf_count > WARP_SIZE:
            raise "single-wave H-PLOC requires 1..WARP_SIZE leaves"
        if (
            len(sorted_morton_codes) != self.leaf_count
            or len(leaf_bounds) != self.leaf_count * AABB.STRIDE
        ):
            raise "single-wave H-PLOC input lengths do not match"
        if search_radius <= 0 or merging_threshold <= 0:
            raise "single-wave H-PLOC parameters must be positive"

        self.leaf_bounds = leaf_bounds
        self.sorted_morton_codes = sorted_morton_codes
        self.sorted_leaf_ids = sorted_leaf_ids

        var node_count = self.leaf_count * 2 - 1
        var guide_count = max(self.leaf_count - 1, 1)
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            node_count * AABB.STRIDE
        )
        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            node_count * HPLOC_NODE_META_STRIDE
        )
        self.guide_tasks = ctx.enqueue_create_buffer[DType.uint32](
            guide_count * HPLOC_GUIDE_STRIDE
        )
        self.cluster_indices = ctx.enqueue_create_buffer[DType.uint32](
            self.leaf_count
        )
        self.cluster_counts = ctx.enqueue_create_buffer[DType.uint32](
            self.leaf_count
        )
        self.nearest_scratch = ctx.enqueue_create_buffer[DType.uint32](
            WARP_SIZE
        )
        self.compact_scratch = ctx.enqueue_create_buffer[DType.uint32](
            WARP_SIZE
        )
        self.allocation_base = ctx.enqueue_create_buffer[DType.uint32](1)
        self.node_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        self.root = ctx.enqueue_create_buffer[DType.uint32](1)
        self.status = ctx.enqueue_create_buffer[DType.uint32](1)

        ctx.enqueue_function[init_hploc_single_wave_kernel[WARP_SIZE]](
            _device_span[mut=False](self.leaf_bounds),
            _device_span[mut=False](self.sorted_morton_codes),
            _device_span[mut=False](self.sorted_leaf_ids),
            _device_span[mut=True](self.node_bounds),
            _device_span[mut=True](self.node_meta),
            _device_span[mut=True](self.guide_tasks),
            _device_span[mut=True](self.cluster_indices),
            _device_span[mut=True](self.cluster_counts),
            _device_span[mut=True](self.node_counter),
            _device_span[mut=True](self.root),
            _device_span[mut=True](self.status),
            grid_dim=1,
            block_dim=WARP_SIZE,
        )
        ctx.enqueue_function[build_hploc_single_wave_kernel[WARP_SIZE]](
            _device_span[mut=False](self.guide_tasks),
            _device_span[mut=True](self.cluster_indices),
            _device_span[mut=True](self.cluster_counts),
            _device_span[mut=True](self.nearest_scratch),
            _device_span[mut=True](self.compact_scratch),
            _device_span[mut=True](self.allocation_base),
            _device_span[mut=True](self.node_bounds),
            _device_span[mut=True](self.node_meta),
            _device_span[mut=True](self.node_counter),
            _device_span[mut=True](self.root),
            _device_span[mut=True](self.status),
            Int32(self.leaf_count),
            Int32(search_radius),
            Int32(merging_threshold),
            grid_dim=1,
            block_dim=WARP_SIZE,
        )

    def result_status(self) raises -> UInt32:
        with self.status.map_to_host() as host:
            return host[0]

    def result_root(self) raises -> UInt32:
        with self.root.map_to_host() as host:
            return host[0]

    def result_node_count(self) raises -> UInt32:
        with self.node_counter.map_to_host() as host:
            return host[0]
