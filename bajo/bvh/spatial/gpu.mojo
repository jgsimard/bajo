"""GPU ownership for the shared spatial-query representation.

Construction and host-input rebuild reuse the CPU spatial builder. Device-input
updates run entirely on device with retained scratch and optional graph replay.
Synchronous wrappers check validation status; enqueue/graph APIs require the
caller to serialize updates/queries and inspect status after completion.
"""

from max.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    DeviceGraph,
    DeviceGraphBuilder,
)
from bajo.gpu.buffers import upload_list, _device_span, copy_to_device
from bajo.core import AABB, Frame
from std.math import ceildiv

from .device_update import (
    UPDATE_BLOCK_SIZE,
    bounds_update_kernel,
    reorder_bounds_kernel,
)
from .parallel_refit import (
    forest_parents_kernel,
    clear_refit_kernel,
    validate_bounds_refit_kernel,
    publish_bounds_refit_kernel,
)
from .bounds import SpatialBoundsBvh


# Retain one launch below the measured multi-block crossover. Parent metadata
# is allocated once for larger owners; updates and graph replay allocate none.
comptime _PARALLEL_BOUNDS_MIN_ITEMS = 1024


@fieldwise_init
struct GpuSpatialBoundsBvh[frame: Frame = .WORLD]:
    var host: SpatialBoundsBvh[Self.frame]
    var bounds: DeviceBuffer[.float32]
    var nodes: DeviceBuffer[.uint32]
    var update_items: DeviceBuffer[.float32]
    var update_order: DeviceBuffer[.uint32]
    var update_parents: DeviceBuffer[.uint32]
    var update_status: DeviceBuffer[.uint32]

    def __init__(
        out self, mut ctx: DeviceContext, var host: SpatialBoundsBvh[Self.frame]
    ) raises:
        self.bounds = upload_list(ctx, host.bounds)
        self.nodes = upload_list(ctx, host.nodes)
        self.update_items = ctx.enqueue_create_buffer[.float32](
            len(host.item_bounds) * 6
        )
        self.update_order = ctx.enqueue_create_buffer[.uint32](
            len(host.nodes) // 2
        )
        self.update_status = upload_list(ctx, List[UInt32](length=1, fill=0))
        self.update_parents = ctx.enqueue_create_buffer[.uint32](
            len(host.nodes) // 2 if len(host.item_bounds)
            >= _PARALLEL_BOUNDS_MIN_ITEMS else 0
        )
        self.host = host^
        self._enqueue_parents(ctx)
        ctx.synchronize()

    def _enqueue_parents(self, ctx: DeviceContext) raises:
        if len(self.update_parents) > 0:
            ctx.enqueue_function[forest_parents_kernel](
                _device_span[mut=False](self.nodes),
                _device_span[mut=True](self.update_parents),
                grid_dim=ceildiv(len(self.update_parents), UPDATE_BLOCK_SIZE),
                block_dim=UPDATE_BLOCK_SIZE,
            )

    def group_root(self, group: UInt32) -> UInt32:
        return self.host.group_root(group)

    def refit(
        mut self, ctx: DeviceContext, boxes: ImmSpan[AABB[Self.frame], _]
    ) raises:
        """Refit host-provided bounds, then update existing device storage."""
        ctx.synchronize()
        self.host.refit(boxes)
        with self.update_items.map_to_host() as data:
            for item in range(len(boxes)):
                for axis in range(3):
                    data[item * 6 + axis] = boxes[item]._min[axis]
                    data[item * 6 + axis + 3] = boxes[item]._max[axis]
        self.refit_device(ctx, self.update_items)

    def rebuild(
        mut self, ctx: DeviceContext, boxes: ImmSpan[AABB[Self.frame], _]
    ) raises:
        """Rebuild with fixed count/groups and stable device buffer addresses.
        """
        ctx.synchronize()
        self.host.rebuild(boxes)
        copy_to_device(self.bounds, self.host.bounds)
        copy_to_device(self.nodes, self.host.nodes)
        self._enqueue_parents(ctx)
        ctx.synchronize()

    def enqueue_update[
        rebuild: Bool, block_size: Int = UPDATE_BLOCK_SIZE
    ](self, ctx: DeviceContext, boxes: DeviceBuffer[.float32]) raises:
        """Enqueue an allocation-free update; inspect status after completion.

        boxes packs [min_x,min_y,min_z,max_x,max_y,max_z] per original item.
        Input and owned buffers must remain alive through execution/replay.
        Device updates leave the host snapshot unchanged. A failed input sets
        update_status[0]=1 and leaves published geometry/topology unchanged.
        """
        comptime assert (
            block_size >= 32
            and block_size <= 1024
            and (block_size & (block_size - 1)) == 0
        )
        if len(boxes) != len(self.update_items):
            raise Error("Device BVH update requires unchanged item count")
        if len(boxes) // 6 >= _PARALLEL_BOUNDS_MIN_ITEMS:
            self._enqueue_parallel_update[rebuild, block_size](ctx, boxes)
            return
        ctx.enqueue_function[bounds_update_kernel[rebuild, block_size]](
            _device_span[mut=False](boxes),
            _device_span[mut=True](self.update_items),
            _device_span[mut=True](self.bounds),
            _device_span[mut=True](self.nodes),
            _device_span[mut=True](self.update_order),
            _device_span[mut=True](self.update_status),
            grid_dim=1,
            block_dim=block_size,
        )

    def _enqueue_parallel_update[
        rebuild: Bool, block_size: Int
    ](self, ctx: DeviceContext, boxes: DeviceBuffer[.float32]) raises:
        # Separate launches validate every item before bounds or leaf IDs can
        # change. update_order is retained scratch, reused as arrival counters.
        ctx.enqueue_function[clear_refit_kernel](
            _device_span[mut=True](self.update_order),
            _device_span[mut=True](self.update_status),
            grid_dim=ceildiv(len(self.update_order), block_size),
            block_dim=block_size,
        )
        ctx.enqueue_function[validate_bounds_refit_kernel](
            _device_span[mut=False](boxes),
            _device_span[mut=True](self.update_items),
            _device_span[mut=True](self.update_status),
            grid_dim=ceildiv(len(boxes) // 6, block_size),
            block_dim=block_size,
        )
        comptime if rebuild:
            ctx.enqueue_function[reorder_bounds_kernel[block_size]](
                _device_span[mut=False](self.update_items),
                _device_span[mut=True](self.nodes),
                _device_span[mut=True](self.update_order),
                _device_span[mut=False](self.update_status),
                grid_dim=1,
                block_dim=block_size,
            )
        ctx.enqueue_function[publish_bounds_refit_kernel](
            _device_span[mut=False](self.update_items),
            _device_span[mut=True](self.bounds),
            _device_span[mut=False](self.nodes),
            _device_span[mut=False](self.update_parents),
            _device_span[mut=True](self.update_order),
            _device_span[mut=False](self.update_status),
            grid_dim=ceildiv(len(self.update_parents), block_size),
            block_dim=block_size,
        )

    def check_update(self, ctx: DeviceContext) raises:
        ctx.synchronize()
        with self.update_status.map_to_host() as status:
            if status[0] != 0:
                raise Error("Device BVH update rejected invalid bounds")

    def refit_device(
        self, ctx: DeviceContext, boxes: DeviceBuffer[.float32]
    ) raises:
        self.enqueue_update[False](ctx, boxes)
        self.check_update(ctx)

    def rebuild_device(
        self, ctx: DeviceContext, boxes: DeviceBuffer[.float32]
    ) raises:
        self.enqueue_update[True](ctx, boxes)
        self.check_update(ctx)

    def create_update_graph[
        rebuild: Bool, block_size: Int = UPDATE_BLOCK_SIZE
    ](
        self, ctx: DeviceContext, input: DeviceBuffer[.float32]
    ) raises -> DeviceGraph:
        """Capture a reusable update graph; input/owner must outlive the graph.

        Replay reads current input contents; check_update reports validation
        status after execution. Buffers and scratch are allocated before capture.
        """
        if len(input) != len(self.update_items):
            raise Error("Device update graph requires unchanged input count")

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            var recording = builder.recording_context()
            self.enqueue_update[rebuild, block_size](recording, input)

        return DeviceGraph.create(ctx, build)
