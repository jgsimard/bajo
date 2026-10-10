"""GPU ownership for the shared spatial-query representation.

Construction and host-input rebuild reuse the CPU median builder. Device-input
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

from .device_update import UPDATE_BLOCK_SIZE, bounds_update_kernel
from .bounds import SpatialBoundsBvh


@fieldwise_init
struct GpuSpatialBoundsBvh[frame: Frame = .WORLD]:
    var host: SpatialBoundsBvh[Self.frame]
    var bounds: DeviceBuffer[.float32]
    var nodes: DeviceBuffer[.uint32]
    var update_items: DeviceBuffer[.float32]
    var update_order: DeviceBuffer[.uint32]
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
        ctx.synchronize()
        self.host = host^

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
        if len(boxes) != len(self.update_items):
            raise Error("Device BVH update requires unchanged item count")
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
        var function = ctx.compile_function[
            bounds_update_kernel[rebuild, block_size]
        ]()
        var arg0 = _device_span[mut=False](input)
        var arg1 = _device_span[mut=True](self.update_items)
        var arg2 = _device_span[mut=True](self.bounds)
        var arg3 = _device_span[mut=True](self.nodes)
        var arg4 = _device_span[mut=True](self.update_order)
        var arg5 = _device_span[mut=True](self.update_status)

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            _ = builder.add_function(
                function,
                arg0,
                arg1,
                arg2,
                arg3,
                arg4,
                arg5,
                grid_dim=1,
                block_dim=block_size,
                dependencies=[],
            )

        return DeviceGraph.create(ctx, build)
