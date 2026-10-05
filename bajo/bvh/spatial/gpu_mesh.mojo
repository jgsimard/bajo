"""Device ownership, validated device-input updates and graph replay."""

from max.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    DeviceGraph,
    DeviceGraphBuilder,
)
from bajo.core import Frame, Point3f32, Vec3f32
from bajo.bvh.gpu.utils import upload_list, _device_span
from .device_update import mesh_update_kernel, velocity_update_kernel
from .mesh import SpatialMesh


struct GpuSpatialMesh[frame: Frame = .WORLD]:
    """Consume a CPU mesh snapshot and publish retained device buffers.

    Kernels construct MeshQueryView from these five buffers. Host-input updates
    synchronize before mutation and on return; externally serialize kernels and
    updates. Device-input updates and reusable graphs use retained scratch.
    """

    var host: SpatialMesh[Self.frame]
    var points: DeviceBuffer[.float32]
    var indices: DeviceBuffer[.uint32]
    var velocities: DeviceBuffer[.float32]
    var bounds: DeviceBuffer[.float32]
    var nodes: DeviceBuffer[.uint32]
    var update_items: DeviceBuffer[.float32]
    var update_order: DeviceBuffer[.uint32]
    var update_status: DeviceBuffer[.uint32]

    def __init__(
        out self, mut ctx: DeviceContext, var host: SpatialMesh[Self.frame]
    ) raises:
        self.points = upload_list(ctx, host.points)
        self.indices = upload_list(ctx, host.indices)
        self.velocities = upload_list(ctx, host.velocities)
        self.bounds = upload_list(ctx, host.bvh.bounds)
        self.nodes = upload_list(ctx, host.bvh.nodes)
        self.update_items = ctx.enqueue_create_buffer[.float32](
            len(host.indices) // 3 * 6
        )
        self.update_order = ctx.enqueue_create_buffer[.uint32](
            len(host.bvh.nodes) // 2
        )
        self.update_status = upload_list(ctx, List[UInt32](length=1, fill=0))
        ctx.synchronize()
        self.host = host^

    def group_root(self, group: UInt32) -> UInt32:
        return self.host.group_root(group)

    def _upload_geometry(mut self, ctx: DeviceContext) raises:
        with self.points.map_to_host() as data:
            for i in range(len(self.host.points)):
                data[i] = self.host.points[i]
        with self.bounds.map_to_host() as data:
            for i in range(len(self.host.bvh.bounds)):
                data[i] = self.host.bvh.bounds[i]
        ctx.synchronize()

    def refit(
        mut self, ctx: DeviceContext, points: ImmSpan[Point3f32[Self.frame], _]
    ) raises:
        ctx.synchronize()
        self.host.refit(points)
        with self.points.map_to_host() as data:
            for i in range(len(self.host.points)):
                data[i] = self.host.points[i]
        self.refit_device(ctx, self.points)

    def rebuild(
        mut self, ctx: DeviceContext, points: ImmSpan[Point3f32[Self.frame], _]
    ) raises:
        ctx.synchronize()
        self.host.rebuild(points)
        self._upload_geometry(ctx)
        with self.nodes.map_to_host() as data:
            for i in range(len(self.host.bvh.nodes)):
                data[i] = self.host.bvh.nodes[i]
        ctx.synchronize()

    def set_velocities(
        mut self,
        ctx: DeviceContext,
        velocities: ImmSpan[Vec3f32[Self.frame], _],
    ) raises:
        ctx.synchronize()
        self.host.set_velocities(velocities)
        with self.velocities.map_to_host() as data:
            for i in range(len(self.host.velocities)):
                data[i] = self.host.velocities[i]
        ctx.synchronize()

    def enqueue_update[
        rebuild: Bool
    ](self, ctx: DeviceContext, points: DeviceBuffer[.float32]) raises:
        """Enqueue a validated device update using flat xyz input positions.

        No allocation, synchronization or host geometry transfer occurs here.
        Keep inputs and owner alive and serialize updates/queries on the context.
        Invalid data sets update_status[0]=1 without modifying published state.
        Device updates leave the last host snapshot unchanged.
        """
        if len(points) != len(self.points):
            raise Error("Device mesh update requires unchanged vertex count")
        ctx.enqueue_function[mesh_update_kernel[rebuild]](
            _device_span[mut=False](points),
            _device_span[mut=True](self.points),
            _device_span[mut=False](self.indices),
            _device_span[mut=True](self.update_items),
            _device_span[mut=True](self.bounds),
            _device_span[mut=True](self.nodes),
            _device_span[mut=True](self.update_order),
            _device_span[mut=True](self.update_status),
            grid_dim=1,
            block_dim=1,
        )

    def check_update(self, ctx: DeviceContext) raises:
        ctx.synchronize()
        with self.update_status.map_to_host() as status:
            if status[0] != 0:
                raise Error("Device mesh update rejected nonfinite data")

    def refit_device(
        self, ctx: DeviceContext, points: DeviceBuffer[.float32]
    ) raises:
        self.enqueue_update[False](ctx, points)
        self.check_update(ctx)

    def rebuild_device(
        self, ctx: DeviceContext, points: DeviceBuffer[.float32]
    ) raises:
        self.enqueue_update[True](ctx, points)
        self.check_update(ctx)

    def set_velocities_device(
        self, ctx: DeviceContext, velocities: DeviceBuffer[.float32]
    ) raises:
        if len(velocities) != 0 and len(velocities) != len(self.velocities):
            raise Error("Device mesh velocity count must match vertex count")
        ctx.enqueue_function[velocity_update_kernel[True]](
            _device_span[mut=False](velocities),
            _device_span[mut=True](self.velocities),
            _device_span[mut=True](self.update_status),
            grid_dim=1,
            block_dim=1,
        )
        self.check_update(ctx)

    def create_update_graph[
        rebuild: Bool
    ](
        self, ctx: DeviceContext, input: DeviceBuffer[.float32]
    ) raises -> DeviceGraph:
        """Capture a reusable update graph; input/owner must outlive the graph.

        Replay reads current input contents; check_update reports validation
        status after execution. Buffers and scratch are allocated before capture.
        """
        if len(input) != len(self.points):
            raise Error("Device update graph requires unchanged input count")
        var function = ctx.compile_function[mesh_update_kernel[rebuild]]()
        var arg0 = _device_span[mut=False](input)
        var arg1 = _device_span[mut=True](self.points)
        var arg2 = _device_span[mut=False](self.indices)
        var arg3 = _device_span[mut=True](self.update_items)
        var arg4 = _device_span[mut=True](self.bounds)
        var arg5 = _device_span[mut=True](self.nodes)
        var arg6 = _device_span[mut=True](self.update_order)
        var arg7 = _device_span[mut=True](self.update_status)

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            _ = builder.add_function(
                function,
                arg0,
                arg1,
                arg2,
                arg3,
                arg4,
                arg5,
                arg6,
                arg7,
                grid_dim=1,
                block_dim=1,
                dependencies=[],
            )

        return DeviceGraph.create(ctx, build)
