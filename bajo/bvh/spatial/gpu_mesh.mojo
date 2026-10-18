"""Device ownership, validated device-input updates and graph replay."""

from max.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    DeviceGraph,
    DeviceGraphBuilder,
)
from std.math import min, max, ceildiv
from .parallel_refit import (
    forest_parents,
    forest_parents_kernel,
    forest_chunks,
    clear_refit_kernel,
    validate_mesh_refit_kernel,
    refit_mesh_chunks_kernel,
    publish_mesh_chunk_parents_kernel,
)
from bajo.gpu.buffers import (
    _copy_staging_slice,
    _device_span,
    copy_to_device,
    upload_list,
)
from bajo.core import Frame, Point3f32, Vec3f32

from .device_update import (
    UPDATE_BLOCK_SIZE,
    mesh_update_kernel,
    velocity_update_kernel,
    validate_velocity_blocks,
    publish_velocity_blocks,
    clear_velocity_kernel,
    forest_rebuild_tiles,
    enqueue_forest_rebuild,
)
from .mesh import SpatialMesh
from .mesh_query import MeshQueryView
from .adjacency import refresh_adjacency_kernel
from .gpu_winding import GpuWindingCache
from .winding import DeviceWindingView
from .query import BoundsQueryView


struct GpuSpatialMesh[frame: Frame = .WORLD, storage_width: SIMDLength = 8]:
    """Consume a CPU mesh snapshot and publish retained device buffers.

    view() borrows device geometry, adjacency and optional winding metadata. Host-input updates
    synchronize before mutation and on return; externally serialize kernels and
    updates. Device-input updates and reusable graphs use retained scratch.
    """

    var host: SpatialMesh[Self.frame, Self.storage_width]
    var adjacency: DeviceBuffer[.uint32]
    var points: DeviceBuffer[.float32]
    var indices: DeviceBuffer[.uint32]
    var velocities: DeviceBuffer[.float32]
    var bounds: DeviceBuffer[.float32]
    var nodes: DeviceBuffer[.uint32]
    var update_items: DeviceBuffer[.float32]
    var velocity_counters: DeviceBuffer[.uint32]
    var update_order: DeviceBuffer[.uint32]
    var update_chunks: DeviceBuffer[.uint32]
    var update_parents: DeviceBuffer[.uint32]
    var rebuild_scratch: DeviceBuffer[.uint32]
    var rebuild_tiles: DeviceBuffer[.uint32]
    var update_status: DeviceBuffer[.uint32]
    var winding: Optional[GpuWindingCache[Self.frame]]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        var host: SpatialMesh[Self.frame, Self.storage_width],
        enable_winding: Bool = False,
    ) raises:
        self.adjacency = upload_list(ctx, host.adjacency)
        var node_count = len(host.bvh.nodes) // 2
        var face_count = len(host.indices) // 3
        var tiles = (
            forest_rebuild_tiles(host.bvh.nodes) if face_count
            >= 256 else List[UInt32]()
        )
        var chunks = forest_chunks(host.bvh.nodes)
        var parents = List[UInt32]()
        if node_count < 4096:
            parents = forest_parents(host.bvh.nodes)
        # One allocation retains disjoint, 64-byte-aligned views. The uploaded
        # prefix contains geometry and initialized metadata; scratch follows it.
        var lengths = Array[Int, 14](fill=0)
        lengths[0] = len(host.points)
        lengths[1] = len(host.indices)
        lengths[2] = len(host.velocities)
        lengths[3] = len(host.bvh.bounds)
        lengths[4] = len(host.bvh.nodes)
        lengths[5] = len(tiles)
        lengths[6] = len(chunks)
        lengths[7] = len(parents)
        lengths[8] = 2
        lengths[9] = 1
        lengths[10] = face_count * (8 if face_count >= 256 else 6)
        lengths[11] = node_count
        lengths[12] = node_count if len(tiles) else 0
        lengths[13] = node_count if node_count >= 4096 else 0
        var offsets = Array[Int, 15](fill=0)
        comptime for i in range(14):
            offsets[i + 1] = offsets[i] + ceildiv(lengths[i], 16) * 16
        var staging = ctx.enqueue_create_host_buffer[.uint32](offsets[10])
        var storage = ctx.enqueue_create_buffer[.uint32](offsets[14])
        var staging_span = staging.as_span()
        _copy_staging_slice(staging_span, offsets[0], host.points)
        _copy_staging_slice(staging_span, offsets[1], host.indices)
        _copy_staging_slice(staging_span, offsets[2], host.velocities)
        _copy_staging_slice(staging_span, offsets[3], host.bvh.bounds)
        _copy_staging_slice(staging_span, offsets[4], host.bvh.nodes)
        _copy_staging_slice(staging_span, offsets[5], tiles)
        _copy_staging_slice(staging_span, offsets[6], chunks)
        _copy_staging_slice(staging_span, offsets[7], parents)
        staging[offsets[8]] = 0
        staging[offsets[8] + 1] = 0
        staging[offsets[9]] = 0
        comptime for i in range(10):
            for padding in range(offsets[i] + lengths[i], offsets[i + 1]):
                staging[padding] = 0
        staging.enqueue_copy_to(
            storage.create_sub_buffer[.uint32](0, offsets[10])
        )
        self.points = storage.create_sub_buffer[.float32](
            offsets[0], lengths[0]
        )
        self.indices = storage.create_sub_buffer[.uint32](
            offsets[1], lengths[1]
        )
        self.velocities = storage.create_sub_buffer[.float32](
            offsets[2], lengths[2]
        )
        self.bounds = storage.create_sub_buffer[.float32](
            offsets[3], lengths[3]
        )
        self.nodes = storage.create_sub_buffer[.uint32](offsets[4], lengths[4])
        self.rebuild_tiles = storage.create_sub_buffer[.uint32](
            offsets[5], lengths[5]
        )
        self.update_chunks = storage.create_sub_buffer[.uint32](
            offsets[6], lengths[6]
        )
        self.velocity_counters = storage.create_sub_buffer[.uint32](
            offsets[8], lengths[8]
        )
        self.update_status = storage.create_sub_buffer[.uint32](
            offsets[9], lengths[9]
        )
        self.update_items = storage.create_sub_buffer[.float32](
            offsets[10], lengths[10]
        )
        self.update_order = storage.create_sub_buffer[.uint32](
            offsets[11], lengths[11]
        )
        self.rebuild_scratch = storage.create_sub_buffer[.uint32](
            offsets[12], lengths[12]
        )
        if node_count >= 4096:
            self.update_parents = storage.create_sub_buffer[.uint32](
                offsets[13], lengths[13]
            )
            ctx.enqueue_function[forest_parents_kernel](
                _device_span[mut=False](self.nodes),
                _device_span[mut=True](self.update_parents),
                grid_dim=ceildiv(node_count, 256),
                block_dim=256,
            )
        else:
            self.update_parents = storage.create_sub_buffer[.uint32](
                offsets[7], lengths[7]
            )
        self.winding = Optional[GpuWindingCache[Self.frame]]()
        if enable_winding:
            host._ensure_winding_cache()
            self.winding = Optional(GpuWindingCache(ctx, host))
        ctx.synchronize()
        self.host = host^

    def view(
        self,
    ) -> MeshQueryView[ImmutAnyOrigin, Self.frame, Self.storage_width]:
        """Borrow device storage for a query kernel; never dereference on host.
        """
        var result = MeshQueryView[
            ImmutAnyOrigin, Self.frame, Self.storage_width
        ](
            _device_span[mut=False](self.points),
            _device_span[mut=False](self.indices),
            _device_span[mut=False](self.velocities),
            BoundsQueryView[ImmutAnyOrigin, Self.frame](
                _device_span[mut=False](self.bounds),
                _device_span[mut=False](self.nodes),
            ),
        )
        result.adjacency = _device_span[mut=False](self.adjacency)
        result.device_winding = self.device_winding_view()
        return result^

    def _enqueue_adjacency(self, ctx: DeviceContext) raises:
        if len(self.adjacency) == 0:
            return
        ctx.enqueue_function[refresh_adjacency_kernel](
            _device_span[mut=False](self.nodes),
            _device_span[mut=True](self.adjacency),
            _device_span[mut=False](self.update_status),
            grid_dim=ceildiv(len(self.nodes) // 2, 256),
            block_dim=256,
        )

    def device_winding_view(
        self,
    ) -> DeviceWindingView[ImmutAnyOrigin, Self.frame]:
        if not self.winding:
            return DeviceWindingView[ImmutAnyOrigin, Self.frame]()
        ref cache = self.winding.value()
        return DeviceWindingView[ImmutAnyOrigin, Self.frame](
            BoundsQueryView[ImmutAnyOrigin, Self.frame](
                _device_span[mut=False](cache.tree.bounds),
                _device_span[mut=False](cache.tree.nodes),
            ),
            _device_span[mut=False](cache.offsets),
            _device_span[mut=False](cache.faces),
        )

    def group_root(self, group: UInt32) -> UInt32:
        return self.host.group_root(group)

    def _upload_geometry(mut self, ctx: DeviceContext) raises:
        copy_to_device(self.points, self.host.points)
        copy_to_device(self.bounds, self.host.bvh.bounds)
        if self.winding:
            with self.update_status.map_to_host() as status:
                status[0] = 0
            self.winding.value().enqueue_refit(
                ctx, self.points, self.indices, self.update_status
            )
        ctx.synchronize()

    def refit(
        mut self, ctx: DeviceContext, points: ImmSpan[Point3f32[Self.frame], _]
    ) raises:
        ctx.synchronize()
        self.host.refit(points)
        copy_to_device(self.points, self.host.points)
        self.refit_device(ctx, self.points)

    def rebuild(
        mut self, ctx: DeviceContext, points: ImmSpan[Point3f32[Self.frame], _]
    ) raises:
        ctx.synchronize()
        self.host.rebuild(points)
        self._upload_geometry(ctx)
        copy_to_device(self.nodes, self.host.bvh.nodes)
        copy_to_device(self.adjacency, self.host.adjacency)
        ctx.synchronize()

    def set_velocities(
        mut self,
        ctx: DeviceContext,
        velocities: ImmSpan[Vec3f32[Self.frame], _],
    ) raises:
        ctx.synchronize()
        self.host.set_velocities(velocities)
        copy_to_device(self.velocities, self.host.velocities)
        ctx.synchronize()

    @always_inline
    def enqueue_update[
        rebuild: Bool, block_size: Int = UPDATE_BLOCK_SIZE
    ](self, ctx: DeviceContext, points: DeviceBuffer[.float32]) raises:
        """Enqueue a validated device update using flat xyz input positions.

        No allocation, synchronization or host geometry transfer occurs here.
        Keep inputs and owner alive and serialize updates/queries on the context.
        Invalid data sets update_status[0]=1 without modifying published state.
        Device updates leave the last host snapshot unchanged.
        """
        self._enqueue_update[rebuild, block_size](ctx, points)

    def _enqueue_update[
        rebuild: Bool,
        block_size: Int,
        winding_block_size: Int = UPDATE_BLOCK_SIZE,
    ](self, ctx: DeviceContext, points: DeviceBuffer[.float32]) raises:
        comptime assert (
            block_size >= 32
            and block_size <= 1024
            and (block_size & (block_size - 1)) == 0
        )
        if len(points) != len(self.points):
            raise Error("Device mesh update requires unchanged vertex count")
        if len(self.indices) // 3 >= (256 if rebuild else 1024):
            self._enqueue_parallel_refit[block_size, rebuild](ctx, points)
            comptime if rebuild:
                self._enqueue_adjacency(ctx)
            if self.winding:
                self.winding.value().enqueue_refit[winding_block_size](
                    ctx, self.points, self.indices, self.update_status
                )
            return
        ctx.enqueue_function[mesh_update_kernel[rebuild, block_size]](
            _device_span[mut=False](points),
            _device_span[mut=True](self.points),
            _device_span[mut=False](self.indices),
            _device_span[mut=True](self.update_items),
            _device_span[mut=True](self.bounds),
            _device_span[mut=True](self.nodes),
            _device_span[mut=True](self.update_order),
            _device_span[mut=True](self.update_status),
            grid_dim=1,
            block_dim=block_size,
        )
        comptime if rebuild:
            if len(self.nodes) > 0:
                self._enqueue_adjacency(ctx)
        if self.winding:
            self.winding.value().enqueue_refit[winding_block_size](
                ctx, self.points, self.indices, self.update_status
            )

    def _enqueue_parallel_refit[
        block_size: Int, rebuild: Bool = False
    ](self, ctx: DeviceContext, input: DeviceBuffer[.float32]) raises:
        comptime chunk_block_size = min(block_size, 128)
        ctx.enqueue_function[clear_refit_kernel](
            _device_span[mut=True](self.update_order),
            _device_span[mut=True](self.update_status),
            grid_dim=ceildiv(len(self.update_order), block_size),
            block_dim=block_size,
        )
        ctx.enqueue_function[validate_mesh_refit_kernel[rebuild]](
            _device_span[mut=False](input),
            _device_span[mut=False](self.indices),
            _device_span[mut=True](self.update_items),
            _device_span[mut=True](self.update_status),
            grid_dim=ceildiv(
                max(len(input), len(self.indices) // 3), block_size
            ),
            block_dim=block_size,
        )
        comptime if rebuild:
            enqueue_forest_rebuild[True](
                ctx,
                self.update_items,
                self.nodes,
                self.rebuild_tiles,
                self.rebuild_scratch,
                self.update_order,
                self.update_status,
                len(self.indices) // 3,
            )
        ctx.enqueue_function[refit_mesh_chunks_kernel[chunk_block_size]](
            _device_span[mut=False](input),
            _device_span[mut=True](self.points),
            _device_span[mut=False](self.update_items),
            _device_span[mut=True](self.bounds),
            _device_span[mut=False](self.nodes),
            _device_span[mut=False](self.update_status),
            _device_span[mut=False](self.update_chunks),
            grid_dim=len(self.update_chunks),
            block_dim=chunk_block_size,
        )
        ctx.enqueue_function[publish_mesh_chunk_parents_kernel](
            _device_span[mut=True](self.bounds),
            _device_span[mut=False](self.nodes),
            _device_span[mut=False](self.update_parents),
            _device_span[mut=True](self.update_order),
            _device_span[mut=False](self.update_status),
            _device_span[mut=False](self.update_chunks),
            grid_dim=ceildiv(len(self.update_chunks), block_size),
            block_dim=block_size,
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

    def enqueue_velocities[
        block_size: Int = 0
    ](self, ctx: DeviceContext, velocities: DeviceBuffer[.float32]) raises:
        """Validate and publish resident velocities; an empty input clears them.

        A zero thread count selects 1,024 threads for medium single-block
        updates (1,025..4,095 components), and 256 for other operations.
        An explicit positive thread count remains unchanged. Rejected inputs
        preserve every published velocity. Large aligned buffers process four
        components per thread, with scalar handling for incomplete final groups.
        Serialize updates and queries on
        this context and keep the owner and input alive.
        """
        comptime if block_size == 0:
            if 1024 < len(velocities) < 4096:
                self._enqueue_velocities_fixed[1024](ctx, velocities)
            else:
                self._enqueue_velocities_fixed[UPDATE_BLOCK_SIZE](
                    ctx, velocities
                )
        else:
            self._enqueue_velocities_fixed[block_size](ctx, velocities)

    def _enqueue_velocities_fixed[
        block_size: Int
    ](self, ctx: DeviceContext, velocities: DeviceBuffer[.float32]) raises:
        comptime assert (
            block_size >= 32
            and block_size <= 1024
            and (block_size & (block_size - 1)) == 0
        )
        if len(velocities) != 0 and len(velocities) != len(self.velocities):
            raise Error("Device mesh velocity count must match vertex count")
        if len(velocities) == 0:
            ctx.enqueue_function[clear_velocity_kernel](
                _device_span[mut=True](self.velocities),
                _device_span[mut=True](self.update_status),
                grid_dim=max(1, ceildiv(len(self.velocities), block_size)),
                block_dim=block_size,
            )
            return
        if len(velocities) >= 4096:
            if (
                len(velocities) >= 65536
                and Int(velocities.unsafe_ptr()) % 16 == 0
                and Int(self.velocities.unsafe_ptr()) % 16 == 0
            ):
                self._enqueue_velocity_blocks[block_size, 4](ctx, velocities)
            else:
                self._enqueue_velocity_blocks[block_size, 1](ctx, velocities)
            return
        ctx.enqueue_function[velocity_update_kernel[True, block_size]](
            _device_span[mut=False](velocities),
            _device_span[mut=True](self.velocities),
            _device_span[mut=True](self.update_status),
            grid_dim=1,
            block_dim=block_size,
        )

    def _enqueue_velocity_blocks[
        block_size: Int, width: SIMDLength
    ](self, ctx: DeviceContext, velocities: DeviceBuffer[.float32]) raises:
        # Packed loads/stores require width*4-byte alignment. The caller checks
        # this before selecting width=4; scalar operations accept every buffer.
        var blocks = ceildiv(len(velocities), block_size * Int(width))
        ctx.enqueue_function[validate_velocity_blocks[block_size, width]](
            _device_span[mut=False](velocities),
            _device_span[mut=True](self.velocity_counters),
            _device_span[mut=True](self.update_status),
            grid_dim=blocks,
            block_dim=block_size,
        )
        ctx.enqueue_function[publish_velocity_blocks[block_size, width]](
            _device_span[mut=False](velocities),
            _device_span[mut=True](self.velocities),
            _device_span[mut=False](self.update_status),
            grid_dim=blocks,
            block_dim=block_size,
        )

    def set_velocities_device(
        self, ctx: DeviceContext, velocities: DeviceBuffer[.float32]
    ) raises:
        self.enqueue_velocities(ctx, velocities)
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
        if len(input) != len(self.points):
            raise Error("Device update graph requires unchanged input count")

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            var recording = builder.recording_context()
            # Graphs historically use the requested block size for small
            # winding refits; direct submissions retain the default size.
            self._enqueue_update[rebuild, block_size, block_size](
                recording, input
            )

        return DeviceGraph.create(ctx, build)
