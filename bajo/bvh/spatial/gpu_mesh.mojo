"""Device ownership, validated device-input updates and graph replay."""

from max.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    DeviceGraph,
    DeviceGraphBuilder,
)
from std.math import min, max, ceildiv
from std.memory import unsafe_memcpy
from .parallel_refit import (
    forest_parents,
    forest_parents_kernel,
    forest_chunks,
    clear_refit_kernel,
    validate_mesh_refit_kernel,
    refit_mesh_chunks_kernel,
    publish_mesh_chunk_parents_kernel,
    publish_bounds_refit_kernel,
)
from bajo.core import Frame, Point3f32, Vec3f32
from bajo.bvh.gpu.utils import upload_list, _device_span
from .device_update import (
    UPDATE_BLOCK_SIZE,
    mesh_update_kernel,
    velocity_update_kernel,
    validate_velocity_blocks,
    publish_velocity_blocks,
    clear_velocity_kernel,
    bounds_update_kernel,
    rebuild_tile_sort_kernel,
    rebuild_merge_order_kernel,
    publish_rebuild_order_kernel,
    clear_rebuild_order_kernel,
)
from .mesh import SpatialMesh
from .gpu_winding import GpuWindingCache, component_bounds_kernel
from .winding import DeviceWindingView
from .query import BoundsQueryView


struct GpuSpatialMesh[frame: Frame = .WORLD, storage_width: SIMDLength = 8]:
    """Consume a CPU mesh snapshot and publish retained device buffers.

    Kernels construct MeshQueryView from these five buffers. Host-input updates
    synchronize before mutation and on return; externally serialize kernels and
    updates. Device-input updates and reusable graphs use retained scratch.
    """

    var host: SpatialMesh[Self.frame, Self.storage_width]
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
        var tiles = List[UInt32]()
        var node_count = len(host.bvh.nodes) // 2
        var face_count = len(host.indices) // 3
        if face_count >= 256:
            var root = 0
            while root < node_count:
                var end = Int(host.bvh.nodes[root * 2])
                var count = (end - root + 1) // 2
                for begin in range(0, count, 256):
                    tiles.extend([UInt32(root), UInt32(count), UInt32(begin)])
                root = end
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
        unsafe_memcpy(
            dest=staging.unsafe_ptr()
            .unsafe_offset(offsets[0])
            .unsafe_bitcast[Float32](),
            src=host.points.unsafe_ptr(),
            count=lengths[0],
        )
        unsafe_memcpy(
            dest=staging.unsafe_ptr().unsafe_offset(offsets[1]),
            src=host.indices.unsafe_ptr(),
            count=lengths[1],
        )
        unsafe_memcpy(
            dest=staging.unsafe_ptr()
            .unsafe_offset(offsets[2])
            .unsafe_bitcast[Float32](),
            src=host.velocities.unsafe_ptr(),
            count=lengths[2],
        )
        unsafe_memcpy(
            dest=staging.unsafe_ptr()
            .unsafe_offset(offsets[3])
            .unsafe_bitcast[Float32](),
            src=host.bvh.bounds.unsafe_ptr(),
            count=lengths[3],
        )
        unsafe_memcpy(
            dest=staging.unsafe_ptr().unsafe_offset(offsets[4]),
            src=host.bvh.nodes.unsafe_ptr(),
            count=lengths[4],
        )
        unsafe_memcpy(
            dest=staging.unsafe_ptr().unsafe_offset(offsets[5]),
            src=tiles.unsafe_ptr(),
            count=lengths[5],
        )
        unsafe_memcpy(
            dest=staging.unsafe_ptr().unsafe_offset(offsets[6]),
            src=chunks.unsafe_ptr(),
            count=lengths[6],
        )
        unsafe_memcpy(
            dest=staging.unsafe_ptr().unsafe_offset(offsets[7]),
            src=parents.unsafe_ptr(),
            count=lengths[7],
        )
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
        with self.points.map_to_host() as data:
            for i in range(len(self.host.points)):
                data[i] = self.host.points[i]
        with self.bounds.map_to_host() as data:
            for i in range(len(self.host.bvh.bounds)):
                data[i] = self.host.bvh.bounds[i]
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
        rebuild: Bool, block_size: Int = UPDATE_BLOCK_SIZE
    ](self, ctx: DeviceContext, points: DeviceBuffer[.float32]) raises:
        """Enqueue a validated device update using flat xyz input positions.

        No allocation, synchronization or host geometry transfer occurs here.
        Keep inputs and owner alive and serialize updates/queries on the context.
        Invalid data sets update_status[0]=1 without modifying published state.
        Device updates leave the last host snapshot unchanged.
        """
        comptime assert (
            block_size >= 32
            and block_size <= 1024
            and (block_size & (block_size - 1)) == 0
        )
        if len(points) != len(self.points):
            raise Error("Device mesh update requires unchanged vertex count")
        if len(self.indices) // 3 >= (256 if rebuild else 1024):
            self._enqueue_parallel_refit[block_size, rebuild](ctx, points)
            if self.winding:
                self.winding.value().enqueue_refit(
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
        if self.winding:
            self.winding.value().enqueue_refit(
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
            ctx.enqueue_function[rebuild_tile_sort_kernel](
                _device_span[mut=False](self.update_items),
                _device_span[mut=False](self.nodes),
                _device_span[mut=False](self.rebuild_tiles),
                _device_span[mut=True](self.rebuild_scratch),
                _device_span[mut=False](self.update_status),
                grid_dim=len(self.rebuild_tiles) // 3,
                block_dim=256,
            )
            var in_scratch = True
            var width = 256
            while width < len(self.indices) // 3:
                ctx.enqueue_function[rebuild_merge_order_kernel](
                    _device_span[mut=False](self.update_items),
                    _device_span[mut=False](self.rebuild_tiles),
                    _device_span[mut=False](
                        self.rebuild_scratch
                    ) if in_scratch else _device_span[mut=False](
                        self.update_order
                    ),
                    _device_span[mut=True](
                        self.update_order
                    ) if in_scratch else _device_span[mut=True](
                        self.rebuild_scratch
                    ),
                    _device_span[mut=False](self.update_status),
                    UInt32(width),
                    grid_dim=len(self.rebuild_tiles) // 3,
                    block_dim=256,
                )
                in_scratch = not in_scratch
                width *= 2
            ctx.enqueue_function[publish_rebuild_order_kernel](
                _device_span[mut=False](self.rebuild_tiles),
                _device_span[mut=False](
                    self.rebuild_scratch
                ) if in_scratch else _device_span[mut=False](self.update_order),
                _device_span[mut=True](self.nodes),
                _device_span[mut=False](self.update_status),
                grid_dim=len(self.rebuild_tiles) // 3,
                block_dim=256,
            )
            ctx.enqueue_function[clear_rebuild_order_kernel](
                _device_span[mut=True](self.update_order),
                grid_dim=ceildiv(len(self.update_order), 256),
                block_dim=256,
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
        if len(self.indices) // 3 >= (256 if rebuild else 1024):
            return self._create_parallel_refit_graph[block_size, rebuild](
                ctx, input
            )
        if self.winding:
            return self._create_winding_update_graph[rebuild, block_size](
                ctx, input
            )
        if len(input) != len(self.points):
            raise Error("Device update graph requires unchanged input count")
        var function = ctx.compile_function[
            mesh_update_kernel[rebuild, block_size]
        ]()
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
                block_dim=block_size,
                dependencies=[],
            )

        return DeviceGraph.create(ctx, build)

    def _create_parallel_refit_graph[
        block_size: Int, rebuild: Bool = False
    ](
        self, ctx: DeviceContext, input: DeviceBuffer[.float32]
    ) raises -> DeviceGraph:
        if len(input) != len(self.points):
            raise Error("Device update graph requires unchanged vertex count")
        comptime chunk_block_size = min(block_size, 128)
        var clear = ctx.compile_function[clear_refit_kernel]()
        var validate = ctx.compile_function[
            validate_mesh_refit_kernel[rebuild]
        ]()
        var tile_sort = ctx.compile_function[rebuild_tile_sort_kernel]()
        var order_merge = ctx.compile_function[rebuild_merge_order_kernel]()
        var order_publish = ctx.compile_function[publish_rebuild_order_kernel]()
        var order_clear = ctx.compile_function[clear_rebuild_order_kernel]()
        var merge = ctx.compile_function[
            refit_mesh_chunks_kernel[chunk_block_size]
        ]()
        var publish = ctx.compile_function[publish_mesh_chunk_parents_kernel]()
        var component_function = ctx.compile_function[
            component_bounds_kernel[128]
        ]()
        var cache_refit_function = ctx.compile_function[
            bounds_update_kernel[False, block_size]
        ]()
        var cache_publish_function = ctx.compile_function[
            publish_bounds_refit_kernel
        ]()
        var counters = _device_span[mut=True](self.update_order)
        var status = _device_span[mut=True](self.update_status)
        var read_status = _device_span[mut=False](self.update_status)
        var source = _device_span[mut=False](input)
        var points = _device_span[mut=True](self.points)
        var indices = _device_span[mut=False](self.indices)
        var items = _device_span[mut=True](self.update_items)
        var read_items = _device_span[mut=False](self.update_items)
        var bounds = _device_span[mut=True](self.bounds)
        var nodes = _device_span[mut=False](self.nodes)
        var chunks = _device_span[mut=False](self.update_chunks)
        var parents = _device_span[mut=False](self.update_parents)
        var clear_grid = ceildiv(len(self.update_order), block_size)
        var validate_grid = ceildiv(
            max(len(input), len(self.indices) // 3), block_size
        )

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            var cleared = builder.add_function(
                clear,
                counters,
                status,
                grid_dim=clear_grid,
                block_dim=block_size,
                dependencies=[],
            )
            var checked = builder.add_function(
                validate,
                source,
                indices,
                items,
                status,
                grid_dim=validate_grid,
                block_dim=block_size,
                dependencies=[cleared],
            )
            var ready = checked
            comptime if rebuild:
                ready = builder.add_function(
                    tile_sort,
                    read_items,
                    nodes,
                    _device_span[mut=False](self.rebuild_tiles),
                    _device_span[mut=True](self.rebuild_scratch),
                    read_status,
                    grid_dim=len(self.rebuild_tiles) // 3,
                    block_dim=256,
                    dependencies=[checked],
                )
                var in_scratch = True
                var width = 256
                while width < len(self.indices) // 3:
                    ready = builder.add_function(
                        order_merge,
                        read_items,
                        _device_span[mut=False](self.rebuild_tiles),
                        _device_span[mut=False](
                            self.rebuild_scratch
                        ) if in_scratch else _device_span[mut=False](
                            self.update_order
                        ),
                        _device_span[mut=True](
                            self.update_order
                        ) if in_scratch else _device_span[mut=True](
                            self.rebuild_scratch
                        ),
                        read_status,
                        UInt32(width),
                        grid_dim=len(self.rebuild_tiles) // 3,
                        block_dim=256,
                        dependencies=[ready],
                    )
                    in_scratch = not in_scratch
                    width *= 2
                ready = builder.add_function(
                    order_publish,
                    _device_span[mut=False](self.rebuild_tiles),
                    _device_span[mut=False](
                        self.rebuild_scratch
                    ) if in_scratch else _device_span[mut=False](
                        self.update_order
                    ),
                    _device_span[mut=True](self.nodes),
                    read_status,
                    grid_dim=len(self.rebuild_tiles) // 3,
                    block_dim=256,
                    dependencies=[ready],
                )
                ready = builder.add_function(
                    order_clear,
                    counters,
                    grid_dim=ceildiv(len(self.update_order), 256),
                    block_dim=256,
                    dependencies=[ready],
                )
            var merged = builder.add_function(
                merge,
                source,
                points,
                read_items,
                bounds,
                nodes,
                read_status,
                chunks,
                grid_dim=len(chunks),
                block_dim=chunk_block_size,
                dependencies=[ready],
            )
            var published = builder.add_function(
                publish,
                bounds,
                nodes,
                parents,
                counters,
                read_status,
                chunks,
                grid_dim=ceildiv(len(chunks), block_size),
                block_dim=block_size,
                dependencies=[merged],
            )
            if self.winding:
                ref cache = self.winding.value()
                var components = len(cache.offsets) - 2
                if components > 0:
                    var cache_ready = published
                    if components >= 1024:
                        cache_ready = builder.add_function(
                            clear,
                            _device_span[mut=True](cache.tree.update_order),
                            _device_span[mut=True](cache.tree.update_status),
                            grid_dim=ceildiv(len(cache.tree.update_order), 128),
                            block_dim=128,
                            dependencies=[published],
                        )
                    var boxes = builder.add_function(
                        component_function,
                        _device_span[mut=False](self.points),
                        indices,
                        _device_span[mut=False](cache.offsets),
                        _device_span[mut=False](cache.faces),
                        _device_span[mut=True](cache.tree.update_items),
                        read_status,
                        grid_dim=(components + 127) // 128,
                        block_dim=128,
                        dependencies=[cache_ready],
                    )
                    if components >= 1024:
                        _ = builder.add_function(
                            cache_publish_function,
                            _device_span[mut=False](cache.tree.update_items),
                            _device_span[mut=True](cache.tree.bounds),
                            _device_span[mut=False](cache.tree.nodes),
                            _device_span[mut=False](cache.parents),
                            _device_span[mut=True](cache.tree.update_order),
                            read_status,
                            grid_dim=ceildiv(len(cache.tree.update_order), 128),
                            block_dim=128,
                            dependencies=[boxes],
                        )
                    else:
                        _ = builder.add_function(
                            cache_refit_function,
                            _device_span[mut=False](cache.tree.update_items),
                            _device_span[mut=True](cache.tree.update_items),
                            _device_span[mut=True](cache.tree.bounds),
                            _device_span[mut=True](cache.tree.nodes),
                            _device_span[mut=True](cache.tree.update_order),
                            _device_span[mut=True](cache.tree.update_status),
                            grid_dim=1,
                            block_dim=block_size,
                            dependencies=[boxes],
                        )

        return DeviceGraph.create(ctx, build)

    def _create_winding_update_graph[
        rebuild: Bool, block_size: Int
    ](
        self, ctx: DeviceContext, input: DeviceBuffer[.float32]
    ) raises -> DeviceGraph:
        if len(input) != len(self.points):
            raise Error("Device update graph requires unchanged vertex count")
        ref cache = self.winding.value()
        var function = ctx.compile_function[
            mesh_update_kernel[rebuild, block_size]
        ]()
        var component_function = ctx.compile_function[
            component_bounds_kernel[128]
        ]()
        var refit_function = ctx.compile_function[
            bounds_update_kernel[False, block_size]
        ]()
        var a0 = _device_span[mut=False](input)
        var a1 = _device_span[mut=True](self.points)
        var a2 = _device_span[mut=False](self.indices)
        var a3 = _device_span[mut=True](self.update_items)
        var a4 = _device_span[mut=True](self.bounds)
        var a5 = _device_span[mut=True](self.nodes)
        var a6 = _device_span[mut=True](self.update_order)
        var a7 = _device_span[mut=True](self.update_status)
        var p = _device_span[mut=False](self.points)
        var offsets = _device_span[mut=False](cache.offsets)
        var faces = _device_span[mut=False](cache.faces)
        var items = _device_span[mut=True](cache.tree.update_items)
        var source = _device_span[mut=False](cache.tree.update_items)
        var status = _device_span[mut=False](self.update_status)
        var cb = _device_span[mut=True](cache.tree.bounds)
        var cn = _device_span[mut=True](cache.tree.nodes)
        var co = _device_span[mut=True](cache.tree.update_order)
        var cs = _device_span[mut=True](cache.tree.update_status)
        var components = len(cache.offsets) - 2

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            var update = builder.add_function(
                function,
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                grid_dim=1,
                block_dim=block_size,
                dependencies=[],
            )
            if components > 0:
                var boxes = builder.add_function(
                    component_function,
                    p,
                    a2,
                    offsets,
                    faces,
                    items,
                    status,
                    grid_dim=(components + 127) // 128,
                    block_dim=128,
                    dependencies=[update],
                )
                _ = builder.add_function(
                    refit_function,
                    source,
                    items,
                    cb,
                    cn,
                    co,
                    cs,
                    grid_dim=1,
                    block_dim=block_size,
                    dependencies=[boxes],
                )

        return DeviceGraph.create(ctx, build)
