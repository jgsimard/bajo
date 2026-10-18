"""Optional exact winding acceleration for topology-certified components."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import min, max, align_up
from bajo.gpu.buffers import _copy_staging_slice, upload_list, _device_span
from bajo.core import Frame
from bajo.bvh.constants import EMPTY_LANE, f32_max

from .device_update import UPDATE_BLOCK_SIZE, forest_rebuild_tiles
from .mesh import SpatialMesh
from .gpu import GpuSpatialBoundsBvh
from .parallel_refit import (
    forest_parents,
    clear_refit_kernel,
    publish_bounds_refit_kernel,
)


def component_bounds_kernel[
    block_size: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    faces: ImmSpan[UInt32, ImmutAnyOrigin],
    items: MutSpan[Float32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var component = Int(global_idx.x)
    if status[0] != 0 or component >= len(offsets) - 2:
        return
    comptime for axis in range(3):
        var lo = Float32(f32_max)
        var hi = -lo
        for slot in range(Int(offsets[component]), Int(offsets[component + 1])):
            var face = Int(faces[slot])
            comptime for corner in range(3):
                var value = points[Int(indices[face * 3 + corner]) * 3 + axis]
                lo = min(lo, value)
                hi = max(hi, value)
        items[component * 6 + axis] = lo
        items[component * 6 + axis + 3] = hi


struct GpuWindingCache[frame: Frame = .WORLD]:
    var tree: GpuSpatialBoundsBvh[Self.frame]
    var parents: DeviceBuffer[.uint32]
    var offsets: DeviceBuffer[.uint32]
    var faces: DeviceBuffer[.uint32]

    def __init__[
        width: SIMDLength
    ](
        out self, mut ctx: DeviceContext, host: SpatialMesh[Self.frame, width]
    ) raises:
        var count = len(host.winding.bvh.item_bounds)
        var offsets = List[UInt32](length=count + 2, fill=0)
        for component in host.winding.face_components:
            var bucket = count if component == EMPTY_LANE else Int(component)
            offsets[bucket + 1] += 1
        for i in range(1, len(offsets)):
            offsets[i] += offsets[i - 1]
        var cursors = offsets.copy()
        var faces = List[UInt32](length=len(host.indices) // 3, fill=0)
        for face in range(len(faces)):
            var component = host.winding.face_components[face]
            var bucket = count if component == EMPTY_LANE else Int(component)
            faces[Int(cursors[bucket])] = UInt32(face)
            cursors[bucket] += 1
        var tree = host.winding.bvh.copy()
        var parents = forest_parents(tree.nodes)
        # One initialized prefix and retained, disjoint scratch views.
        var lengths = Array[Int, 8](fill=0)
        lengths[0] = len(offsets)
        lengths[1] = len(faces)
        lengths[2] = len(tree.bounds)
        lengths[3] = len(tree.nodes)
        lengths[4] = len(parents)
        lengths[5] = 1
        lengths[6] = len(tree.item_bounds) * 6
        lengths[7] = len(tree.nodes) // 2
        var starts = Array[Int, 9](fill=0)
        comptime for i in range(8):
            starts[i + 1] = starts[i] + Int(align_up(lengths[i], 16))
        var staging = ctx.enqueue_create_host_buffer[.uint32](starts[6])
        var storage = ctx.enqueue_create_buffer[.uint32](starts[8])
        var staging_span = staging.as_span()
        _copy_staging_slice(staging_span, starts[0], offsets)
        _copy_staging_slice(staging_span, starts[1], faces)
        _copy_staging_slice(staging_span, starts[2], tree.bounds)
        _copy_staging_slice(staging_span, starts[3], tree.nodes)
        _copy_staging_slice(staging_span, starts[4], parents)
        staging[starts[5]] = 0
        comptime for i in range(6):
            staging_span[starts[i] + lengths[i] : starts[i + 1]].fill(0)
        staging.enqueue_copy_to(
            storage.create_sub_buffer[.uint32](0, starts[6])
        )
        self.offsets = storage.create_sub_buffer[.uint32](starts[0], lengths[0])
        self.faces = storage.create_sub_buffer[.uint32](starts[1], lengths[1])
        self.parents = storage.create_sub_buffer[.uint32](starts[4], lengths[4])
        var tiles = (
            forest_rebuild_tiles(tree.nodes) if count
            >= 1024 else List[UInt32]()
        )
        var rebuild_tiles = upload_list(ctx, tiles)
        var rebuild_scratch = ctx.enqueue_create_buffer[.uint32](
            len(tree.nodes) // 2 if count >= 1024 else 0
        )
        self.tree = GpuSpatialBoundsBvh(
            host=tree^,
            rebuild_tiles=rebuild_tiles^,
            rebuild_scratch=rebuild_scratch^,
            update_parents=self.parents,
            bounds=storage.create_sub_buffer[.float32](starts[2], lengths[2]),
            nodes=storage.create_sub_buffer[.uint32](starts[3], lengths[3]),
            update_status=storage.create_sub_buffer[.uint32](
                starts[5], lengths[5]
            ),
            update_items=storage.create_sub_buffer[.float32](
                starts[6], lengths[6]
            ),
            update_order=storage.create_sub_buffer[.uint32](
                starts[7], lengths[7]
            ),
        )
        ctx.synchronize()

    def enqueue_refit[
        block_size: Int = UPDATE_BLOCK_SIZE
    ](
        self,
        ctx: DeviceContext,
        points: DeviceBuffer[.float32],
        indices: DeviceBuffer[.uint32],
        status: DeviceBuffer[.uint32],
    ) raises:
        var count = len(self.offsets) - 2
        if count == 0:
            return
        if count >= 1024:
            ctx.enqueue_function[clear_refit_kernel](
                _device_span[mut=True](self.tree.update_order),
                _device_span[mut=True](self.tree.update_status),
                grid_dim=(len(self.tree.update_order) + 127) // 128,
                block_dim=128,
            )
        ctx.enqueue_function[component_bounds_kernel[128]](
            _device_span[mut=False](points),
            _device_span[mut=False](indices),
            _device_span[mut=False](self.offsets),
            _device_span[mut=False](self.faces),
            _device_span[mut=True](self.tree.update_items),
            _device_span[mut=False](status),
            grid_dim=(count + 127) // 128,
            block_dim=128,
        )
        # Refit is allocation-free; geometry validation is already completed by
        # the mesh update. Failed mesh inputs leave these component boxes intact.
        if count >= 1024:
            ctx.enqueue_function[publish_bounds_refit_kernel](
                _device_span[mut=False](self.tree.update_items),
                _device_span[mut=True](self.tree.bounds),
                _device_span[mut=False](self.tree.nodes),
                _device_span[mut=False](self.parents),
                _device_span[mut=True](self.tree.update_order),
                _device_span[mut=False](status),
                grid_dim=(len(self.tree.update_order) + 127) // 128,
                block_dim=128,
            )
        else:
            self.tree.enqueue_update[False, block_size](
                ctx, self.tree.update_items
            )
