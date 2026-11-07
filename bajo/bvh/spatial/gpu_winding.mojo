"""Optional exact winding acceleration for topology-certified components."""

from max.gpu import global_idx, block_idx, thread_idx
from max.gpu.primitives import block
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


comptime WINDING_COMPONENT_PARALLEL_MIN_FACES = 256
comptime WINDING_COMPONENT_TILED_MIN_FACES = 65536
comptime WINDING_COMPONENT_TILE_FACES = 2048


def component_bounds_kernel[
    block_size: Int, skip_large: Bool = False
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
    comptime if skip_large:
        if (
            offsets[component + 1] - offsets[component]
            >= WINDING_COMPONENT_PARALLEL_MIN_FACES
        ):
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


def component_bounds_parallel_kernel[
    block_size: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    faces: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
    large_components: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var component = Int(large_components[Int(block_idx.x)])
    var lane = Int(thread_idx.x)
    if component >= len(offsets) - 2 or status[0] != 0:
        return
    comptime for axis in range(3):
        var lo = Float32(f32_max)
        var hi = -lo
        for slot in range(
            Int(offsets[component]) + lane,
            Int(offsets[component + 1]),
            block_size,
        ):
            var face = Int(faces[slot])
            comptime for corner in range(3):
                var value = points[3 * Int(indices[3 * face + corner]) + axis]
                lo = min(lo, value)
                hi = max(hi, value)
        lo = block.min[block_size=block_size, broadcast=True](lo)
        hi = block.max[block_size=block_size, broadcast=True](hi)
        if lane == 0:
            output[6 * component + axis] = lo
            output[6 * component + axis + 3] = hi


def component_bounds_tiles_kernel[
    block_size: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    faces: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
    tiles: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var tile = Int(block_idx.x)
    var component = Int(tiles[3 * tile])
    var lane = Int(thread_idx.x)
    if component >= len(offsets) - 2 or status[0] != 0:
        return
    comptime for axis in range(3):
        var lo = Float32(f32_max)
        var hi = -lo
        for slot in range(
            Int(tiles[3 * tile + 1]) + lane,
            Int(tiles[3 * tile + 2]),
            block_size,
        ):
            var face = Int(faces[slot])
            comptime for corner in range(3):
                var value = points[3 * Int(indices[3 * face + corner]) + axis]
                lo = min(lo, value)
                hi = max(hi, value)
        lo = block.min[block_size=block_size, broadcast=True](lo)
        hi = block.max[block_size=block_size, broadcast=True](hi)
        if lane == 0:
            output[6 * tile + axis] = lo
            output[6 * tile + axis + 3] = hi


def component_bounds_finish_tiles_kernel[
    block_size: Int
](
    partials: ImmSpan[Float32, ImmutAnyOrigin],
    components: ImmSpan[UInt32, ImmutAnyOrigin],
    offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    if status[0] != 0:
        return
    var group = Int(block_idx.x)
    var component = Int(components[group])
    var lane = Int(thread_idx.x)
    comptime for axis in range(3):
        var lo = Float32(f32_max)
        var hi = -lo
        for tile in range(
            Int(offsets[group]) + lane, Int(offsets[group + 1]), block_size
        ):
            lo = min(lo, partials[6 * tile + axis])
            hi = max(hi, partials[6 * tile + axis + 3])
        lo = block.min[block_size=block_size, broadcast=True](lo)
        hi = block.max[block_size=block_size, broadcast=True](hi)
        if lane == 0:
            output[6 * component + axis] = lo
            output[6 * component + axis + 3] = hi


struct GpuWindingCache[frame: Frame = .WORLD]:
    var tree: GpuSpatialBoundsBvh[Self.frame]
    var parents: DeviceBuffer[.uint32]
    var offsets: DeviceBuffer[.uint32]
    var faces: DeviceBuffer[.uint32]
    var large_components: DeviceBuffer[.uint32]
    var tiled_components: DeviceBuffer[.uint32]
    var tile_offsets: DeviceBuffer[.uint32]
    var tiles: DeviceBuffer[.uint32]
    var partial_bounds: DeviceBuffer[.float32]

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
        var large = [
            UInt32(i)
            for i in range(count)
            if offsets[i + 1] - offsets[i]
            >= WINDING_COMPONENT_PARALLEL_MIN_FACES
            and offsets[i + 1] - offsets[i] < WINDING_COMPONENT_TILED_MIN_FACES
        ]
        self.large_components = upload_list(ctx, large)
        var tiled_components = List[UInt32]()
        var tiles = List[UInt32]()
        var tile_offsets = List[UInt32](length=1, fill=0)
        for component in range(count):
            var begin = Int(offsets[component])
            var end = Int(offsets[component + 1])
            if end - begin >= WINDING_COMPONENT_TILED_MIN_FACES:
                tiled_components.append(UInt32(component))
                for first in range(begin, end, WINDING_COMPONENT_TILE_FACES):
                    tiles.extend(
                        [
                            UInt32(component),
                            UInt32(first),
                            UInt32(
                                min(first + WINDING_COMPONENT_TILE_FACES, end)
                            ),
                        ]
                    )
                tile_offsets.append(UInt32(len(tiles) // 3))
        self.tiled_components = upload_list(ctx, tiled_components)
        self.tile_offsets = upload_list(ctx, tile_offsets)
        self.tiles = upload_list(ctx, tiles)
        self.partial_bounds = ctx.enqueue_create_buffer[.float32](
            len(tiles) * 2
        )
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
        var rebuild_tiles_host = (
            forest_rebuild_tiles(tree.nodes) if count
            >= 1024 else List[UInt32]()
        )
        var rebuild_tiles = upload_list(ctx, rebuild_tiles_host)
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
        if len(self.large_components) == 0 and len(self.tiled_components) == 0:
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
        else:
            if len(self.large_components) + len(self.tiled_components) < count:
                ctx.enqueue_function[component_bounds_kernel[128, True]](
                    _device_span[mut=False](points),
                    _device_span[mut=False](indices),
                    _device_span[mut=False](self.offsets),
                    _device_span[mut=False](self.faces),
                    _device_span[mut=True](self.tree.update_items),
                    _device_span[mut=False](status),
                    grid_dim=(count + 127) // 128,
                    block_dim=128,
                )
            if len(self.large_components) != 0:
                ctx.enqueue_function[component_bounds_parallel_kernel[128]](
                    _device_span[mut=False](points),
                    _device_span[mut=False](indices),
                    _device_span[mut=False](self.offsets),
                    _device_span[mut=False](self.faces),
                    _device_span[mut=True](self.tree.update_items),
                    _device_span[mut=False](status),
                    _device_span[mut=False](self.large_components),
                    grid_dim=len(self.large_components),
                    block_dim=128,
                )
            if len(self.tiled_components) != 0:
                ctx.enqueue_function[component_bounds_tiles_kernel[128]](
                    _device_span[mut=False](points),
                    _device_span[mut=False](indices),
                    _device_span[mut=False](self.offsets),
                    _device_span[mut=False](self.faces),
                    _device_span[mut=True](self.partial_bounds),
                    _device_span[mut=False](status),
                    _device_span[mut=False](self.tiles),
                    grid_dim=len(self.tiles) // 3,
                    block_dim=128,
                )
                ctx.enqueue_function[component_bounds_finish_tiles_kernel[128]](
                    _device_span[mut=False](self.partial_bounds),
                    _device_span[mut=False](self.tiled_components),
                    _device_span[mut=False](self.tile_offsets),
                    _device_span[mut=True](self.tree.update_items),
                    _device_span[mut=False](status),
                    grid_dim=len(self.tiled_components),
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
