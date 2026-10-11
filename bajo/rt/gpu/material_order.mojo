"""Group coating shading by exact material ID, reusing invariant queue planes."""
from std.math import ceildiv
from max.gpu import global_idx, grid_dim, block_dim, lane_id
from max.gpu.host import DeviceContext
from bajo.rt.wavefront_contract import (
    WAVE_COUNTER,
)
from bajo.rt.gpu.queue_order import (
    GpuRtQueueOrderScratch,
    gpu_rt_queue_order_enabled,
    gpu_rt_queue_order_group,
    gpu_rt_queue_order_init_kernel,
    gpu_rt_queue_order_scan_kernel,
)
from bajo.rt.gpu.views import GpuRtTraceQueueView
from bajo.rt.material_types import SurfaceId


def gpu_rt_material_order_rays_kernel[
    scatter: Bool
](queues: GpuRtTraceQueueView, coated_count: Int32):
    var active = Int(queues.counters[unsafe_offset=WAVE_COUNTER.SHADE])
    var scratch = GpuRtQueueOrderScratch.from_queues(queues)
    var idx = global_idx.x
    var lane = lane_id()
    # Every lane participates in the match and ballot, including the inactive
    # tail of the last warp. Mask it out before counting or reserving slots.
    while idx - lane < active:
        var valid = idx < active
        var key = UInt32(0)
        if valid:
            var surface = queues.shade_surfaces[unsafe_offset=idx]
            key = SurfaceId.index_from_raw(surface) if SurfaceId.kind_from_raw(
                surface
            ) == .COATED_DIFFUSE else UInt32(coated_count)
        gpu_rt_queue_order_group[scatter](scratch, idx, key, valid)
        idx += Int(grid_dim.x * block_dim.x)


def gpu_rt_material_order_enabled(
    capacity: Int, bounce: UInt32, coated_count: Int
) -> Bool:
    return (
        gpu_rt_queue_order_enabled(capacity, bounce)
        and coated_count > 1
        and coated_count < 64
    )


def enqueue_gpu_rt_material_order(
    ctx: DeviceContext, queues: GpuRtTraceQueueView, coated_count: Int
) raises:
    ctx.enqueue_function[gpu_rt_queue_order_init_kernel[64]](
        queues, grid_dim=1, block_dim=64
    )
    var blocks = min(ceildiv(Int(queues.capacity), 256), 1024)
    ctx.enqueue_function[gpu_rt_material_order_rays_kernel[False]](
        queues, Int32(coated_count), grid_dim=blocks, block_dim=256
    )
    ctx.enqueue_function[gpu_rt_queue_order_scan_kernel[64]](
        queues, grid_dim=1, block_dim=64
    )
    ctx.enqueue_function[gpu_rt_material_order_rays_kernel[True]](
        queues, Int32(coated_count), grid_dim=blocks, block_dim=256
    )
