"""Group visibility work by the actual sampled light ID."""
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


@always_inline
def gpu_rt_light_order_enabled(capacity: Int, bounce: UInt32) -> Bool:
    return gpu_rt_queue_order_enabled(capacity, bounce)


def gpu_rt_light_order_rays_kernel[scatter: Bool](queues: GpuRtTraceQueueView):
    var active = Int(queues.counters[unsafe_offset=WAVE_COUNTER.SHADOW])
    var scratch = GpuRtQueueOrderScratch.from_queues(queues)
    var idx = global_idx.x
    var lane = lane_id()
    # Every lane participates in the match and ballot, including the inactive
    # tail of the last warp. Mask it out before counting or reserving slots.
    while idx - lane < active:
        var valid = idx < active
        var key = UInt32(0)
        if valid:
            key = scratch.light_ids[unsafe_offset=idx]
        gpu_rt_queue_order_group[scatter](scratch, idx, key, valid)
        idx += Int(grid_dim.x * block_dim.x)


def enqueue_gpu_rt_light_order(
    ctx: DeviceContext, queues: GpuRtTraceQueueView, bounce: UInt32
) raises:
    if not gpu_rt_light_order_enabled(Int(queues.capacity), bounce):
        return
    ctx.enqueue_function[gpu_rt_queue_order_init_kernel[64]](
        queues, grid_dim=1, block_dim=64
    )
    var blocks = min(ceildiv(Int(queues.capacity), 256), 1024)
    ctx.enqueue_function[gpu_rt_light_order_rays_kernel[False]](
        queues, grid_dim=blocks, block_dim=256
    )
    ctx.enqueue_function[gpu_rt_queue_order_scan_kernel[64]](
        queues, grid_dim=1, block_dim=64
    )
    ctx.enqueue_function[gpu_rt_light_order_rays_kernel[True]](
        queues, grid_dim=blocks, block_dim=256
    )
