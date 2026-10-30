"""Group visibility work by the actual sampled light ID."""
from max.gpu.host import DeviceContext
from bajo.rt.wavefront_contract import (
    WAVE_COUNTER,
)
from bajo.rt.gpu.queue_order import (
    GpuRtQueueOrderScratch,
    gpu_rt_queue_order_enabled,
    gpu_rt_queue_order_rays,
    enqueue_gpu_rt_queue_order,
)
from bajo.rt.gpu.views import GpuRtTraceQueueView


def gpu_rt_light_order_rays_kernel[scatter: Bool](queues: GpuRtTraceQueueView):
    var active = Int(queues.counters[unsafe_offset=WAVE_COUNTER.SHADOW])
    if active == 0:
        return
    var scratch = GpuRtQueueOrderScratch.from_queues(queues)

    @always_inline
    def key(idx: Int) {imm} -> UInt32:
        return scratch.light_ids[unsafe_offset=idx]

    gpu_rt_queue_order_rays[scatter](scratch, active, key)


def enqueue_gpu_rt_light_order(
    ctx: DeviceContext, queues: GpuRtTraceQueueView, bounce: UInt32
) raises:
    if not gpu_rt_queue_order_enabled(Int(queues.capacity), bounce):
        return
    enqueue_gpu_rt_queue_order[
        bin_count=64,
        counter=WAVE_COUNTER.SHADOW,
        count_kernel=gpu_rt_light_order_rays_kernel[False],
        scatter_kernel=gpu_rt_light_order_rays_kernel[True],
    ](ctx, queues)
