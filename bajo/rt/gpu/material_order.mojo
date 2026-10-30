"""Group coating shading by exact material ID, reusing invariant queue planes."""
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
from bajo.rt.material_types import SurfaceId


def gpu_rt_material_order_rays_kernel[
    scatter: Bool
](queues: GpuRtTraceQueueView, coated_count: Int32):
    var active = Int(queues.counters[unsafe_offset=WAVE_COUNTER.SHADE])
    if active == 0:
        return
    var scratch = GpuRtQueueOrderScratch.from_queues(queues)

    @always_inline
    def key(idx: Int) {imm} -> UInt32:
        var surface = queues.shade_surfaces[unsafe_offset=idx]
        return SurfaceId.index_from_raw(surface) if SurfaceId.kind_from_raw(
            surface
        ) == .COATED_DIFFUSE else UInt32(coated_count)

    gpu_rt_queue_order_rays[scatter](scratch, active, key)


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
    enqueue_gpu_rt_queue_order[
        bin_count=64,
        counter=WAVE_COUNTER.SHADE,
        count_kernel=gpu_rt_material_order_rays_kernel[False],
        scatter_kernel=gpu_rt_material_order_rays_kernel[True],
    ](ctx, queues, Int32(coated_count))
