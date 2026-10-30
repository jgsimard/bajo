"""Group secondary rays by position and direction for every geometry mix."""
from std.math import max, clamp
from max.gpu.host import DeviceContext
from bajo.rt.gpu.queue_layout import gpu_rt_path_tile
from bajo.rt.wavefront_contract import (
    WAVE_COUNTER,
    WavePathFloatAbi,
)
from bajo.rt.gpu.queue_order import (
    GpuRtQueueOrderScratch,
    gpu_rt_queue_order_enabled,
    gpu_rt_queue_order_rays,
    enqueue_gpu_rt_queue_order,
)
from bajo.rt.gpu.views import GpuRtTraceQueueView, GpuRtSceneView
from bajo.rt.gpu.config import GpuRtSceneKind, GpuRtBvhFormat
from bajo.rt.gpu.order_bounds import gpu_rt_order_bounds
from bajo.core.morton import expand_bits_3d


@always_inline
def gpu_rt_spatial_order_enabled(capacity: Int, bounce: UInt32) -> Bool:
    return gpu_rt_queue_order_enabled(capacity, bounce) and bounce > 0


def gpu_rt_spatial_order_rays_kernel[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    scatter: Bool,
](queues: GpuRtTraceQueueView, scene: GpuRtSceneView):
    var capacity = Int(queues.capacity)
    var active = Int(queues.counters[unsafe_offset=WAVE_COUNTER.ACTIVE])
    if active == 0:
        return
    var paths = gpu_rt_path_tile(queues.src_path_fields, capacity)
    var scratch = GpuRtQueueOrderScratch.from_queues(queues)
    var low, high = gpu_rt_order_bounds[
        kind, sphere_format, triangle_format, tlas_format
    ](scene)
    var inv_x = 4.0 / max(high.x - low.x, Float32(1e-10))
    var inv_y = 4.0 / max(high.y - low.y, Float32(1e-10))
    var inv_z = 4.0 / max(high.z - low.z, Float32(1e-10))

    @always_inline
    def key(idx: Int) {imm} -> UInt32:
        var dx = paths[(WavePathFloatAbi.DX, idx)]
        var dy = paths[(WavePathFloatAbi.DY, idx)]
        var dz = paths[(WavePathFloatAbi.DZ, idx)]
        var direction = (
            UInt32(dx < 0.0) | (UInt32(dy < 0.0) << 1) | (UInt32(dz < 0.0) << 2)
        )
        var ox = paths[(WavePathFloatAbi.OX, idx)]
        var oy = paths[(WavePathFloatAbi.OY, idx)]
        var oz = paths[(WavePathFloatAbi.OZ, idx)]
        var x = UInt32(clamp((ox - low.x) * inv_x, Float32(0), Float32(3)))
        var y = UInt32(clamp((oy - low.y) * inv_y, Float32(0), Float32(3)))
        var z = UInt32(clamp((oz - low.z) * inv_z, Float32(0), Float32(3)))
        var position = (
            expand_bits_3d[1](x)
            | (expand_bits_3d[1](y) << 1)
            | (expand_bits_3d[1](z) << 2)
        )
        return (direction << 6) | position

    gpu_rt_queue_order_rays[scatter](scratch, active, key)


def enqueue_gpu_rt_spatial_order[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
](
    ctx: DeviceContext,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    bounce: UInt32,
) raises:
    if not gpu_rt_spatial_order_enabled(Int(queues.capacity), bounce):
        return
    enqueue_gpu_rt_queue_order[
        bin_count=512,
        counter=WAVE_COUNTER.ACTIVE,
        count_kernel=gpu_rt_spatial_order_rays_kernel[
            kind, sphere_format, triangle_format, tlas_format, False
        ],
        scatter_kernel=gpu_rt_spatial_order_rays_kernel[
            kind, sphere_format, triangle_format, tlas_format, True
        ],
    ](ctx, queues, scene)
