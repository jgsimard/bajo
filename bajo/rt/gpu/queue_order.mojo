"""Shared histogram, scan and index scatter for GPU wavefront ordering.

The RT queue adapters reconstruct constant ray bounds rather than storing them.
Until the next bounce, destination path T_MIN holds histogram offsets, shadow
T_MIN holds work indices, and destination path T_MAX holds sampled light IDs.
These aliases leave every ray, shade record, path ID and random stream intact.
Stages reuse the index plane in stream order: traversal, coating, visibility.
"""

from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros, pop_count
from max.gpu import global_idx, lane_id, WARP_SIZE
from max.gpu.primitives import block, warp

from bajo.rt.wavefront_contract import WavePathFloatAbi, WaveShadowFloatAbi
from bajo.rt.gpu.views import GpuRtTraceQueueView


comptime GPU_RT_ORDER_MIN_CAPACITY = 1024
comptime GPU_RT_ORDER_LAST_BOUNCE = 8


@fieldwise_init
struct GpuRtQueueOrderScratch(TrivialRegisterPassable):
    """Typed non-owning aliases for the invariant queue planes."""

    var bins: Pointer[UInt32, MutUntrackedOrigin]
    var indices: Pointer[UInt32, MutUntrackedOrigin]
    var light_ids: Pointer[UInt32, MutUntrackedOrigin]

    @staticmethod
    @always_inline
    def from_queues(queues: GpuRtTraceQueueView) -> Self:
        var capacity = Int(queues.capacity)
        return Self(
            queues.dst_path_fields.unsafe_offset(
                WavePathFloatAbi.T_MIN * capacity
            ).unsafe_bitcast[UInt32](),
            queues.shadow_fields.unsafe_offset(
                WaveShadowFloatAbi.T_MIN * capacity
            ).unsafe_bitcast[UInt32](),
            queues.dst_path_fields.unsafe_offset(
                WavePathFloatAbi.T_MAX * capacity
            ).unsafe_bitcast[UInt32](),
        )


@always_inline
def gpu_rt_queue_order_enabled(capacity: Int, bounce: UInt32) -> Bool:
    return (
        capacity >= GPU_RT_ORDER_MIN_CAPACITY
        and bounce < GPU_RT_ORDER_LAST_BOUNCE
    )


def gpu_rt_queue_order_init_kernel[bin_count: Int](queues: GpuRtTraceQueueView):
    var idx = global_idx.x
    if idx < bin_count:
        GpuRtQueueOrderScratch.from_queues(queues).bins[unsafe_offset=idx] = 0


def gpu_rt_queue_order_scan_kernel[bin_count: Int](queues: GpuRtTraceQueueView):
    var idx = global_idx.x
    var scratch = GpuRtQueueOrderScratch.from_queues(queues)
    var count = UInt32(0)
    if idx < bin_count:
        count = scratch.bins[unsafe_offset=idx]
    var offset = block.prefix_sum[block_size=bin_count, exclusive=True](count)
    if idx < bin_count:
        scratch.bins[unsafe_offset=idx] = offset


@always_inline
def gpu_rt_queue_order_group[
    scatter: Bool
](scratch: GpuRtQueueOrderScratch, idx: Int, key: UInt32, valid: Bool,):
    """Reserve once per distinct warp key; all lanes participate, including tails.
    """
    comptime mask_dtype = DType.uint32 if WARP_SIZE <= 32 else DType.uint64
    comptime Mask = Scalar[mask_dtype]
    var lane = lane_id()
    var peers = Mask(warp.match_any(key)) & warp.vote[mask_dtype](valid)
    var leader = UInt32(0)
    if valid:
        leader = UInt32(count_trailing_zeros(peers))
    var offset = UInt32(0)
    if valid and UInt32(lane) == leader:
        offset = Atomic.fetch_add[ordering=Ordering.RELAXED](
            scratch.bins.unsafe_offset(Int(key)), UInt32(pop_count(peers))
        )
    comptime if scatter:
        offset = warp.shuffle_idx(offset, leader)
        var rank = UInt32(pop_count(peers & ((Mask(1) << Mask(lane)) - 1)))
        if valid:
            scratch.indices[unsafe_offset=Int(offset + rank)] = UInt32(idx)
