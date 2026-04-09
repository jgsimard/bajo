from std.bit import pop_count, count_trailing_zeros
from std.gpu import WARP_SIZE
from std.gpu.primitives import warp, block
from std.os.atomic import Atomic


@fieldwise_init
struct DoubleBuffer[dtype: DType](Copyable):
    var current: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var alternate: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    def swap(mut self):
        swap(self.current, self.alternate)


def circular_shift(val: UInt32, lid: UInt32) -> UInt32:
    comptime WARP_MASK = UInt32(WARP_SIZE - 1)
    return warp.shuffle_idx(val, (lid + WARP_MASK) & WARP_MASK)


@always_inline
def warp_level_multi_split[
    keys_dtype: DType, BITS_PER_PASS: Int, KEYS_PER_THREAD: Int
](
    keys: InlineArray[Scalar[keys_dtype], KEYS_PER_THREAD],
    lid: Int,
    radix_shift: Scalar[keys_dtype],
    s_warp_hist_ptr: UnsafePointer[
        UInt32, MutExternalOrigin, address_space=AddressSpace.SHARED
    ],
) -> InlineArray[UInt32, KEYS_PER_THREAD]:
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)

    comptime mask_dtype = DType.uint64 if WARP_SIZE > 32 else DType.uint32
    comptime MaskInt = SIMD[mask_dtype, 1]

    var offsets = InlineArray[UInt32, KEYS_PER_THREAD](uninitialized=True)
    var lane_mask_lt = (MaskInt(1) << MaskInt(lid)) - 1

    comptime for i in range(KEYS_PER_THREAD):
        var warp_flags: MaskInt = ~MaskInt(0)
        var key = keys[i]

        comptime for k in range(BITS_PER_PASS):
            var t2 = ((key >> (radix_shift + Scalar[keys_dtype](k))) & 1) == 1
            var ballot = warp.vote[mask_dtype](t2)
            var match_mask = ballot if t2 else ~ballot
            warp_flags &= match_mask

        var bits = UInt32(pop_count(warp_flags & lane_mask_lt))
        var pre_increment_val: UInt32 = 0

        if bits == 0:
            var digit = Int((key >> radix_shift) & RADIX_MASK)
            var count = UInt32(pop_count(warp_flags))
            pre_increment_val = Atomic.fetch_add(s_warp_hist_ptr + digit, count)

        var leader_lane = count_trailing_zeros(warp_flags)
        pre_increment_val = warp.shuffle_idx(
            pre_increment_val, UInt32(leader_lane)
        )

        offsets[i] = pre_increment_val + UInt32(bits)
    return offsets^
