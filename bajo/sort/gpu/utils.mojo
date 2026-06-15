from std.bit import pop_count, count_trailing_zeros
from std.gpu import WARP_SIZE, lane_id
from std.gpu.primitives import warp, block
from std.atomic import Atomic, Ordering


def circular_shift(val: UInt32) -> UInt32:
    comptime WARP_MASK = UInt32(WARP_SIZE - 1)
    var lid = UInt32(lane_id())
    return warp.shuffle_idx(val, (lid + WARP_MASK) & WARP_MASK)


def warp_level_multi_split[
    origin: MutOrigin,
    address_space: AddressSpace,
    //,
    keys_dtype: DType,
    BITS_PER_PASS: Int,
    KEYS_PER_THREAD: Int,
    USE_MATCH_ANY: Bool = False,
](
    keys: InlineArray[Scalar[keys_dtype], KEYS_PER_THREAD],
    lid: Int,
    radix_shift: Scalar[keys_dtype],
    s_warp_hist_ptr: UnsafePointer[UInt32, origin, address_space=address_space],
) -> InlineArray[UInt32, KEYS_PER_THREAD]:
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)

    comptime mask_dtype = DType.uint64 if WARP_SIZE > 32 else DType.uint32
    comptime MaskInt = SIMD[mask_dtype, 1]

    var offsets = InlineArray[UInt32, KEYS_PER_THREAD](uninitialized=True)
    var lane_mask_lt = (MaskInt(1) << MaskInt(lid)) - 1

    comptime for i in range(KEYS_PER_THREAD):
        var warp_flags: MaskInt
        var key = keys[i]

        comptime if USE_MATCH_ANY:
            warp_flags = MaskInt(
                warp.match_any(UInt32((key >> radix_shift) & RADIX_MASK))
            )
        else:
            warp_flags = ~MaskInt(0)
            comptime for k in range(BITS_PER_PASS):
                var bit_is_set = (
                    (key >> (radix_shift + Scalar[keys_dtype](k))) & 1
                ) == 1
                var ballot = warp.vote[mask_dtype](bit_is_set)
                var match_mask = ballot if bit_is_set else ~ballot
                warp_flags &= match_mask

        var bits = UInt32(pop_count(warp_flags & lane_mask_lt))
        var pre_increment_val: UInt32 = 0

        if bits == 0:
            var digit = Int((key >> radix_shift) & RADIX_MASK)
            var count = UInt32(pop_count(warp_flags))
            pre_increment_val = Atomic.fetch_add[ordering=Ordering.RELAXED](
                s_warp_hist_ptr + digit, count
            )

        var leader_lane = count_trailing_zeros(warp_flags)
        pre_increment_val = warp.shuffle_idx(
            pre_increment_val, UInt32(leader_lane)
        )

        offsets[i] = pre_increment_val + bits
    return offsets^
