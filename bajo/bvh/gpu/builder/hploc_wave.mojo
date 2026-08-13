from std.bit import count_trailing_zeros, pop_count
from std.gpu import WARP_SIZE
from std.gpu.primitives import warp


# Kernels using these helpers must have a comptime specialization parameter,
# matching the existing OneSweep kernels, so Mojo lowers subgroup intrinsics
# for the active accelerator target rather than materializing a host version.
comptime HPLOC_WAVE_SIZE = WARP_SIZE
comptime HPLOC_INVALID_LANE = WARP_SIZE


@always_inline
def hploc_wave_ballot(predicate: Bool) -> UInt64:
    """Return the subgroup vote as a backend-independent UInt64 mask."""

    comptime if WARP_SIZE > 32:
        return UInt64(warp.vote[DType.uint64](predicate))
    else:
        return UInt64(warp.vote[DType.uint32](predicate))


@always_inline
def hploc_wave_rank(mask: UInt64, lane: Int) -> UInt32:
    """Count set lanes preceding this lane in a previously computed ballot."""

    var lane_mask_lt = (UInt64(1) << UInt64(lane)) - UInt64(1)
    return UInt32(pop_count(mask & lane_mask_lt))


@always_inline
def hploc_wave_first_lane(mask: UInt64) -> Int:
    """Return the first set lane or HPLOC_INVALID_LANE for an empty mask."""

    if mask == 0:
        return HPLOC_INVALID_LANE
    return Int(count_trailing_zeros(mask))


@fieldwise_init
struct HplocWaveMinimum(TrivialRegisterPassable):
    """Lexicographically ordered candidate for deterministic wave arg-min."""

    var cost_key: UInt32
    var tie_key: UInt32


@always_inline
def hploc_wave_is_better(
    candidate: HplocWaveMinimum,
    current: HplocWaveMinimum,
) -> Bool:
    """Compare H-PLOC candidates by cost, then by deterministic tie key."""

    return candidate.cost_key < current.cost_key or (
        candidate.cost_key == current.cost_key
        and candidate.tie_key < current.tie_key
    )
