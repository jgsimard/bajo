"""Stable two-pass sort for the 30-bit keys emitted by morton3(dim=1024)."""
from bajo.bvh.cpu.builder.lbvh import MortonItem


def _pass15[
    shift: Int
](src: ImmSpan[MortonItem, _], dst: MutSpan[MortonItem, _]):
    comptime buckets = 1 << 15
    var counts = Array[UInt32, buckets](fill=0)
    for item in src:
        var bucket = Int((item.code >> UInt32(shift)) & UInt32(buckets - 1))
        counts[bucket] += 1
    var offsets = Array[Int, buckets](fill=0)
    var offset = 0
    for bucket in range(buckets):
        offsets[bucket] = offset
        offset += Int(counts[bucket])
    for item in src:
        var bucket = Int((item.code >> UInt32(shift)) & UInt32(buckets - 1))
        dst.unsafe_get(offsets[bucket]) = item
        offsets[bucket] += 1


def sort_morton30(pairs: MutSpan[MortonItem, _]):
    """Sort original records stably; every code must be below 2^30.

    Spatial construction supplies only canonical 10-bit-per-axis Morton keys.
    Counts and prefix sums bound each scatter offset by the destination length.
    Other CPU builders retain their general 32-bit sorting implementation.
    """
    var scratch = List[MortonItem](unsafe_uninit_length=len(pairs))
    _pass15[0](pairs, scratch)
    _pass15[15](scratch, pairs)
