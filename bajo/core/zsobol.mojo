# SPDX-License-Identifier: Apache-2.0
# Index permutation and hashing adapted from pbrt, Copyright (c) 1998-2020
# Matt Pharr, Wenzel Jakob, and Greg Humphreys.
# See bajo/parser/pbrt/LICENSE-pbrt.txt.
"""Morton-ordered Z-Sobol index permutation shared by CPU SIMD and GPU code.

The permutation follows pbrt-v4's ZSobolSampler::GetSampleIndex. Adjacent
dimensions form a two-dimensional Sobol net. Non-power-of-two sequence lengths
use a prefix of the next power-of-two domain, without changing the render's
requested sample count. Indices retain all 52 bits supported by PBRT's net.
"""

from std.bit import count_leading_zeros
from std.builtin.globals import global_constant


@always_inline
def zsobol_log2_samples(count: UInt32) -> Int:
    return 32 - Int(count_leading_zeros(max(count, UInt32(1)) - UInt32(1)))


@always_inline
def _mix_bits[
    length: SIMDLength
](value: SIMD[.uint64, length]) -> SIMD[.uint64, length]:
    var bits = value
    bits ^= bits >> UInt64(31)
    bits *= UInt64(0x7FB5D329728EA185)
    bits ^= bits >> UInt64(27)
    bits *= UInt64(0x81DADEF4BC2DD44D)
    return bits ^ (bits >> UInt64(33))


@always_inline
def _spread_bits[
    length: SIMDLength
](value: SIMD[.uint32, length]) -> SIMD[.uint64, length]:
    var bits = value.cast[.uint64]()
    bits = (bits ^ (bits << UInt64(16))) & UInt64(0x0000FFFF0000FFFF)
    bits = (bits ^ (bits << UInt64(8))) & UInt64(0x00FF00FF00FF00FF)
    bits = (bits ^ (bits << UInt64(4))) & UInt64(0x0F0F0F0F0F0F0F0F)
    bits = (bits ^ (bits << UInt64(2))) & UInt64(0x3333333333333333)
    return (bits ^ (bits << UInt64(1))) & UInt64(0x5555555555555555)


@always_inline
def zsobol_morton_index[
    length: SIMDLength
](
    pixel_id: SIMD[.uint32, length],
    image_width: UInt32,
    sample_index: SIMD[.uint64, length],
    log2_samples: Int,
) -> SIMD[.uint64, length]:
    var width = max(image_width, UInt32(1))
    var morton = _spread_bits(pixel_id % width) | (
        _spread_bits(pixel_id / width) << UInt64(1)
    )
    return (morton << UInt64(log2_samples)) | sample_index


# Four 2-bit entries per byte, in PBRT's permutation order.
comptime _PERMUTATIONS: Array[UInt32, 24] = [
    0xE4,
    0xB4,
    0xD8,
    0x78,
    0x6C,
    0x9C,
    0xE1,
    0xB1,
    0xC9,
    0x39,
    0x2D,
    0x8D,
    0xC6,
    0x36,
    0xD2,
    0x72,
    0x4E,
    0x1E,
    0x27,
    0x87,
    0x1B,
    0x4B,
    0x63,
    0x93,
]


@always_inline
def zsobol_sample_index[
    length: SIMDLength
](
    morton: SIMD[.uint64, length],
    log2_samples: Int,
    base4_digits: Int,
    dimension: UInt32,
) -> SIMD[.uint64, length]:
    ref permutations = global_constant[_PERMUTATIONS]()
    var result = SIMD[.uint64, length](0)
    var odd = log2_samples & 1
    var salt = UInt64(UInt32(0x55555555) * dimension)
    for index in range(base4_digits - 1, odd - 1, -1):
        var shift = 2 * index - odd
        var digit = ((morton >> UInt64(shift)) & UInt64(3)).cast[.uint32]()
        var higher = morton >> UInt64(shift + 2)
        var permutation = (_mix_bits(higher ^ salt) >> UInt64(24)) % UInt64(24)
        var packed = permutations.unsafe_ptr().unsafe_gather(
            permutation.cast[.int]()
        )
        var permuted = (packed >> (digit * UInt32(2))) & UInt32(3)
        result |= permuted.cast[.uint64]() << UInt64(shift)
    if odd:
        result |= (morton & UInt64(1)) ^ (
            _mix_bits((morton >> UInt64(1)) ^ salt) & UInt64(1)
        )
    return result


@always_inline
def zsobol_scramble(dimension: UInt32, seed: UInt32) -> UInt64:
    """PBRT Hash(dimension, seed): MurmurHash64A of two little-endian Int32s."""
    var multiplier = UInt64(0xC6A4A7935BD1E995)
    var word = UInt64(dimension) | (UInt64(seed) << UInt64(32))
    word *= multiplier
    word ^= word >> UInt64(47)
    word *= multiplier
    var hash = ((UInt64(8) * multiplier) ^ word) * multiplier
    hash ^= hash >> UInt64(47)
    hash *= multiplier
    return hash ^ (hash >> UInt64(47))
