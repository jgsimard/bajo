from std.math import clamp


def expand_bits_2d[
    size: Int
](n_in: SIMD[DType.uint32, size]) -> SIMD[DType.uint32, size]:
    # 0000 0000 0000 0000 gfed cba9 8675 4321
    # 0g0f 0e0d 0c0b 0a09 0807 0605 0403 0201
    n = n_in
    n &= 0x0000FFFF
    n = (n ^ (n << 8)) & 0x00FF00FF
    n = (n ^ (n << 4)) & 0x0F0F0F0F
    n = (n ^ (n << 2)) & 0x33333333
    n = (n ^ (n << 1)) & 0x55555555
    return n


def expand_bits_3d[
    size: Int
](n_in: SIMD[DType.uint32, size]) -> SIMD[DType.uint32, size]:
    # 0000 0000 0000 0000 0000 00a9 8675 4321
    # 0000 a009 0080 0700 6005 0040 0300 2001
    n = n_in
    n = (n ^ (n << 16)) & 0xFF0000FF
    n = (n ^ (n << 8)) & 0x0300F00F
    n = (n ^ (n << 4)) & 0x030C30C3
    n = (n ^ (n << 2)) & 0x09249249
    return n


def morton3[
    size: Int, dim: UInt32 = 1024
](
    x: SIMD[DType.float32, size],
    y: SIMD[DType.float32, size],
    z: SIMD[DType.float32, size],
) -> SIMD[DType.uint32, size]:
    """Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*lwp2(dim) bits.
    """

    # lwp2(dim):stands for Log Width base 2.
    # function that returns the number of bits needed to represent the value dim.
    # lwp2(1024) = 10, because 2^10 = 1024

    # masks for ux, uy, uz = use 3*10 = 30 bits
    # 0000 1001 0010 0100 1001 0010 0100 1001
    # 0001 0010 0100 1001 0010 0100 1001 0010
    # 0010 0100 1001 0010 0100 1001 0010 0100

    comptime dimf = Float32(dim)
    comptime T = SIMD[DType.uint32, size]
    ux = clamp(T(x * dimf), 0, dim - 1)
    uy = clamp(T(y * dimf), 0, dim - 1)
    uz = clamp(T(z * dimf), 0, dim - 1)
    return (
        (expand_bits_3d(uz) << 2)
        | (expand_bits_3d(uy) << 1)
        | expand_bits_3d(ux)
    )
