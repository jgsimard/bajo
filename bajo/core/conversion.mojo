from std.math import pi


def degrees_to_radians[
    dtype: DType, size: Int
](degrees: SIMD[dtype, size]) -> SIMD[dtype, size]:
    comptime factor = pi / 180.0
    return degrees * factor


def radians_to_degrees[
    dtype: DType, size: Int
](radians: SIMD[dtype, size]) -> SIMD[dtype, size]:
    comptime factor = 180.0 / pi
    return radians * factor
