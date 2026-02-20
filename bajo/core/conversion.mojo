from math import pi


fn degrees_to_radians[
    type: DType, size: Int
](degrees: SIMD[type, size]) -> SIMD[type, size]:
    comptime factor = pi / 180.0
    return degrees * factor


fn radians_to_degrees[
    type: DType, size: Int
](radian: SIMD[type, size]) -> SIMD[type, size]:
    comptime factor = 180.0 / pi
    return radian * factor
