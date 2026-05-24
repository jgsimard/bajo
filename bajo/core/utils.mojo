from std.math import pi
from std.sys.info import size_of
from std.bit import count_trailing_zeros
from std.memory import pack_bits

from bajo.core.vec import Vec3, Vec3f32
from bajo.obj import read_obj, triangulated_indices


def print_size_of[type: AnyType]():
    comptime name = reflect[type].name()
    size_bytes = size_of[type]()
    size_32 = size_bytes / 4
    print(t"{name}: {size_bytes} bytes, {size_32} x 32 bits")


def is_power_of_2(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0


def degrees_to_radians[
    dtype: DType, size: Int
](degrees: SIMD[dtype, size]) -> SIMD[dtype, size]:
    return degrees * comptime (pi / 180.0)


def radians_to_degrees[
    dtype: DType, size: Int
](radians: SIMD[dtype, size]) -> SIMD[dtype, size]:
    return radians * comptime (180.0 / pi)


def ns_to_ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


def ns_to_mrays_per_s(ns: Int, ray_count: Int) -> Float64:
    var seconds = Float64(ns) * 1.0e-9
    if seconds <= 0.0:
        return 0.0
    return (Float64(ray_count) / seconds) / 1_000_000.0


def print_vec3_rounded[dtype: DType](name: String, v: Vec3[dtype]):
    var x = round(v.x, 3)
    var y = round(v.y, 3)
    var z = round(v.z, 3)
    print(t"{name} ({x}, {y}, {z})")


def min_argmin[
    dtype: DType, width: Int
](x: SIMD[dtype, width]) -> Tuple[Scalar[dtype], Int]:
    comptime if width == 2:
        var best = x[0]
        var index = 0
        if x[1] < best:
            best = x[1]
            index = 1
        return (best, index)
    else:
        var _min = x.reduce_min()
        var is_min = x.eq(_min)
        var mask = pack_bits(is_min)
        var index = Int(count_trailing_zeros(mask))
        return (_min, index)


@always_inline
def fmax[
    dtype: DType, width: Int
](a: SIMD[dtype, width], b: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return a.gt(b).select(a, b)


@always_inline
def fmin[
    dtype: DType, width: Int
](a: SIMD[dtype, width], b: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return a.lt(b).select(a, b)
