from std.math import sqrt, clamp, pi
from std.sys.info import size_of

from bajo.core.vec import Vec3, vclamp, Vec3f32
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


@always_inline
def ns_to_ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


@always_inline
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


def pack_obj_triangles(path: String) raises -> List[Vec3f32]:
    var mesh = read_obj(path)
    var idx = triangulated_indices(mesh)

    var out = List[Vec3f32](capacity=len(idx))
    for i in range(len(idx)):
        var p = Int(idx[i].p)
        var base = p * 3
        out.append(
            Vec3f32(
                mesh.positions[base + 0],
                mesh.positions[base + 1],
                mesh.positions[base + 2],
            )
        )

    return out^
