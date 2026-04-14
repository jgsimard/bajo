from std.benchmark import run, Unit, keep
from std.random import random_float64
from std.memory import UnsafePointer
from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.math import fma

from bajo.core.quat import Quat
from bajo.core.utils import degrees_to_radians
from bajo.core.random import PhiloxRNG
from bajo.core.vec import Vec3f32

comptime f32 = DType.float32
comptime num_elements = 100000


def quat_mul_1(q1: Quat, q2: Quat) -> Quat:
    x = q1.w() * q2.x() + q1.x() * q2.w() + q1.y() * q2.z() - q1.z() * q2.y()
    y = q1.w() * q2.y() - q1.x() * q2.z() + q1.y() * q2.w() + q1.z() * q2.x()
    z = q1.w() * q2.z() + q1.x() * q2.y() - q1.y() * q2.x() + q1.z() * q2.w()
    w = q1.w() * q2.w() - q1.x() * q2.x() - q1.y() * q2.y() - q1.z() * q2.z()
    return Quat(x, y, z, w)


# def quat_mul_2(q1: Quat, q2: Quat) -> Quat:
#     # Version 2 : more SIMD: 3 shuffles, 4 mul, 12 add
#     var t0 = q1.data[0] * q2.data
#     var t1 = q1.data[1] * q2.data.shuffle[1, 0, 3, 2]()
#     var t2 = q1.data[2] * q2.data.shuffle[2, 3, 0, 1]()
#     var t3 = q1.data[3] * q2.data.shuffle[3, 2, 1, 0]()

#     # Apply the standard Hamilton sign patterns
#     w = t0[0] - t1[0] - t2[0] - t3[0]
#     x = t0[1] + t1[1] + t2[1] - t3[1]
#     y = t0[2] - t1[2] + t2[2] + t3[2]
#     z = t0[3] + t1[3] - t2[3] + t3[3]

#     return Quat(w, x, y, z)

# # These are in the wxyz format
# def quat_mul_3(q1: Quat, q2: Quat) -> Quat:
#     # Version 3: SIMD FMA : 3 shuffles, 4 mul, 3 fma
#     comptime s1 = SIMD[DType.float32, 4](-1.0, 1.0, -1.0, 1.0)
#     comptime s2 = SIMD[DType.float32, 4](-1.0, 1.0, 1.0, -1.0)
#     comptime s3 = SIMD[DType.float32, 4](-1.0, -1.0, 1.0, 1.0)

#     res = q1.data[0] * q2.data
#     res = fma(q1.data[1] * s1, q2.data.shuffle[1, 0, 3, 2](), res)
#     res = fma(q1.data[2] * s2, q2.data.shuffle[2, 3, 0, 1](), res)
#     res = fma(q1.data[3] * s3, q2.data.shuffle[3, 2, 1, 0](), res)
#     return Quat(res)


struct BenchmarkData(Copyable):
    var src_a: UnsafePointer[Quat, MutAnyOrigin]
    var src_b: UnsafePointer[Quat, MutAnyOrigin]
    var dst: UnsafePointer[Quat, MutAnyOrigin]

    def __init__(out self):
        self.src_a = alloc[Quat](num_elements)
        self.src_b = alloc[Quat](num_elements)
        self.dst = alloc[Quat](num_elements)
        rng = PhiloxRNG(123, 123)

        for i in range(num_elements):
            self.src_a[i] = Quat.from_axis_angle(
                Vec3f32(1, 0, 0), rng.next_f32()
            )
            self.src_b[i] = Quat.from_axis_angle(
                Vec3f32(0, 1, 0), rng.next_f32()
            )

    def __del__(deinit self):
        self.src_a.free()
        self.src_b.free()
        self.dst.free()


@always_inline
def dispatch_mul[version: Int](q1: Quat, q2: Quat) -> Quat:
    comptime if version == 1:
        return quat_mul_1(q1, q2)
    # elif version == 2:
    #     return quat_mul_2(q1, q2)
    # elif version == 3:
    #     return quat_mul_3(q1, q2)
    else:
        return q1 * q2  # Quat.__mul__


def main() raises:
    def bench_throughput[version: Int]() raises:
        data = BenchmarkData()

        # bounds checking makes this benchmars 3X slower !
        def wrapper() raises capturing:
            for i in range(num_elements):
                data.dst[i] = dispatch_mul[version](
                    data.src_a[i], data.src_b[i]
                )
            keep(data.dst[0].data)

        report = run[func3=wrapper](max_iters=1000)
        avg_time_us = round(report.mean(Unit.us), 2)
        mops = round(num_elements / avg_time_us, 2)

        print(t"Throughput: {mops} Mops/s | Avg Time: {avg_time_us} us")

    bench_throughput[1]()
    # bench_throughput[2]()
    # bench_throughput[3]()
    bench_throughput[4]()

    def bench_latency[version: Int]() raises:
        angle = degrees_to_radians(Float32(45))
        q2 = Quat.from_axis_angle(Vec3f32(0, 1, 0), angle)
        q3 = Quat.from_axis_angle(Vec3f32(1, 0, 0), angle)
        a = dispatch_mul[version](q2, q3)
        b = Quat(0.353553, 0.353553, -0.146447, 0.853553)
        assert_almost_equal(a.data, b.data, atol=1e-6)

        q = a

        def bench_fn() raises capturing:
            for _ in range(1e6):
                q = dispatch_mul[version](q, q2)
                q = dispatch_mul[version](q, q3)
            keep(q)

        var time_us = round(run[func3=bench_fn](max_iters=100).mean(Unit.us), 1)

        print(t"v{version} : {time_us} us")

    bench_latency[1]()
    # bench_latency[2]()
    # bench_latency[3]()
    bench_latency[4]()

    # Throughput: 1176.84 Mops/s | Avg Time: 84.97 us
    # Throughput: 1380.21 Mops/s | Avg Time: 72.45 us
    # Throughput: 1519.97 Mops/s | Avg Time: 65.79 us
    # Throughput: 1514.84 Mops/s | Avg Time: 66.01 us
    # v1 : 6668.1 us
    # v2 : 6079.7 us
    # v3 : 6342.2 us
    # v4 : 6061.5 us
