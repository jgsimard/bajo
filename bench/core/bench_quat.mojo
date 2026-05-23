from std.benchmark import run, Unit, keep
from std.random import random_float64
from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.math import fma

from bajo.core.quat import Quaternion
from bajo.core.utils import degrees_to_radians
from bajo.core.random import Rng
from bajo.core.vec import Vec3

comptime dtype = DType.float32
comptime num_elements = 100000


def quat_mul_0[
    width: Int
](q1: Quaternion[dtype, width], q2: Quaternion[dtype, width]) -> Quaternion[
    dtype, width
]:
    x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    return Quaternion[dtype, width](x, y, z, w)


struct BenchmarkData[width: Int](Copyable):
    var src_a: List[Quaternion[dtype, Self.width]]
    var src_b: List[Quaternion[dtype, Self.width]]
    var dst: List[Quaternion[dtype, Self.width]]

    def __init__(out self):
        rng = Rng(123, 123)

        self.src_a = [
            Quaternion[dtype, Self.width].from_axis_angle(
                Vec3[dtype, Self.width](1, 0, 0), rng.f32()
            )
            for _ in range(num_elements / Self.width)
        ]
        self.src_b = [
            Quaternion[dtype, Self.width].from_axis_angle(
                Vec3[dtype, Self.width](0, 1, 0), rng.f32()
            )
            for _ in range(num_elements / Self.width)
        ]
        self.dst = [
            Quaternion[dtype, Self.width].identity()
            for _ in range(num_elements / Self.width)
        ]


def dispatch_mul[
    version: Int, width: Int
](q1: Quaternion[dtype, width], q2: Quaternion[dtype, width]) -> Quaternion[
    dtype, width
]:
    comptime if version == 0:
        return quat_mul_0(q1, q2)
    else:
        return q1 * q2  # Quat.__mul__


def main() raises:
    def bench_throughput[version: Int, width: Int]() raises:
        data = BenchmarkData[width]()

        # bounds checking makes this benchmars 3X slower !
        def wrapper() raises capturing:
            for i in range(num_elements / width):
                data.dst.unsafe_ptr()[i] = dispatch_mul[version](
                    data.src_a.unsafe_ptr()[i], data.src_b.unsafe_ptr()[i]
                )
            keep(data.dst[0].z)

        report = run[func3=wrapper](max_iters=1000)
        avg_time_us = round(report.mean(Unit.us), 2)
        mops = round(num_elements / avg_time_us, 2)

        print(t"Throughput: {mops} Mops/s | Avg Time: {avg_time_us} us")

    def bench_latency[version: Int, width: Int]() raises:
        angle = degrees_to_radians(Float32(45))
        q2 = Quaternion[dtype, width].from_axis_angle(
            Vec3[dtype, width](0, 1, 0), angle
        )
        q3 = Quaternion[dtype, width].from_axis_angle(
            Vec3[dtype, width](1, 0, 0), angle
        )
        a = dispatch_mul[version](q2, q3)
        b = Quaternion[dtype, width](0.353553, 0.353553, -0.146447, 0.853553)
        assert_almost_equal(a.x, b.x, atol=1e-6)
        assert_almost_equal(a.y, b.y, atol=1e-6)
        assert_almost_equal(a.z, b.z, atol=1e-6)
        assert_almost_equal(a.w, b.w, atol=1e-6)

        q = a

        def bench_fn() raises capturing:
            for _ in range(1e6):
                q = dispatch_mul[version](q, q2)
                q = dispatch_mul[version](q, q3)
            keep(q)

        var time_us = round(run[func3=bench_fn](max_iters=100).mean(Unit.us), 1)

        print(t"v{version} : {time_us} us")

    comptime for w in [1, 2, 4, 8]:
        print(t"width = {w}")
        bench_throughput[0, w]()
        bench_throughput[1, w]()

        bench_latency[0, w]()
        bench_latency[1, w]()
