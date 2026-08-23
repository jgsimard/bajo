from std.benchmark import run, Unit, keep
from std.random import random_float64
from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.math import fma

from bajo.core.quat import Quaternion
from bajo.core.utils import degrees_to_radians
from bajo.core.random import Rng
from bajo.core.vec import Vec3
from bajo.core.frame import Frame

comptime dtype = DType.float32
comptime num_elements = 100000


def quat_mul_0[
    frame: Frame, width: SIMDLength
](
    q1: Quaternion[dtype, frame, width], q2: Quaternion[dtype, frame, width]
) -> Quaternion[dtype, frame, width]:
    var x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    var y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    var z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    var w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    return Quaternion[dtype, frame, width](x, y, z, w)


struct BenchmarkData[width: SIMDLength](Copyable):
    var src_a: List[Quaternion[dtype, .WORLD, Self.width]]
    var src_b: List[Quaternion[dtype, .WORLD, Self.width]]
    var dst: List[Quaternion[dtype, .WORLD, Self.width]]

    def __init__(out self):
        var rng = Rng(123, 123)

        self.src_a = [
            Quaternion[dtype, .WORLD, Self.width].from_axis_angle(
                Vec3[dtype, .WORLD, Self.width](1, 0, 0), rng.f32()
            )
            for _ in range(num_elements / Self.width)
        ]
        self.src_b = [
            Quaternion[dtype, .WORLD, Self.width].from_axis_angle(
                Vec3[dtype, .WORLD, Self.width](0, 1, 0), rng.f32()
            )
            for _ in range(num_elements / Self.width)
        ]
        self.dst = [
            Quaternion[dtype, .WORLD, Self.width].identity()
            for _ in range(num_elements / Self.width)
        ]


def dispatch_mul[
    version: Int, frame: Frame, width: SIMDLength
](
    q1: Quaternion[dtype, frame, width], q2: Quaternion[dtype, frame, width]
) -> Quaternion[dtype, frame, width]:
    comptime if version == 0:
        return quat_mul_0(q1, q2)
    else:
        return q1 * q2  # Quat.__mul__


def main() raises:
    def bench_throughput[version: Int, width: SIMDLength]() raises:
        var data = BenchmarkData[width]()

        # bounds checking makes this benchmars 3X slower !
        def wrapper() raises {mut data}:
            for i in range(num_elements / width):
                data.dst.unsafe_ptr()[unsafe_offset=i] = dispatch_mul[version](
                    data.src_a.unsafe_ptr()[unsafe_offset=i],
                    data.src_b.unsafe_ptr()[unsafe_offset=i],
                )
            keep(data.dst[0].z)

        var report = run(wrapper, max_iters=1000)
        var avg_time_us = round(report.mean(Unit.us), 2)
        var mops = round(num_elements / avg_time_us, 2)

        print(t"Throughput: {mops} Mops/s | Avg Time: {avg_time_us} us")

    comptime for w in [1, 2, 4, 8]:
        print(t"width = {w}")
        bench_throughput[0, w]()
        bench_throughput[1, w]()
