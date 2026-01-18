from benchmark import run, Unit, keep
from random import random_float64
from memory import UnsafePointer
from testing import TestSuite, assert_equal, assert_almost_equal
from math import fma

from bajo.bmath import Quat, degrees_to_radians, Vec3f

comptime f32 = DType.float32
comptime num_elements = 100000


fn quat_mul_1(q1: Quat, q2: Quat) -> Quat:
    var w = (
        q1.w() * q2.w() - q1.x() * q2.x() - q1.y() * q2.y() - q1.z() * q2.z()
    )
    var x = (
        q1.w() * q2.x() + q1.x() * q2.w() + q1.y() * q2.z() - q1.z() * q2.y()
    )
    var y = (
        q1.w() * q2.y() - q1.x() * q2.z() + q1.y() * q2.w() + q1.z() * q2.x()
    )
    var z = (
        q1.w() * q2.z() + q1.x() * q2.y() - q1.y() * q2.x() + q1.z() * q2.w()
    )
    return Quat(w, x, y, z)


fn quat_mul_2(q1: Quat, q2: Quat) -> Quat:
    # Version 2 : more SIMD: 3 shuffles, 4 mul, 12 add
    var t0 = q1.data[0] * q2.data
    var t1 = q1.data[1] * q2.data.shuffle[1, 0, 3, 2]()
    var t2 = q1.data[2] * q2.data.shuffle[2, 3, 0, 1]()
    var t3 = q1.data[3] * q2.data.shuffle[3, 2, 1, 0]()

    # Apply the standard Hamilton sign patterns
    var w = t0[0] - t1[0] - t2[0] - t3[0]
    var x = t0[1] + t1[1] + t2[1] - t3[1]
    var y = t0[2] - t1[2] + t2[2] + t3[2]
    var z = t0[3] + t1[3] - t2[3] + t3[3]

    return Quat(w, x, y, z)


fn quat_mul_3(q1: Quat, q2: Quat) -> Quat:
    # Version 3: SIMD FMA : 3 shuffles, 4 mul, 3 fma
    comptime s1 = SIMD[DType.float32, 4](-1.0, 1.0, -1.0, 1.0)
    comptime s2 = SIMD[DType.float32, 4](-1.0, 1.0, 1.0, -1.0)
    comptime s3 = SIMD[DType.float32, 4](-1.0, -1.0, 1.0, 1.0)

    var res = q1.data[0] * q2.data
    res = fma(q1.data[1] * s1, q2.data.shuffle[1, 0, 3, 2](), res)
    res = fma(q1.data[2] * s2, q2.data.shuffle[2, 3, 0, 1](), res)
    res = fma(q1.data[3] * s3, q2.data.shuffle[3, 2, 1, 0](), res)
    return Quat(res)


struct BenchmarkData(Copyable):
    var src_a: List[Quat]
    var src_b: List[Quat]
    var dst: List[Quat]

    fn __init__(out self):
        self.src_a = List[Quat](length=num_elements, fill=Quat.identity())
        self.src_b = List[Quat](length=num_elements, fill=Quat.identity())
        self.dst = List[Quat](length=num_elements, fill=Quat.identity())

        for i in range(num_elements):
            self.src_a[i] = Quat.angle_axis(
                Float32(random_float64()), Vec3f(1, 0, 0)
            )
            self.src_b[i] = Quat.angle_axis(
                Float32(random_float64()), Vec3f(0, 1, 0)
            )


@always_inline
fn dispatch_mul[version: Int](q1: Quat, q2: Quat) -> Quat:
    @parameter
    if version == 1:
        return quat_mul_1(q1, q2)
    elif version == 2:
        return quat_mul_2(q1, q2)
    elif version == 3:
        return quat_mul_3(q1, q2)
    else:
        return q1 * q2  # Quat.__mul__


fn main() raises:
    @parameter
    fn bench_throughput[version: Int]() raises:
        var data = BenchmarkData()

        fn wrapper() raises capturing:
            for i in range(num_elements):
                data.dst[i] = dispatch_mul[version](
                    data.src_a[i], (data.src_b[i])
                )
            keep(data.dst[0].data)

        var report = run[func3=wrapper](max_iters=1000)
        var avg_time_us = report.mean(Unit.us)
        var mops = num_elements / avg_time_us

        print(
            "Throughput:",
            round(mops, 2),
            "Mops/s | Avg Time:",
            round(avg_time_us, 2),
            "us",
        )

    bench_throughput[1]()
    bench_throughput[2]()
    bench_throughput[3]()
    bench_throughput[4]()

    @parameter
    fn bench_latency[version: Int]() raises:
        var angle = degrees_to_radians(Scalar[DType.float32](45))
        var q2 = Quat.angle_axis(angle, Vec3f(0, 1, 0))
        var q3 = Quat.angle_axis(angle, Vec3f(1, 0, 0))
        var a = dispatch_mul[version](q2, q3)
        var b = Quat(0.853553, 0.353553, 0.353553, -0.146447)
        assert_almost_equal(a.data, b.data, atol=1e-6)

        var q = a

        fn bench_fn() raises capturing:
            for _ in range(1e6):
                q = dispatch_mul[version](q, q2)
                q = dispatch_mul[version](q, q3)
            keep(q)

        var time_us = round(run[func3=bench_fn](max_iters=100).mean(Unit.us), 1)

        print("v{} : {} us".format(version, time_us))

    bench_latency[1]()
    bench_latency[2]()
    bench_latency[3]()
    bench_latency[4]()

    # Throughput: 1176.84 Mops/s | Avg Time: 84.97 us
    # Throughput: 1380.21 Mops/s | Avg Time: 72.45 us
    # Throughput: 1519.97 Mops/s | Avg Time: 65.79 us
    # Throughput: 1514.84 Mops/s | Avg Time: 66.01 us
    # v1 : 6668.1 us
    # v2 : 6079.7 us
    # v3 : 6342.2 us
    # v4 : 6061.5 us
