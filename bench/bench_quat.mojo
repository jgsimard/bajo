from benchmark import run, Unit, keep
from random import random_float64
from memory import UnsafePointer
from testing import TestSuite, assert_equal, assert_almost_equal

from bajo.bmath import Quat, deg_to_radians, Vec3f

comptime f32 = DType.float32
comptime num_elements = 100000


struct BenchmarkData(Copyable):
    var src_a: UnsafePointer[Quat, MutAnyOrigin]
    var src_b: UnsafePointer[Quat, MutAnyOrigin]
    var dst: UnsafePointer[Quat, MutAnyOrigin]

    fn __init__(out self):
        self.src_a = alloc[Quat](num_elements)
        self.src_b = alloc[Quat](num_elements)
        self.dst = alloc[Quat](num_elements)

        for i in range(num_elements):
            self.src_a[i] = Quat.angle_axis(
                Float32(random_float64()), Vec3f(1, 0, 0)
            )
            self.src_b[i] = Quat.angle_axis(
                Float32(random_float64()), Vec3f(0, 1, 0)
            )

    fn __del__(deinit self):
        self.src_a.free()
        self.src_b.free()
        self.dst.free()


fn main() raises:

    @parameter
    fn bench_throughput[version: Int]() raises:
        var data = BenchmarkData()

        fn wrapper() raises capturing:
            for i in range(num_elements):
                data.dst[i] = data.src_a[i].__mul__[version](data.src_b[i])
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
        var q2 = Quat.angle_axis(
            deg_to_radians(Scalar[DType.float32](45)), Vec3f(0, 1, 0)
        )

        var q3 = Quat.angle_axis(
            deg_to_radians(Scalar[DType.float32](45)), Vec3f(1, 0, 0)
        )
        var a = q2.__mul__[version](q3)
        var b = Quat(0.853553, 0.353553, 0.353553, -0.146447)
        assert_almost_equal(a.data, b.data, atol=1e-6)

        var q = a

        fn bench_fn() raises capturing:
            for _ in range(1e6):
                q = q.__mul__[version](q2)
                q = q.__mul__[version](q3)
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