from std.benchmark import run, Unit, keep
from std.random import random_float64
from std.reflection import get_function_name


from bajo.core import (
    Affine3f32,
    Affine3,
    vmin,
    vmax,
    Vec3f32,
    Vec3,
    Point3,
    Point3f32,
    Quat,
    Quaternion,
    AABB,
    AxisAlignedBoundingBox,
    Mat33f32,
    Frame,
    GeoKind,
)
from bajo.core.random import Rng


comptime num_elements = 100_000

comptime BENCH_FRAME = Frame.WORLD
comptime BenchAABB = AABB[BENCH_FRAME]
comptime BenchVec3f32 = Vec3f32[BENCH_FRAME]
comptime BenchPoint3f32 = Point3f32[BENCH_FRAME]
comptime BenchQuat = Quat[BENCH_FRAME]
comptime BenchMat33f32 = Mat33f32[BENCH_FRAME]


def apply_trs_naive_________(
    box: BenchAABB,
    translation: BenchVec3f32,
    rotation: BenchQuat,
    scale: BenchVec3f32,
) -> BenchAABB:
    rot_mat = BenchMat33f32.from_rotation_scale(rotation, scale)
    txfmed = BenchAABB(
        translation.copy().unsafe_convert_kind[GeoKind.POINT](),
        translation.copy().unsafe_convert_kind[GeoKind.POINT](),
    )

    for i in range(3):
        for j in range(3):
            e = rot_mat[i][j] * box._min[j]
            f = rot_mat[i][j] * box._max[j]

            if e < f:
                txfmed._min[i] += e
                txfmed._max[i] += f
            else:
                txfmed._min[i] += f
                txfmed._max[i] += e
    return txfmed


def apply_trs_naive_comptime(
    box: BenchAABB,
    translation: BenchVec3f32,
    rotation: BenchQuat,
    scale: BenchVec3f32,
) -> BenchAABB:
    rot_mat = BenchMat33f32.from_rotation_scale(rotation, scale)
    txfmed = BenchAABB(
        translation.copy().unsafe_convert_kind[GeoKind.POINT](),
        translation.copy().unsafe_convert_kind[GeoKind.POINT](),
    )

    comptime for j in range(3):
        comptime for i in range(3):
            e = rot_mat[i][j] * box._min[j]
            f = rot_mat[i][j] * box._max[j]

            if e[0] < f[0]:
                txfmed._min.add_axis[i](e)
                txfmed._max.add_axis[i](f)
            else:
                txfmed._min.add_axis[i](f)
                txfmed._max.add_axis[i](e)
    return txfmed


def apply_trs_arvo_v0_______(
    box: BenchAABB,
    translation: BenchVec3f32,
    rotation: BenchQuat,
    scale: BenchVec3f32,
) -> BenchAABB:
    mat = BenchMat33f32.from_rotation_scale(rotation, scale)
    new_min = translation.copy()
    new_max = translation.copy()

    # X column
    c0_a = mat.col[0]() * box._min.x
    c0_b = mat.col[0]() * box._max.x
    new_min += vmin(c0_a, c0_b)
    new_max += vmax(c0_a, c0_b)

    # Y column
    c1_a = mat.col[1]() * box._min.y
    c1_b = mat.col[1]() * box._max.y
    new_min += vmin(c1_a, c1_b)
    new_max += vmax(c1_a, c1_b)

    # Z column
    c2_a = mat.col[2]() * box._min.z
    c2_b = mat.col[2]() * box._max.z
    new_min += vmin(c2_a, c2_b)
    new_max += vmax(c2_a, c2_b)

    return BenchAABB(
        new_min.copy().unsafe_convert_kind[GeoKind.POINT](),
        new_max.copy().unsafe_convert_kind[GeoKind.POINT](),
    )


def apply_trs_arvo_v1_______(
    box: BenchAABB,
    translation: BenchVec3f32,
    rotation: BenchQuat,
    scale: BenchVec3f32,
) -> BenchAABB:
    mat = BenchMat33f32.from_rotation_scale(rotation, scale)
    new_min = translation.copy()
    new_max = translation.copy()

    comptime for i in range(3):
        c_a = mat.col[i]() * box._min[i]
        c_b = mat.col[i]() * box._max[i]
        new_min += vmin(c_a, c_b)
        new_max += vmax(c_a, c_b)

    return BenchAABB(
        new_min.copy().unsafe_convert_kind[GeoKind.POINT](),
        new_max.copy().unsafe_convert_kind[GeoKind.POINT](),
    )


def apply_trs_affine3_v0_width[
    width: SIMDSize
](
    box: AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width],
    translation: Vec3[DType.float32, BENCH_FRAME, width],
    rotation: Quaternion[DType.float32, BENCH_FRAME, width],
    scale: Vec3[DType.float32, BENCH_FRAME, width],
) -> AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width]:
    transform = Affine3[
        DType.float32,
        BENCH_FRAME,
        BENCH_FRAME,
        width,
    ].from_rotation_scale(rotation, scale)

    var new_min = translation.copy()
    var new_max = translation.copy()

    # X column
    var c0 = Vec3[DType.float32, BENCH_FRAME, width](
        transform.m00, transform.m10, transform.m20
    )
    var c0_a = c0 * box._min.x
    var c0_b = c0 * box._max.x
    new_min += vmin(c0_a, c0_b)
    new_max += vmax(c0_a, c0_b)

    # Y column
    var c1 = Vec3[DType.float32, BENCH_FRAME, width](
        transform.m01, transform.m11, transform.m21
    )
    var c1_a = c1 * box._min.y
    var c1_b = c1 * box._max.y
    new_min += vmin(c1_a, c1_b)
    new_max += vmax(c1_a, c1_b)

    # Z column
    var c2 = Vec3[DType.float32, BENCH_FRAME, width](
        transform.m02, transform.m12, transform.m22
    )
    var c2_a = c2 * box._min.z
    var c2_b = c2 * box._max.z
    new_min += vmin(c2_a, c2_b)
    new_max += vmax(c2_a, c2_b)

    return AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width](
        new_min.copy().unsafe_convert_kind[GeoKind.POINT](),
        new_max.copy().unsafe_convert_kind[GeoKind.POINT](),
    )


def apply_trs_affine3_v1_width[
    width: SIMDSize
](
    box: AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width],
    translation: Vec3[DType.float32, BENCH_FRAME, width],
    rotation: Quaternion[DType.float32, BENCH_FRAME, width],
    scale: Vec3[DType.float32, BENCH_FRAME, width],
) -> AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width]:
    transform = Affine3[
        DType.float32,
        BENCH_FRAME,
        BENCH_FRAME,
        width,
    ].from_rotation_scale(rotation, scale)

    var new_min = translation.copy()
    var new_max = translation.copy()

    def _add_transformed_axis_width[
        i: Int,
    ](
        m: SIMD[DType.float32, width],
        lo: SIMD[DType.float32, width],
        hi: SIMD[DType.float32, width],
    ) capturing:
        var e = m * lo
        var f = m * hi

        comptime if width == 1:
            if e[0] < f[0]:
                new_min.add_axis[i](e)
                new_max.add_axis[i](f)
            else:
                new_min.add_axis[i](f)
                new_max.add_axis[i](e)
        else:
            var mask = e.lt(f)
            new_min.add_axis[i](mask.select(e, f))
            new_max.add_axis[i](mask.select(f, e))

    # X column
    _add_transformed_axis_width[0](transform.m00, box._min.x, box._max.x)
    _add_transformed_axis_width[1](transform.m10, box._min.x, box._max.x)
    _add_transformed_axis_width[2](transform.m20, box._min.x, box._max.x)

    # Y column
    _add_transformed_axis_width[0](transform.m01, box._min.y, box._max.y)
    _add_transformed_axis_width[1](transform.m11, box._min.y, box._max.y)
    _add_transformed_axis_width[2](transform.m21, box._min.y, box._max.y)

    # Z column
    _add_transformed_axis_width[0](transform.m02, box._min.z, box._max.z)
    _add_transformed_axis_width[1](transform.m12, box._min.z, box._max.z)
    _add_transformed_axis_width[2](transform.m22, box._min.z, box._max.z)

    return AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width](
        new_min.copy().unsafe_convert_kind[GeoKind.POINT](),
        new_max.copy().unsafe_convert_kind[GeoKind.POINT](),
    )


def dispatch_affine3_width[
    version: Int,
    width: SIMDSize,
](
    box: AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width],
    translation: Vec3[DType.float32, BENCH_FRAME, width],
    rotation: Quaternion[DType.float32, BENCH_FRAME, width],
    scale: Vec3[DType.float32, BENCH_FRAME, width],
) -> AxisAlignedBoundingBox[DType.float32, BENCH_FRAME, width]:
    comptime if version == 0:
        return apply_trs_affine3_v0_width[width](
            box, translation, rotation, scale
        )
    else:
        return apply_trs_affine3_v1_width[width](
            box, translation, rotation, scale
        )


struct Affine3WidthBenchmarkData[width: SIMDSize]:
    comptime aabb = AxisAlignedBoundingBox[
        DType.float32,
        BENCH_FRAME,
        Self.width,
    ]

    var boxes: List[Self.aabb]
    var translations: List[Vec3[DType.float32, BENCH_FRAME, Self.width]]
    var rotations: List[Quaternion[DType.float32, BENCH_FRAME, Self.width]]
    var scales: List[Vec3[DType.float32, BENCH_FRAME, Self.width]]
    var dst: List[Self.aabb]

    def __init__(out self):
        comptime packet_count = num_elements / Self.width

        self.boxes = []
        self.translations = []
        self.rotations = []
        self.scales = []

        rng = Rng(123, 123)

        for _ in range(packet_count):
            self.boxes.append(
                Self.aabb(
                    Point3[DType.float32, BENCH_FRAME, Self.width](-1),
                    Point3[DType.float32, BENCH_FRAME, Self.width](1),
                )
            )

            self.translations.append(
                Vec3[DType.float32, BENCH_FRAME, Self.width](
                    rng.f32(),
                    rng.f32(),
                    rng.f32(),
                )
            )

            self.rotations.append(
                Quaternion[
                    DType.float32,
                    BENCH_FRAME,
                    Self.width,
                ].from_axis_angle(
                    Vec3[DType.float32, BENCH_FRAME, Self.width](0, 1, 0),
                    rng.f32(),
                )
            )

            self.scales.append(
                Vec3[DType.float32, BENCH_FRAME, Self.width](
                    rng.f32(),
                    rng.f32(),
                    rng.f32(),
                )
            )

        self.dst = self.boxes.copy()


def bench_affine3_width[
    version: Int,
    width: SIMDSize,
]() raises:
    comptime packet_count = num_elements / width
    data = Affine3WidthBenchmarkData[width]()

    def wrapper() raises capturing:
        for i in range(packet_count):
            data.dst.unsafe_ptr()[i] = dispatch_affine3_width[version, width](
                data.boxes.unsafe_ptr()[i],
                data.translations.unsafe_ptr()[i],
                data.rotations.unsafe_ptr()[i],
                data.scales.unsafe_ptr()[i],
            )

        keep(data.dst[0]._min)

    report = run[func3=wrapper](max_iters=200)
    avg_time_us = round(report.mean(Unit.us), 2)
    throughput = round(num_elements / avg_time_us, 1)

    comptime if version == 0:
        name = String("apply_trs_affine3_v0_width")
    else:
        name = String("apply_trs_affine3_v1_width")

    print(t"{name}| Throughput:{throughput}, Mops/s | Avg: {avg_time_us} us")


# ----------------------------------------------------------------------
# Benchmark Harness
# ----------------------------------------------------------------------


struct AABBBenchmarkData:
    var boxes: List[BenchAABB]
    var translations: List[BenchVec3f32]
    var rotations: List[BenchQuat]
    var scales: List[BenchVec3f32]
    var dst: List[BenchAABB]

    def __init__(out self):
        self.boxes = []
        self.translations = []
        self.rotations = []
        self.scales = []
        rng = Rng(123, 123)

        for _ in range(num_elements):
            self.boxes.append(BenchAABB(BenchPoint3f32(-1), BenchPoint3f32(1)))
            self.translations.append(
                BenchVec3f32(rng.f32(), rng.f32(), rng.f32())
            )
            self.rotations.append(
                BenchQuat.from_axis_angle(
                    BenchVec3f32(0, 1, 0),
                    rng.f32(),
                )
            )

            self.scales.append(BenchVec3f32(rng.f32(), rng.f32(), rng.f32()))

        self.dst = self.boxes.copy()


def main() raises:
    data = AABBBenchmarkData()
    print("Benchmarking AABB Transform (apply_trs) - Elements:", num_elements)

    def bench[
        f: def(
            BenchAABB,
            BenchVec3f32,
            BenchQuat,
            BenchVec3f32,
        ) thin -> BenchAABB
    ]() capturing raises:
        def wrapper() raises capturing:
            for i in range(num_elements):
                data.dst.unsafe_ptr()[i] = f(
                    data.boxes.unsafe_ptr()[i],
                    data.translations.unsafe_ptr()[i],
                    data.rotations.unsafe_ptr()[i],
                    data.scales.unsafe_ptr()[i],
                )
            keep(data.dst[0]._min)

        report = run[wrapper](max_iters=200)
        avg_time = report.mean(Unit.us)
        name = get_function_name[f]()
        throughput = round(num_elements / avg_time, 1)
        mops = round(avg_time, 2)
        print(t"{name}| Throughput:{throughput}, Mops/s | Avg: {mops} us")

    bench[apply_trs_naive_________]()
    bench[apply_trs_naive_comptime]()
    bench[apply_trs_arvo_v0_______]()
    bench[apply_trs_arvo_v1_______]()

    print("\nBenchmarking Affine3 packet widths")
    comptime for w in [1, 2, 4, 8]:
        print(t"width = {w}")
        bench_affine3_width[0, w]()
        bench_affine3_width[1, w]()


# Benchmarking AABB Transform (apply_trs) - Elements: 100000
# apply_trs_naive_________| Throughput:34.6, Mops/s | Avg: 2887.25 us
# apply_trs_naive_comptime| Throughput:199.4, Mops/s | Avg: 501.58 us
# apply_trs_arvo_v0_______| Throughput:253.9, Mops/s | Avg: 393.86 us
# apply_trs_arvo_v1_______| Throughput:254.0, Mops/s | Avg: 393.73 us

# Benchmarking Affine3 packet widths
# width = 1
# apply_trs_affine3_v0_width| Throughput:252.7, Mops/s | Avg: 395.75 us
# apply_trs_affine3_v1_width| Throughput:201.1, Mops/s | Avg: 497.37 us
# width = 2
# apply_trs_affine3_v0_width| Throughput:338.8, Mops/s | Avg: 295.16 us
# apply_trs_affine3_v1_width| Throughput:335.5, Mops/s | Avg: 298.09 us
# width = 4
# apply_trs_affine3_v0_width| Throughput:670.2, Mops/s | Avg: 149.2 us
# apply_trs_affine3_v1_width| Throughput:668.9, Mops/s | Avg: 149.51 us
# width = 8
# apply_trs_affine3_v0_width| Throughput:1268.1, Mops/s | Avg: 78.86 us
# apply_trs_affine3_v1_width| Throughput:1262.0, Mops/s | Avg: 79.24 us
