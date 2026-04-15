from std.benchmark import run, Unit, keep
from std.random import random_float64
from std.memory import UnsafePointer
from std.reflection import get_function_name


from bajo.core.mat import Mat33f32
from bajo.core.vec import vmin, vmax, Vec3f32
from bajo.core.quat import Quat
from bajo.core.aabb import AABB
from bajo.core.random import Rng


comptime num_elements = 100_000


def apply_trs_naive_________(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    rot_mat = Mat33f32.from_rotation_scale(rotation, scale)
    txfmed = AABB(translation.copy(), translation.copy())

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
    return txfmed^


def apply_trs_naive_comptime(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    rot_mat = Mat33f32.from_rotation_scale(rotation, scale)
    txfmed = AABB(translation.copy(), translation.copy())

    comptime for j in range(3):
        comptime for i in range(3):
            e = rot_mat[i][j] * box._min[j]
            f = rot_mat[i][j] * box._max[j]

            if e < f:
                txfmed._min[i] += e
                txfmed._max[i] += f
            else:
                txfmed._min[i] += f
                txfmed._max[i] += e
    return txfmed^


def apply_trs_arvo_v0_______(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    mat = Mat33f32.from_rotation_scale(rotation, scale)
    new_min = translation.copy()
    new_max = translation.copy()

    # X column
    c0_a = mat[0] * box._min.x()
    c0_b = mat[0] * box._max.x()
    new_min += vmin(c0_a, c0_b)
    new_max += vmax(c0_a, c0_b)

    # Y column
    c1_a = mat[1] * box._min.y()
    c1_b = mat[1] * box._max.y()
    new_min += vmin(c1_a, c1_b)
    new_max += vmax(c1_a, c1_b)

    # Z column
    c2_a = mat[2] * box._min.z()
    c2_b = mat[2] * box._max.z()
    new_min += vmin(c2_a, c2_b)
    new_max += vmax(c2_a, c2_b)

    return AABB(new_min^, new_max^)


def apply_trs_arvo_v1_______(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    mat = Mat33f32.from_rotation_scale(rotation, scale)
    new_min = translation.copy()
    new_max = translation.copy()

    comptime for i in range(3):
        c_a = mat[i] * box._min[i]
        c_b = mat[i] * box._max[i]
        new_min += vmin(c_a, c_b)
        new_max += vmax(c_a, c_b)

    return AABB(new_min^, new_max^)


# ----------------------------------------------------------------------
# Benchmark Harness
# ----------------------------------------------------------------------


struct AABBBenchmarkData:
    var boxes: UnsafePointer[AABB, MutAnyOrigin]
    var translations: UnsafePointer[Vec3f32, MutAnyOrigin]
    var rotations: UnsafePointer[Quat, MutAnyOrigin]
    var scales: UnsafePointer[Vec3f32, MutAnyOrigin]
    var dst: UnsafePointer[AABB, MutAnyOrigin]

    def __init__(out self):
        self.boxes = alloc[AABB](num_elements)
        self.translations = alloc[Vec3f32](num_elements)
        self.rotations = alloc[Quat](num_elements)
        self.scales = alloc[Vec3f32](num_elements)
        self.dst = alloc[AABB](num_elements)
        rng = Rng(123, 123)

        for i in range(num_elements):
            self.boxes[i] = AABB(Vec3f32(-1), Vec3f32(1))
            self.translations[i] = rng.vec3f32()
            self.rotations[i] = Quat.from_axis_angle(
                Vec3f32(0, 1, 0),
                rng.f32(),
            )

            self.scales[i] = rng.vec3f32()

    def __del__(deinit self):
        self.boxes.free()
        self.translations.free()
        self.rotations.free()
        self.scales.free()
        self.dst.free()


def main() raises:
    data = AABBBenchmarkData()
    print("Benchmarking AABB Transform (apply_trs) - Elements:", num_elements)

    def bench[
        f: def(AABB, Vec3f32, Quat, Vec3f32) thin -> AABB
    ]() capturing raises:
        def wrapper() raises capturing:
            for i in range(num_elements):
                data.dst[i] = f(
                    data.boxes[i],
                    data.translations[i],
                    data.rotations[i],
                    data.scales[i],
                )
            keep(data.dst[0]._min.data)

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


# Benchmarking AABB Transform (apply_trs) - Elements: 100000
# apply_trs_naive_________| Throughput: 60.4, Mops/s | Avg: 1654.9 us
# apply_trs_naive_comptime| Throughput:198.0, Mops/s | Avg: 504.96 us # 3.3 x faster !!!
# apply_trs_arvo_v0_______| Throughput:169.8, Mops/s | Avg: 588.8 us
# apply_trs_arvo_v1_______| Throughput:169.3, Mops/s | Avg: 590.83 us
