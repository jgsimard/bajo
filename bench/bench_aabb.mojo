from std.benchmark import run, Unit, keep
from std.random import random_float64
from std.memory import UnsafePointer
from std.reflection import get_function_name


from bajo.core.mat import Mat33f32
from bajo.core.vec import vmin, vmax, Vec3f32
from bajo.core.quat import Quat
from bajo.core.aabb import AABB
from bajo.core.random import PhiloxRNG


comptime num_elements = 100_000


fn apply_trs_naive_________(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    rot_mat = Mat33f32.from_rotation_scale(rotation, scale)
    txfmed = AABB(translation.copy(), translation.copy())

    for i in range(3):
        for j in range(3):
            val_min: Scalar[DType.float32]
            val_max: Scalar[DType.float32]
            if j == 0:
                val_min = box._min.x()
                val_max = box._max.x()
            elif j == 1:
                val_min = box._min.y()
                val_max = box._max.y()
            else:
                val_min = box._min.z()
                val_max = box._max.z()

            ref col_j = rot_mat[j]
            mat_val: Scalar[DType.float32]
            if i == 0:
                mat_val = col_j.x()
            elif i == 1:
                mat_val = col_j.y()
            else:
                mat_val = col_j.z()

            e = mat_val * val_min
            f = mat_val * val_max

            if e < f:
                txfmed._min[i] += e
                txfmed._max[i] += f
            else:
                txfmed._min[i] += f
                txfmed._max[i] += e
    return txfmed^


fn apply_trs_naive_comptime(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    rot_mat = Mat33f32.from_rotation_scale(rotation, scale)
    txfmed = AABB(translation.copy(), translation.copy())

    comptime for i in range(3):
        comptime for j in range(3):
            val_min: Scalar[DType.float32]
            val_max: Scalar[DType.float32]
            if j == 0:
                val_min = box._min.x()
                val_max = box._max.x()
            elif j == 1:
                val_min = box._min.y()
                val_max = box._max.y()
            else:
                val_min = box._min.z()
                val_max = box._max.z()

            ref col_j = rot_mat[j]
            mat_val: Scalar[DType.float32]
            if i == 0:
                mat_val = col_j.x()
            elif i == 1:
                mat_val = col_j.y()
            else:
                mat_val = col_j.z()

            e = mat_val * val_min
            f = mat_val * val_max

            if e < f:
                txfmed._min[i] += e
                txfmed._max[i] += f
            else:
                txfmed._min[i] += f
                txfmed._max[i] += e
    return txfmed^


fn apply_trs_arvo_v0_______(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    mat = Mat33f32.from_rotation_scale(rotation, scale).transpose()
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


fn apply_trs_arvo_v1_______(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    mat = Mat33f32.from_rotation_scale(rotation, scale).transpose()
    aabb = AABB(translation.copy(), translation.copy())

    comptime for i in range(3):
        ref c = mat[0]
        c_a = c * box._min[i]
        c_b = c * box._max[i]
        aabb._min += vmin(c_a, c_b)
        aabb._max += vmax(c_a, c_b)

    return aabb^


# ----------------------------------------------------------------------
# Benchmark Harness
# ----------------------------------------------------------------------


struct AABBBenchmarkData:
    var boxes: UnsafePointer[AABB, MutAnyOrigin]
    var translations: UnsafePointer[Vec3f32, MutAnyOrigin]
    var rotations: UnsafePointer[Quat, MutAnyOrigin]
    var scales: UnsafePointer[Vec3f32, MutAnyOrigin]
    var dst: UnsafePointer[AABB, MutAnyOrigin]

    fn __init__(out self):
        self.boxes = alloc[AABB](num_elements)
        self.translations = alloc[Vec3f32](num_elements)
        self.rotations = alloc[Quat](num_elements)
        self.scales = alloc[Vec3f32](num_elements)
        self.dst = alloc[AABB](num_elements)
        rng = PhiloxRNG(123, 123)

        for i in range(num_elements):
            self.boxes[i] = AABB(Vec3f32(-1), Vec3f32(1))
            self.translations[i] = rng.next_Vec3f32()
            self.rotations[i] = Quat.from_axis_angle(
                Vec3f32(0, 1, 0),
                rng.next_f32(),
            )

            self.scales[i] = rng.next_Vec3f32()

    fn __del__(deinit self):
        self.boxes.free()
        self.translations.free()
        self.rotations.free()
        self.scales.free()
        self.dst.free()


fn main() raises:
    data = AABBBenchmarkData()
    print("Benchmarking AABB Transform (apply_trs) - Elements:", num_elements)

    fn bench[f: fn(AABB, Vec3f32, Quat, Vec3f32) -> AABB]() capturing raises:
        fn wrapper() raises capturing:
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
        # name = "Naive (Loop)    " if version == 0 else "Naive (comptime)" if version == 1 else "Arvo (Vector)   "
        throughput = round(num_elements / avg_time, 2)
        mops = round(avg_time, 2)
        print(t"{name}| Throughput:{throughput}, Mops/s | Avg: {mops} us")

    bench[apply_trs_naive_________]()
    bench[apply_trs_naive_comptime]()
    bench[apply_trs_arvo_v0_______]()
    bench[apply_trs_arvo_v1_______]()


# Benchmarking AABB Transform (apply_trs) - Elements: 100000
# apply_trs_naive_________| Throughput:55.66, Mops/s | Avg: 1796.74 us
# apply_trs_naive_comptime| Throughput:190.21, Mops/s | Avg: 525.73 us
# apply_trs_arvo_v0_______| Throughput:167.25, Mops/s | Avg: 597.9 us
# apply_trs_arvo_v1_______| Throughput:154.43, Mops/s | Avg: 647.55 us
