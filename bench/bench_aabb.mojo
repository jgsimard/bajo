from benchmark import run, Unit, keep
from random import random_float64
from memory import UnsafePointer


from bajo.core.mat import Mat33f32
from bajo.core.vec import vmin, vmax, Vec3f32
from bajo.core.quat import Quat
from bajo.core.aabb import AABB
from bajo.core.random import PhiloxRNG


comptime num_elements = 100_000


fn apply_trs_naive(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    rot_mat = Mat33f32.from_rotation_scale(rotation, scale)
    txfmed = AABB(translation.copy(), translation.copy())

    for i in range(3):
        for j in range(3):
            val_min: Scalar[DType.float32]
            val_max: Scalar[DType.float32]
            if j == 0:
                val_min = box.min.x()
                val_max = box.max.x()
            elif j == 1:
                val_min = box.min.y()
                val_max = box.max.y()
            else:
                val_min = box.min.z()
                val_max = box.max.z()

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
                txfmed.min[i] += e
                txfmed.max[i] += f
            else:
                txfmed.min[i] += f
                txfmed.max[i] += e
    return txfmed^


fn apply_trs_arvo(
    box: AABB, translation: Vec3f32, rotation: Quat, scale: Vec3f32
) -> AABB:
    mat = Mat33f32.from_rotation_scale(rotation, scale).transpose()
    new_min = translation.copy()
    new_max = translation.copy()

    # X column
    c0_a = mat[0] * box.min.x()
    c0_b = mat[0] * box.max.x()
    new_min += vmin(c0_a, c0_b)
    new_max += vmax(c0_a, c0_b)

    # Y column
    c1_a = mat[1] * box.min.y()
    c1_b = mat[1] * box.max.y()
    new_min += vmin(c1_a, c1_b)
    new_max += vmax(c1_a, c1_b)

    # Z column
    c2_a = mat[2] * box.min.z()
    c2_b = mat[2] * box.max.z()
    new_min += vmin(c2_a, c2_b)
    new_max += vmax(c2_a, c2_b)

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

    fn __init__(out self):
        self.boxes = alloc[AABB](num_elements)
        self.translations = alloc[Vec3f32](num_elements)
        self.rotations = alloc[Quat](num_elements)
        self.scales = alloc[Vec3f32](num_elements)
        self.dst = alloc[AABB](num_elements)
        rng = PhiloxRNG(123, 123)

        for i in range(num_elements):
            self.boxes[i] = AABB(Vec3f32(-1), Vec3f32(1))
            self.translations[i] = Vec3f32(
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
            )
            self.rotations[i] = Quat.from_axis_angle(
                Vec3f32(0, 1, 0),
                rng.next_f32(),
            )

            self.scales[i] = Vec3f32(
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
            )

    fn __del__(deinit self):
        self.boxes.free()
        self.translations.free()
        self.rotations.free()
        self.scales.free()
        self.dst.free()


fn main() raises:
    data = AABBBenchmarkData()
    print("Benchmarking AABB Transform (apply_trs) - Elements:", num_elements)

    @parameter
    fn bench[version: Int]() raises:
        fn wrapper() raises capturing:
            f = apply_trs_naive if version == 0 else apply_trs_arvo
            for i in range(num_elements):
                data.dst[i] = f(
                    data.boxes[i],
                    data.translations[i],
                    data.rotations[i],
                    data.scales[i],
                )
            keep(data.dst[0].min.data)

        report = run[wrapper](max_iters=200)
        avg_time = report.mean(Unit.us)
        name = "Naive (Loop)  " if version == 0 else "Arvo (Vector) "
        throughput = round(num_elements / avg_time, 2)
        mops = round(avg_time, 2)
        print(t"{name}| Throughput:{throughput}, Mops/s | Avg: {mops} us")

    bench[0]()
    bench[1]()


# Benchmarking AABB Transform (apply_trs) - Elements: 100000
# Naive (Loop)  | Throughput:54.88, Mops/s | Avg: 1822.12 us
# Arvo (Vector) | Throughput:156.35, Mops/s | Avg: 639.6 us
