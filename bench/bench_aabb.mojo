from benchmark import run, Unit, keep
from random import random_float64
from memory import UnsafePointer

from bajo.bmath import AABB, Vec3f, Quat, Diag3f, Mat3f, Vector

comptime num_elements = 100_000


fn apply_trs_naive(
    box: AABB, translation: Vec3f, rotation: Quat, scale: Diag3f
) -> AABB:
    var rot_mat = Mat3f.from_rs(rotation, scale)
    var txfmed = AABB(translation, translation)

    for i in range(3):
        for j in range(3):
            var val_min: Scalar[DType.float32]
            var val_max: Scalar[DType.float32]
            if j == 0:
                val_min = box.pMin.x()
                val_max = box.pMax.x()
            elif j == 1:
                val_min = box.pMin.y()
                val_max = box.pMax.y()
            else:
                val_min = box.pMin.z()
                val_max = box.pMax.z()

            var col_j = rot_mat[j]
            var mat_val: Scalar[DType.float32]
            if i == 0:
                mat_val = col_j.x()
            elif i == 1:
                mat_val = col_j.y()
            else:
                mat_val = col_j.z()

            var e = mat_val * val_min
            var f = mat_val * val_max

            if e < f:
                txfmed.pMin[i] += e
                txfmed.pMax[i] += f
            else:
                txfmed.pMin[i] += f
                txfmed.pMax[i] += e
    return txfmed


fn apply_trs_arvo(
    box: AABB, translation: Vec3f, rotation: Quat, scale: Diag3f
) -> AABB:
    var mat = Mat3f.from_rs(rotation, scale)
    var new_min = translation
    var new_max = translation

    # X column
    var c0_a = mat.c0 * box.pMin.x()
    var c0_b = mat.c0 * box.pMax.x()
    new_min += Vector.min(c0_a, c0_b)
    new_max += Vector.max(c0_a, c0_b)

    # Y column
    var c1_a = mat.c1 * box.pMin.y()
    var c1_b = mat.c1 * box.pMax.y()
    new_min += Vector.min(c1_a, c1_b)
    new_max += Vector.max(c1_a, c1_b)

    # Z column
    var c2_a = mat.c2 * box.pMin.z()
    var c2_b = mat.c2 * box.pMax.z()
    new_min += Vector.min(c2_a, c2_b)
    new_max += Vector.max(c2_a, c2_b)

    return AABB(new_min, new_max)


# ----------------------------------------------------------------------
# Benchmark Harness
# ----------------------------------------------------------------------


struct AABBBenchmarkData:
    var boxes: UnsafePointer[AABB, MutAnyOrigin]
    var translations: UnsafePointer[Vec3f, MutAnyOrigin]
    var rotations: UnsafePointer[Quat, MutAnyOrigin]
    var scales: UnsafePointer[Diag3f, MutAnyOrigin]
    var dst: UnsafePointer[AABB, MutAnyOrigin]

    fn __init__(out self):
        self.boxes = alloc[AABB](num_elements)
        self.translations = alloc[Vec3f](num_elements)
        self.rotations = alloc[Quat](num_elements)
        self.scales = alloc[Diag3f](num_elements)
        self.dst = alloc[AABB](num_elements)

        for i in range(num_elements):
            self.boxes[i] = AABB(Vec3f(-1), Vec3f(1))
            self.translations[i] = Vec3f(
                Float32(random_float64()),
                Float32(random_float64()),
                Float32(random_float64()),
            )
            self.rotations[i] = Quat.angle_axis(
                Float32(random_float64()), Vec3f(0, 1, 0)
            )
            self.scales[i] = Diag3f.uniform(Float32(random_float64() * 2.0))

    fn __del__(deinit self):
        self.boxes.free()
        self.translations.free()
        self.rotations.free()
        self.scales.free()
        self.dst.free()


fn main() raises:
    var data = AABBBenchmarkData()
    print("Benchmarking AABB Transform (apply_trs) - Elements:", num_elements)

    @parameter
    fn bench[version: Int]() raises:
        fn wrapper() raises capturing:
            var f = apply_trs_naive if version == 0 else apply_trs_arvo
            for i in range(num_elements):
                data.dst[i] = f(
                    data.boxes[i],
                    data.translations[i],
                    data.rotations[i],
                    data.scales[i],
                )
            keep(data.dst[0].pMin.data)

        var report = run[wrapper](max_iters=200)
        var avg_time = report.mean(Unit.us)
        var name = "Naive (Loop)  " if version == 0 else "Arvo (Vector) "
        print(
            "{}| Throughput:".format(name),
            round(num_elements / avg_time, 2),
            "Mops/s | Avg:",
            round(avg_time, 2),
            "us",
        )

    bench[0]()
    bench[1]()
