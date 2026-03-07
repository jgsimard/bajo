from std.random import randint, seed, random_si64, random_float64
from std.time import perf_counter_ns

from bajo.core.random import PhiloxRNG
from bajo.core.vec import Vec3f32, vclamp, length
from bajo.core.aabb import AABB
from bajo.core.primitives import Trianglef32

from bajo.core.bvh.base import BVH


comptime TRIANGLE_COUNT = 10_000_000
comptime GRID_SIZE = 1000
comptime cell_size = Float32(10.0) / Float32(GRID_SIZE)


fn main():
    print("hello bench_bvh")
    print("building test data....")

    seed()

    # rng = PhiloxRNG(123, 123)

    bounds = AABB(Vec3f32(0), Vec3f32(0))
    bounds.clear()

    triangles = List[Trianglef32](capacity=TRIANGLE_COUNT)

    for _ in range(TRIANGLE_COUNT):
        x = random_si64(0, GRID_SIZE - 1)
        y = random_si64(0, GRID_SIZE - 1)
        z = random_si64(0, GRID_SIZE - 1)

        base_x = Float32(x) * cell_size
        base_y = Float32(y) * cell_size
        base_z = Float32(z) * cell_size

        offset_x = Float32(
            random_float64(Float64(0.1 * cell_size), Float64(0.9 * cell_size))
        )
        offset_y = Float32(
            random_float64(Float64(0.1 * cell_size), Float64(0.9 * cell_size))
        )
        offset_z = Float32(
            random_float64(Float64(0.1 * cell_size), Float64(0.9 * cell_size))
        )

        edge_dist_x_1 = Float32(
            random_float64(Float64(-0.4 * cell_size), Float64(0.4 * cell_size))
        )
        edge_dist_y_1 = Float32(
            random_float64(Float64(-0.4 * cell_size), Float64(0.4 * cell_size))
        )
        edge_dist_z_1 = Float32(
            random_float64(Float64(-0.4 * cell_size), Float64(0.4 * cell_size))
        )

        edge_dist_x_2 = Float32(
            random_float64(Float64(-0.4 * cell_size), Float64(0.4 * cell_size))
        )
        edge_dist_y_2 = Float32(
            random_float64(Float64(-0.4 * cell_size), Float64(0.4 * cell_size))
        )
        edge_dist_z_2 = Float32(
            random_float64(Float64(-0.4 * cell_size), Float64(0.4 * cell_size))
        )

        v0 = Vec3f32(base_x + offset_x, base_y + offset_y, base_z + offset_z)
        v1 = v0 + Vec3f32(edge_dist_x_1, edge_dist_y_1, edge_dist_z_1)
        v2 = v0 + Vec3f32(edge_dist_x_2, edge_dist_y_2, edge_dist_z_2)

        # Ensure v1 and v2 are not too close to v0 (avoid degeneracy)
        if length(v1 - v0) < 0.1 * cell_size:
            v1[0] += 0.2 * cell_size
        if length(v2 - v0) < 0.1 * cell_size:
            v2[1] += 0.2 * cell_size

        triangles.append(Trianglef32(v0.copy(), v1.copy(), v2.copy()))

        prim_bounds = AABB(v0, v1, v2)
        bounds.grow(prim_bounds)

    print("building test data....DONE")
    print("building BVH....")
    t0 = perf_counter_ns()
    _bvh = BVH(triangles^)
    t1 = perf_counter_ns()
    delta_t = t1 - t0
    delta_t_ms = Float32(delta_t) / 1_000_000.0
    print(t"building BVH....DONE in {delta_t_ms} ms")

    # basic : single threaded on CPU
    #  1_000_000 =  1332.778 ms
    # 10_000_000 = 19715.723 ms

    # hploc : on GPU (expected)
    # 10_000_000 = ~20 ms
