from std.algorithm import parallelize
from std.collections import List
from std.time import perf_counter_ns
from std.math import min
from std.utils import Variant

from bajo.core.vec import Vec3f32, normalize, cross, length
from bajo.core.bvh.tinybvh import BVH, WideBVH, Ray
from bajo.core.random import PhiloxRNG, random_unit_vector


struct Timer:
    var start: UInt

    @always_inline
    def __init__(out self):
        self.start = perf_counter_ns()

    @always_inline
    def reset(mut self):
        self.start = perf_counter_ns()

    @always_inline
    def elapsed(self) -> Float64:
        return Float64(perf_counter_ns() - self.start) / 1e9


comptime PASSES = 5
comptime BVHVariant = Variant[BVH, WideBVH[4], WideBVH[8], WideBVH[16]]


def run_bench[
    is_occlusion: Bool, reset_t: Bool
](bvh_var: BVHVariant, mut rays: List[Ray], label: String):
    """
    Unified benchmark implementation using Variant dispatch.
    """
    var n_rays = len(rays)
    var timer = Timer()

    for _ in range(PASSES):

        @parameter
        def worker(i: Int):
            ref ray = rays[i]
            comptime if reset_t:
                ray.hit.t = 1e30

            if bvh_var.isa[BVH]():
                ref b = bvh_var[BVH]
                comptime if is_occlusion:
                    _ = b.is_occluded(ray)
                else:
                    b.traverse(ray)
            elif bvh_var.isa[WideBVH[4]]():
                ref b = bvh_var[WideBVH[4]]
                comptime if is_occlusion:
                    _ = b.is_occluded(ray)
                else:
                    b.traverse(ray)
            elif bvh_var.isa[WideBVH[8]]():
                ref b = bvh_var[WideBVH[8]]
                comptime if is_occlusion:
                    _ = b.is_occluded(ray)
                else:
                    b.traverse(ray)
            elif bvh_var.isa[WideBVH[16]]():
                ref b = bvh_var[WideBVH[16]]
                comptime if is_occlusion:
                    _ = b.is_occluded(ray)
                else:
                    b.traverse(ray)

        parallelize[worker](n_rays)

    var total_sec = timer.elapsed()
    var mrays = (Float64(n_rays) * PASSES / 1_000_000.0) / total_sec
    print("    ", label, ": ", round(mrays, 2), " MRays/s")


def main():
    print("--- TinyBVH Mojo Port: Traverse Speedtest ---")

    var vertices = List[Vec3f32]()
    print(
        "Generating procedural scene (30x30x30 cube grid, ~324k triangles)..."
    )

    for x in range(30):
        for y in range(30):
            for z in range(30):
                var px = Float32(x) * 2.0
                var py = Float32(y) * 2.0
                var pz = Float32(z) * 2.0
                vertices.append(Vec3f32(px, py, pz))
                vertices.append(Vec3f32(px + 1, py, pz))
                vertices.append(Vec3f32(px + 1, py + 1, pz))
                vertices.append(Vec3f32(px, py, pz))
                vertices.append(Vec3f32(px + 1, py + 1, pz))
                vertices.append(Vec3f32(px, py + 1, pz))
                vertices.append(Vec3f32(px, py, pz + 1))
                vertices.append(Vec3f32(px + 1, py + 1, pz + 1))
                vertices.append(Vec3f32(px + 1, py, pz + 1))
                vertices.append(Vec3f32(px, py, pz + 1))
                vertices.append(Vec3f32(px, py + 1, pz + 1))
                vertices.append(Vec3f32(px + 1, py + 1, pz + 1))
                vertices.append(Vec3f32(px, py, pz))
                vertices.append(Vec3f32(px, py + 1, pz + 1))
                vertices.append(Vec3f32(px, py, pz + 1))
                vertices.append(Vec3f32(px, py, pz))
                vertices.append(Vec3f32(px, py + 1, pz))
                vertices.append(Vec3f32(px, py + 1, pz + 1))
                vertices.append(Vec3f32(px + 1, py, pz))
                vertices.append(Vec3f32(px + 1, py, pz + 1))
                vertices.append(Vec3f32(px + 1, py + 1, pz + 1))
                vertices.append(Vec3f32(px + 1, py, pz))
                vertices.append(Vec3f32(px + 1, py + 1, pz + 1))
                vertices.append(Vec3f32(px + 1, py + 1, pz))
                vertices.append(Vec3f32(px, py + 1, pz))
                vertices.append(Vec3f32(px + 1, py + 1, pz + 1))
                vertices.append(Vec3f32(px, py + 1, pz + 1))
                vertices.append(Vec3f32(px, py + 1, pz))
                vertices.append(Vec3f32(px + 1, py + 1, pz))
                vertices.append(Vec3f32(px + 1, py + 1, pz + 1))
                vertices.append(Vec3f32(px, py, pz))
                vertices.append(Vec3f32(px, py, pz + 1))
                vertices.append(Vec3f32(px + 1, py, pz + 1))
                vertices.append(Vec3f32(px, py, pz))
                vertices.append(Vec3f32(px + 1, py, pz + 1))
                vertices.append(Vec3f32(px + 1, py, pz))

    var tri_count = UInt32(len(vertices) // 3)

    var SCREEN_WIDTH = 480
    var SCREEN_HEIGHT = 320
    var Nsmall = SCREEN_WIDTH * SCREEN_HEIGHT

    print("\n[1] Building BVH...")
    var timer = Timer()
    var bvh = BVH(
        vertices.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), tri_count
    )
    # bvh.build["median", False]()
    # bvh.build["median", True]()
    # bvh.build["sah", False]()
    bvh.build["sah", True]()

    print(
        t"    - Built in {round(timer.elapsed() * 1000.0, 2)} ms | SAH cost:"
        t" {bvh.tree_quality()} | Nodes: {bvh.nodes_used}"
    )

    print("\n[2] Collapsing Wide SIMD BVHs...")
    timer.reset()
    var bvh4 = WideBVH[4](bvh)
    print(
        t"    - BVH4 collapse: {round(timer.elapsed() * 1000.0, 2)} ms | Nodes:"
        t" {len(bvh4.nodes)}"
    )

    timer.reset()
    var bvh8 = WideBVH[8](bvh)
    print(
        t"    - BVH8 collapse: {round(timer.elapsed() * 1000.0, 2)} ms | Nodes:"
        t" {len(bvh8.nodes)}"
    )

    timer.reset()
    var bvh16 = WideBVH[16](bvh)
    print(
        t"    - BVH16 collapse: {round(timer.elapsed() * 1000.0, 2)} ms |"
        t" Nodes: {len(bvh16.nodes)}"
    )

    print("\n[3] Generating test rays (Primary, Shadow, Diffuse)...")
    var primary_rays = List[Ray](capacity=Nsmall)
    var shadow_rays = List[Ray](capacity=Nsmall)
    var diffuse_rays = List[Ray](capacity=Nsmall)

    var eye = Vec3f32(-20.0, 30.0, -20.0)
    var view = normalize(Vec3f32(30.0, 30.0, 30.0) - eye)
    var right = normalize(cross(Vec3f32(0.0, 1.0, 0.0), view))
    var up = cross(view, right) * 0.8
    var C = eye + view * 2.0
    var p1 = C - right + up
    var p2 = C + right + up
    var p3 = C - right - up

    for y in range(SCREEN_HEIGHT):
        for x in range(SCREEN_WIDTH):
            var u = Float32(x) / Float32(SCREEN_WIDTH)
            var v = Float32(y) / Float32(SCREEN_HEIGHT)
            var P = p1 + (p2 - p1) * u + (p3 - p1) * v
            primary_rays.append(Ray(eye, normalize(P - eye)))

    var lightPos = Vec3f32(30.0, 100.0, 30.0)
    var rng = PhiloxRNG(123, 321)

    for i in range(Nsmall):
        ref r = primary_rays[i]
        bvh.traverse(r)

        var hit_t = min(Float32(100.0), r.hit.t)
        var I = r.O + r.D * hit_t

        var L = lightPos - I
        var dist = length(L)
        var shadow_dir = L / dist
        shadow_rays.append(Ray(I + shadow_dir * 1e-4, shadow_dir, dist - 1e-4))

        var diff_dir = random_unit_vector(rng)
        diffuse_rays.append(Ray(I + diff_dir * 1e-4, diff_dir))

    # traversals
    var bvh_list = List[BVHVariant]()
    var labels = List[String]()

    bvh_list.append(BVHVariant(bvh^))
    labels.append("BVH2")

    bvh_list.append(BVHVariant(bvh4^))
    labels.append("BVH4")

    bvh_list.append(BVHVariant(bvh8^))
    labels.append("BVH8")

    bvh_list.append(BVHVariant(bvh16^))
    labels.append("BVH16")

    print("\n--- Traversals (Multi-threaded, Averaged over 5 passes) ---")

    print("\n[Primary Rays - Coherent Rays]")
    for i in range(len(bvh_list)):
        run_bench[is_occlusion=False, reset_t=True](
            bvh_list[i], primary_rays, labels[i]
        )

    print("\n[Shadow Rays - Early Out Occlusion]")
    for i in range(len(bvh_list)):
        run_bench[is_occlusion=True, reset_t=False](
            bvh_list[i], shadow_rays, labels[i]
        )

    print("\n[Diffuse Rays - Incoherent Bounces]")
    for i in range(len(bvh_list)):
        run_bench[is_occlusion=False, reset_t=False](
            bvh_list[i], diffuse_rays, labels[i]
        )


# --- TinyBVH Mojo Port: Traverse Speedtest ---
# Generating procedural scene (30x30x30 cube grid, ~324k triangles)...

# [1] Building BVH...
#     - Built in 59.84 ms | SAH cost: 106.674324 | Nodes: 207495

# [2] Collapsing Wide SIMD BVHs...
#     - BVH4 collapse: 9.03 ms | Nodes: 40018
#     - BVH8 collapse: 21.17 ms | Nodes: 21695
#     - BVH16 collapse: 34.41 ms | Nodes: 19444

# [3] Generating test rays (Primary, Shadow, Diffuse)...

# --- Traversals (Multi-threaded, Averaged over 5 passes) ---

# [Primary Rays - Coherent Rays]
#      BVH2 :  12.0  MRays/s
#      BVH4 :  25.31  MRays/s
#      BVH8 :  32.63  MRays/s
#      BVH16 :  26.98  MRays/s

# [Shadow Rays - Early Out Occlusion]
#      BVH2 :  14.54  MRays/s
#      BVH4 :  28.73  MRays/s
#      BVH8 :  43.85  MRays/s
#      BVH16 :  40.79  MRays/s

# [Diffuse Rays - Incoherent Bounces]
#      BVH2 :  15.78  MRays/s
#      BVH4 :  41.48  MRays/s
#      BVH8 :  52.7  MRays/s
#      BVH16 :  43.65  MRays/s
