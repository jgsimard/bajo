from std.collections import List
from std.time import perf_counter_ns
from std.math import min

from bajo.core.vec import Vec3f32, normalize, cross, length
from bajo.core.bvh.tinybvh import BVH, WideBVH, Ray


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
    # bvh.build_quick()
    # bvh.build_sah()
    bvh.build_sah_mt()

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
    var seed = UInt32(0x12345678)

    for i in range(Nsmall):
        var r = primary_rays[i].copy()
        bvh.traverse(r)
        primary_rays[i] = r.copy()

        var hit_t = min(Float32(100.0), r.hit.t)
        var I = r.O + r.D * hit_t

        var L = lightPos - I
        var dist = length(L)
        var shadow_dir = L / dist
        shadow_rays.append(Ray(I + shadow_dir * 1e-4, shadow_dir, dist - 1e-4))

        seed ^= seed << 13
        seed ^= seed >> 17
        seed ^= seed << 5
        var rx = (Float32(seed % 1000) / 500.0) - 1.0
        seed ^= seed << 13
        seed ^= seed >> 17
        seed ^= seed << 5
        var ry = (Float32(seed % 1000) / 500.0) - 1.0
        seed ^= seed << 13
        seed ^= seed >> 17
        seed ^= seed << 5
        var rz = (Float32(seed % 1000) / 500.0) - 1.0
        var diff_dir = normalize(Vec3f32(rx, ry, rz))
        diffuse_rays.append(Ray(I + diff_dir * 1e-4, diff_dir))

    var PASSES = 5
    var M = Float64(Nsmall) / 1_000_000.0

    print("\n--- Traversals (Averaged over 5 passes) ---")

    print("\n[Primary Rays - Coherent Rays]")
    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            var r = primary_rays[i].copy()
            r.hit.t = 1e30
            bvh.traverse(r)
    print(
        t"    BVH2 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            var r = primary_rays[i].copy()
            r.hit.t = 1e30
            bvh4.traverse(r)
    print(
        t"    BVH4 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            var r = primary_rays[i].copy()
            r.hit.t = 1e30
            bvh8.traverse(r)
    print(
        t"    BVH8 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    print("\n[Shadow Rays - Early Out Occlusion]")
    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            _ = bvh.is_occluded(shadow_rays[i])
    print(
        t"    BVH2 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            _ = bvh4.is_occluded(shadow_rays[i])
    print(
        t"    BVH4 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            _ = bvh8.is_occluded(shadow_rays[i])
    print(
        t"    BVH8 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    print("\n[Diffuse Rays - Incoherent Bounces]")
    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            bvh.traverse(diffuse_rays[i])
    print(
        t"    BVH2 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            bvh4.traverse(diffuse_rays[i])
    print(
        t"    BVH4 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )

    timer.reset()
    for _ in range(PASSES):
        for i in range(Nsmall):
            bvh8.traverse(diffuse_rays[i])
    print(
        t"    BVH8 : {round(M / (timer.elapsed()/Float64(PASSES)), 2)} MRays/s"
    )
