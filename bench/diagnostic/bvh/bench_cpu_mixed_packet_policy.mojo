"""Controlled mixed packet/scalar traversal policy sweep."""

from std.benchmark import keep
from std.math import round
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import (
    make_camera_rays_and_params,
    make_grid_triangles,
    make_hit_and_miss_rays,
)
from bajo.bvh.constants import f32_max
from bajo.bvh.cpu import CpuBlasSet, CpuBvhBuildMethod
from bajo.bvh.cpu.blas_set import (
    _trace_blas_set_packet_policy,
    build_cpu_triangle_blas_set,
)
from bajo.bvh.cpu.triangle_bvh import TrianglePacketConfig
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.core import Point3, Point3f32, Ray, Rayf32, Vec3
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.parser.obj.pack import pack_obj_triangles


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime WIDTH = 16
comptime REPEATS = 7
comptime DRAGON_WIDTH = 1024
comptime DRAGON_HEIGHT = 576
comptime FOV_SCALE = 0.2


@fieldwise_init
struct Timing(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def _trace_policy[
    config: TrianglePacketConfig,
    common_octant: Bool = True,
    packet_width: SIMDLength = WIDTH,
](
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = Float64(0.0)
    var hits = 0
    for base in range(0, len(rays), packet_width):
        var ox = SIMD[.float32, packet_width](0.0)
        var oy = SIMD[.float32, packet_width](0.0)
        var oz = SIMD[.float32, packet_width](0.0)
        var dx = SIMD[.float32, packet_width](0.0)
        var dy = SIMD[.float32, packet_width](0.0)
        var dz = SIMD[.float32, packet_width](1.0)
        var t_min = SIMD[.float32, packet_width](0.0)
        var t_max = SIMD[.float32, packet_width](f32_max)
        var valid = SIMD[.bool, packet_width](fill=False)
        var lane_count = min(packet_width, len(rays) - base)
        comptime for lane in range(packet_width):
            if lane < lane_count:
                ref ray = rays[base + lane]
                ox[lane] = ray.o.x
                oy[lane] = ray.o.y
                oz[lane] = ray.o.z
                dx[lane] = ray.d.x
                dy[lane] = ray.d.y
                dz[lane] = ray.d.z
                t_min[lane] = ray.t_min
                t_max[lane] = ray.t_max
                valid[lane] = True

        var packet = Ray[.float32, .WORLD, packet_width](
            Point3[.float32, .WORLD, packet_width](ox, oy, oz),
            Vec3[.float32, .WORLD, packet_width](dx, dy, dz),
            t_min,
            t_max,
        )
        var packet_hit = _trace_blas_set_packet_policy[
            WIDTH,
            WIDTH,
            packet_width,
            common_octant,
            .WORLD,
            config,
        ](bvh, UInt32(0), packet, valid)
        comptime for lane in range(packet_width):
            if lane < lane_count and packet_hit.prim[lane] != UInt32(
                0xFFFFFFFF
            ):
                checksum += (
                    Float64(packet_hit.t[lane])
                    + Float64(packet_hit.u[lane])
                    + Float64(packet_hit.v[lane])
                    + Float64(packet_hit.normal.x[lane])
                    + Float64(packet_hit.normal.y[lane])
                    + Float64(packet_hit.normal.z[lane])
                    + Float64(packet_hit.prim[lane])
                )
                hits += 1
    return (checksum, hits)


def _benchmark[
    config: TrianglePacketConfig,
    common_octant: Bool = True,
    packet_width: SIMDLength = WIDTH,
](
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Timing:
    var summary = _trace_policy[config, common_octant, packet_width](bvh, rays)
    var best = Int.MAX
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        summary = _trace_policy[config, common_octant, packet_width](bvh, rays)
        best = min(best, Int(perf_counter_ns() - start))
    keep(summary[0])
    keep(summary[1])
    return Timing(best, summary[0], summary[1])


def _print(label: String, timing: Timing, ray_count: Int):
    print(
        t"{label}\t{round(ns_to_ms(timing.ns), 3)}\t"
        t"{round(ns_to_mrays_per_s(timing.ns, ray_count), 3)}\t"
        t"{timing.hits}\t{round(timing.checksum, 3)}"
    )


def _run_sweep(
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
):
    print("Policy\tTrace ms\tMRay/s\tHits\tChecksum")
    _print(
        "production",
        _benchmark[TrianglePacketConfig.PRODUCTION](bvh, rays),
        len(rays),
    )
    _print(
        "pure-packet",
        _benchmark[TrianglePacketConfig.PURE](bvh, rays),
        len(rays),
    )
    comptime for threshold in [1, 2, 4, 6, 8, 10, 12, 15]:
        _print(
            String(t"scalar-leaves-{threshold}"),
            _benchmark[TrianglePacketConfig.scalar_leaves[threshold]()](
                bvh, rays
            ),
            len(rays),
        )
    comptime for threshold in [1, 2, 4, 6, 8, 10, 12, 15]:
        _print(
            String(t"scalar-continuation-{threshold}"),
            _benchmark[TrianglePacketConfig.scalar_continuation[threshold]()](
                bvh, rays
            ),
            len(rays),
        )
    comptime for threshold in [1, 2, 4, 6, 8, 10, 12, 15]:
        _print(
            String(t"scalar-both-{threshold}"),
            _benchmark[TrianglePacketConfig.scalar_both[threshold]()](
                bvh, rays
            ),
            len(rays),
        )
    comptime for task_count in [1, 2, 3, 4, 6, 8, 12, 16]:
        _print(
            String(t"scalar-root-{task_count}"),
            _benchmark[TrianglePacketConfig.scalar_root[task_count]()](
                bvh, rays
            ),
            len(rays),
        )


def _run_selected(
    label: String,
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
):
    print(t"\n{label}")
    print("Policy\tTrace ms\tMRay/s\tHits\tChecksum")
    _print(
        "production",
        _benchmark[TrianglePacketConfig.PRODUCTION](bvh, rays),
        len(rays),
    )
    _print(
        "pure-packet",
        _benchmark[TrianglePacketConfig.PURE](bvh, rays),
        len(rays),
    )
    _print(
        "scalar-leaves-10",
        _benchmark[TrianglePacketConfig.scalar_leaves[10]()](bvh, rays),
        len(rays),
    )
    _print(
        "scalar-leaves-12",
        _benchmark[TrianglePacketConfig.scalar_leaves[12]()](bvh, rays),
        len(rays),
    )
    _print(
        "scalar-leaves-15",
        _benchmark[TrianglePacketConfig.scalar_leaves[15]()](bvh, rays),
        len(rays),
    )
    _print(
        "scalar-continuation-6",
        _benchmark[TrianglePacketConfig.scalar_continuation[6]()](bvh, rays),
        len(rays),
    )
    _print(
        "scalar-both-6",
        _benchmark[TrianglePacketConfig.scalar_both[6]()](bvh, rays),
        len(rays),
    )
    _print(
        "scalar-root-1",
        _benchmark[TrianglePacketConfig.scalar_root[1]()](bvh, rays),
        len(rays),
    )


def _run_narrow_sweep[
    packet_width: SIMDLength,
](
    label: String,
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
):
    print(t"\n{label} packet{Int(packet_width)}")
    print("Policy\tTrace ms\tMRay/s\tHits\tChecksum")
    _print(
        "production",
        _benchmark[TrianglePacketConfig.PRODUCTION, True, packet_width](
            bvh, rays
        ),
        len(rays),
    )
    _print(
        "pure-packet",
        _benchmark[TrianglePacketConfig.PURE, True, packet_width](bvh, rays),
        len(rays),
    )
    comptime if packet_width == 8:
        comptime for threshold in [2, 4, 6, 7]:
            _print(
                String(t"scalar-leaves-{threshold}"),
                _benchmark[
                    TrianglePacketConfig.scalar_leaves[threshold](),
                    True,
                    packet_width,
                ](bvh, rays),
                len(rays),
            )
            _print(
                String(t"scalar-continuation-{threshold}"),
                _benchmark[
                    TrianglePacketConfig.scalar_continuation[threshold](),
                    True,
                    packet_width,
                ](bvh, rays),
                len(rays),
            )
    elif packet_width == 4:
        comptime for threshold in [1, 2, 3]:
            _print(
                String(t"scalar-leaves-{threshold}"),
                _benchmark[
                    TrianglePacketConfig.scalar_leaves[threshold](),
                    True,
                    packet_width,
                ](bvh, rays),
                len(rays),
            )
            _print(
                String(t"scalar-continuation-{threshold}"),
                _benchmark[
                    TrianglePacketConfig.scalar_continuation[threshold](),
                    True,
                    packet_width,
                ](bvh, rays),
                len(rays),
            )
        comptime for tasks in [1, 2, 3, 4]:
            _print(
                String(t"scalar-root-{tasks}"),
                _benchmark[
                    TrianglePacketConfig.scalar_root[tasks](),
                    True,
                    packet_width,
                ](bvh, rays),
                len(rays),
            )


def _run_narrow_selected(
    label: String,
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
):
    print(t"\n{label} narrow packet selection")
    print("Policy\tTrace ms\tMRay/s\tHits\tChecksum")
    _print(
        "packet8-production",
        _benchmark[TrianglePacketConfig.PRODUCTION, True, 8](bvh, rays),
        len(rays),
    )
    _print(
        "packet8-pure",
        _benchmark[TrianglePacketConfig.PURE, True, 8](bvh, rays),
        len(rays),
    )
    _print(
        "packet8-scalar-leaves-4",
        _benchmark[TrianglePacketConfig.scalar_leaves[4](), True, 8](bvh, rays),
        len(rays),
    )
    _print(
        "packet4-production",
        _benchmark[TrianglePacketConfig.PRODUCTION, True, 4](bvh, rays),
        len(rays),
    )
    _print(
        "packet4-pure",
        _benchmark[TrianglePacketConfig.PURE, True, 4](bvh, rays),
        len(rays),
    )
    _print(
        "packet4-scalar-leaves-3",
        _benchmark[TrianglePacketConfig.scalar_leaves[3](), True, 4](bvh, rays),
        len(rays),
    )


def main() raises:
    var vertices = pack_obj_triangles[.WORLD](OBJ_PATH)
    var bounds = compute_bounds(vertices)
    var camera = make_camera_rays_and_params(
        bounds, DRAGON_WIDTH, DRAGON_HEIGHT, 1, FOV_SCALE
    )
    var rays = camera[0].copy()
    var bvh = build_cpu_triangle_blas_set[
        WIDTH, WIDTH, CpuBvhBuildMethod.SAH, .WORLD
    ]([vertices.copy()])
    print("Dragon SAH/BVH16 leaf16 coherent packet16 policy sweep; best of 7")
    _run_sweep(bvh, rays)
    _run_narrow_sweep[8]("Dragon SAH/BVH16 leaf16", bvh, rays)
    _run_narrow_sweep[4]("Dragon SAH/BVH16 leaf16", bvh, rays)

    var median = build_cpu_triangle_blas_set[
        WIDTH, WIDTH, CpuBvhBuildMethod.MEDIAN, .WORLD
    ]([vertices.copy()])
    _run_selected("Dragon Median/BVH16 leaf16", median, rays)
    _run_narrow_selected("Dragon Median/BVH16 leaf16", median, rays)
    var lbvh = build_cpu_triangle_blas_set[
        WIDTH, WIDTH, CpuBvhBuildMethod.LBVH, .WORLD
    ]([vertices.copy()])
    _run_selected("Dragon LBVH/BVH16 leaf16", lbvh, rays)
    _run_narrow_selected("Dragon LBVH/BVH16 leaf16", lbvh, rays)
    var hploc = build_cpu_triangle_blas_set[
        WIDTH, WIDTH, CpuBvhBuildMethod.HPLOC, .WORLD
    ]([vertices^])
    _run_selected("Dragon H-PLOC/BVH16 leaf16", hploc, rays)
    _run_narrow_selected("Dragon H-PLOC/BVH16 leaf16", hploc, rays)

    var grid_vertices = make_grid_triangles()
    var grid_rays = make_hit_and_miss_rays()
    var grid_hploc = build_cpu_triangle_blas_set[
        WIDTH, WIDTH, CpuBvhBuildMethod.HPLOC, .WORLD
    ]([grid_vertices^])
    _run_selected("Grid H-PLOC/BVH16 leaf16", grid_hploc, grid_rays)
