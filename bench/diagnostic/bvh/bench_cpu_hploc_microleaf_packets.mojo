"""Packet/adaptive traversal quality for CPU H-PLOC microleaf candidates."""

from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.bvh.cpu import CpuBvhBuildMethod
from bajo.bvh.cpu.blas_set import build_cpu_triangle_blas_set
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Point3f32, Rayf32
from bajo.parser.obj.pack import pack_obj_triangles
from bench.bvh.bench_cpu_bvh_packets import (
    benchmark,
    benchmark_adaptive,
    print_timing,
)


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime WIDTH = 16


def _benchmark_candidate[
    microleaf_size: Int,
](vertices: List[Point3f32[.WORLD]], rays: List[Rayf32[.WORLD]]):
    print(t"\nDragon H-PLOC/BVH16 leaf16 microleaf {microleaf_size}")
    var bvh = build_cpu_triangle_blas_set[
        WIDTH,
        WIDTH,
        CpuBvhBuildMethod.HPLOC,
        .WORLD,
        microleaf_size,
    ]([vertices.copy()])
    print_timing("scalar", benchmark[WIDTH, WIDTH, 1](bvh, rays), len(rays))
    print_timing(
        "coh-packet4", benchmark[WIDTH, WIDTH, 4, True](bvh, rays), len(rays)
    )
    print_timing(
        "coh-packet8", benchmark[WIDTH, WIDTH, 8, True](bvh, rays), len(rays)
    )
    print_timing(
        "coh-packet16",
        benchmark[WIDTH, WIDTH, 16, True](bvh, rays),
        len(rays),
    )
    print_timing(
        "adaptive-16-8-scalar",
        benchmark_adaptive[WIDTH, WIDTH, 16, 8](bvh, rays),
        len(rays),
    )


def main() raises:
    print("CPU H-PLOC microleaf packet-quality experiment")
    var vertices = pack_obj_triangles[.WORLD](OBJ_PATH)
    var bounds = compute_bounds(vertices)
    var camera = make_camera_rays_and_params(bounds, 1024, 576, 1, 0.2)
    var rays = camera[0].copy()
    comptime for microleaf_size in [1, 4, 8, 16]:
        _benchmark_candidate[microleaf_size](vertices, rays)
