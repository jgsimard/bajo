"""Geometry and ray fixtures shared by BVH benchmarks."""

from std.math import max

from bajo.bvh import Camera
from bajo.bvh.cpu import CpuBlasSet, trace_blas_set
from bajo.core import AABB, Point3f32, Rayf32, Vec3f32


comptime GRID_SIDE = 256
comptime PRIM_COUNT = GRID_SIDE * GRID_SIDE
comptime RAY_REPEATS_PER_PRIM = 4
comptime RAY_COUNT = PRIM_COUNT * RAY_REPEATS_PER_PRIM
comptime DEPTH_LAYER_COUNT = 4096
comptime DEPTH_STRESS_RAY_COUNT = 512 * 288


def _grid_x(i: Int) -> Float32:
    return (Float32(i % GRID_SIDE) - Float32(GRID_SIDE) * 0.5) * 3.0


def _grid_y(i: Int) -> Float32:
    return (Float32(i / GRID_SIDE) - Float32(GRID_SIDE) * 0.5) * 3.0


def make_grid_triangles() -> List[Point3f32[.WORLD]]:
    var vertices = List[Point3f32[.WORLD]](capacity=PRIM_COUNT * 3)
    for i in range(PRIM_COUNT):
        var cx = _grid_x(i)
        var cy = _grid_y(i)
        vertices.append(Point3f32[.WORLD](cx - 0.75, cy - 0.75, 2.0))
        vertices.append(Point3f32[.WORLD](cx + 0.75, cy - 0.75, 2.0))
        vertices.append(Point3f32[.WORLD](cx, cy + 0.75, 2.0))
    return vertices^


def make_hit_and_miss_rays() -> List[Rayf32[.WORLD]]:
    var rays = List[Rayf32[.WORLD]](capacity=RAY_COUNT)
    for i in range(RAY_COUNT):
        var prim_idx = i % PRIM_COUNT
        if i % RAY_REPEATS_PER_PRIM == 0:
            rays.append(
                Rayf32[.WORLD](
                    Point3f32[.WORLD](10000.0 + Float32(i), 10000.0, 0.0),
                    Vec3f32[.WORLD](0.0, 0.0, 1.0),
                )
            )
        else:
            rays.append(
                Rayf32[.WORLD](
                    Point3f32[.WORLD](
                        _grid_x(prim_idx), _grid_y(prim_idx), 0.0
                    ),
                    Vec3f32[.WORLD](0.0, 0.0, 1.0),
                )
            )
    return rays^


def make_depth_overlap_triangles() -> List[Point3f32[.WORLD]]:
    """Separated depth layers whose bounds all overlap benchmark rays."""
    var vertices = List[Point3f32[.WORLD]](capacity=DEPTH_LAYER_COUNT * 3)
    for i in range(DEPTH_LAYER_COUNT):
        var z = 2.0 + 0.01 * Float32(i)
        vertices.append(Point3f32[.WORLD](-4.0, -4.0, z))
        vertices.append(Point3f32[.WORLD](4.0, -4.0, z))
        vertices.append(Point3f32[.WORLD](0.0, 4.0, z))
    return vertices^


def make_depth_overlap_rays() -> List[Rayf32[.WORLD]]:
    var rays = List[Rayf32[.WORLD]](capacity=DEPTH_STRESS_RAY_COUNT)
    for i in range(DEPTH_STRESS_RAY_COUNT):
        var x = (Float32(i % 256) / 255.0 - 0.5) * 0.5
        var y = (Float32((i / 256) % 256) / 255.0 - 0.5) * 0.5
        rays.append(
            Rayf32[.WORLD](
                Point3f32[.WORLD](x, y, 0.0),
                Vec3f32[.WORLD](0.0, 0.0, 1.0),
            )
        )
    return rays^


def permute_rays(
    rays: List[Rayf32[.WORLD]],
) -> List[Rayf32[.WORLD]]:
    """Return a deterministic cache-unfriendly permutation."""
    var permuted = List[Rayf32[.WORLD]](capacity=len(rays))
    for i in range(len(rays)):
        permuted.append(rays[(i * 104729) % len(rays)].copy())
    return permuted^


def select_and_repeat_hit_rays(
    bvh: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> List[Rayf32[.WORLD]]:
    """Retain hit rays in order and repeat them to the original count."""
    var hit_rays = List[Rayf32[.WORLD]](capacity=len(rays))
    for ray in rays:
        if trace_blas_set[16, 16, .CLOSEST_HIT, .WORLD](
            bvh, UInt32(0), ray
        ).is_hit():
            hit_rays.append(ray.copy())
    debug_assert["safe", _use_compiler_assume=True](
        len(hit_rays) > 0, "high-hit benchmark found no intersections"
    )
    var repeated = List[Rayf32[.WORLD]](capacity=len(rays))
    for i in range(len(rays)):
        repeated.append(hit_rays[i % len(hit_rays)].copy())
    return repeated^


def make_camera_rays_and_params(
    bounds: AABB[.WORLD],
    width: Int,
    height: Int,
    views: Int,
    fov_scale: Float32 = 0.75,
) -> Tuple[List[Rayf32[.WORLD]], List[Float32]]:
    var center = bounds.centroid()
    var extent = bounds.extent()

    var scene_w = max(max(extent.x, extent.y), extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var rays = List[Rayf32[.WORLD]](capacity=width * height * views)
    var params = List[Float32](capacity=views * Camera.STRIDE)

    for view in range(views):
        var view_offset = Float32(view) - Float32(views - 1) * 0.5
        var eye = center + Vec3f32[.WORLD](
            view_offset * scene_w * 0.30,
            extent.y * 0.20,
            -scene_w * 2.50,
        )
        var camera = Camera(
            eye,
            center,
            Vec3f32[.WORLD](0.0, 1.0, 0.0),
            fov_scale,
        )
        params.extend(camera.flatten())

        for py in range(height):
            for px in range(width):
                rays.append(camera.make_ray(px, py, width, height))

    return (rays^, params^)
