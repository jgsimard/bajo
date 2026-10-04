"""Large static MIS queues preserve radiance through the traversal call boundary."""
from std.math import max
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext
from bajo.bvh.host_utils import compute_bounds
from bajo.rt import Camera, Color, RenderSettings, SceneBuilder
from bajo.rt.gpu.scene import prepare_gpu_scene
from bajo.rt.gpu.render import enqueue_render_gpu
from bajo.rt.gpu.resources import download_gpu_pixels
from bajo.core import Vec3f32
from test.bvh.fixtures import _make_small_scene


def _check(coated: Bool) raises:
    var vertices = _make_small_scene[.WORLD]()
    var bounds = compute_bounds(vertices)
    var extent = bounds.extent()
    var center = bounds.centroid()
    var builder = SceneBuilder()
    var surface = builder.add_coated_diffuse(
        Color(0.65), 0.2, 1.5
    ) if coated else builder.add_lambertian(Color(0.65))
    builder.add_triangle_mesh(vertices, surface)
    var data = builder^.finish()
    var camera = Camera.from_vfov(
        center
        + Vec3f32[.WORLD](
            0.0, 0.0, 3.0 * max(max(extent.x, extent.y), extent.z)
        ),
        center,
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        40.0,
    )
    var settings = RenderSettings(512, 128, 9, UInt64(673), 5)
    with DeviceContext() as ctx:
        var world = prepare_gpu_scene[.TRIANGLES](ctx, data)
        var small = world.make_render_target(ctx, settings, camera, 262144)
        var large = world.make_render_target(ctx, settings, camera, 589824)
        assert_true(small.arena.capacity < 524288)
        assert_true(large.arena.capacity >= 524288)
        enqueue_render_gpu[.MIS](ctx, small, world, settings)
        ctx.synchronize()
        var a = download_gpu_pixels(ctx, small)
        enqueue_render_gpu[.MIS](ctx, large, world, settings)
        ctx.synchronize()
        var b = download_gpu_pixels(ctx, large)
        assert_equal(len(a), len(b))
        for i in range(len(a)):
            assert_equal(a[i].x, b[i].x)
            assert_equal(a[i].y, b[i].y)
            assert_equal(a[i].z, b[i].z)


def test_static_mis_call_boundary_lambertian() raises:
    _check(False)


def test_static_mis_call_boundary_coated() raises:
    _check(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
