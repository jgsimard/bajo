"""Scene-sized target allocation preserves explicit bounds and pixel alignment."""

from std.testing import TestSuite, assert_equal
from max.gpu.host import DeviceContext
from bajo.core import Affine3f32, Point3f32, Vec3f32
from bajo.bvh.host_utils import compute_bounds
from bajo.rt import Camera, Color, RenderSettings, SceneBuilder
from bajo.rt.scene_data import SceneData
from bajo.rt.gpu import (
    GpuRtPreparedRenderer,
    prepare_gpu_scene,
    render_gpu_configured,
    GPU_RT_BVH_WIDE4,
)


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 3.0),
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        45.0,
    )


def test_gpu_scene_target_bounds_and_alignment() raises:
    var plain_builder = SceneBuilder()
    var matte = plain_builder.add_lambertian(Color(0.6))
    plain_builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, matte)
    var plain = plain_builder^.finish()
    var coated_builder = SceneBuilder()
    var coat = coated_builder.add_coated_diffuse(Color(0.6), 0.2, 1.5)
    coated_builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, coat)
    var coated = coated_builder^.finish()
    var settings = RenderSettings(512, 128, 13, UInt64(653), 4)
    var camera = _camera()
    with DeviceContext() as ctx:
        var plain_scene = prepare_gpu_scene[.SPHERES](ctx, plain)
        var coated_scene = prepare_gpu_scene[.SPHERES](ctx, coated)
        var small = plain_scene.make_render_target(ctx, settings, camera)
        var large = coated_scene.make_render_target(ctx, settings, camera)
        var bounded = coated_scene.make_render_target(
            ctx, settings, camera, 100003
        )
        assert_equal(small.arena.capacity, 262132)
        # The large recommendation is clamped to the full render's 851968 samples.
        assert_equal(large.arena.capacity, 512 * 128 * 13)
        assert_equal(bounded.arena.capacity, 99996)
        assert_equal(len(small.arena.path_a.fields), 12 * small.arena.capacity)
        assert_equal(len(large.arena.shade.fields), 6 * large.arena.capacity)
        assert_equal(
            len(bounded.arena.shadow.fields), 11 * bounded.arena.capacity
        )
        var tiny_settings = RenderSettings(2, 3, 17, UInt64(659), 4)
        var tiny = coated_scene.make_render_target(
            ctx, tiny_settings, camera, 1
        )
        assert_equal(tiny.arena.capacity, 17)
        var whole = coated_scene.make_render_target(ctx, tiny_settings, camera)
        assert_equal(whole.arena.capacity, 102)


def test_gpu_instance_and_static_target_recommendations() raises:
    var mesh: List[Point3f32[.LOCAL]] = [
        Point3f32[.LOCAL](-0.5, -0.5, 0.0),
        Point3f32[.LOCAL](0.5, -0.5, 0.0),
        Point3f32[.LOCAL](0.0, 0.5, 0.0),
    ]
    var inst_builder = SceneBuilder()
    var inst_matte = inst_builder.add_lambertian(Color(0.6))
    _ = inst_builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].identity(),
        compute_bounds(mesh),
        inst_matte,
    )
    var instances = inst_builder^.finish()
    var tri_builder = SceneBuilder()
    var tri_matte = tri_builder.add_lambertian(Color(0.6))
    tri_builder.add_triangle(
        Point3f32[.WORLD](-0.5, -0.5, 0.0),
        Point3f32[.WORLD](0.5, -0.5, 0.0),
        Point3f32[.WORLD](0.0, 0.5, 0.0),
        tri_matte,
    )
    var triangles = tri_builder^.finish()
    with DeviceContext() as ctx:
        var inst_scene = prepare_gpu_scene[
            .INSTANCES, blas_format=GPU_RT_BVH_WIDE4
        ](ctx, instances)
        var tri_scene = prepare_gpu_scene[
            .TRIANGLES, triangle_format=GPU_RT_BVH_WIDE4
        ](ctx, triangles)
        var tiny_settings = RenderSettings(2, 3, 17, UInt64(661), 4)
        var target = inst_scene.make_render_target(
            ctx, tiny_settings, _camera()
        )
        assert_equal(target.arena.capacity, 102)
        assert_equal(tri_scene.preferred_path_capacity(), 262144)
        assert_equal(inst_scene.preferred_path_capacity(), 4194304)


def _triangle_grid(with_sphere: Bool) raises -> SceneData:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.6))
    if with_sphere:
        builder.add_sphere(Point3f32[.WORLD](0.0), 0.25, matte)
    for i in range(4096):
        var x = Float32(i % 64)
        var y = Float32(i // 64)
        builder.add_triangle(
            Point3f32[.WORLD](x, y, 0.0),
            Point3f32[.WORLD](x + 0.5, y, 0.0),
            Point3f32[.WORLD](x, y + 0.5, 0.0),
            matte,
        )
    return builder^.finish()


def test_gpu_large_static_and_mixed_target_recommendations() raises:
    var static_data = _triangle_grid(False)
    var mixed_data = _triangle_grid(True)
    with DeviceContext() as ctx:
        var static_scene = prepare_gpu_scene[.TRIANGLES](ctx, static_data)
        var mixed_scene = prepare_gpu_scene[.SPHERES_TRIANGLES](ctx, mixed_data)
        assert_equal(static_scene.preferred_path_capacity(), 8388608)
        assert_equal(mixed_scene.preferred_path_capacity(), 1048576)
        var settings = RenderSettings(2, 3, 17, UInt64(673), 4)
        var target = static_scene.make_render_target(ctx, settings, _camera())
        assert_equal(target.arena.capacity, 102)


def test_gpu_prepared_renderer_retains_scene_budget_on_resize() raises:
    var builder = SceneBuilder()
    var coat = builder.add_coated_diffuse(Color(0.6), 0.2, 1.5)
    builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, coat)
    var data = builder^.finish()
    var camera = _camera()
    var initial = RenderSettings(512, 256, 9, UInt64(677), 4)
    var renderer = GpuRtPreparedRenderer[.SPHERES](initial, camera, data)
    assert_equal(renderer.target.arena.capacity, 1048572)
    var resized = RenderSettings(256, 128, 9, UInt64(677), 4)
    var actual = renderer.render[.NORMALS](resized, camera)
    assert_equal(renderer.target.arena.capacity, 256 * 128 * 9)
    var expected = render_gpu_configured[kind=.SPHERES, integrator=.NORMALS](
        resized, camera, data
    )
    var repeated = renderer.render[.NORMALS](resized, camera)
    assert_equal(len(actual.pixels), len(expected.pixels))
    for i in range(len(actual.pixels)):
        assert_equal(actual.pixels[i].x, expected.pixels[i].x)
        assert_equal(actual.pixels[i].y, expected.pixels[i].y)
        assert_equal(actual.pixels[i].z, expected.pixels[i].z)
        assert_equal(repeated.pixels[i].x, actual.pixels[i].x)
        assert_equal(repeated.pixels[i].y, actual.pixels[i].y)
        assert_equal(repeated.pixels[i].z, actual.pixels[i].z)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
