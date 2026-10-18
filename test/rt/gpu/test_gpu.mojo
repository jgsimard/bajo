from test.helpers import assert_vec_equal
from test.rt.helpers import assert_pixels_close, assert_pixels_equal

from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
)
from max.gpu.host import DeviceContext

from bajo.bvh.camera import Camera
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Point3f32, Point3W, Vec3f32, Vec3W
from bajo.core.random import Sampler
from bajo.rt.cpu import (
    CpuScene,
    CpuSchedulerMode,
    render_wavefront,
)
from bajo.rt.gpu import (
    GPU_RT_BVH_WIDE4,
    GpuRtBvhFormat,
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_render_gpu,
    prepare_gpu_scene,
    render_gpu,
    render_gpu_configured,
    update_gpu_camera,
)
from bajo.bvh.constants import PrimitiveKind
from bajo.rt.render_types import (
    Color,
    RenderResult,
    RenderSettings,
)
from bajo.rt.material_types import (
    Environment,
    ImageTexture,
    Integrator,
    SurfaceId,
)
from bajo.rt.scene_data import (
    SceneBuilder,
    SceneData,
)
from examples.cornell_box import make_cornell_scene
from examples.stress_scenes import (
    make_indirect_hall_scene,
    make_many_lights_scene,
    make_specular_transport_scene,
)


def _sphere_scene_data() raises -> SceneData:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5, 0.6, 0.7))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    builder.add_sphere(
        Point3f32[.WORLD](0.0, -100.5, -1.0),
        100.0,
        matte,
    )
    return builder^.finish()


def _sphere_world() raises -> CpuScene[4, 8]:
    return CpuScene[4, 8](_sphere_scene_data())


def _image_environment_world() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.65, 0.55, 0.45))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    builder.add_sphere(
        Point3f32[.WORLD](0.0, -100.5, -1.0),
        100.0,
        matte,
    )
    var pixels: List[Float32] = [
        4.0,
        0.2,
        0.1,
        0.1,
        2.0,
        0.2,
        0.2,
        0.3,
        3.0,
        1.0,
        1.0,
        1.0,
    ]
    var texture_index = builder.surfaces.add_image_texture(
        ImageTexture(2, 2, pixels^)
    )
    builder.set_environment(
        Environment.image(
            texture_index,
            Color(0.75),
            Affine3f32[.WORLD, .LOCAL].identity(),
            Affine3f32[.LOCAL, .WORLD].identity(),
        )
    )
    return CpuScene[4, 8](builder^.finish())


def _material_sphere_world() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    var ground = builder.add_lambertian(Color(0.45, 0.45, 0.45))
    var metal = builder.add_metal(Color(0.8, 0.65, 0.3), 0.2)
    var glass = builder.add_dielectric(1.5)
    var light = builder.add_emissive(Color(3.0, 2.0, 1.0))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, -100.5, -1.5),
        100.0,
        ground,
    )
    builder.add_sphere(
        Point3f32[.WORLD](-0.7, 0.0, -1.5),
        0.45,
        metal,
    )
    builder.add_sphere(
        Point3f32[.WORLD](0.25, 0.0, -1.25),
        0.45,
        glass,
    )
    builder.add_sphere(
        Point3f32[.WORLD](0.9, 0.35, -1.5),
        0.3,
        light,
    )
    var scene = builder^.finish()
    return CpuScene[4, 8](scene^)


def _coated_sphere_world() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    builder.set_environment(Environment.black())
    var albedo_pixels: List[Float32] = [
        0.9,
        0.2,
        0.1,
        0.2,
        0.8,
        0.3,
        0.1,
        0.3,
        0.9,
        0.8,
        0.7,
        0.2,
    ]
    var bump_pixels: List[Float32] = [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.25,
        0.25,
        0.25,
        0.75,
        0.75,
        0.75,
    ]
    var albedo_texture = builder.surfaces.add_image_texture(
        ImageTexture(2, 2, albedo_pixels^)
    )
    var bump_texture = builder.surfaces.add_image_texture(
        ImageTexture(2, 2, bump_pixels^)
    )
    var coated = builder.surfaces.add_coated_diffuse(
        Color(0.8, 0.7, 0.6),
        0.2,
        1.5,
        albedo_texture,
        bump_texture,
        0.08,
        1.5,
        1.5,
    )
    var light = builder.add_emissive(Color(4.0, 3.0, 2.0))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        coated,
    )
    builder.add_sphere(
        Point3f32[.WORLD](0.9, 0.8, -0.5),
        0.2,
        light,
    )
    return CpuScene[4, 8](builder^.finish())


def _mixed_light_world() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.55, 0.58, 0.62))
    var sphere_light = builder.add_emissive(Color(5.0, 4.0, 3.0))
    var triangle_light = builder.add_emissive(Color(3.0, 4.0, 5.0))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.2),
        0.45,
        matte,
    )
    builder.add_sphere(
        Point3f32[.WORLD](-0.8, 0.9, -1.0),
        0.2,
        sphere_light,
    )
    builder.add_triangle(
        Point3f32[.WORLD](0.45, 1.1, -0.8),
        Point3f32[.WORLD](1.05, 1.1, -0.8),
        Point3f32[.WORLD](0.75, 0.65, -0.8),
        triangle_light,
    )
    var scene = builder^.finish()
    return CpuScene[4, 8](scene^)


def _triangle_scene_data() raises -> SceneData:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.35, 0.65, 0.25))
    builder.add_triangle(
        Point3f32[.WORLD](-1.5, -1.0, -1.0),
        Point3f32[.WORLD](1.5, -1.0, -1.0),
        Point3f32[.WORLD](1.5, 1.0, -1.0),
        matte,
    )
    builder.add_triangle(
        Point3f32[.WORLD](-1.5, -1.0, -1.0),
        Point3f32[.WORLD](1.5, 1.0, -1.0),
        Point3f32[.WORLD](-1.5, 1.0, -1.0),
        matte,
    )
    return builder^.finish()


def _triangle_world() raises -> CpuScene[4, 8]:
    return CpuScene[4, 8](_triangle_scene_data())


def _mixed_world() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    var sphere_matte = builder.add_lambertian(Color(0.7, 0.25, 0.2))
    var back_matte = builder.add_lambertian(Color(0.2, 0.35, 0.7))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.4,
        sphere_matte,
    )
    builder.add_triangle(
        Point3f32[.WORLD](-2.0, -1.5, -2.0),
        Point3f32[.WORLD](2.0, -1.5, -2.0),
        Point3f32[.WORLD](2.0, 1.5, -2.0),
        back_matte,
    )
    builder.add_triangle(
        Point3f32[.WORLD](-2.0, -1.5, -2.0),
        Point3f32[.WORLD](2.0, 1.5, -2.0),
        Point3f32[.WORLD](-2.0, 1.5, -2.0),
        back_matte,
    )
    var scene = builder^.finish()
    return CpuScene[4, 8](scene^)


def _textured_instance_world[coated: Bool = False]() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    var pixels: List[Float32] = [
        0.9,
        0.1,
        0.1,
        0.1,
        0.9,
        0.1,
        0.1,
        0.1,
        0.9,
        0.9,
        0.9,
        0.1,
    ]
    var texture_idx = builder.surfaces.add_image_texture(
        ImageTexture(2, 2, pixels^)
    )
    var matte: SurfaceId[1]
    comptime if coated:
        matte = builder.surfaces.add_coated_diffuse(
            Color(0.8, 0.7, 0.6), 0.2, 1.5, texture_idx
        )
    else:
        matte = builder.surfaces.add_lambertian(
            Color(0.8, 0.7, 0.6), texture_idx
        )
    var mesh: List[Point3f32[.LOCAL]] = [
        Point3f32[.LOCAL](-1.25, -1.0, -1.0),
        Point3f32[.LOCAL](1.25, -1.0, -1.0),
        Point3f32[.LOCAL](0.0, 1.0, -1.0),
    ]
    var mesh_idx = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].identity(),
        compute_bounds(mesh),
        matte,
    )
    var normals: List[Float32] = [
        -0.2,
        0.0,
        0.98,
        0.2,
        0.0,
        0.98,
        0.0,
        0.2,
        0.98,
    ]
    var texcoords: List[Float32] = [0.0, 0.0, 1.0, 0.0, 0.5, 1.0]
    builder.triangle_mesh_normals[Int(mesh_idx)] = normals^
    builder.triangle_mesh_texcoords[Int(mesh_idx)] = texcoords^
    return CpuScene[4, 8](builder^.finish())


def _combined_instance_world() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    var red = builder.add_lambertian(Color(0.7, 0.2, 0.2))
    var blue = builder.add_lambertian(Color(0.2, 0.3, 0.7))
    var green = builder.add_lambertian(Color(0.2, 0.7, 0.3))
    builder.add_sphere(
        Point3f32[.WORLD](-0.55, 0.0, -1.1),
        0.3,
        red,
    )
    builder.add_triangle(
        Point3f32[.WORLD](-2.0, -1.2, -2.0),
        Point3f32[.WORLD](2.0, -1.2, -2.0),
        Point3f32[.WORLD](0.0, 1.5, -2.0),
        blue,
    )
    var mesh = List[Point3f32[.LOCAL]]()
    mesh.append(Point3f32[.LOCAL](-0.25, -0.35, -1.0))
    mesh.append(Point3f32[.LOCAL](0.25, -0.35, -1.0))
    mesh.append(Point3f32[.LOCAL](0.0, 0.35, -1.0))
    var bounds = compute_bounds(mesh)
    _ = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].from_translation(
            Vec3f32[.WORLD](0.55, 0.0, 0.0)
        ),
        bounds,
        green,
    )
    var scene = builder^.finish()
    return CpuScene[4, 8](scene^)


def _emissive_instance_world() raises -> CpuScene[4, 8]:
    var builder = SceneBuilder()
    var floor = builder.add_lambertian(Color(0.65, 0.65, 0.65))
    var light = builder.add_emissive(Color(8.0, 6.0, 4.0))
    builder.add_quad(
        Point3f32[.WORLD](-2.0, 0.0, -2.0),
        Point3f32[.WORLD](-2.0, 0.0, 2.0),
        Point3f32[.WORLD](2.0, 0.0, 2.0),
        Point3f32[.WORLD](2.0, 0.0, -2.0),
        floor,
    )

    var mesh = List[Point3f32[.LOCAL]]()
    mesh.append(Point3f32[.LOCAL](-0.5, 0.0, -0.5))
    mesh.append(Point3f32[.LOCAL](0.5, 0.0, -0.5))
    mesh.append(Point3f32[.LOCAL](0.5, 0.0, 0.5))
    mesh.append(Point3f32[.LOCAL](-0.5, 0.0, -0.5))
    mesh.append(Point3f32[.LOCAL](0.5, 0.0, 0.5))
    mesh.append(Point3f32[.LOCAL](-0.5, 0.0, 0.5))
    _ = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD](
            1.5,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.75,
            0.0,
        ),
        compute_bounds(mesh),
        light,
    )
    var scene = builder^.finish()
    return CpuScene[4, 8](scene^)


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )


def _cornell_camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 1.0, 3.2),
        Point3f32[.WORLD](0.0, 1.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        28.0,
        4.2,
    )


def _emissive_instance_camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 1.0, 3.0),
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        35.0,
        1.5,
    )


def test_gpu_sphere_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(4, 3, 2, UInt64(91))
    var world = _sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](settings, camera, world)
    var gpu = render_gpu_configured[
        kind=.SPHERES,
        integrator=.PATH,
        sphere_format=GPU_RT_BVH_WIDE4,
    ](settings, camera, world.scene_data())

    assert_equal(
        gpu.timings.pixel_count, settings.image_width * settings.image_height
    )
    assert_equal(
        gpu.timings.sample_count,
        settings.image_width
        * settings.image_height
        * settings.samples_per_pixel,
    )
    assert_pixels_close(gpu.pixels, cpu.pixels)


def test_gpu_sample_sequences_are_batch_invariant() raises:
    var world = _sphere_world()
    var camera = _camera()
    for sampler_idx in range(1, 6):
        var sampler = Sampler(UInt32(sampler_idx))
        var full_settings = RenderSettings(
            4, 3, 4, UInt64(97), 4, sampler, 0, 4
        )
        var first_settings = RenderSettings(
            4, 3, 2, UInt64(97), 4, sampler, 0, 4
        )
        var second_settings = RenderSettings(
            4, 3, 2, UInt64(97), 4, sampler, 2, 4
        )
        var full = render_gpu_configured[
            kind=.SPHERES,
            integrator=.PATH,
            sphere_format=GPU_RT_BVH_WIDE4,
        ](full_settings, camera, world.scene_data())
        var first = render_gpu_configured[
            kind=.SPHERES,
            integrator=.PATH,
            sphere_format=GPU_RT_BVH_WIDE4,
        ](first_settings, camera, world.scene_data())
        var second = render_gpu_configured[
            kind=.SPHERES,
            integrator=.PATH,
            sphere_format=GPU_RT_BVH_WIDE4,
        ](second_settings, camera, world.scene_data())
        var cpu = render_wavefront[.PATH, 1, 64, False](
            full_settings, camera, world
        )
        for pixel_idx in range(len(full.pixels)):
            var accumulated = 0.5 * (
                first.pixels[pixel_idx] + second.pixels[pixel_idx]
            )
            assert_vec_equal(accumulated, full.pixels[pixel_idx], atol=1.0e-5)
            assert_vec_equal(
                cpu.pixels[pixel_idx], full.pixels[pixel_idx], atol=1.0e-5
            )


def test_gpu_materials_match_cpu_wavefront() raises:
    var settings = RenderSettings(8, 4, 2, UInt64(119))
    var world = _material_sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](settings, camera, world)
    var gpu = render_gpu_configured[
        kind=.SPHERES,
        integrator=.PATH,
        sphere_format=GPU_RT_BVH_WIDE4,
    ](settings, camera, world.scene_data())

    assert_pixels_close(gpu.pixels, cpu.pixels)


def test_gpu_coated_diffuse_and_bump_match_cpu() raises:
    var settings = RenderSettings(8, 4, 4, UInt64(127))
    var world = _coated_sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[.MIS, 1, 64, False](settings, camera, world)
    var gpu = render_gpu_configured[
        kind=.SPHERES,
        integrator=.MIS,
        sphere_format=GPU_RT_BVH_WIDE4,
    ](settings, camera, world.scene_data())
    # GPU transcendental rounding can send a layered random walk down a
    # different internal branch; compare the resulting radiance rather
    # than requiring bit-near identity for every stochastic path.
    assert_pixels_close(gpu.pixels, cpu.pixels, atol=5.0e-3)

    var cpu_normals = render_wavefront[
        .NORMALS, 1, 1, False, CpuSchedulerMode.RUNTIME_DEFAULT, 4, 8
    ](settings, camera, world)
    var gpu_normals = render_gpu_configured[
        kind=.SPHERES,
        integrator=.NORMALS,
        sphere_format=GPU_RT_BVH_WIDE4,
    ](settings, camera, world.scene_data())
    assert_pixels_close(gpu_normals.pixels, cpu_normals.pixels)


def test_gpu_triangle_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(211))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](settings, camera, world)
    var gpu = render_gpu_configured[
        kind=.TRIANGLES,
        integrator=.PATH,
        triangle_format=GPU_RT_BVH_WIDE4,
        triangle_build_method=.LBVH,
    ](settings, camera, world.scene_data())

    assert_pixels_close(gpu.pixels, cpu.pixels)


def test_gpu_triangle_default_hploc_cwbvh8_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(227))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](settings, camera, world)
    var gpu = render_gpu_configured[
        kind=.TRIANGLES,
        integrator=.PATH,
    ](settings, camera, world.scene_data())

    assert_pixels_close(gpu.pixels, cpu.pixels)


def test_gpu_chunked_triangle_nee_matches_full_capacity() raises:
    var settings = RenderSettings(6, 4, 2, UInt64(229))
    var world = _triangle_world()
    var camera = _camera()
    with DeviceContext() as ctx:
        var gpu_world = prepare_gpu_scene[
            kind=.TRIANGLES,
            triangle_format=GPU_RT_BVH_WIDE4,
            triangle_build_method=.LBVH,
        ](ctx, world.scene_data())
        var full = GpuRtRenderTarget(
            ctx,
            settings,
            camera,
            settings.image_width
            * settings.image_height
            * settings.samples_per_pixel,
        )
        enqueue_render_gpu[.NEE](ctx, full, gpu_world, settings)
        var full_pixels = download_gpu_pixels(ctx, full)

        # Two pixels per chunk exercises global path IDs and disjoint resolve.
        var chunked = GpuRtRenderTarget(
            ctx, settings, camera, 2 * settings.samples_per_pixel
        )
        enqueue_render_gpu[.NEE](ctx, chunked, gpu_world, settings)
        var chunked_pixels = download_gpu_pixels(ctx, chunked)

        assert_pixels_equal(chunked_pixels, full_pixels)


def test_gpu_host_boundaries_raise_validation_errors() raises:
    var settings = RenderSettings(2, 2, 1, UInt64(230))
    var camera = _camera()
    var data = _triangle_scene_data()
    with DeviceContext() as ctx:
        with assert_raises():
            _ = prepare_gpu_scene[.SPHERES](ctx, data)

        var target = GpuRtRenderTarget(ctx, settings, camera)
        var incompatible = settings.copy()
        incompatible.samples_per_pixel = 2
        with assert_raises():
            target.validate_compatible(incompatible)


def test_prepared_gpu_scene_is_stable_across_repeated_renders() raises:
    var settings = RenderSettings(3, 2, 1, UInt64(237))
    var data = _sphere_scene_data()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[.SPHERES](ctx, data)
        var target = GpuRtRenderTarget(ctx, settings, _camera())

        enqueue_render_gpu[.NORMALS](ctx, target, scene, settings)
        var before = download_gpu_pixels(ctx, target)

        enqueue_render_gpu[.NORMALS](ctx, target, scene, settings)
        var after = download_gpu_pixels(ctx, target)
        assert_pixels_equal(after, before)

        # Reuse this render for the sphere-normal range and hit checks.
        assert_equal(len(before), settings.image_width * settings.image_height)
        var nonzero = False
        for pixel in before:
            assert_true(pixel.x >= 0.0 and pixel.x <= 1.0)
            assert_true(pixel.y >= 0.0 and pixel.y <= 1.0)
            assert_true(pixel.z >= 0.0 and pixel.z <= 1.0)
            nonzero |= pixel.x > 0.0 or pixel.y > 0.0 or pixel.z > 0.0
        assert_true(nonzero)


def test_gpu_camera_update_is_ordered_before_render() raises:
    var settings = RenderSettings(5, 3, 1, UInt64(238))
    var data = _sphere_scene_data()
    var initial_camera = _camera()
    var updated_camera = Camera.from_vfov(
        Point3f32[.WORLD](0.4, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        70.0,
    )
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[.SPHERES](ctx, data)

        var updated = GpuRtRenderTarget(ctx, settings, initial_camera)
        update_gpu_camera(ctx, updated, updated_camera)
        enqueue_render_gpu[.NORMALS](ctx, updated, scene, settings)
        var updated_pixels = download_gpu_pixels(ctx, updated)

        var reference = GpuRtRenderTarget(ctx, settings, updated_camera)
        enqueue_render_gpu[.NORMALS](ctx, reference, scene, settings)
        var reference_pixels = download_gpu_pixels(ctx, reference)

        assert_pixels_equal(updated_pixels, reference_pixels)


def test_gpu_mixed_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(6, 4, 2, UInt64(307))
    var world = _mixed_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](settings, camera, world)
    var gpu = render_gpu[.PATH](settings, camera, world.scene_data())

    assert_pixels_close(gpu.pixels, cpu.pixels)


def test_gpu_textured_instance_matches_cpu_uvs_and_normals() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(409))
    var world = _textured_instance_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](settings, camera, world)
    var gpu = render_gpu[.PATH](settings, camera, world.scene_data())
    assert_pixels_close(gpu.pixels, cpu.pixels)

    var cpu_normals = render_wavefront[
        .NORMALS, 1, 1, False, CpuSchedulerMode.RUNTIME_DEFAULT, 4, 8
    ](settings, camera, world)
    var gpu_normals = render_gpu[.NORMALS](settings, camera, world.scene_data())
    assert_pixels_close(gpu_normals.pixels, cpu_normals.pixels)


def test_gpu_combined_instances_match_cpu_wavefront() raises:
    var settings = RenderSettings(7, 4, 2, UInt64(457))
    var world = _combined_instance_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](settings, camera, world)
    var gpu = render_gpu[.PATH](settings, camera, world.scene_data())
    var gpu_hploc_tlas = render_gpu_configured[
        kind=.ALL,
        integrator=.PATH,
        tlas_format=GpuRtBvhFormat(2, 2, .WIDE),
        tlas_build_method=.HPLOC,
    ](settings, camera, world.scene_data())
    assert_pixels_close(gpu.pixels, cpu.pixels)
    assert_pixels_close(gpu_hploc_tlas.pixels, cpu.pixels)


def test_gpu_combined_scene_instantiates_every_integrator() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(463))
    var world = _combined_instance_world()
    var camera = _camera()
    comptime for integrator in [
        Integrator.NORMALS,
        Integrator.AO,
        Integrator.NEE,
        Integrator.MIS,
    ]:
        var result = render_gpu[integrator](
            settings, camera, world.scene_data()
        )
        assert_equal(len(result.pixels), 1)


def test_gpu_ao_matches_unoccluded_cpu_reference() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(509))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[
        .AO, 1, 1, False, CpuSchedulerMode.RUNTIME_DEFAULT, 4, 8
    ](settings, camera, world)
    var gpu = render_gpu[.AO](settings, camera, world.scene_data())
    assert_vec_equal(gpu.pixels[0], cpu.pixels[0], atol=1.0e-5)


def _test_gpu_direct_light_matches_cpu[integrator: Integrator]() raises:
    var settings = RenderSettings(3, 3, 2, UInt64(601))
    var world = CpuScene[16, 16](make_cornell_scene())
    var camera = _cornell_camera()
    var cpu = render_wavefront[integrator, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[integrator](settings, camera, world.scene_data())
    assert_pixels_close(gpu.pixels, cpu.pixels, atol=1.0e-4)


def test_gpu_nee_matches_cpu_wavefront() raises:
    _test_gpu_direct_light_matches_cpu[.NEE]()


def test_gpu_sphere_nee_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(619))
    var world = _material_sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[.NEE, 1, 64, False](settings, camera, world)
    var gpu = render_gpu[.NEE](settings, camera, world.scene_data())
    assert_pixels_close(gpu.pixels, cpu.pixels, atol=1.0e-4)


def test_gpu_image_environment_nee_and_mis_match_cpu() raises:
    var settings = RenderSettings(4, 3, 4, UInt64(620))
    var world = _image_environment_world()
    var camera = _camera()
    var cpu_nee = render_wavefront[.NEE, 1, 64, False](settings, camera, world)
    var gpu_nee = render_gpu[.NEE](settings, camera, world.scene_data())
    var cpu_mis = render_wavefront[.MIS, 1, 64, False](settings, camera, world)
    var gpu_mis = render_gpu[.MIS](settings, camera, world.scene_data())
    assert_pixels_close(gpu_nee.pixels, cpu_nee.pixels, atol=1.0e-4)
    assert_pixels_close(gpu_mis.pixels, cpu_mis.pixels, atol=1.0e-4)


def _test_gpu_mixed_lights_match_cpu[integrator: Integrator]() raises:
    var settings = RenderSettings(5, 3, 3, UInt64(623))
    var world = _mixed_light_world()
    ref data = world.scene_data()
    assert_equal(len(data.lights().records), 2)
    var has_sphere_light = False
    var has_triangle_light = False
    for light in data.lights().records:
        has_sphere_light |= light.primitive.kind() == PrimitiveKind.SPHERE
        has_triangle_light |= light.primitive.kind() == PrimitiveKind.TRIANGLE
    assert_true(has_sphere_light)
    assert_true(has_triangle_light)
    var camera = _camera()
    var cpu = render_wavefront[integrator, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[integrator](settings, camera, data)
    assert_pixels_close(gpu.pixels, cpu.pixels, atol=1.0e-4)


def test_gpu_mixed_lights_nee_matches_cpu_wavefront() raises:
    _test_gpu_mixed_lights_match_cpu[.NEE]()


def test_gpu_mixed_lights_mis_matches_cpu_wavefront() raises:
    _test_gpu_mixed_lights_match_cpu[.MIS]()


def _test_gpu_emissive_instance_matches_cpu[integrator: Integrator]() raises:
    var settings = RenderSettings(4, 3, 4, UInt64(631))
    var world = _emissive_instance_world()
    assert_equal(len(world.scene_data().lights().records), 2)
    for light in world.scene_data().lights().records:
        assert_equal(light.primitive.kind(), PrimitiveKind.TRIANGLE_INSTANCE)
    var camera = _emissive_instance_camera()
    var cpu = render_wavefront[integrator, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[integrator](settings, camera, world.scene_data())
    var energy = Float32(0.0)
    for cpu_pixel in cpu.pixels:
        energy += cpu_pixel.x + cpu_pixel.y + cpu_pixel.z
    assert_true(energy > 0.0)
    assert_pixels_close(gpu.pixels, cpu.pixels, atol=1.0e-4)


def test_gpu_emissive_instance_nee_matches_cpu_wavefront() raises:
    _test_gpu_emissive_instance_matches_cpu[.NEE]()


def test_gpu_emissive_instance_mis_matches_cpu_wavefront() raises:
    _test_gpu_emissive_instance_matches_cpu[.MIS]()


def test_gpu_mis_matches_cpu_wavefront() raises:
    _test_gpu_direct_light_matches_cpu[.MIS]()


def test_gpu_direct_light_zero_depth_stays_black() raises:
    var settings = RenderSettings(3, 2, 1, UInt64(701), 0)
    var scene = _triangle_scene_data()
    var camera = _camera()
    var nee = render_gpu_configured[
        kind=.TRIANGLES,
        integrator=.NEE,
        triangle_format=GPU_RT_BVH_WIDE4,
        triangle_build_method=.LBVH,
    ](settings, camera, scene)
    var mis = render_gpu_configured[
        kind=.TRIANGLES,
        integrator=.MIS,
        triangle_format=GPU_RT_BVH_WIDE4,
        triangle_build_method=.LBVH,
    ](settings, camera, scene)
    for pixel in nee.pixels:
        assert_equal(pixel.x, 0.0)
        assert_equal(pixel.y, 0.0)
        assert_equal(pixel.z, 0.0)
    for pixel in mis.pixels:
        assert_equal(pixel.x, 0.0)
        assert_equal(pixel.y, 0.0)
        assert_equal(pixel.z, 0.0)


def _assert_nonzero_result(result: RenderResult) raises:
    assert_equal(len(result.pixels), 40)
    var nonzero = False
    for pixel in result.pixels:
        assert_true(pixel.x >= 0.0 and pixel.y >= 0.0 and pixel.z >= 0.0)
        nonzero |= pixel.x > 0.0 or pixel.y > 0.0 or pixel.z > 0.0
    assert_true(nonzero)


def test_gpu_viewer_renders_all_procedural_stress_scenes() raises:
    var settings = RenderSettings(8, 5, 1, UInt64(2026), 6, .OWEN_SOBOL)

    var many_lights = make_many_lights_scene()
    var many_camera = Camera.from_vfov(
        Point3W(0.0, 4.3, 11.0),
        Point3W(0.0, 1.8, -2.0),
        Vec3W(0.0, 1.0, 0.0),
        52.0,
        18.0,
    )
    var many = render_gpu[.MIS](settings, many_camera, many_lights)

    var indirect_hall = make_indirect_hall_scene()
    var indirect_camera = Camera.from_vfov(
        Point3W(0.0, 2.2, 8.0),
        Point3W(0.0, 1.8, -5.0),
        Vec3W(0.0, 1.0, 0.0),
        55.0,
        22.0,
    )
    var indirect = render_gpu[.MIS](settings, indirect_camera, indirect_hall)

    var specular = make_specular_transport_scene()
    var specular_camera = Camera.from_vfov(
        Point3W(0.0, 2.7, 9.5),
        Point3W(0.0, 1.4, -4.5),
        Vec3W(0.0, 1.0, 0.0),
        48.0,
        20.0,
    )
    var difficult = render_gpu[.MIS](settings, specular_camera, specular)

    _assert_nonzero_result(many)
    _assert_nonzero_result(indirect)
    _assert_nonzero_result(difficult)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
