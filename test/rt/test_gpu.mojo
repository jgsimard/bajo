from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from max.gpu.host import DeviceContext

from bajo.bvh.camera import Camera
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Point3f32, Vec3f32
from bajo.rt.cpu import CpuScene, render_depth_first, render_wavefront
from bajo.rt.gpu import (
    GPU_RT_BVH_WIDE4_LBVH,
    GpuRtBvhPolicy,
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_render_gpu,
    prepare_gpu_scene,
    render_gpu,
    render_gpu_scene,
)
from bajo.rt.gpu.render import (
    _prefer_cwbvh8_blases,
    _prefer_cwbvh8_triangles,
)
from bajo.rt.types import (
    Color,
    PrimitiveKind,
    Integrator,
    RenderSettings,
    SceneBuilder,
    SceneData,
)
from examples.cornell_box import make_cornell_world


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


def _instance_scene_data() raises -> SceneData:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.55, 0.3, 0.75))
    var mesh = List[Point3f32[.LOCAL]]()
    mesh.append(Point3f32[.LOCAL](-1.25, -1.0, -1.0))
    mesh.append(Point3f32[.LOCAL](1.25, -1.0, -1.0))
    mesh.append(Point3f32[.LOCAL](0.0, 1.0, -1.0))
    var bounds = compute_bounds(mesh)
    _ = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].identity(),
        bounds,
        matte,
    )
    return builder^.finish()


def _instance_world() raises -> CpuScene[4, 8]:
    return CpuScene[4, 8](_instance_scene_data())


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
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_scene[
        kind=.SPHERES,
        integrator=.PATH,
        sphere_policy=GPU_RT_BVH_WIDE4_LBVH,
    ](
        settings, camera, world.scene_data()
    )

    assert_equal(len(gpu.pixels), len(cpu.pixels))
    assert_equal(
        gpu.timings.pixel_count, settings.image_width * settings.image_height
    )
    assert_equal(
        gpu.timings.sample_count,
        settings.image_width
        * settings.image_height
        * settings.samples_per_pixel,
    )
    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_materials_match_cpu_wavefront() raises:
    var settings = RenderSettings(8, 4, 2, UInt64(119))
    var world = _material_sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_scene[
        kind=.SPHERES,
        integrator=.PATH,
        sphere_policy=GPU_RT_BVH_WIDE4_LBVH,
    ](
        settings, camera, world.scene_data()
    )

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(211))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_scene[
        kind=.TRIANGLES,
        integrator=.PATH,
        triangle_policy=GPU_RT_BVH_WIDE4_LBVH,
    ](
        settings, camera, world.scene_data()
    )

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_hploc_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(223))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_scene[
        kind=.TRIANGLES,
        integrator=.PATH,
        triangle_policy=GpuRtBvhPolicy(4, 4, .HPLOC, .WIDE),
    ](settings, camera, world.scene_data())

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_default_hploc_cwbvh8_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(227))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_scene[
        kind=.TRIANGLES,
        integrator=.PATH,
    ](settings, camera, world.scene_data())

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_chunked_triangle_nee_matches_full_capacity() raises:
    var settings = RenderSettings(6, 4, 2, UInt64(229))
    var world = _triangle_world()
    var camera = _camera()
    with DeviceContext() as ctx:
        var gpu_world = prepare_gpu_scene[
            kind=.TRIANGLES,
            triangle_policy=GPU_RT_BVH_WIDE4_LBVH,
        ](ctx, world.scene_data())
        var full = GpuRtRenderTarget(
            ctx,
            settings,
            camera,
            settings.image_width
            * settings.image_height
            * settings.samples_per_pixel,
        )
        enqueue_render_gpu[.NEE](
            ctx, full, gpu_world, settings
        )
        var full_pixels = download_gpu_pixels(ctx, full)

        # Two pixels per chunk exercises global path IDs and disjoint resolve.
        var chunked = GpuRtRenderTarget(
            ctx, settings, camera, 2 * settings.samples_per_pixel
        )
        enqueue_render_gpu[.NEE](
            ctx, chunked, gpu_world, settings
        )
        var chunked_pixels = download_gpu_pixels(ctx, chunked)

        assert_equal(len(chunked_pixels), len(full_pixels))
        for i, full_pixel in enumerate(full_pixels):
            assert_equal(chunked_pixels[i].x, full_pixel.x)
            assert_equal(chunked_pixels[i].y, full_pixel.y)
            assert_equal(chunked_pixels[i].z, full_pixel.z)


def test_prepared_scene_policy_and_common_enqueue_api() raises:
    var settings = RenderSettings(3, 2, 1, UInt64(233))
    var world = _triangle_world()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[
            kind=.TRIANGLES,
            triangle_policy=GPU_RT_BVH_WIDE4_LBVH,
        ](
            ctx, world.scene_data()
        )
        var target = GpuRtRenderTarget(ctx, settings, _camera())
        enqueue_render_gpu[.NORMALS](ctx, target, scene, settings)
        var pixels = download_gpu_pixels(ctx, target)
        assert_equal(len(pixels), settings.image_width * settings.image_height)


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
        for i, pixel in enumerate(before):
            assert_equal(after[i].x, pixel.x)
            assert_equal(after[i].y, pixel.y)
            assert_equal(after[i].z, pixel.z)


def test_common_prepared_api_instantiates_every_scene_kind() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(239), 1)
    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, _camera())

        var sphere_data = _sphere_world()
        var spheres = prepare_gpu_scene[.SPHERES](
            ctx, sphere_data.scene_data()
        )
        enqueue_render_gpu(ctx, target, spheres, settings)

        var mixed_data = _mixed_world()
        var mixed = prepare_gpu_scene[.SPHERES_TRIANGLES](ctx, mixed_data.scene_data())
        enqueue_render_gpu(ctx, target, mixed, settings)

        var instance_data = _instance_world()
        var instances = prepare_gpu_scene[.INSTANCES](
            ctx, instance_data.scene_data()
        )
        enqueue_render_gpu(ctx, target, instances, settings)

        var combined_data = _combined_instance_world()
        var combined = prepare_gpu_scene[.ALL](
            ctx, combined_data.scene_data()
        )
        enqueue_render_gpu[.NORMALS](ctx, target, combined, settings)

        var pixels = download_gpu_pixels(ctx, target)
        assert_equal(len(pixels), 1)


def test_gpu_mixed_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(6, 4, 2, UInt64(307))
    var world = _mixed_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[.PATH, 4, 4](
        settings, camera, world.scene_data()
    )

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_instances_match_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(401))
    var world = _instance_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[.PATH, 4, 4](
        settings, camera, world.scene_data()
    )

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_instance_default_policy_keeps_micro_blas_wide() raises:
    var data = _instance_scene_data()
    assert_true(not _prefer_cwbvh8_blases(data))
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.55, 0.3, 0.75))
    var mesh = List[Point3f32[.LOCAL]]()
    for _ in range(32):
        mesh.append(Point3f32[.LOCAL](-1.25, -1.0, -1.0))
        mesh.append(Point3f32[.LOCAL](1.25, -1.0, -1.0))
        mesh.append(Point3f32[.LOCAL](0.0, 1.0, -1.0))
    var bounds = compute_bounds(mesh)
    _ = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].identity(),
        bounds,
        matte,
    )
    var large = builder^.finish()
    assert_true(_prefer_cwbvh8_blases(large))


def test_gpu_static_default_policy_keeps_micro_geometry_wide() raises:
    var data = _triangle_scene_data()
    assert_true(not _prefer_cwbvh8_triangles(data))
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.35, 0.65, 0.25))
    for _ in range(32):
        builder.add_triangle(
            Point3f32[.WORLD](-1.5, -1.0, -1.0),
            Point3f32[.WORLD](1.5, -1.0, -1.0),
            Point3f32[.WORLD](0.0, 1.0, -1.0),
            matte,
        )
    var large = builder^.finish()
    assert_true(_prefer_cwbvh8_triangles(large))


def test_gpu_default_hploc_cwbvh8_instances_match_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(401))
    var world = _instance_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_scene[
        kind=.INSTANCES,
        integrator=.PATH,
        tlas_policy=GpuRtBvhPolicy(2, 2, .LBVH, .WIDE),
    ](settings, camera, world.scene_data())

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_combined_instances_match_cpu_wavefront() raises:
    var settings = RenderSettings(7, 4, 2, UInt64(457))
    var world = _combined_instance_world()
    var camera = _camera()
    var cpu = render_wavefront[.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[.PATH, 4, 4](
        settings, camera, world.scene_data()
    )
    var gpu_hploc_tlas = render_gpu_scene[
        kind=.ALL,
        integrator=.PATH,
        tlas_policy=GpuRtBvhPolicy(2, 2, .HPLOC, .WIDE),
    ](settings, camera, world.scene_data())
    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)
        assert_almost_equal(
            gpu_hploc_tlas.pixels[i].x, cpu_pixel.x, atol=1.0e-5
        )
        assert_almost_equal(
            gpu_hploc_tlas.pixels[i].y, cpu_pixel.y, atol=1.0e-5
        )
        assert_almost_equal(
            gpu_hploc_tlas.pixels[i].z, cpu_pixel.z, atol=1.0e-5
        )


def test_gpu_combined_scene_instantiates_every_integrator() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(463))
    var world = _combined_instance_world()
    var camera = _camera()
    var normals = render_gpu[.NORMALS, 4, 4](
        settings, camera, world.scene_data()
    )
    var ao = render_gpu[.AO, 4, 4](settings, camera, world.scene_data())
    var nee = render_gpu[.NEE, 4, 4](settings, camera, world.scene_data())
    var mis = render_gpu[.MIS, 4, 4](settings, camera, world.scene_data())
    assert_equal(len(normals.pixels), 1)
    assert_equal(len(ao.pixels), 1)
    assert_equal(len(nee.pixels), 1)
    assert_equal(len(mis.pixels), 1)


def test_gpu_ao_matches_unoccluded_cpu_reference() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(509))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_depth_first[.AO, 1, 1, 0, 4, 8](
        settings, camera, world
    )
    var gpu = render_gpu[.AO, 4, 4](settings, camera, world.scene_data())
    assert_almost_equal(gpu.pixels[0].x, cpu.pixels[0].x, atol=1.0e-5)
    assert_almost_equal(gpu.pixels[0].y, cpu.pixels[0].y, atol=1.0e-5)
    assert_almost_equal(gpu.pixels[0].z, cpu.pixels[0].z, atol=1.0e-5)


def _test_gpu_direct_light_matches_cpu[integrator: Integrator]() raises:
    var settings = RenderSettings(3, 3, 2, UInt64(601))
    var world = make_cornell_world()
    var camera = _cornell_camera()
    var cpu = render_wavefront[integrator, 1, 64, False](settings, camera, world)
    var gpu = render_gpu[integrator](settings, camera, world.scene_data())
    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-4)


def test_gpu_nee_matches_cpu_wavefront() raises:
    _test_gpu_direct_light_matches_cpu[.NEE]()


def test_gpu_sphere_nee_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(619))
    var world = _material_sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[.NEE, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[.NEE, 4, 4](settings, camera, world.scene_data())
    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-4)


def _test_gpu_emissive_instance_matches_cpu[integrator: Integrator]() raises:
    var settings = RenderSettings(4, 3, 4, UInt64(631))
    var world = _emissive_instance_world()
    assert_equal(len(world.scene_data().lights().records), 2)
    for light in world.scene_data().lights().records:
        assert_equal(light.primitive.kind(), PrimitiveKind.TRIANGLE_INSTANCE)
    var camera = _emissive_instance_camera()
    var cpu = render_wavefront[integrator, 1, 64, False](settings, camera, world)
    var gpu = render_gpu[integrator, 4, 4](settings, camera, world.scene_data())
    var energy = Float32(0.0)
    for i, cpu_pixel in enumerate(cpu.pixels):
        energy += cpu_pixel.x + cpu_pixel.y + cpu_pixel.z
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-4)
    assert_true(energy > 0.0)


def test_gpu_emissive_instance_nee_matches_cpu_wavefront() raises:
    _test_gpu_emissive_instance_matches_cpu[.NEE]()


def test_gpu_emissive_instance_mis_matches_cpu_wavefront() raises:
    _test_gpu_emissive_instance_matches_cpu[.MIS]()


def test_gpu_mis_matches_cpu_wavefront() raises:
    _test_gpu_direct_light_matches_cpu[.MIS]()


def test_gpu_sphere_normals_render() raises:
    var settings = RenderSettings(3, 2, 1, UInt64(7))
    var scene = _sphere_scene_data()
    var result = render_gpu_scene[
        kind=.SPHERES,
        integrator=.NORMALS,
        sphere_policy=GPU_RT_BVH_WIDE4_LBVH,
    ](
        settings, _camera(), scene
    )
    assert_equal(len(result.pixels), 6)
    var nonzero = False
    for pixel in result.pixels:
        assert_true(pixel.x >= 0.0 and pixel.x <= 1.0)
        assert_true(pixel.y >= 0.0 and pixel.y <= 1.0)
        assert_true(pixel.z >= 0.0 and pixel.z <= 1.0)
        nonzero |= pixel.x > 0.0 or pixel.y > 0.0 or pixel.z > 0.0
    assert_true(nonzero)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
