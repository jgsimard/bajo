from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from max.gpu.host import DeviceContext

from bajo.bvh.camera import Camera
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Instance, Sphere
from bajo.core import Affine3f32, Frame, Point3f32, Vec3f32
from bajo.rt.cpu import render_depth_first, render_wavefront
from bajo.rt.gpu import (
    GPU_RT_BVH_WIDE4_LBVH,
    GpuRtRenderTarget,
    GpuRtTriangleScene,
    download_gpu_pixels,
    enqueue_render_gpu,
    enqueue_render_gpu_triangles,
    prepare_gpu_combined_instance_scene,
    prepare_gpu_mixed_scene,
    prepare_gpu_sphere_scene,
    prepare_gpu_triangle_instance_scene,
    prepare_gpu_triangle_scene,
    render_gpu,
    render_gpu_combined_instances,
    render_gpu_spheres,
    render_gpu_triangles,
    render_gpu_triangle_instances,
)
from bajo.rt.gpu.render import (
    _prefer_cwbvh8_blases,
    _prefer_cwbvh8_triangles,
)
from bajo.rt.types import (
    Color,
    CpuScene,
    RENDER,
    RenderSettings,
    SceneData,
    SurfaceId,
    SurfaceStore,
    World,
    add_sphere,
    add_triangle,
    add_triangle_mesh_instance,
)
from examples.cornell_box import make_cornell_world


def _sphere_scene_data() -> SceneData:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.5, 0.6, 0.7))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, -100.5, -1.0),
        100.0,
        matte,
    )
    return SceneData(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def _sphere_world() -> CpuScene[4, 8]:
    return CpuScene[4, 8](_sphere_scene_data())


def _material_sphere_world() -> World[4, 8]:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var ground = surfaces.add_lambertian(Color(0.45, 0.45, 0.45))
    var metal = surfaces.add_metal(Color(0.8, 0.65, 0.3), 0.2)
    var glass = surfaces.add_dielectric(1.5)
    var light = surfaces.add_emissive(Color(3.0, 2.0, 1.0))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, -100.5, -1.5),
        100.0,
        ground,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](-0.7, 0.0, -1.5),
        0.45,
        metal,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.25, 0.0, -1.25),
        0.45,
        glass,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.9, 0.35, -1.5),
        0.3,
        light,
    )
    return World[4, 8](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def _triangle_world() -> World[4, 8]:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.35, 0.65, 0.25))
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-1.5, -1.0, -1.0),
        Point3f32[Frame.WORLD](1.5, -1.0, -1.0),
        Point3f32[Frame.WORLD](1.5, 1.0, -1.0),
        matte,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-1.5, -1.0, -1.0),
        Point3f32[Frame.WORLD](1.5, 1.0, -1.0),
        Point3f32[Frame.WORLD](-1.5, 1.0, -1.0),
        matte,
    )
    return World[4, 8](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def _mixed_world() -> World[4, 8]:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var sphere_matte = surfaces.add_lambertian(Color(0.7, 0.25, 0.2))
    var back_matte = surfaces.add_lambertian(Color(0.2, 0.35, 0.7))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.4,
        sphere_matte,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-2.0, -1.5, -2.0),
        Point3f32[Frame.WORLD](2.0, -1.5, -2.0),
        Point3f32[Frame.WORLD](2.0, 1.5, -2.0),
        back_matte,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-2.0, -1.5, -2.0),
        Point3f32[Frame.WORLD](2.0, 1.5, -2.0),
        Point3f32[Frame.WORLD](-2.0, 1.5, -2.0),
        back_matte,
    )
    return World[4, 8](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def _instance_world() -> World[4, 8]:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.55, 0.3, 0.75))
    var mesh = List[Point3f32[Frame.LOCAL]]()
    mesh.append(Point3f32[Frame.LOCAL](-1.25, -1.0, -1.0))
    mesh.append(Point3f32[Frame.LOCAL](1.25, -1.0, -1.0))
    mesh.append(Point3f32[Frame.LOCAL](0.0, 1.0, -1.0))
    var bounds = compute_bounds(mesh)
    _ = add_triangle_mesh_instance(
        triangle_meshes,
        triangle_instances,
        triangle_instance_surfaces,
        mesh,
        Affine3f32[Frame.LOCAL, Frame.WORLD].identity(),
        bounds,
        matte,
    )
    return World[4, 8](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def _combined_instance_world() -> World[4, 8]:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var red = surfaces.add_lambertian(Color(0.7, 0.2, 0.2))
    var blue = surfaces.add_lambertian(Color(0.2, 0.3, 0.7))
    var green = surfaces.add_lambertian(Color(0.2, 0.7, 0.3))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](-0.55, 0.0, -1.1),
        0.3,
        red,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-2.0, -1.2, -2.0),
        Point3f32[Frame.WORLD](2.0, -1.2, -2.0),
        Point3f32[Frame.WORLD](0.0, 1.5, -2.0),
        blue,
    )
    var mesh = List[Point3f32[Frame.LOCAL]]()
    mesh.append(Point3f32[Frame.LOCAL](-0.25, -0.35, -1.0))
    mesh.append(Point3f32[Frame.LOCAL](0.25, -0.35, -1.0))
    mesh.append(Point3f32[Frame.LOCAL](0.0, 0.35, -1.0))
    var bounds = compute_bounds(mesh)
    _ = add_triangle_mesh_instance(
        triangle_meshes,
        triangle_instances,
        triangle_instance_surfaces,
        mesh,
        Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(
            Vec3f32[Frame.WORLD](0.55, 0.0, 0.0)
        ),
        bounds,
        green,
    )
    return World[4, 8](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 0.0, 0.0),
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        90.0,
    )


def _cornell_camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 1.0, 3.2),
        Point3f32[Frame.WORLD](0.0, 1.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        28.0,
        4.2,
    )


def test_gpu_sphere_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(4, 3, 2, UInt64(91))
    var world = _sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_spheres[RENDER.PATH, 4, 4](
        settings, camera, world.scene
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
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_spheres[RENDER.PATH, 4, 4](
        settings, camera, world.scene
    )

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(211))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_triangles[RENDER.PATH, 4, 4](
        settings, camera, world.scene
    )

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_hploc_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(223))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_triangles[
        RENDER.PATH,
        4,
        4,
        GpuBvhBuildMethod.HPLOC,
    ](settings, camera, world.scene)

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_default_hploc_cwbvh8_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(227))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_triangles[
        RENDER.PATH,
        8,
        4,
    ](settings, camera, world.scene)

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_chunked_triangle_nee_matches_full_capacity() raises:
    var settings = RenderSettings(6, 4, 2, UInt64(229))
    var world = _triangle_world()
    var camera = _camera()
    with DeviceContext() as ctx:
        var gpu_world = GpuRtTriangleScene[4, 4](ctx, world.scene)
        var full = GpuRtRenderTarget(
            ctx,
            settings,
            camera,
            settings.image_width
            * settings.image_height
            * settings.samples_per_pixel,
        )
        enqueue_render_gpu_triangles[RENDER.NEE, 4, 4](
            ctx, full, gpu_world, settings
        )
        var full_pixels = download_gpu_pixels(ctx, full)

        # Two pixels per chunk exercises global path IDs and disjoint resolve.
        var chunked = GpuRtRenderTarget(
            ctx, settings, camera, 2 * settings.samples_per_pixel
        )
        enqueue_render_gpu_triangles[RENDER.NEE, 4, 4](
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
        var scene = prepare_gpu_triangle_scene[GPU_RT_BVH_WIDE4_LBVH](
            ctx, world.scene
        )
        var target = GpuRtRenderTarget(ctx, settings, _camera())
        enqueue_render_gpu[RENDER.NORMALS](ctx, target, scene, settings)
        var pixels = download_gpu_pixels(ctx, target)
        assert_equal(len(pixels), settings.image_width * settings.image_height)


def test_common_prepared_api_instantiates_every_scene_kind() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(239), 1)
    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, _camera())

        var sphere_data = _sphere_world()
        var spheres = prepare_gpu_sphere_scene(ctx, sphere_data.scene)
        enqueue_render_gpu(ctx, target, spheres, settings)

        var mixed_data = _mixed_world()
        var mixed = prepare_gpu_mixed_scene(ctx, mixed_data.scene)
        enqueue_render_gpu(ctx, target, mixed, settings)

        var instance_data = _instance_world()
        var instances = prepare_gpu_triangle_instance_scene(
            ctx, instance_data.scene
        )
        enqueue_render_gpu(ctx, target, instances, settings)

        var combined_data = _combined_instance_world()
        var combined = prepare_gpu_combined_instance_scene[True, True](
            ctx, combined_data.scene
        )
        enqueue_render_gpu[RENDER.NORMALS](ctx, target, combined, settings)

        var pixels = download_gpu_pixels(ctx, target)
        assert_equal(len(pixels), 1)


def test_gpu_mixed_path_matches_cpu_wavefront() raises:
    var settings = RenderSettings(6, 4, 2, UInt64(307))
    var world = _mixed_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[RENDER.PATH, 4, 4](settings, camera, world.scene)

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_triangle_instances_match_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(401))
    var world = _instance_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[RENDER.PATH, 4, 4](settings, camera, world.scene)

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_instance_default_policy_keeps_micro_blas_wide() raises:
    var world = _instance_world()
    assert_true(not _prefer_cwbvh8_blases(world.scene))
    for _ in range(31):
        world.scene.triangle_meshes[0].append(
            Point3f32[Frame.LOCAL](-1.25, -1.0, -1.0)
        )
        world.scene.triangle_meshes[0].append(
            Point3f32[Frame.LOCAL](1.25, -1.0, -1.0)
        )
        world.scene.triangle_meshes[0].append(
            Point3f32[Frame.LOCAL](0.0, 1.0, -1.0)
        )
    assert_true(_prefer_cwbvh8_blases(world.scene))


def test_gpu_static_default_policy_keeps_micro_geometry_wide() raises:
    var world = _triangle_world()
    assert_true(not _prefer_cwbvh8_triangles(world.scene))
    for _ in range(30):
        world.scene.triangle_vertices.append(
            Point3f32[Frame.WORLD](-1.5, -1.0, -1.0)
        )
        world.scene.triangle_vertices.append(
            Point3f32[Frame.WORLD](1.5, -1.0, -1.0)
        )
        world.scene.triangle_vertices.append(
            Point3f32[Frame.WORLD](0.0, 1.0, -1.0)
        )
    assert_true(_prefer_cwbvh8_triangles(world.scene))


def test_gpu_default_hploc_cwbvh8_instances_match_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(401))
    var world = _instance_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu_triangle_instances[
        RENDER.PATH,
        2,
        2,
        8,
        4,
    ](settings, camera, world.scene)

    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-5)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-5)


def test_gpu_combined_instances_match_cpu_wavefront() raises:
    var settings = RenderSettings(7, 4, 2, UInt64(457))
    var world = _combined_instance_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.PATH, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[RENDER.PATH, 4, 4](settings, camera, world.scene)
    var gpu_hploc_tlas = render_gpu_combined_instances[
        RENDER.PATH,
        True,
        True,
        4,
        4,
        2,
        2,
        8,
        4,
        GpuBvhBuildMethod.HPLOC,
        True,
        8,
        4,
        GpuBvhBuildMethod.HPLOC,
        True,
        GpuBvhBuildMethod.HPLOC,
    ](settings, camera, world.scene)
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


def test_gpu_combined_scene_instantiates_every_algorithm() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(463))
    var world = _combined_instance_world()
    var camera = _camera()
    var normals = render_gpu[RENDER.NORMALS, 4, 4](
        settings, camera, world.scene
    )
    var ao = render_gpu[RENDER.AO, 4, 4](settings, camera, world.scene)
    var nee = render_gpu[RENDER.NEE, 4, 4](settings, camera, world.scene)
    var mis = render_gpu[RENDER.MIS, 4, 4](settings, camera, world.scene)
    assert_equal(len(normals.pixels), 1)
    assert_equal(len(ao.pixels), 1)
    assert_equal(len(nee.pixels), 1)
    assert_equal(len(mis.pixels), 1)


def test_gpu_ao_matches_unoccluded_cpu_reference() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(509))
    var world = _triangle_world()
    var camera = _camera()
    var cpu = render_depth_first[RENDER.AO, 1, 1, 0, 4, 8](
        settings, camera, world
    )
    var gpu = render_gpu[RENDER.AO, 4, 4](settings, camera, world.scene)
    assert_almost_equal(gpu.pixels[0].x, cpu.pixels[0].x, atol=1.0e-5)
    assert_almost_equal(gpu.pixels[0].y, cpu.pixels[0].y, atol=1.0e-5)
    assert_almost_equal(gpu.pixels[0].z, cpu.pixels[0].z, atol=1.0e-5)


def _test_gpu_direct_light_matches_cpu[ALGORITHM: RENDER]() raises:
    var settings = RenderSettings(3, 3, 2, UInt64(601))
    var world = make_cornell_world()
    var camera = _cornell_camera()
    var cpu = render_wavefront[ALGORITHM, 1, 64, False](settings, camera, world)
    var gpu = render_gpu[ALGORITHM](settings, camera, world.scene)
    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-4)


def test_gpu_nee_matches_cpu_wavefront() raises:
    _test_gpu_direct_light_matches_cpu[RENDER.NEE]()


def test_gpu_sphere_nee_matches_cpu_wavefront() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(619))
    var world = _material_sphere_world()
    var camera = _camera()
    var cpu = render_wavefront[RENDER.NEE, 1, 64, False](
        settings, camera, world
    )
    var gpu = render_gpu[RENDER.NEE, 4, 4](settings, camera, world.scene)
    for i, cpu_pixel in enumerate(cpu.pixels):
        assert_almost_equal(gpu.pixels[i].x, cpu_pixel.x, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].y, cpu_pixel.y, atol=1.0e-4)
        assert_almost_equal(gpu.pixels[i].z, cpu_pixel.z, atol=1.0e-4)


def test_gpu_mis_matches_cpu_wavefront() raises:
    _test_gpu_direct_light_matches_cpu[RENDER.MIS]()


def test_gpu_sphere_normals_render() raises:
    var settings = RenderSettings(3, 2, 1, UInt64(7))
    var scene = _sphere_scene_data()
    var result = render_gpu_spheres[RENDER.NORMALS, 4, 4](
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
