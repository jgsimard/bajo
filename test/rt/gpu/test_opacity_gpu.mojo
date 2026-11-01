"""Device acceptance before wide/compressed hit reduction and shadow exit."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.core import Point3f32, Vec3f32, Rayf32
from bajo.bvh.types import Hit
from bajo.parser.pbrt import parse_pbrt, read_pbrt
from bajo.parser import MemoryAssetLoader
from bajo.rt import Color, CpuScene, render_wavefront
from bajo.rt.gpu import (
    prepare_gpu_scene,
    GpuRtRenderTarget,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from test.rt.helpers import assert_pixels_close
from bajo.rt.gpu.config import (
    GpuRtSceneKind,
    GpuRtBvhFormat,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_WIDE4,
    GPU_RT_BVH_TLAS4,
)
from bajo.rt.gpu.views import GpuRtSceneView
from bajo.rt.gpu.opacity import (
    trace_gpu_opacity_triangles,
    trace_gpu_opacity_instances,
    trace_gpu_opacity_spheres,
)
from test.parser.png.test_png import _encode
from test.rt.cpu.test_opacity import (
    _coplanar_textured_source,
    _textured_emitter_source,
)


def _opacity_trace_kernel[
    kind: GpuRtSceneKind, format: GpuRtBvhFormat, tlas_format: GpuRtBvhFormat
](scene: GpuRtSceneView, output: Pointer[Float32, MutAnyOrigin], count: Int32,):
    var index = global_idx.x
    if index >= Int(count):
        return
    var t_max = Float32(1.99) if index == 1 else Float32(100.0)
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.2 + Float32(index) / 8192.0, 0.3, 2),
        Vec3f32[.WORLD](0, 0, -1),
        1e-4,
        t_max,
    )
    var hit: Hit[.WORLD]
    var shadow: Hit[.WORLD]
    comptime if kind == .INSTANCES:
        hit = trace_gpu_opacity_instances[tlas_format, format, .CLOSEST_HIT](
            scene, ray
        )
        shadow = trace_gpu_opacity_instances[tlas_format, format, .ANY_HIT](
            scene, ray
        )
    elif kind == .TRIANGLES:
        hit = trace_gpu_opacity_triangles[format, .CLOSEST_HIT](scene, ray)
        shadow = trace_gpu_opacity_triangles[format, .ANY_HIT](scene, ray)
    else:
        hit = trace_gpu_opacity_spheres[format, .CLOSEST_HIT](scene, ray)
        shadow = trace_gpu_opacity_spheres[format, .ANY_HIT](scene, ray)
    output[unsafe_offset=6 * index] = Float32(hit.is_hit())
    output[unsafe_offset=6 * index + 1] = hit.t
    output[unsafe_offset=6 * index + 2] = hit.u
    output[unsafe_offset=6 * index + 3] = hit.v
    output[unsafe_offset=6 * index + 4] = Float32(shadow.is_occluded())
    output[unsafe_offset=6 * index + 5] = Float32(hit.inst)


def _check_device[
    kind: GpuRtSceneKind,
    format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS4,
](world: CpuScene[4, 4], count: Int = 3,) raises:
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[
            kind=kind,
            sphere_format=format if kind == .SPHERES else GPU_RT_BVH_WIDE4,
            triangle_format=format,
            blas_format=format,
            tlas_format=tlas_format,
        ](ctx, world.scene_data())
        var output = ctx.enqueue_create_buffer[.float32](6 * count)
        ctx.enqueue_function[_opacity_trace_kernel[kind, format, tlas_format]](
            scene.view(),
            output,
            Int32(count),
            grid_dim=(count + 127) // 128,
            block_dim=128,
        )
        ctx.synchronize()
        with output.map_to_host() as values:
            for index in range(count):
                var t_max = Float32(1.99) if index == 1 else Float32(100.0)
                var ray = Rayf32[.WORLD](
                    Point3f32[.WORLD](0.2 + Float32(index) / 8192.0, 0.3, 2),
                    Vec3f32[.WORLD](0, 0, -1),
                    1e-4,
                    t_max,
                )
                var expected = world.trace_surface(ray)
                assert_equal(values[6 * index], Float32(expected.hit))
                assert_equal(
                    values[6 * index + 4], Float32(world.occluded(ray))
                )
                if expected.hit:
                    assert_almost_equal(
                        values[6 * index + 1], expected.t[0], atol=1e-5
                    )
                    # GPU Hit UV fields are barycentrics; the public CPU hit is
                    # already interpolated. Their hit identity/visibility is the gate.


def test_coplanar_flat_opacity_on_wide_and_compressed_gpu_trees() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Shape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0] "float alpha" 0
Shape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0]
"""
    )
    var world = CpuScene[4, 4](parsed^.take_data())
    _check_device[.TRIANGLES, GPU_RT_BVH_WIDE4](world)
    _check_device[.TRIANGLES, GPU_RT_BVH_CWBVH8](world)


def test_coplanar_png_alpha_in_one_instance_on_gpu() raises:
    var loader = MemoryAssetLoader()
    loader.add_file("main.pbrt", _coplanar_textured_source())
    loader.add_image_file(
        "alpha.png",
        _encode([UInt8(0), 255, 0, 0, 0, 255, 0, 0, 255], 2, 1, color=6),
    )
    var parsed = read_pbrt("main.pbrt", loader)
    var world = CpuScene[4, 4](parsed^.take_data())
    _check_device[.INSTANCES, GPU_RT_BVH_WIDE4](world)
    _check_device[.INSTANCES, GPU_RT_BVH_CWBVH8](world)
    _check_device[.INSTANCES, GPU_RT_BVH_WIDE4, GPU_RT_BVH_WIDE4](world)
    _check_device[.INSTANCES, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_WIDE4](world)


def test_fractional_gpu_alpha_matches_cpu_per_ray() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Shape "trianglemesh" "point3 P" [-10 -10 0 10 -10 0 0 10 0] "float alpha" .25
"""
    )
    var world = CpuScene[4, 4](parsed^.take_data())
    _check_device[.TRIANGLES, GPU_RT_BVH_WIDE4](world, 1024)
    _check_device[.TRIANGLES, GPU_RT_BVH_CWBVH8](world, 1024)


def test_sphere_alpha_rejection_preserves_coplanar_opaque_sphere() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Shape "sphere" "float radius" .5 "float alpha" 0
Shape "sphere" "float radius" .5
"""
    )
    var world = CpuScene[4, 4](parsed^.take_data())
    _check_device[.SPHERES, GPU_RT_BVH_WIDE4](world)
    _check_device[.SPHERES, GpuRtBvhFormat(2, 1, .WIDE)](world)


def test_png_masked_emitters_match_cpu_and_do_not_emit_through_holes() raises:
    for mirrored in [False, True]:
        for opaque in [False, True]:
            var loader = MemoryAssetLoader()
            loader.add_file(
                "main.pbrt",
                _textured_emitter_source(".75" if opaque else ".25", mirrored),
            )
            loader.add_image_file(
                "alpha.png",
                _encode(
                    [UInt8(0), 255, 255, 255, 0, 255, 255, 255, 255],
                    2,
                    1,
                    color=6,
                ),
            )
            var parsed = read_pbrt("main.pbrt", loader)
            var settings = parsed.settings.copy()
            var camera = parsed.camera
            var world = CpuScene[4, 4](parsed^.take_data())
            var cpu = render_wavefront[.MIS, 4](settings, camera, world)
            with DeviceContext() as ctx:
                var scene = prepare_gpu_scene[kind=.INSTANCES](
                    ctx, world.scene_data()
                )
                var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 16)
                enqueue_render_gpu[.MIS](ctx, target, scene, settings)
                var gpu = download_gpu_pixels(ctx, target)
                assert_pixels_close(gpu, cpu.pixels, atol=1e-3, rtol=1e-3)
                var energy = Float32(0.0)
                for pixel in gpu:
                    energy += pixel.x + pixel.y + pixel.z
                if opaque:
                    assert_true(energy > 0.1)
                else:
                    assert_almost_equal(energy, 0.0, atol=1e-7)


def test_transparent_canopy_preserves_full_mis_render_and_shadow_visibility() raises:
    var expected = List[Color]()
    for masked in [False, True]:
        var source = """Film "rgb" "integer xresolution" 4 "integer yresolution" 3
Sampler "halton" "integer pixelsamples" 16
Integrator "path" "integer maxdepth" 3
LookAt 0 0 1.8 0 0 0 0 1 0
Camera "perspective" "float fov" 10
WorldBegin
LightSource "infinite" "rgb L" [.1 .1 .1]
"""
        if masked:
            source += """Shape "trianglemesh" "float alpha" 0
 "point3 P" [-10 -10 1 10 -10 1 0 10 1]
 "point2 uv" [0 0 1 0 0 1]
"""
        source += """Material "diffuse" "rgb reflectance" [.7 .4 .2]
Shape "trianglemesh" "point3 P" [-10 -10 0 10 -10 0 0 10 0]
 "point2 uv" [0 0 1 0 0 1]
AreaLightSource "diffuse" "rgb L" [4 4 4]
Shape "trianglemesh" "point3 P" [-1 -1 2 0 1 2 1 -1 2]
 "point2 uv" [0 0 1 0 0 1]
"""
        var parsed = parse_pbrt(source)
        var settings = parsed.settings.copy()
        var camera = parsed.camera
        var world = CpuScene[4, 4](parsed^.take_data())
        var cpu = render_wavefront[.MIS, 4](settings, camera, world)
        with DeviceContext() as ctx:
            var scene = prepare_gpu_scene[kind=.INSTANCES](
                ctx, world.scene_data()
            )
            var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 16)
            enqueue_render_gpu[.MIS](ctx, target, scene, settings)
            var gpu = download_gpu_pixels(ctx, target)
            assert_pixels_close(gpu, cpu.pixels, atol=1e-3, rtol=1e-3)
            if masked:
                assert_pixels_close(gpu, expected, atol=1e-4, rtol=1e-4)
            else:
                expected = gpu^


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
