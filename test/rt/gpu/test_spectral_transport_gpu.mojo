"""Actual GPU spectral transport, physical Canon film and graph reuse."""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from std.math import abs, sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core.color import Color
from bajo.core import Point3f32, Vec3, Affine3f32
from bajo.core.camera import Camera
from bajo.core.image import ImageTexture
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths
from bajo.parser.pbrt import parse_pbrt
from bajo.rt.film import FilmResponse
from bajo.rt.scene_data import SceneBuilder
from bajo.rt.material_types import Environment
from bajo.rt.render_types import RenderSettings
from bajo.rt.cpu import CpuScene, CpuWavefrontWorkspace
from bajo.rt.cpu.wavefront import render_wavefront_configured
from bajo.rt.gpu.prepared_renderer import GpuRtPreparedRenderer
from bajo.rt.gpu import (
    prepare_gpu_scene,
    enqueue_render_gpu,
    download_gpu_pixels,
)


def _canon_kernel[
    _device: Bool = True
](film: FilmResponse, output: Pointer[Float32, MutAnyOrigin]):
    if global_idx.x != 0:
        return
    var wavelengths = SampledWavelengths[1](
        SampledSpectrum[1](360, 460.3, 550.7, 830),
        SampledSpectrum[1](0.1, 0.2, 0.3, 0.4),
    )
    var value = film.apply_spectral_sample(
        SampledSpectrum[1](2, 3, 5, 7), wavelengths
    )
    comptime for channel in range(3):
        output[unsafe_offset=channel] = value[channel]


def test_gpu_canon_measured_response_and_complete_sample_clamp() raises:
    var film = FilmResponse(3, max_component_value=2)
    film.sensor_kind = .CANON_EOS_5D_MKIV
    film.white_balance_temperature = 3200
    film = film.prepared_spectral()
    var wavelengths = SampledWavelengths[1](
        SampledSpectrum[1](360, 460.3, 550.7, 830),
        SampledSpectrum[1](0.1, 0.2, 0.3, 0.4),
    )
    var expected = film.apply_spectral_sample(
        SampledSpectrum[1](2, 3, 5, 7), wavelengths
    )
    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[.float32](3)
        ctx.enqueue_function[_canon_kernel[True]](
            film, output, grid_dim=1, block_dim=1
        )
        with output.map_to_host() as actual:
            comptime for channel in range(3):
                assert_almost_equal(
                    actual[channel], expected[channel], atol=2e-5
                )


def test_gpu_spectral_primary_tail_chunking_graph_and_mode_changes() raises:
    var scene = parse_pbrt(
        'Film "rgb" "integer xresolution" [5] "integer yresolution" [3] "string'
        ' sensor" "canon_eos_5d_mkiv" Sampler "zsobol" "integer pixelsamples"'
        ' [17] LookAt 0 0 5 0 0 0 0 1 0 Camera "perspective" WorldBegin'
        ' LightSource "infinite" "rgb L" [.3 .7 .9] Translate 10 0 0 Shape'
        ' "sphere"'
    )
    var settings = scene.settings.copy()
    settings.color_mode = .SPECTRAL
    settings.max_depth = 0
    settings.sample_sequence_length = 34
    var camera = scene.camera
    var renderer = GpuRtPreparedRenderer[.SPHERES](settings, camera, scene.data)
    assert_true(not renderer.target.spectral)
    assert_true(not renderer.scene.materials.spectral.device)
    renderer.target = renderer.scene.make_render_target(
        renderer.ctx, settings, camera, path_capacity=34
    )
    var first = renderer.render[.MIS](settings, camera)
    var replay = renderer.render[.MIS](settings, camera)
    var world = CpuScene[16, 16](scene^.take_data())
    var workspace = CpuWavefrontWorkspace[4, .MIS]()
    var expected = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 17, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    settings.sample_offset = 17
    var later = renderer.render[.MIS](settings, camera)
    var expected_later = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 17, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    settings.sample_offset = 0
    for pixel in range(len(first.pixels)):
        comptime for channel in range(3):
            assert_almost_equal(
                first.pixels[pixel][channel],
                expected.pixels[pixel][channel],
                atol=2e-5,
            )
            assert_equal(
                first.pixels[pixel][channel], replay.pixels[pixel][channel]
            )
            assert_almost_equal(
                later.pixels[pixel][channel],
                expected_later.pixels[pixel][channel],
                atol=2e-5,
            )
    settings.color_mode = .RGB
    var rgb = renderer.render[.MIS](settings, camera)
    settings.color_mode = .SPECTRAL
    var returned = renderer.render[.MIS](settings, camera)
    assert_true(rgb.pixels[0].is_finite()[0])
    comptime for channel in range(3):
        assert_equal(first.pixels[0][channel], returned.pixels[0][channel])
    var preview = renderer.preview[.MIS](settings, camera, 1)
    settings.color_mode = .RGB
    var reset = renderer.preview[.MIS](settings, camera, 1)
    assert_equal(preview.batches, 1)
    assert_equal(reset.batches, 1)


def test_gpu_programmatic_environment_product_matches_cpu() raises:
    var builder = SceneBuilder()
    var material = builder.add_lambertian(Color(1))
    builder.add_sphere(Point3f32[.WORLD](100, 0, 0), 1, material)
    var texture = builder.surfaces.add_image_texture(
        ImageTexture(1, 1, [Float32(0), Float32(1), Float32(0)])
    )
    builder.set_environment(
        Environment.image(
            texture,
            Color(1, 0, 0),
            Affine3f32[.WORLD, .LOCAL].identity(),
            Affine3f32[.LOCAL, .WORLD].identity(),
        )
    )
    var data = builder^.finish()
    var settings = RenderSettings(
        2, 2, 256, 0, max_depth=0, color_mode=.SPECTRAL
    )
    var camera = Camera(
        Point3f32[.WORLD](0),
        Point3f32[.WORLD](0, 0, 1),
        Vec3[.float32, .WORLD](0, 1, 0),
        1,
    )
    var renderer = GpuRtPreparedRenderer[.SPHERES](settings, camera, data)
    var actual = renderer.render[.MIS](settings, camera)
    var world = CpuScene[16, 16](data^)
    var workspace = CpuWavefrontWorkspace[4, .MIS]()
    var expected = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 1024, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    for pixel in range(len(actual.pixels)):
        assert_true(actual.pixels[pixel].y > 0)
        comptime for channel in range(3):
            assert_almost_equal(
                actual.pixels[pixel][channel],
                expected.pixels[pixel][channel],
                atol=2e-5,
            )


def test_gpu_spectral_all_current_bsdfs_absorption_and_cpu_region_agreement() raises:
    var scene = parse_pbrt(
        'Film "rgb" "integer xresolution" [12] "integer yresolution" [8]'
        ' Sampler "zsobol" "integer pixelsamples" [64] LookAt 0 1 8 0 0 0 0 1 0'
        ' Camera "perspective" WorldBegin LightSource "infinite" "rgb L" [.4 .6'
        ' .9] AttributeBegin AreaLightSource "diffuse" "blackbody L" [4500]'
        ' "float scale" [2] Translate 0 5 0 Shape "sphere" AttributeEnd'
        ' AttributeBegin Material "diffuse" "spectrum reflectance" [360 .2 830'
        ' .8] Translate -2.3 0 0 Shape "sphere" AttributeEnd AttributeBegin'
        ' Material "metal" "rgb reflectance" [.4 .7 .8] "float roughness" [.3]'
        ' Translate -.8 0 0 Shape "sphere" "float radius" [.6] AttributeEnd'
        ' MakeNamedMedium "absorption" "string type" "homogeneous" "spectrum'
        ' sigma_a" [360 .01 830 .7] AttributeBegin MediumInterface "absorption"'
        ' "" Material "dielectric" "spectrum eta" [360 1.6 830 1.4] "float'
        ' roughness" [.1] Translate .8 0 0 Shape "sphere" "float radius" [.7]'
        ' AttributeEnd AttributeBegin Material "coatedconductor" "spectrum'
        ' conductor.eta" "metal-Cu-eta" "spectrum conductor.k" "metal-Cu-k"'
        ' "float interface.roughness" [.2] Translate 2.3 0 0 Shape "sphere"'
        ' AttributeEnd AttributeBegin Material "coateddiffuse" "rgb'
        ' reflectance" [.6 .4 .2] "rgb albedo" [.1 .2 .3] "float roughness"'
        ' [.2] Translate 0 -2 0 Shape "sphere" AttributeEnd'
    )
    var settings = scene.settings.copy()
    settings.color_mode = .SPECTRAL
    settings.max_depth = 3
    var camera = scene.camera
    var renderer = GpuRtPreparedRenderer[.SPHERES](settings, camera, scene.data)
    var world = CpuScene[16, 16](scene^.take_data())
    var workspace = CpuWavefrontWorkspace[4, .MIS]()
    var cpu_means = List[Color]()
    var gpu_means = List[Color]()
    var cpu_total = Color(0)
    var gpu_total = Color(0)
    for seed in range(8):
        settings.rng_seed = UInt64(seed)
        var cpu = render_wavefront_configured[
            .AUTO_COHERENT,
            .MIS,
            4,
            1024,
            False,
            .TASK_PARTITIONS,
            16,
            16,
            16,
            8,
            4,
        ](settings, camera, world, workspace)
        var gpu = renderer.render[.MIS](settings, camera)
        var a = Color(0)
        var b = Color(0)
        for pixel in range(len(cpu.pixels)):
            assert_true(cpu.pixels[pixel].is_finite()[0])
            assert_true(gpu.pixels[pixel].is_finite()[0])
            a += cpu.pixels[pixel]
            b += gpu.pixels[pixel]
        a = a / Float32(len(cpu.pixels))
        b = b / Float32(len(gpu.pixels))
        cpu_means.append(a)
        gpu_means.append(b)
        cpu_total += a
        gpu_total += b
    cpu_total = cpu_total / 8
    gpu_total = gpu_total / 8
    comptime for channel in range(3):
        var variance = Float32(0)
        for seed in range(8):
            var a = cpu_means[seed][channel] - cpu_total[channel]
            var b = gpu_means[seed][channel] - gpu_total[channel]
            variance += a * a + b * b
        var tolerance = max(
            Float32(0.02) * cpu_total.max_component(), 4 * sqrt(variance / 56)
        )
        assert_almost_equal(
            cpu_total[channel], gpu_total[channel], atol=Float64(tolerance)
        )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
