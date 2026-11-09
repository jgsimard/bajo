"""Spectral authoring, physical sensors and complete CPU transport."""

from std.builtin.globals import global_constant
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from bajo.core import Vec3, Point3f32, Affine3f32, dot
from bajo.core.camera import Camera
from bajo.core.image import ImageTexture
from bajo.core.color import Color
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    cie_xyz,
)
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.core.optics import fresnel_conductor
from bajo.parser.pbrt import parse_pbrt
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.cpu.wavefront import (
    render_wavefront,
    render_wavefront_configured,
    CpuWavefrontWorkspace,
)
from bajo.rt.cpu.spectral_resources import CpuSpectralResources
from bajo.rt.render_types import RenderSettings
from bajo.rt.scene_data import SceneBuilder
from bajo.rt.material_types import Environment
from bajo.rt.film import FilmResponse
from bajo.rt.sensor import FilmSensorKind, sensor_to_xyz, canon_sensor_sample
from bajo.rt.spectrum_sources import blackbody_spectrum, SpectrumSource

# Independent NumPy references computed from the original PBRT sources.
# Original spectrum.cpp SHA-256: 4c907c7655ad5431bd2797692b963ac34bc6a533c7a053a16e6988aea03c750b
# Original film.cpp SHA-256: 745740d09fe985eba62874712df982746eb132b0e3a76ade72310b6bb790192c
comptime SENSOR_REFERENCE_TEMPERATURES: Array[Float32, 3] = [3200, 6500, 9000]
comptime SENSOR_REFERENCE_MATRICES: Array[Array[Float64, 9], 3] = [
    [
        1.15905727171846,
        0.158794873070181,
        0.433626412785519,
        0.26949683883968,
        1.32111079632277,
        -0.516104029635075,
        0.138457570445982,
        -1.13128503548223,
        4.6556286197457,
    ],
    [
        1.88637526748162,
        -0.0690457916557337,
        0.248101644982663,
        0.744239312163809,
        0.967555885355727,
        -0.400997738799006,
        0.178473417622179,
        -0.582058265487783,
        2.06636051444375,
    ],
    [
        2.12237400141953,
        -0.123481317692531,
        0.218666541002477,
        0.908458359710631,
        0.882968372122977,
        -0.36244346512294,
        0.164383137965073,
        -0.472991190629949,
        1.65277381629226,
    ],
]
comptime CANON_REFERENCE_SAMPLE = Color(
    0.844700267604, 3.53093047806, 3.36564651765
)

comptime ANALYTICAL_DISPERSIVE_ABSORPTION = Color(
    0.562153081528, 0.533530727064, 0.634357256217
)
comptime ANALYTICAL_CONDUCTOR_FRESNEL = SIMD[.float32, 4](
    0.940831565476, 0.850699950995, 0.237057430566, 0.139974757253
)


def test_spectral_conductor_fresnel_matches_independent_complex_reference() raises:
    comptime for width in [1, 4, 16]:
        var fresnel = fresnel_conductor(
            SIMD[.float32, width](0.6),
            SampledSpectrum[width](0.18, 0.35, 1.4, 1.8),
            SampledSpectrum[width](3.4, 2.8, 1.1, 0.5),
        )
        comptime for channel in range(4):
            for lane in range(width):
                assert_almost_equal(
                    fresnel[channel][lane],
                    ANALYTICAL_CONDUCTOR_FRESNEL[channel],
                    atol=1e-6,
                )


def test_measured_canon_sensor_and_calibration_match_original_source_references() raises:
    ref temperatures = global_constant[SENSOR_REFERENCE_TEMPERATURES]()
    ref matrices = global_constant[SENSOR_REFERENCE_MATRICES]()
    for index in range(3):
        var matrix = sensor_to_xyz(.CANON_EOS_5D_MKIV, temperatures[index])
        for row in range(3):
            for column in range(3):
                assert_almost_equal(
                    matrix[row][column],
                    matrices[index][3 * row + column],
                    atol=2e-5,
                )
    comptime for width in [1, 4, 16]:
        var wavelengths = SampledWavelengths[width](
            SampledSpectrum[width](360, 460.3, 550.7, 830),
            SampledSpectrum[width](0.1, 0.2, 0.3, 0.4),
        )
        var response = canon_sensor_sample(
            SampledSpectrum[width](2, 3, 5, 7), wavelengths
        )
        for lane in range(width):
            comptime for channel in range(3):
                assert_almost_equal(
                    response[channel][lane],
                    CANON_REFERENCE_SAMPLE[channel],
                    atol=2e-6,
                )
        wavelengths.pdf = SampledSpectrum[width]()
        assert_equal(
            canon_sensor_sample(
                SampledSpectrum[width](1), wavelengths
            ).max_component()[0],
            Float32(0),
        )


def test_complete_sample_uses_final_hero_pdf_before_sensor_clamping() raises:
    var film = FilmResponse(2, max_component_value=3)
    film.use_cie1931()
    var wavelengths = SampledWavelengths[1].visible(0.3)
    var old_pdf = wavelengths.pdf.s0
    var earlier = SampledSpectrum[1](100, 20, 30, 40)
    wavelengths.terminate_secondary(True)
    wavelengths.terminate_secondary(True)
    var complete = earlier + SampledSpectrum[1](2, 0, 0, 0)
    var rgb = film.apply_spectral_sample(complete, wavelengths)
    var sensor = cie_xyz(wavelengths.lambda_nm.s0) * (102 * 2 / old_pdf)
    sensor = sensor * (3 / sensor.max_component())
    var expected = Color(
        dot(film.xyz_red, sensor),
        dot(film.xyz_green, sensor),
        dot(film.xyz_blue, sensor),
    )
    comptime for channel in range(3):
        assert_almost_equal(rgb[channel], expected[channel], atol=2e-6)


def test_authored_spectral_products_and_blackbodies_are_retained() raises:
    var parsed = parse_pbrt(
        'Texture "a" "spectrum" "constant" "rgb value" [.2 .4 .8] Texture "b"'
        ' "spectrum" "constant" "spectrum value" [360 .1 830 .9] Texture "ab"'
        ' "spectrum" "scale" "texture tex" "a" "texture scale" "b" WorldBegin'
        ' Material "diffuse" "texture reflectance" "ab" Shape "sphere"'
    )
    var resources = CpuSpectralResources(parsed.data.surfaces())
    ref surfaces = parsed.data.surfaces()
    var source = surfaces.albedo_spectra[
        parsed.data.sphere_surfaces()[0].value[0]
    ]
    assert_true(
        surfaces.spectrum_sources[Int(source)].kind == SpectrumSource.PRODUCT
    )
    var wavelengths = SampledWavelengths[1](
        SampledSpectrum[1](400, 500, 600, 700), SampledSpectrum[1](1)
    )
    var value = resources.view().sample(
        source, wavelengths, RgbSpectrumView.ALBEDO
    )
    var rgb = resources.rgb.view().sample(
        Color(0.2, 0.4, 0.8), wavelengths, RgbSpectrumView.ALBEDO
    )
    comptime for channel in range(4):
        var curve = 0.1 + 0.8 * (wavelengths.lambda_nm[channel] - 360) / 470
        assert_almost_equal(value[channel], rgb[channel] * curve, atol=1e-6)
    var peak = Float32(2.8977721e6 / 6500)
    wavelengths.lambda_nm = SampledSpectrum[1](peak)
    assert_almost_equal(
        blackbody_spectrum(SIMD[.float32, 1](6500), wavelengths).s0,
        Float32(1),
        atol=2e-5,
    )
    _ = resources^


def test_spectral_scattering_between_legacy_wavelengths_is_rejected() raises:
    with assert_raises(contains="scattering"):
        _ = parse_pbrt(
            'WorldBegin MakeNamedMedium "bad" "string type" "homogeneous"'
            ' "spectrum sigma_s" [400 0 405 1 410 0] Shape "sphere"'
        )
    with assert_raises(contains="constant"):
        _ = parse_pbrt(
            'WorldBegin Material "coateddiffuse" "spectrum eta" [360 1.6 830'
            ' 1.4] Shape "sphere"'
        )


def test_reused_workspace_refreshes_bindings_for_replaced_scene() raises:
    var workspace = CpuWavefrontWorkspace[4, .MIS]()
    for iteration in range(2):
        var light = String("360 .8 830 .2") if iteration == 0 else String(
            "360 .2 830 .9"
        )
        var scene = parse_pbrt(
            'Film "rgb" "integer xresolution" [2] "integer yresolution" [2]'
            ' Sampler "zsobol" "integer pixelsamples" [64] WorldBegin'
            ' LightSource "infinite" "spectrum L" ['
            + light
            + '] Translate 100 0 0 Shape "sphere"'
        )
        var settings = scene.settings.copy()
        settings.color_mode = .SPECTRAL
        settings.max_depth = 0
        var camera = scene.camera
        var world = CpuScene[16, 16](scene^.take_data())
        var reused = render_wavefront_configured[
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
        var fresh = render_wavefront[.MIS, 4, PARALLEL=False](
            settings, camera, world
        )
        comptime for channel in range(3):
            assert_almost_equal(
                reused.pixels[0][channel], fresh.pixels[0][channel], atol=1e-6
            )
        var invalid = SceneBuilder()
        var material = invalid.add_lambertian(Color(1))
        invalid.add_sphere(Point3f32[.WORLD](100, 0, 0), 1, material)
        invalid.surfaces.spectrum_sources.append(SpectrumSource.product(99, 99))
        var invalid_world = CpuScene[16, 16](invalid^.finish())
        with assert_raises(contains="earlier sources"):
            _ = render_wavefront_configured[
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
            ](settings, camera, invalid_world, workspace)
        var recovered = render_wavefront_configured[
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
        comptime for channel in range(3):
            assert_almost_equal(
                recovered.pixels[0][channel],
                fresh.pixels[0][channel],
                atol=1e-6,
            )


def test_spectral_cpu_width_chunk_batch_and_mode_invariance() raises:
    var scene = parse_pbrt(
        'Film "rgb" "integer xresolution" [5] "integer yresolution" [3] Sampler'
        ' "zsobol" "integer pixelsamples" [32] WorldBegin LightSource'
        ' "infinite" "rgb L" [.8 .6 .4] Material "diffuse" "spectrum'
        ' reflectance" [360 .2 830 .8] Translate 0 0 3 Shape "sphere"'
    )
    var camera = scene.camera
    var settings = scene.settings.copy()
    settings.max_depth = 3
    settings.color_mode = .SPECTRAL
    var world = CpuScene[16, 16](scene^.take_data())
    var scalar = render_wavefront[.MIS, 1, CHUNK_PATHS=32, PARALLEL=False](
        settings, camera, world
    )
    var wide = render_wavefront[.MIS, 4, CHUNK_PATHS=96, PARALLEL=False](
        settings, camera, world
    )
    for pixel in range(len(scalar.pixels)):
        assert_true(wide.pixels[pixel].is_finite()[0])
        comptime for channel in range(3):
            assert_almost_equal(
                scalar.pixels[pixel][channel],
                wide.pixels[pixel][channel],
                atol=1e-4,
            )
    var first = settings.copy()
    first.samples_per_pixel = 16
    var second = first.copy()
    second.sample_offset = 16
    var a = render_wavefront[.MIS, 4, PARALLEL=False](first, camera, world)
    var b = render_wavefront[.MIS, 4, PARALLEL=False](second, camera, world)
    for pixel in range(len(wide.pixels)):
        comptime for channel in range(3):
            assert_almost_equal(
                wide.pixels[pixel][channel],
                (a.pixels[pixel][channel] + b.pixels[pixel][channel]) * 0.5,
                atol=1e-5,
            )
    settings.color_mode = .RGB
    var rgb = render_wavefront[.MIS, 4, PARALLEL=False](settings, camera, world)
    assert_true(rgb.pixels[0].is_finite()[0])


def test_narrow_spectral_emission_has_visible_sampling_support() raises:
    var scene = parse_pbrt(
        'Film "rgb" "integer xresolution" [4] "integer yresolution" [4] Sampler'
        ' "zsobol" "integer pixelsamples" [256] LookAt 0 0 0 0 0 1 0 1 0 Camera'
        ' "perspective" WorldBegin AreaLightSource "diffuse" "spectrum L" [399'
        ' 0 400 1 401 0] Translate 0 0 101 Shape "sphere" "float radius" [100]'
    )
    assert_true(scene.data.total_light_weight() > 0)
    var settings = scene.settings.copy()
    settings.color_mode = .SPECTRAL
    settings.max_depth = 0
    var camera = scene.camera
    var world = CpuScene[16, 16](scene^.take_data())
    var image = render_wavefront[.MIS, 4, PARALLEL=False](
        settings, camera, world
    )
    var blue = Float32(0)
    for pixel in image.pixels:
        assert_true(pixel.is_finite()[0])
        blue += pixel.z
    assert_true(blue > 0)


def test_programmatic_environment_preserves_spectral_texture_product() raises:
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
    var settings = RenderSettings(2, 2, 256, 0)
    settings.max_depth = 0
    settings.color_mode = .SPECTRAL
    var camera = Camera(
        Point3f32[.WORLD](0),
        Point3f32[.WORLD](0, 0, 1),
        Vec3[.float32, .WORLD](0, 1, 0),
        1,
    )
    var world = CpuScene[16, 16](builder^.finish())
    var workspace = CpuWavefrontWorkspace[4, .MIS]()
    var image = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 1024, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    var energy = Float32(0)
    for pixel in image.pixels:
        assert_true(pixel.is_finite()[0])
        energy += pixel.y
    assert_true(energy > 0)


def test_partial_dielectric_curve_extends_endpoints_in_spectral_mode() raises:
    var scene = parse_pbrt(
        'WorldBegin Material "dielectric" "spectrum eta" [400 1.6 700 1.4]'
        ' Shape "sphere"'
    )
    var resources = CpuSpectralResources(scene.data.surfaces())
    var source = UInt32(0)
    for entry in scene.data.surfaces().dielectric_eta_sources.items():
        source = entry.value
    var wavelengths = SampledWavelengths[1](
        SampledSpectrum[1](360, 400, 700, 830), SampledSpectrum[1](1)
    )
    var eta = resources.view().sample(
        source, wavelengths, RgbSpectrumView.UNBOUNDED
    )
    assert_almost_equal(eta.s0, Float32(1.6), atol=1e-6)
    assert_almost_equal(eta.s3, Float32(1.4), atol=1e-6)
    _ = resources^


def test_dispersive_absorbing_sphere_matches_independent_analytic_series() raises:
    var scene = parse_pbrt(
        'Film "rgb" "integer xresolution" [1] "integer yresolution" [1] Sampler'
        ' "zsobol" "integer pixelsamples" [32768] PixelFilter "box" LookAt 0 0'
        ' 3 0 0 0 0 1 0 Camera "perspective" "float fov" [.000001] WorldBegin'
        ' LightSource "infinite" "spectrum L" [360 1 830 1] MakeNamedMedium "a"'
        ' "string type" "homogeneous" "spectrum sigma_a" [360 .1 830 .6] "rgb'
        ' sigma_s" [0 0 0] MediumInterface "a" "" Material "dielectric"'
        ' "spectrum eta" [360 1.65 830 1.4] Shape "sphere"'
    )
    var camera = scene.camera
    var settings = scene.settings.copy()
    settings.color_mode = .SPECTRAL
    settings.max_depth = 4
    var world = CpuScene[16, 16](scene^.take_data())
    var total = Color(0)
    for seed in range(8):
        settings.rng_seed = UInt64(seed)
        var image = render_wavefront[.MIS, 4, PARALLEL=False](
            settings, camera, world
        )
        total += image.pixels[0] / 8
    comptime for channel in range(3):
        assert_almost_equal(
            total[channel], ANALYTICAL_DISPERSIVE_ABSORPTION[channel], atol=0.01
        )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
