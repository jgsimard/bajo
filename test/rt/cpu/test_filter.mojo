"""PBRT Gaussian sampling, reconstruction footprints, and film/filter order."""

from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
    assert_raises,
)
from bajo.rt.filter import ReconstructionFilter, FilterSampler
from bajo.rt import CpuScene, render_wavefront
from bajo.rt.material_types import Integrator
from bajo.rt.film import FilmResponse
from bajo.rt.render_types import RenderSettings, SamplingConfig
from bajo.core.random import Sampler
from bajo.rt.camera_sampling import sample_camera
from std.math import iota
from bajo.parser.pbrt import parse_pbrt
from bajo.rt.scene_description import SceneDescription
from test.rt.cpu.test_film import make_film_path
from bajo.core.color import Color


def make_filter_edge() raises -> SceneDescription:
    return parse_pbrt(
        """Film "rgb" "integer xresolution" 1 "integer yresolution" 1
Sampler "halton" "integer pixelsamples" 4096
Integrator "path" "integer maxdepth" 0
LookAt 0 0 1 0 0 0 0 1 0
Camera "perspective" "float fov" 90
WorldBegin
Material "diffuse" "rgb reflectance" [0 0 0]
AreaLightSource "diffuse" "rgb L" [1 1 1]
Shape "trianglemesh" "point3 P" [1.4 -100 0 100 -100 0 100 100 0 1.4 100 0]
 "integer indices" [0 1 2 0 2 3]
"""
    )


def test_gaussian_sampler_matches_reference_quantiles_and_table_mass() raises:
    var sampler = FilterSampler(ReconstructionFilter.gaussian())
    assert_equal(sampler.nx, Int32(48))
    assert_equal(sampler.ny, Int32(48))
    # Independent Float32 implementation of PBRT's Gaussian/PiecewiseConstant2D.
    var u: Array[Float32, 5] = [0.01, 0.125, 0.5, 0.8, 0.999]
    var v: Array[Float32, 5] = [0.99, 0.9, 0.5, 0.2, 0.001]
    var x: Array[Float32, 5] = [
        -0.590743542,
        -0.060583591,
        0.499999940,
        0.912225842,
        1.839033604,
    ]
    var y: Array[Float32, 5] = [
        1.590743542,
        1.123264790,
        0.499999940,
        0.087773800,
        -0.839037061,
    ]
    var view = sampler.view()
    for i in range(5):
        var sample = view.sample(
            SIMD[.float32, 1](u[i]), SIMD[.float32, 1](v[i])
        )
        assert_almost_equal(sample.pixel_u, x[i], atol=6e-6)
        assert_almost_equal(sample.pixel_v, y[i], atol=6e-6)
        assert_almost_equal(sample.weight, Float32(0.942309558), atol=1e-6)
    assert_almost_equal(sampler.data[0], Float32(0.942309558), atol=1e-6)
    var rectangular = FilterSampler(
        ReconstructionFilter.gaussian(2.25, 0.75, 0.35)
    )
    assert_equal(rectangular.nx, Int32(72))
    assert_equal(rectangular.ny, Int32(24))
    var filter = ReconstructionFilter.gaussian()
    assert_equal(filter.evaluate(1.5, 0), Float32(0))
    assert_equal(filter.evaluate(2, 0), Float32(0))


def test_filter_sampling_widths_boundaries_and_analytic_filters() raises:
    for config in [
        ReconstructionFilter(),
        ReconstructionFilter(x_radius=1.2, y_radius=0.75),
        ReconstructionFilter.triangle(),
        ReconstructionFilter.gaussian(),
    ]:
        var owner = FilterSampler(config)
        var view = owner.view()
        comptime for width in [1, 4, 16]:
            var u = SIMD[.float32, width](0)
            var v = SIMD[.float32, width](0)
            for lane in range(width):
                u[lane] = Float32(lane) / Float32(width)
                v[lane] = Float32(width - lane - 1) / Float32(width)
            var packet = view.sample(u, v)
            for lane in range(width):
                var scalar = view.sample(
                    SIMD[.float32, 1](u[lane]), SIMD[.float32, 1](v[lane])
                )
                assert_almost_equal(
                    packet.pixel_u[lane], scalar.pixel_u, atol=1e-6
                )
                assert_almost_equal(
                    packet.pixel_v[lane], scalar.pixel_v, atol=1e-6
                )
                assert_true(
                    packet.pixel_u[lane] >= 0.5 - config.x_radius - 1e-6
                    and packet.pixel_u[lane] <= 0.5 + config.x_radius + 1e-6
                )
                assert_true(packet.weight[lane] > 0)
        _ = owner^
    var triangle = FilterSampler(ReconstructionFilter.triangle(2, 1))
    var tent = triangle.view().sample(
        SIMD[.float32, 1](0.125), SIMD[.float32, 1](0.875)
    )
    assert_almost_equal(tent.pixel_u, Float32(-0.5))
    assert_almost_equal(tent.pixel_v, Float32(1))
    assert_equal(tent.weight, Float32(1))


def test_filter_parser_defaults_parameters_and_invalid_values() raises:
    var implicit = parse_pbrt('WorldBegin\nShape "sphere"')
    assert_equal(implicit.settings.filter.kind, ReconstructionFilter.GAUSSIAN)
    assert_equal(implicit.settings.filter.x_radius, Float32(1.5))
    assert_equal(implicit.settings.filter.sigma, Float32(0.5))
    var explicit = parse_pbrt(
        'PixelFilter "gaussian" "float xradius" 2.25 "float yradius" .75 "float'
        ' sigma" .35\nWorldBegin\nShape "sphere"'
    )
    assert_equal(explicit.settings.filter.y_radius, Float32(0.75))
    assert_equal(explicit.settings.filter.sigma, Float32(0.35))
    var box = parse_pbrt('PixelFilter "box"\nWorldBegin\nShape "sphere"')
    assert_equal(box.settings.filter.kind, ReconstructionFilter.BOX)
    var triangle = parse_pbrt(
        'PixelFilter "triangle"\nWorldBegin\nShape "sphere"'
    )
    assert_equal(triangle.settings.filter.x_radius, Float32(2))
    for invalid in [
        ReconstructionFilter(x_radius=-1),
        ReconstructionFilter(y_radius=bitcast[.float32](UInt32(0x7FC00000))),
        ReconstructionFilter.gaussian(sigma=0),
        ReconstructionFilter.gaussian(x_radius=0.001),
        ReconstructionFilter(kind=99),
    ]:
        with assert_raises():
            _ = FilterSampler(invalid)
    with assert_raises():
        _ = parse_pbrt('PixelFilter "missing"\nWorldBegin\nShape "sphere"')


def test_gaussian_reconstructs_light_outside_the_nominal_pixel() raises:
    var parsed = make_filter_edge()
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var box = settings.copy()
    box.filter = ReconstructionFilter()
    var unfiltered = render_wavefront[.PATH, 4](box, camera, world)
    assert_equal(unfiltered.pixels[0].x, Float32(0))
    comptime for integrator in [
        Integrator.PATH,
        Integrator.NEE,
        Integrator.MIS,
    ]:
        comptime for width in [1, 4, 16]:
            var result = render_wavefront[integrator, width](
                settings, camera, world
            )
            # Exact integral of the reference 48-cell 1D step distribution
            # above film offset 0.7; the emitter starts at world x=1.4.
            assert_almost_equal(
                result.pixels[0].x, Float32(0.074630116), atol=0.003
            )


def test_filter_weight_is_applied_after_sample_clamping() raises:
    var parsed = make_film_path()
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var expected = settings.film.apply_sample(Color(36, 12, 12))
    comptime for integrator in [
        Integrator.PATH,
        Integrator.NEE,
        Integrator.MIS,
    ]:
        var result = render_wavefront[integrator, 4, 7](settings, camera, world)
        for pixel in result.pixels:
            comptime for channel in range(3):
                assert_almost_equal(
                    pixel[channel], expected[channel], atol=1e-5
                )


def test_pinhole_shortcut_keeps_filter_samples_and_rng_dimensions() raises:
    for config in [
        ReconstructionFilter(),
        ReconstructionFilter.gaussian(),
        ReconstructionFilter.triangle(),
    ]:
        var filter = FilterSampler(config)
        for sampler in range(7):
            var settings = RenderSettings(8, 4, 16, 2026, 1, Sampler(sampler))
            var sampling = SamplingConfig.from_settings(settings)
            sampling.filter = filter.view()
            var ids = iota[.uint32, 4]() + UInt32(37)
            var expected = sample_camera(sampling, ids)
            var actual = sample_camera(sampling, ids, True)
            assert_equal(actual.lens_u, SIMD[.float32, 4](0))
            assert_equal(actual.lens_v, SIMD[.float32, 4](0))
            for lane in range(4):
                assert_equal(
                    bitcast[.uint32](actual.pixel_u[lane]),
                    bitcast[.uint32](expected.pixel_u[lane]),
                )
                assert_equal(
                    bitcast[.uint32](actual.pixel_v[lane]),
                    bitcast[.uint32](expected.pixel_v[lane]),
                )
                assert_equal(
                    bitcast[.uint32](actual.filter_weight[lane]),
                    bitcast[.uint32](expected.filter_weight[lane]),
                )
        _ = filter^


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
