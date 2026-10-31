"""Complete-sample sensor clamping, film parsing, and ordered CPU reduction."""

from std.memory import bitcast
from std.math import abs
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
    assert_raises,
)
from bajo.core import Point3f32, Vec3f32, Affine3f32
from bajo.core.camera import Camera
from bajo.core.color import Color
from bajo.rt import CpuScene, SceneBuilder, render_wavefront
from bajo.rt.film import FilmResponse
from bajo.rt.render_types import RenderSettings
from bajo.rt.material_types import Environment, ImageTexture, Integrator
from bajo.rt.scene_description import SceneDescription
from bajo.parser.pbrt import parse_pbrt
from test.rt.helpers import assert_pixels_close


def make_film_path() raises -> SceneDescription:
    return parse_pbrt(
        """Film "rgb" "integer xresolution" 3 "integer yresolution" 2
 "float iso" 200 "float maxcomponentvalue" 20
Sampler "halton" "integer pixelsamples" 5
Integrator "path" "integer maxdepth" 2
LookAt 0 0 3 0 0 0 0 1 0
Camera "perspective" "float fov" 5
WorldBegin
LightSource "infinite" "rgb L" [12 0 6]
Material "dielectric" "float eta" 1
AreaLightSource "diffuse" "rgb L" [12 3 6]
Shape "trianglemesh" "point3 P" [-20 -20 2 20 -20 2 0 20 2]
 "point2 uv" [0 0 1 0 .5 1]
AreaLightSource "diffuse" "rgb L" [12 9 0]
Shape "trianglemesh" "point3 P" [-20 -20 1 20 -20 1 0 20 1]
 "point2 uv" [0 0 1 0 .5 1]
"""
    )


def make_varying_film_world() raises -> CpuScene[4, 4]:
    var builder = SceneBuilder()
    var black = builder.add_lambertian(Color(0))
    builder.add_sphere(Point3f32[.WORLD](1000, 0, 0), 1, black)
    var texels: List[Float32] = [
        100,
        2,
        1,
        1,
        2,
        3,
        50,
        20,
        5,
        0.1,
        0.2,
        0.3,
        1,
        2,
        3,
        200,
        10,
        30,
        0.1,
        0.2,
        0.3,
        100,
        30,
        2,
    ]
    var image = builder.surfaces.add_image_texture(ImageTexture(4, 2, texels^))
    builder.set_environment(
        Environment.image(
            image,
            Color(1),
            Affine3f32[.WORLD, .LOCAL].identity(),
            Affine3f32[.LOCAL, .WORLD].identity(),
        )
    )
    return CpuScene[4, 4](builder^.finish())


def film_camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0),
        Point3f32[.WORLD](0, 0, -1),
        Vec3f32[.WORLD](0, 1, 0),
        120,
    )


def test_film_parser_retains_sensor_limit_and_rejects_invalid_limits() raises:
    var default = parse_pbrt('WorldBegin\nShape "sphere"')
    assert_equal(
        default.settings.film.max_component_value,
        bitcast[.float32](UInt32(0x7F800000)),
    )
    assert_almost_equal(
        default.settings.film.sensor_red.x, Float32(0.412457692), atol=1e-7
    )
    var parsed = make_film_path()
    ref film = parsed.settings.film
    assert_equal(film.max_component_value, Float32(20))
    assert_equal(film.exposure, Float32(2))
    assert_almost_equal(
        film.sensor_green.x + film.sensor_green.y + film.sensor_green.z,
        Float32(1),
        atol=1e-7,
    )
    # Independent high-precision integration of the original CIE/D65 tables.
    var expected: Array[Color, 3] = [
        Color(0.412457692, 0.357575812, 0.180437194),
        Color(0.212673497, 0.715151625, 0.072174878),
        Color(0.019333954, 0.119191937, 0.950302556),
    ]
    var actual: Array[Color, 3] = [
        film.sensor_red,
        film.sensor_green,
        film.sensor_blue,
    ]
    for row in range(3):
        comptime for channel in range(3):
            assert_almost_equal(
                actual[row][channel], expected[row][channel], atol=1e-7
            )
    for invalid in [
        Float32(-1),
        bitcast[.float32](UInt32(0xFF800000)),
        bitcast[.float32](UInt32(0x7FC00000)),
    ]:
        var value = FilmResponse(max_component_value=invalid)
        with assert_raises():
            value.validate()
    with assert_raises():
        _ = parse_pbrt(
            'Film "rgb" "float maxcomponentvalue" -1\nWorldBegin\nShape'
            ' "sphere"'
        )


def test_film_sensor_clamp_preserves_channel_ratios_and_exposure_order() raises:
    var film = FilmResponse(2, max_component_value=20)
    var rgb = film.apply_sample(Color(30, 10, 5))
    assert_almost_equal(rgb.x, Float32(20))
    assert_almost_equal(rgb.y, Float32(20.0 / 3.0))
    assert_almost_equal(rgb.z, Float32(10.0 / 3.0))
    assert_equal(film.apply_sample(Color(0)).x, Float32(0))
    film.use_cie1931()
    var red = film.apply_sample(Color(100, 0, 0))
    # Pure red's largest CIE sensor component is X, not its output R channel.
    assert_almost_equal(red.x * film.sensor_red.x, Float32(20), atol=2e-6)
    assert_true(red.x > 40)
    assert_equal(red.y, Float32(0))
    comptime for width in [1, 4, 16]:
        var color = Vec3f32[.WORLD](30, 10, 5).broadcast[width]()
        var packet = film.apply_sample(color)
        var scalar = film.apply_sample(Color(30, 10, 5))
        for lane in range(width):
            comptime for channel in range(3):
                assert_almost_equal(
                    packet[channel][lane], scalar[channel], atol=1e-6
                )


def test_complete_path_is_clamped_after_all_emission_and_environment_contributions() raises:
    var parsed = make_film_path()
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var total = Color(36, 12, 12) * 2
    # Independent scalar sensor calculation for this known complete path.
    var x = (
        total.x * settings.film.sensor_red.x
        + total.y * settings.film.sensor_red.y
        + total.z * settings.film.sensor_red.z
    )
    var y = (
        total.x * settings.film.sensor_green.x
        + total.y * settings.film.sensor_green.y
        + total.z * settings.film.sensor_green.z
    )
    var z = (
        total.x * settings.film.sensor_blue.x
        + total.y * settings.film.sensor_blue.y
        + total.z * settings.film.sensor_blue.z
    )
    var expected = total * (Float32(20) / max(x, max(y, z)))
    comptime for integrator in [
        Integrator.PATH,
        Integrator.NEE,
        Integrator.MIS,
    ]:
        comptime for width in [1, 4, 16]:
            var result = render_wavefront[integrator, width, 7](
                settings, camera, world
            )
            for pixel in result.pixels:
                comptime for channel in range(3):
                    assert_almost_equal(
                        pixel[channel], expected[channel], atol=5e-6
                    )


def test_varying_samples_are_clamped_before_pixel_average_and_survive_batching() raises:
    var world = make_varying_film_world()
    var camera = film_camera()
    for spp in [5, 8]:
        var settings = RenderSettings(3, 2, spp, 2026, 0, sampler=.HALTON)
        settings.film = FilmResponse(1.7, max_component_value=20)
        settings.film.use_cie1931()
        var expected = List[Color](length=6, fill=Color(0))
        var unclamped_mean = expected.copy()
        for index in range(spp):
            var one = settings.copy()
            one.samples_per_pixel = 1
            one.sample_offset = index
            one.film = FilmResponse()
            var raw = render_wavefront[.PATH, 1](one, camera, world)
            for pixel in range(6):
                expected[pixel] += settings.film.apply_sample(
                    raw.pixels[pixel]
                ) / Float32(spp)
                unclamped_mean[pixel] += raw.pixels[pixel] / Float32(spp)
        var difference = Float32(0)
        for pixel in range(6):
            difference += abs(
                expected[pixel].x
                - settings.film.apply_sample(unclamped_mean[pixel]).x
            )
        assert_true(difference > 0.1)
        comptime for width in [1, 4, 16]:
            var result = render_wavefront[.PATH, width, 7](
                settings, camera, world
            )
            assert_pixels_close(result.pixels, expected, atol=1e-5)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
