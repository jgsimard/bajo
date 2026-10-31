"""Alpha rejection before hit reduction, including coplanar primitives."""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.core import Point3f32, Vec3f32, Rayf32, Point3, Vec3, Ray
from bajo.parser.pbrt import parse_pbrt, read_pbrt
from bajo.parser import MemoryAssetLoader
from bajo.rt import CpuScene
from bajo.rt.material_types import ImageTexture
from bajo.rt.shading_types import ShadingPoint
from bajo.rt.cpu.lighting import _sample_direct_light_candidate
from bajo.core.random import Rng
from bajo.rt import Color, render_wavefront
from bajo.rt.opacity import opacity_accepts, opacity_point_sample
from bajo.rt.lighting import _sample_triangle_light_surface, light_sample_uv
from test.parser.png.test_png import _encode


def _ray(t_max: Float32 = 100.0) -> Rayf32[.WORLD]:
    return Rayf32[.WORLD](
        Point3f32[.WORLD](0.2, 0.3, 2), Vec3f32[.WORLD](0, 0, -1), 1e-4, t_max
    )


def test_coplanar_flat_alpha_rejection_preserves_opaque_hit() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Material "diffuse" "rgb reflectance" [1 0 0]
Shape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0] "float alpha" 0
Material "diffuse" "rgb reflectance" [0 0 1]
Shape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0]
"""
    )
    assert_equal(len(parsed.data.opacity().records), 1)
    var world = CpuScene[4, 4](parsed^.take_data())
    var hit = world.trace_surface(_ray())
    assert_equal(hit.hit[0], True)
    assert_almost_equal(hit.t[0], 2.0)
    assert_equal(hit.surface.index()[0], UInt32(2))
    assert_equal(world.occluded(_ray())[0], True)
    assert_equal(world.occluded(_ray(1.99))[0], False)


def _coplanar_textured_source(two_triangles: Bool = True) -> String:
    var source = """WorldBegin
Texture "alpha" "float" "imagemap" "string filename" "alpha.png"
Shape "trianglemesh" "texture alpha" "alpha"
 "point3 P" [0 0 0 1 0 0 0 1 0"""
    source += " 0 0 0 1 0 0 0 1 0]\n" if two_triangles else "]\n"
    source += ' "point2 uv" [.25 .5 .25 .5 .25 .5'
    source += " .75 .5 .75 .5 .75 .5]\n" if two_triangles else "]\n"
    return source


def test_coplanar_candidates_in_one_mesh_use_png_alpha() raises:
    var loader = MemoryAssetLoader()
    loader.add_file("main.pbrt", _coplanar_textured_source())
    loader.add_image_file(
        "alpha.png",
        _encode([UInt8(0), 255, 0, 0, 0, 255, 0, 0, 255], 2, 1, color=6),
    )
    var parsed = read_pbrt("main.pbrt", loader)
    var world = CpuScene[4, 4](parsed^.take_data())
    var hit = world.trace_surface(_ray())
    assert_equal(hit.hit[0], True)
    assert_almost_equal(hit.t[0], 2.0)
    assert_almost_equal(hit.uv_u[0], 0.75)
    assert_equal(world.occluded(_ray())[0], True)
    assert_equal(world.occluded(_ray(1.99))[0], False)


def test_transparent_only_instance_is_invisible_to_primary_and_shadow_packets() raises:
    var loader = MemoryAssetLoader()
    loader.add_file("main.pbrt", _coplanar_textured_source(False))
    loader.add_image_file(
        "alpha.png",
        _encode([UInt8(0), 255, 0, 0, 0, 255, 0, 0, 255], 2, 1, color=6),
    )
    var parsed = read_pbrt("main.pbrt", loader)
    var world = CpuScene[4, 4](parsed^.take_data())
    var rays = Ray[.float32, .WORLD, 4](
        Point3[.float32, .WORLD, 4](0.2, 0.3, 2),
        Vec3[.float32, .WORLD, 4](0, 0, -1),
    )
    var valid = SIMD[.bool, 4](True, False, True, False)
    var hit = world.trace_surface(rays, valid)
    var blocked = world.occluded(rays, valid)
    comptime for lane in range(4):
        assert_equal(hit.hit[lane], False)
        assert_equal(blocked[lane], False)


def test_fractional_alpha_has_expected_coverage_and_visibility_agreement() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Shape "trianglemesh" "point3 P" [-10 -10 0 10 -10 0 0 10 0] "float alpha" .25
"""
    )
    var world = CpuScene[4, 4](parsed^.take_data())
    var accepted = 0
    for index in range(4096):
        var ray = Rayf32[.WORLD](
            Point3f32[.WORLD](Float32(index) / 8192.0, 0.1, 2),
            Vec3f32[.WORLD](0, 0, -1),
        )
        var primary = world.trace_surface(ray)
        var shadow = world.occluded(ray)
        assert_equal(primary.hit[0], shadow[0])
        accepted += Int(primary.hit[0])
    assert_true(accepted > 940 and accepted < 1110)


def test_scalar_image_alpha_and_rgb_average_rules() raises:
    var alpha = ImageTexture(1, 1, [1, 0.2, 0.3], [0.25])
    assert_almost_equal(alpha.sample_scalar_bilinear(0.5, 0.5)[0], 0.25)
    var opaque = ImageTexture(1, 1, [0.9, 0.3, 0.6], [1.0])
    assert_almost_equal(opaque.sample_scalar_bilinear(0.5, 0.5)[0], 0.6)


def _emitter_source(alpha: String) -> String:
    return (
        """WorldBegin
AreaLightSource "diffuse" "rgb L" [4 4 4]
Shape "trianglemesh" "point3 P" [-1 -1 2 0 1 2 1 -1 2]
 "float alpha" """
        + alpha
    )


def test_masked_emitter_sampling_preserves_pdf_and_fractional_coverage() raises:
    var opaque_parsed = parse_pbrt(_emitter_source("1"))
    var masked_parsed = parse_pbrt(_emitter_source(".25"))
    var transparent_parsed = parse_pbrt(_emitter_source("0"))
    var opaque = CpuScene[4, 4](opaque_parsed^.take_data())
    var masked = CpuScene[4, 4](masked_parsed^.take_data())
    var transparent = CpuScene[4, 4](transparent_parsed^.take_data())
    var point = ShadingPoint[4](
        Point3[.float32, .WORLD, 4](0.0),
        Vec3[.float32, .WORLD, 4](0, 0, 1),
        SIMD[.bool, 4](fill=True),
    )
    var accepted = 0
    for index in range(1024):
        var ids = SIMD[.uint64, 4](
            UInt64(4 * index),
            UInt64(4 * index + 1),
            UInt64(4 * index + 2),
            UInt64(4 * index + 3),
        )
        var rng = Rng[4](17, ids)
        var masked_rng = Rng[4](17, ids)
        var transparent_rng = Rng[4](17, ids)
        var full = _sample_direct_light_candidate(opaque, point, rng)
        var sample = _sample_direct_light_candidate(masked, point, masked_rng)
        var zero = _sample_direct_light_candidate(
            transparent, point, transparent_rng
        )
        for lane in range(4):
            assert_equal(full.valid[lane], True)
            assert_equal(zero.valid[lane], False)
            if sample.valid[lane]:
                accepted += 1
                assert_almost_equal(
                    sample.light_pdf[lane], full.light_pdf[lane], atol=1e-6
                )
                assert_almost_equal(sample.emission.x[lane], 4.0)
    assert_true(accepted > 900 and accepted < 1150)


def _textured_emitter_source(uv_u: String, mirrored: Bool = False) -> String:
    var source = """Film "rgb" "integer xresolution" 4 "integer yresolution" 3
Sampler "halton" "integer pixelsamples" 16
Integrator "path" "integer maxdepth" 3
LookAt 0 0 1.8 0 0 0 0 1 0
Camera "perspective" "float fov" 10
WorldBegin
Material "diffuse" "rgb reflectance" [.7 .4 .2]
Shape "trianglemesh" "point3 P" [-10 -10 0 10 -10 0 0 10 0]
 "point2 uv" [0 0 1 0 0 1]
Texture "alpha" "float" "imagemap" "string filename" "alpha.png"
AreaLightSource "diffuse" "rgb L" [4 4 4]
"""
    if mirrored:
        source += "Scale -1 1 1\n"
    source += (
        """Shape "trianglemesh" "texture alpha" "alpha"
 "point3 P" [-1 -1 2 0 1 2 1 -1 2]
 "point2 uv" ["""
        + uv_u
        + " .5 "
        + uv_u
        + " .5 "
        + uv_u
        + " .5]\n"
    )
    return source


def test_png_masked_emitter_is_dark_and_opaque_half_still_illuminates() raises:
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
            var image = render_wavefront[.MIS, 4](settings, camera, world)
            var energy = Float32(0.0)
            for pixel in image.pixels:
                energy += pixel.x + pixel.y + pixel.z
            if opaque:
                assert_true(energy > 0.1)
            else:
                assert_almost_equal(energy, 0.0, atol=1e-7)


def test_fractional_emissive_hit_uses_position_stable_alpha() raises:
    var parsed = parse_pbrt(_emitter_source(".5"))
    var world = CpuScene[4, 4](parsed^.take_data())
    var tested = 0
    var masked = 0
    for index in range(1024):
        var ray = Rayf32[.WORLD](
            Point3f32[.WORLD](Float32(index) / 4096.0, 0, 0),
            Vec3f32[.WORLD](0, 0, 1),
        )
        var hit = world.trace_surface(ray)
        if hit.hit:
            tested += 1
            var visible = opacity_accepts(
                0.5, opacity_point_sample(ray.at(hit.t))
            )
            assert_equal(hit.emission_visible[0], visible)
            masked += Int(not visible)
    assert_true(tested > 450 and tested < 580)
    assert_true(masked > 180 and masked < 330)


def test_mirrored_emitter_uvs_follow_the_swapped_geometry_corners() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Scale -1 1 1
AreaLightSource "diffuse" "rgb L" [4 4 4]
Shape "trianglemesh" "point3 P" [-1 -1 2 0 1 2 1 -1 2]
 "point2 uv" [0 0 0 1 1 0]
"""
    )
    ref light = parsed.data.lights().records[0]
    var sample = _sample_triangle_light_surface(
        light.p0, light.p1, light.p2, Float32(0.36), Float32(0.25)
    )
    var uv = light_sample_uv(
        light.uv0_u,
        light.uv0_v,
        light.uv1_u,
        light.uv1_v,
        light.uv2_u,
        light.uv2_v,
        sample.u,
        sample.v,
    )
    assert_almost_equal(uv[0], 0.45, atol=1e-6)
    assert_almost_equal(uv[1], 0.15, atol=1e-6)
    # Undo the mirror and reconstruct the original triangle from its UVs.
    assert_almost_equal(-sample.point.x, -1.0 + uv[1] + 2.0 * uv[0], atol=1e-6)
    assert_almost_equal(sample.point.y, -1.0 + 2.0 * uv[1], atol=1e-6)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
