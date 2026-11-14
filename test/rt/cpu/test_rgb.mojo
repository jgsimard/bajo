"""Explicit RGB entry points share transport and retain workspace ownership."""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)
from bajo.core import Point3f32, Vec3f32
from bajo.rt import Camera, Color, RenderSettings, SceneBuilder
from bajo.rt.cpu import CpuScene, CpuWavefrontWorkspace
from bajo.rt.cpu.rgb import render_wavefront, render_wavefront_configured
from bajo.rt.cpu.wavefront import (
    render_wavefront as render_mode_aware,
    render_wavefront_configured as render_configured_mode_aware,
)
from test.rt.helpers import assert_pixels_equal


def _world() raises -> CpuScene[16, 16]:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.7, 0.3, 0.2))
    builder.add_sphere(Point3f32[.WORLD](0, 0, -1), 0.5, matte)
    builder.add_sphere(Point3f32[.WORLD](0, -100.5, -1), 100, matte)
    return CpuScene[16, 16](builder^.finish())


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0, 0, 1),
        Point3f32[.WORLD](0, 0, -1),
        Vec3f32[.WORLD](0, 1, 0),
        45,
    )


def test_cpu_rgb_entry_points_match_mode_aware_and_reuse_workspace() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(809), 3)
    var world = _world()
    var camera = _camera()
    var expected = render_mode_aware[.MIS, 4, PARALLEL=False](
        settings, camera, world
    )
    var rgb = render_wavefront[.MIS, 4, PARALLEL=False](settings, camera, world)
    assert_pixels_equal(rgb.pixels, expected.pixels)
    var fresh = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 1024, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world)
    assert_pixels_equal(fresh.pixels, expected.pixels)
    var workspace = CpuWavefrontWorkspace[4, .MIS]()
    var retained = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 1024, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    assert_pixels_equal(retained.pixels, expected.pixels)
    assert_false(workspace.spectral)

    settings.color_mode = .SPECTRAL
    var spectral = render_configured_mode_aware[
        .AUTO_COHERENT, .MIS, 4, 1024, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    assert_equal(len(spectral.pixels), len(expected.pixels))
    assert_true(workspace.spectral)
    settings.color_mode = .RGB
    var again = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 1024, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    assert_pixels_equal(again.pixels, expected.pixels)
    assert_true(workspace.spectral)


def test_cpu_rgb_rejects_spectral_before_workspace_mutation() raises:
    var settings = RenderSettings(5, 3, 2, UInt64(811), 3)
    var world = _world()
    var camera = _camera()
    var workspace = CpuWavefrontWorkspace[4, .MIS]()
    _ = render_wavefront_configured[
        .AUTO_COHERENT, .MIS, 4, 1024, False, .TASK_PARTITIONS, 16, 16, 16, 8, 4
    ](settings, camera, world, workspace)
    var arenas = Int(workspace.arenas.unsafe_ptr())
    var samples = Int(workspace.arenas[0].samples.unsafe_ptr())
    var count = len(workspace.arenas[0].samples)
    settings.color_mode = .SPECTRAL
    settings.image_width = 64
    settings.image_height = 64
    with assert_raises(contains="RGB"):
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
        ](settings, camera, world, workspace)
    assert_equal(Int(workspace.arenas.unsafe_ptr()), arenas)
    assert_equal(Int(workspace.arenas[0].samples.unsafe_ptr()), samples)
    assert_equal(len(workspace.arenas[0].samples), count)
    assert_false(workspace.spectral)
    with assert_raises(contains="RGB"):
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
        ](settings, camera, world)
    with assert_raises(contains="RGB"):
        _ = render_wavefront[.MIS, 4, PARALLEL=False](settings, camera, world)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
