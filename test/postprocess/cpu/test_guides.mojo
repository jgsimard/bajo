"""Guide rays respect alpha and shading maps without consuming camera samples."""
from std.math import abs
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.core import Point3W, Vec3W
from bajo.core.camera import Camera
from bajo.core.image import ImageTexture
from bajo.rt import SceneBuilder, CpuScene, Color
from bajo.rt.material_types import PrimitiveId, NO_TEXTURE
from bajo.rt.opacity import Opacity
from bajo.rt.cpu.denoise_guides import make_cpu_denoise_guides
from bajo.postprocess import (
    GuideLayout,
    DenoiseFrame,
    AtrousWorkspace,
    atrous_denoise,
)
from examples.cornell_box import make_cornell_scene


def cutout_world() raises -> CpuScene[16, 16]:
    var builder = SceneBuilder()
    var front = builder.add_lambertian(Color(1, 0, 0))
    var back = builder.add_lambertian(Color(0.2, 0.4, 0.8))
    var normal = builder.surfaces.add_image_texture(
        ImageTexture(1, 1, [0.7, 0.5, 1])
    )
    builder.surfaces.normal_maps[back.value[0]] = normal
    builder.add_sphere(Point3W(0), 1, front)
    builder.add_sphere(Point3W(0, 0, -3), 1, back)
    builder.opacity.records[PrimitiveId(.SPHERE, 0).value] = Opacity(
        0, NO_TEXTURE, 1, 1
    )
    return CpuScene[16, 16](builder^.finish())


def test_cutout_shading_maps_and_lens_center_guides() raises:
    var world = cutout_world()
    var camera = Camera.from_vfov(
        Point3W(0, 0, 3), Point3W(0), Vec3W(0, 1, 0), 45, 3, 0.1
    )
    var guides = make_cpu_denoise_guides(world, camera, 1, 1)
    assert_equal(
        guides.surfaces[0], world.scene_data().sphere_surfaces()[1].value[0]
    )
    assert_almost_equal(guides.fields[2], Float32(-2), atol=1e-5)
    assert_almost_equal(guides.fields[6], Float32(0.2), atol=1e-6)
    assert_almost_equal(guides.fields[8], Float32(0.8), atol=1e-6)
    assert_true(abs(guides.fields[3]) + abs(guides.fields[4]) > 0.1)
    assert_true(guides.fields[9] > 0)
    var again = make_cpu_denoise_guides(world, camera, 1, 1)
    for i in range(GuideLayout.STRIDE):
        assert_equal(again.fields[i], guides.fields[i])


def test_flat_wall_hit_roundoff_does_not_stop_denoising() raises:
    var world = CpuScene[16, 16](make_cornell_scene())
    var camera = Camera.from_vfov(
        Point3W(0, 1, 3.2), Point3W(0, 1, -1), Vec3W(0, 1, 0), 40
    )
    var guides = make_cpu_denoise_guides(world, camera, 160, 160)
    var frame = DenoiseFrame(160 * 160, 16)
    for i in range(160 * 160):
        var a = guides.fields[i * GuideLayout.STRIDE + GuideLayout.ALBEDO]
        var noise = Float32(0.2) if i % 2 else Float32(-0.2)
        frame.diffuse[i] = Color(a * (1 + noise))
        frame.variance[i] = 0.04
    var workspace = AtrousWorkspace()
    atrous_denoise(frame, guides, workspace)
    var before = Float32(0)
    var after = Float32(0)
    for y in range(25, 60):
        for x in range(60, 100):
            var i = y * 160 + x
            var a = guides.fields[i * GuideLayout.STRIDE + GuideLayout.ALBEDO]
            var raw = frame.diffuse[i].x - a
            var filtered = workspace.ping[i].x - a
            before += raw * raw
            after += filtered * filtered
    assert_true(after < before * 0.25)


def _check_packet_guides[
    packet: SIMDLength
](world: CpuScene[16, 16], camera: Camera) raises:
    var expected = make_cpu_denoise_guides[16, 16, 1](world, camera, 17, 9)
    var actual = make_cpu_denoise_guides[16, 16, packet](world, camera, 17, 9)
    for i in range(17 * 9):
        assert_equal(actual.surfaces[i], expected.surfaces[i])
        for field in range(GuideLayout.STRIDE):
            assert_almost_equal(
                actual.fields[i * GuideLayout.STRIDE + field],
                expected.fields[i * GuideLayout.STRIDE + field],
                atol=2e-4,
            )


def test_packet_guides_match_scalar_with_cutouts_background_and_partial_tails() raises:
    var world = cutout_world()
    var camera = Camera.from_vfov(
        Point3W(0, 0, 3), Point3W(0), Vec3W(0, 1, 0), 60, 3, 0.1
    )
    comptime for width in [4, 8, 16, 32]:
        _check_packet_guides[width](world, camera)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
