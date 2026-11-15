"""CPU/device guide agreement and presentation-only GPU refresh."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from std.memory import bitcast
from max.gpu.host import DeviceContext
from bajo.core import Point3W, Vec3W
from bajo.core.camera import Camera
from bajo.rt import CpuScene, RenderSettings
from bajo.rt.cpu.denoise_guides import make_cpu_denoise_guides
from bajo.rt.gpu.prepared_renderer import GpuRtPreparedRenderer
from bajo.rt.gpu.scene import prepare_gpu_scene
from bajo.rt.gpu.config import GpuRtSceneKind, _gpu_rt_kernel_formats
from bajo.rt.gpu.denoise_guides import enqueue_gpu_denoise_guides
from bajo.rt.gpu.upload import _upload_nonempty
from bajo.bvh.gpu.utils import upload_camera
from bajo.postprocess.gpu import GpuDenoiseGuides
from bajo.postprocess.types import GuideLayout
from test.postprocess.cpu.test_guides import cutout_world
from test.rt.gpu.test_gpu import (
    _textured_instance_world,
    _sphere_scene_data,
    _triangle_scene_data,
)


def _camera() -> Camera:
    # Avoid discontinuities of nearest-neighbor texture sampling: a UV exactly
    # on a texel boundary can round to opposite sides on the two backends.
    return Camera.from_vfov(
        Point3W(0.037, 0.021, 3), Point3W(0.037, 0.021, 0), Vec3W(0, 1, 0), 45
    )


def _compare[kind: GpuRtSceneKind](world: CpuScene[16, 16]) raises:
    var camera = _camera()
    var host = make_cpu_denoise_guides(world, camera, 9, 7)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind](ctx, world.scene_data())
        var cam = upload_camera(ctx, camera)
        var metals = _upload_nonempty(
            ctx, [m.texture_index for m in world.scene_data().surfaces().metals]
        )
        var guides = GpuDenoiseGuides(ctx, 9, 7)
        comptime Formats = _gpu_rt_kernel_formats[kind]
        enqueue_gpu_denoise_guides[kind, Formats](
            ctx, scene.view(), cam, metals, guides
        )
        with guides.surfaces.map_to_host() as ids:
            for i in range(63):
                assert_equal(ids[i], host.surfaces[i])
        with guides.fields.map_to_host() as fields:
            for i in range(63):
                if host.surfaces[i] != GuideLayout.MISS:
                    for channel in range(GuideLayout.STRIDE):
                        assert_almost_equal(
                            fields[i * GuideLayout.STRIDE + channel],
                            host.fields[i * GuideLayout.STRIDE + channel],
                            atol=2e-5,
                            # Planar derivatives amplify tiny hit-point differences
                            # near grazing angles; retain strict guide tolerances.
                            rtol=2e-4 if channel
                            == GuideLayout.DEPTH_FWIDTH else 1e-5,
                        )


def test_sphere_triangle_and_textured_instance_guides() raises:
    var sphere = CpuScene[16, 16](_sphere_scene_data())
    _compare[.SPHERES](sphere)
    var cutout = cutout_world()
    _compare[.SPHERES](cutout)
    var triangles = CpuScene[16, 16](_triangle_scene_data())
    _compare[.TRIANGLES](triangles)
    var instance = _textured_instance_world()
    _compare[.INSTANCES](instance)
    var coated = _textured_instance_world[True]()
    _compare[.INSTANCES](coated)


def test_gpu_refresh_preserves_raw_accumulation_and_counts() raises:
    var data = _sphere_scene_data()
    var camera = _camera()
    var settings = RenderSettings(9, 7, 2, 2026, 2, sample_sequence_length=3)
    var renderer = GpuRtPreparedRenderer[.SPHERES](settings, camera, data)
    var first = renderer.preview[.MIS](settings, camera, 4)
    var raw = first.rgba.copy()
    assert_true(first.preparation_ns > 0)
    assert_true(first.batch_ns > 0)
    assert_true(first.postprocess_ns > 0)
    var words = List[Float32]()
    with renderer._preview.value().sums.map_to_host() as sums:
        for i in range(len(sums)):
            words.append(sums[i])
    for enabled in [True, False, True, False]:
        var refreshed = renderer.refresh_preview(camera, 4, enabled)
        assert_equal(refreshed.samples, 2)
        assert_equal(refreshed.batch_ns, 0)
        assert_true(refreshed.postprocess_ns > 0)
        assert_equal(refreshed.batches, 1)
        if not enabled:
            for i in range(len(raw)):
                assert_equal(refreshed.rgba[i], raw[i])
        with renderer._preview.value().sums.map_to_host() as sums:
            for i in range(len(words)):
                assert_equal(
                    bitcast[.uint32](sums[i]), bitcast[.uint32](words[i])
                )
    # A reconstruction failure still publishes the retained raw image.
    renderer._preview.value().guides.value().width = 0
    var recovered = renderer.refresh_preview(camera, 4, True)
    assert_true(recovered.postprocess_error != "")
    assert_equal(recovered.samples, 2)
    for i in range(len(raw)):
        assert_equal(recovered.rgba[i], raw[i])
    renderer._preview.value().guides.value().width = 9
    settings.samples_per_pixel = 1
    settings.sample_offset = 2
    var last = renderer.preview[.MIS](settings, camera, 4, True)
    assert_equal(last.samples, 3)
    assert_equal(last.batches, 2)
    settings.color_mode = .SPECTRAL
    var spectral = renderer.preview[.MIS](settings, camera, 5, True)
    assert_equal(spectral.samples, 1)
    assert_equal(spectral.batches, 1)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
