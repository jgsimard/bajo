"""Mixed coated conductors retain instance attributes and chunk invariance."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from max.gpu.host import DeviceContext
from bajo.parser import MemoryAssetLoader
from bajo.parser.pbrt import read_pbrt
from bajo.rt import CpuScene, render_wavefront
from bajo.rt.gpu import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_render_gpu,
    prepare_gpu_scene,
)
from test.parser.pbrt.test_pbrt import _triangle_ply


def test_mixed_coated_conductor_instances_match_cpu_and_gpu_chunks() raises:
    var loader = MemoryAssetLoader()
    loader.add_ply_file("scene/mesh.ply", _triangle_ply())
    loader.add_file(
        "scene/main.pbrt",
        """Film "rgb" "integer xresolution" 12 "integer yresolution" 8
Sampler "independent" "integer pixelsamples" 16
LookAt .3 .3 2 .3 .3 0 0 1 0
Camera "perspective" "float fov" 35
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
MakeNamedMaterial "red" "string type" "diffuse" "rgb reflectance" [.7 .05 .02]
MakeNamedMaterial "rim" "string type" "coatedconductor" "spectrum conductor.eta" "metal-Al-eta" "spectrum conductor.k" "metal-Al-k" "float conductor.roughness" .05
Material "mix" "string materials" ["red" "rim"] "float amount" .3
Shape "plymesh" "string filename" "mesh.ply"
""",
    )
    var parsed = read_pbrt("scene/main.pbrt", loader)
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var cpu = render_wavefront[.MIS, 4](settings, camera, world)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var full = GpuRtRenderTarget(ctx, settings, camera, 96 * 16)
        enqueue_render_gpu[.MIS](ctx, full, scene, settings)
        var expected = download_gpu_pixels(ctx, full)
        var chunked = GpuRtRenderTarget(ctx, settings, camera, 2 * 16)
        enqueue_render_gpu[.MIS](ctx, chunked, scene, settings)
        var actual = download_gpu_pixels(ctx, chunked)
        for i in range(len(expected)):
            assert_true(expected[i].is_finite()[0])
            assert_equal(actual[i].x, expected[i].x)
            assert_equal(actual[i].y, expected[i].y)
            assert_equal(actual[i].z, expected[i].z)
            assert_almost_equal(
                expected[i].x[0], cpu.pixels[i].x[0], atol=1.0e-3, rtol=1.0e-3
            )
            assert_almost_equal(
                expected[i].y[0], cpu.pixels[i].y[0], atol=1.0e-3, rtol=1.0e-3
            )
            assert_almost_equal(
                expected[i].z[0], cpu.pixels[i].z[0], atol=1.0e-3, rtol=1.0e-3
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
