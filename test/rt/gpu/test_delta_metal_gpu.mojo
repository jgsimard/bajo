"""Delta mirrors retain reflected emission under direct-light integrators."""

from std.testing import TestSuite, assert_almost_equal
from max.gpu.host import DeviceContext

from bajo.core import Point3f32, Vec3f32
from bajo.rt import Camera, Color, Integrator, RenderSettings, SceneBuilder
from bajo.rt.material_types import Environment
from bajo.rt.gpu import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_render_gpu,
    prepare_gpu_scene,
)
from bajo.rt.gpu.config import GPU_RT_BVH_WIDE4


def test_gpu_delta_metal_direct_lighting_preserves_reflected_emission() raises:
    var settings = RenderSettings(8, 6, 3, UInt64(641), 2)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 1.0),
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        10.0,
    )
    var albedo = Color(0.8, 0.6, 0.4)
    var emission = Color(3.0, 2.0, 1.0)
    var expected = albedo * emission
    var fuzz_values: List[Float32] = [0.0, -0.0, 1.0e-4]
    with DeviceContext() as ctx:
        # Both signed zeros and the inclusive delta cutoff must retain full
        # reflected emission. A sampled light contributes no mirror energy.
        for fuzz in fuzz_values:
            for environment_light in [False, True]:
                var builder = SceneBuilder()
                builder.set_environment(
                    Environment.uniform(
                        Color(0.5)
                    ) if environment_light else Environment.black()
                )
                var mirror = builder.add_metal(albedo, fuzz)
                var light = builder.add_emissive(emission)
                builder.add_quad(
                    Point3f32[.WORLD](-10.0, -10.0, 0.0),
                    Point3f32[.WORLD](10.0, -10.0, 0.0),
                    Point3f32[.WORLD](10.0, 10.0, 0.0),
                    Point3f32[.WORLD](-10.0, 10.0, 0.0),
                    mirror,
                )
                builder.add_quad(
                    Point3f32[.WORLD](-10.0, -10.0, 2.0),
                    Point3f32[.WORLD](-10.0, 10.0, 2.0),
                    Point3f32[.WORLD](10.0, 10.0, 2.0),
                    Point3f32[.WORLD](10.0, -10.0, 2.0),
                    light,
                )
                var world = builder^.finish()
                var scene = prepare_gpu_scene[
                    kind=.TRIANGLES, triangle_format=GPU_RT_BVH_WIDE4
                ](ctx, world)
                comptime for integrator_index in range(2):
                    comptime integrator = Integrator.NEE if integrator_index == 0 else Integrator.MIS
                    for capacity in [8 * 6 * 3, 7 * 3]:
                        var target = GpuRtRenderTarget(
                            ctx, settings, camera, capacity
                        )
                        enqueue_render_gpu[integrator](
                            ctx, target, scene, settings
                        )
                        var pixels = download_gpu_pixels(ctx, target)
                        for pixel in pixels:
                            assert_almost_equal(
                                pixel.x, expected.x, atol=2.0e-5
                            )
                            assert_almost_equal(
                                pixel.y, expected.y, atol=2.0e-5
                            )
                            assert_almost_equal(
                                pixel.z, expected.z, atol=2.0e-5
                            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
