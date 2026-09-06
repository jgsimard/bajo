from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_true

from bajo.core import Vec3f32
from bajo.rt.coated_diffuse import _sample_coated_diffuse
from bajo.rt.types import BsdfSample


def _coated_sample() -> BsdfSample[1]:
    return _sample_coated_diffuse(
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
        Vec3f32[.WORLD](0.7, 0.5, 0.25),
        SIMD[.float32, 1](0.2),
        SIMD[.float32, 1](1.5),
        SIMD[.float32, 1](0.01),
        Vec3f32[.WORLD](0.1, 0.05, 0.02),
        SIMD[.float32, 1](0.2),
        SIMD[.float32, 1](10.0),
        SIMD[.float32, 1](2.0),
        SIMD[.float32, 1](0.7),
        SIMD[.float32, 1](0.3),
    )


def _coated_sample_kernel(output: Pointer[Float32, MutAnyOrigin]):
    var sampled = _coated_sample()
    output[unsafe_offset=0] = sampled.direction.x[0]
    output[unsafe_offset=1] = sampled.direction.y[0]
    output[unsafe_offset=2] = sampled.direction.z[0]
    output[unsafe_offset=3] = sampled.weight.x[0]
    output[unsafe_offset=4] = sampled.weight.y[0]
    output[unsafe_offset=5] = sampled.weight.z[0]
    output[unsafe_offset=6] = sampled.pdf[0]
    output[unsafe_offset=7] = Float32(sampled.delta[0])
    output[unsafe_offset=8] = Float32(sampled.ok[0])


def test_layered_coated_diffuse_matches_on_gpu() raises:
    var expected = _coated_sample()
    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[.float32](9)
        ctx.enqueue_function[_coated_sample_kernel](
            output, grid_dim=1, block_dim=1
        )
        ctx.synchronize()
        with output.map_to_host() as host:
            assert_almost_equal(host[0], expected.direction.x[0], atol=1.0e-6)
            assert_almost_equal(host[1], expected.direction.y[0], atol=1.0e-6)
            assert_almost_equal(host[2], expected.direction.z[0], atol=1.0e-6)
            assert_almost_equal(host[3], expected.weight.x[0], atol=1.0e-6)
            assert_almost_equal(host[4], expected.weight.y[0], atol=1.0e-6)
            assert_almost_equal(host[5], expected.weight.z[0], atol=1.0e-6)
            assert_almost_equal(host[6], expected.pdf[0], atol=1.0e-6)
            assert_true((host[7] != 0.0) == expected.delta[0])
            assert_true((host[8] != 0.0) == expected.ok[0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
