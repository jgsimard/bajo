"""Device decoding preserves exact texels across palette and raw images."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_almost_equal
from bajo.rt.gpu.texture_storage import pack_texture_tables
from bajo.rt.gpu.textures import _GpuImageTextureView
from bajo.rt.gpu.upload import _upload_nonempty
from test.rt.cpu.test_texture_storage import make_textures


def _decode_kernel[
    _device: Bool = True
](
    descriptors: Pointer[UInt32, ImmutAnyOrigin],
    words: Pointer[Float32, ImmutAnyOrigin],
    output: Pointer[UInt32, MutAnyOrigin],
):
    var index = global_idx.x
    if index >= 289:
        return
    for image in range(4):
        var view = _GpuImageTextureView.load(UInt32(image), descriptors, words)
        if index >= view.width * view.height:
            continue
        var u = Float32(index) / 31.0 - 2.0
        var v = Float32(index % 53) / 19.0 - 1.0
        var texel = view.texel(index)
        var nearest = view.sample(u, v)
        var bilinear = view.sample_bilinear(u, v)
        var environment = view.sample_environment(u, v)
        var base = (image * 289 + index) * 12
        comptime for channel in range(3):
            output[unsafe_offset=base + channel] = bitcast[.uint32](
                texel[channel]
            )
            output[unsafe_offset=base + 3 + channel] = bitcast[.uint32](
                nearest[channel]
            )
            output[unsafe_offset=base + 6 + channel] = bitcast[.uint32](
                bilinear[channel]
            )
            output[unsafe_offset=base + 9 + channel] = bitcast[.uint32](
                environment[channel]
            )


def test_device_palette_and_raw_fallback_recover_texels_and_sample_domains() raises:
    var surfaces = make_textures()
    var packed = pack_texture_tables(surfaces)
    with DeviceContext() as ctx:
        var descriptors = _upload_nonempty(ctx, packed.descriptors)
        var words = _upload_nonempty(ctx, packed.words)
        var output = ctx.enqueue_create_buffer[.uint32](4 * 289 * 12)
        ctx.enqueue_function[_decode_kernel[True]](
            descriptors, words, output, grid_dim=3, block_dim=128
        )
        ctx.synchronize()
        with output.map_to_host() as values:
            for image in range(4):
                ref source = surfaces.image_textures[image]
                for pixel in range(source.width * source.height):
                    var base = (image * 289 + pixel) * 12
                    var u = Float32(pixel) / 31.0 - 2.0
                    var v = Float32(pixel % 53) / 19.0 - 1.0
                    var nearest = source.sample(u, v)
                    var bilinear = source.sample_bilinear(u, v)
                    var environment = source.sample_environment(u, v)
                    comptime for channel in range(3):
                        assert_equal(
                            values[base + channel],
                            bitcast[.uint32](
                                source.pixels[3 * pixel + channel]
                            ),
                        )
                        assert_equal(
                            values[base + 3 + channel],
                            bitcast[.uint32](nearest[channel]),
                        )
                        assert_almost_equal(
                            bitcast[.float32](values[base + 6 + channel]),
                            bilinear[channel],
                            atol=2e-6,
                        )
                        assert_equal(
                            values[base + 9 + channel],
                            bitcast[.uint32](environment[channel]),
                        )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
