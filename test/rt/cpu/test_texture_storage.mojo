"""Lossless texture storage preserves texel bits and sampling behavior."""

from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.rt.material_types import SurfaceStore, ImageTexture
from bajo.rt.material_storage import TextureDescLayout
from bajo.rt.gpu.texture_storage import pack_texture_tables
from bajo.rt.gpu.textures import _GpuImageTextureView
from bajo.parser.png.parser import _srgb_u8_table


def make_textures() -> SurfaceStore:
    var surfaces = SurfaceStore()
    _ = surfaces.add_image_texture(ImageTexture(1, 1, [3.5, 0.125, 0.75]))
    var pixels = List[Float32]()
    var table = _srgb_u8_table()
    for index in range(17 * 17 * 3):
        pixels.append(table[index % 256])
    _ = surfaces.add_image_texture(ImageTexture(17, 17, pixels^))
    pixels = List[Float32]()
    for index in range(257 * 3):
        # More than 256 distinct values, with the overflow late enough to
        # exercise rollback after packed texels have already been appended.
        pixels.append(Float32(index // 3) / Float32(257.0) + 0.000001)
    _ = surfaces.add_image_texture(ImageTexture(257, 1, pixels^))
    pixels = List[Float32]()
    for index in range(17 * 17 * 3):
        # Arbitrary Float32 bits, including distinct signs of zero and HDR.
        var code = index % 256
        var word = UInt32(0x3E800001) + UInt32(code * 67991)
        if code == 0:
            word = UInt32(0x80000000)
        elif code == 1:
            word = 0
        pixels.append(bitcast[.float32](word))
    _ = surfaces.add_image_texture(ImageTexture(17, 17, pixels^))
    return surfaces^


def test_palette_and_raw_fallback_preserve_every_float32_bit() raises:
    var surfaces = make_textures()
    var packed = pack_texture_tables(surfaces)
    assert_equal(len(packed.descriptors), 4 * TextureDescLayout.STRIDE)
    assert_equal(
        packed.descriptors[TextureDescLayout.FORMAT], TextureDescLayout.RGB_F32
    )
    assert_equal(
        packed.descriptors[TextureDescLayout.STRIDE + TextureDescLayout.FORMAT],
        TextureDescLayout.PALETTE_RGB8,
    )
    assert_equal(
        packed.descriptors[
            2 * TextureDescLayout.STRIDE + TextureDescLayout.FORMAT
        ],
        TextureDescLayout.RGB_F32,
    )
    assert_equal(
        packed.descriptors[
            3 * TextureDescLayout.STRIDE + TextureDescLayout.FORMAT
        ],
        TextureDescLayout.PALETTE_RGB8,
    )
    var expected_words = 3 + 2 * (256 + 17 * 17) + 257 * 3
    assert_equal(len(packed.words), expected_words)
    var descriptors = (
        packed.descriptors.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var words = (
        packed.words.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for image in range(len(surfaces.image_textures)):
        var view = _GpuImageTextureView.load(UInt32(image), descriptors, words)
        ref source = surfaces.image_textures[image]
        for pixel in range(source.width * source.height):
            var actual = view.texel(pixel)
            comptime for channel in range(3):
                assert_equal(
                    bitcast[.uint32](actual[channel]),
                    bitcast[.uint32](source.pixels[3 * pixel + channel]),
                )


def test_paletted_nearest_bilinear_and_environment_addressing_match_source() raises:
    var surfaces = make_textures()
    var packed = pack_texture_tables(surfaces)
    var descriptors = (
        packed.descriptors.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var words = (
        packed.words.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for image in range(len(surfaces.image_textures)):
        var view = _GpuImageTextureView.load(UInt32(image), descriptors, words)
        ref source = surfaces.image_textures[image]
        for index in range(257):
            var u = Float32(index) / 31.0 - 2.0
            var v = Float32(index % 53) / 19.0 - 1.0
            var nearest = view.sample(u, v)
            var expected_nearest = source.sample(u, v)
            var bilinear = view.sample_bilinear(u, v)
            var expected_bilinear = source.sample_bilinear(u, v)
            var environment = view.sample_environment(u, v)
            var expected_environment = source.sample_environment(u, v)
            comptime for channel in range(3):
                assert_equal(
                    bitcast[.uint32](nearest[channel]),
                    bitcast[.uint32](expected_nearest[channel]),
                )
                assert_equal(
                    bitcast[.uint32](bilinear[channel]),
                    bitcast[.uint32](expected_bilinear[channel]),
                )
                assert_equal(
                    bitcast[.uint32](environment[channel]),
                    bitcast[.uint32](expected_environment[channel]),
                )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
