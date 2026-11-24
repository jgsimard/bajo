"""Lossless texture storage preserves texel bits and sampling behavior."""

from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
)
from bajo.rt.material_types import SurfaceStore, ImageTexture, NO_TEXTURE
from bajo.rt.material_storage import TextureDescLayout
from bajo.rt.gpu.texture_storage import pack_texture_tables, plan_texture_upload
from bajo.rt.gpu.textures import _GpuImageTextureView
from std.builtin.globals import global_constant
from bajo.core.color import _SRGB_U8_VALUES


def make_textures() -> SurfaceStore:
    var surfaces = SurfaceStore()
    _ = surfaces.add_image_texture(ImageTexture(1, 1, [3.5, 0.125, 0.75]))
    var pixels = List[Float32]()
    ref table = global_constant[_SRGB_U8_VALUES]()
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


def _check_texture_groups[
    width: SIMDLength, bilinear: Bool
](surfaces: SurfaceStore) raises:
    for pattern in range(6):
        var ids = SIMD[.uint32, width](NO_TEXTURE)
        var u = SIMD[.float32, width](0)
        var v = SIMD[.float32, width](0)
        for lane in range(Int(width)):
            var index = (
                lane // 3 if pattern
                == 0 else lane if pattern
                == 1 else 2 if pattern
                == 2 else lane % 2
            ) % 4
            if pattern != 4 and (pattern != 3 or lane % 3):
                ids[lane] = UInt32(index)
                u[lane] = Float32(lane) / 7 - 1.3
                v[lane] = Float32(lane) / 11 + 0.2
            else:
                u[lane] = bitcast[.float32](UInt32(0x7FC00000))
                v[lane] = u[lane]
        var actual = surfaces.sample_image_textures[width, bilinear](ids, u, v)
        for lane in range(Int(width)):
            if ids[lane] == NO_TEXTURE:
                for channel in range(3):
                    assert_equal(actual[channel][lane], Float32(1))
            else:
                ref image = surfaces.image_textures[Int(ids[lane])]
                var expected = image.sample_bilinear(
                    u[lane], v[lane]
                ) if bilinear else image.sample(u[lane], v[lane])
                for channel in range(3):
                    assert_almost_equal(
                        actual[channel][lane], expected[channel], atol=1e-6
                    )


def test_cpu_texture_packets_preserve_mixed_groups_missing_maps_and_bilinear_samples() raises:
    var surfaces = make_textures()
    comptime for width in [1, 4, 8, 16, 32]:
        _check_texture_groups[width, False](surfaces)
        _check_texture_groups[width, True](surfaces)


def test_upload_plan_leaves_raw_images_in_their_original_storage() raises:
    var surfaces = make_textures()
    var plan = plan_texture_upload(surfaces)
    var packed = pack_texture_tables(surfaces)
    assert_equal(plan.descriptors, packed.descriptors)
    assert_equal(plan.word_count, len(packed.words))
    assert_equal(len(plan.packed_images[0]), 0)
    assert_equal(len(plan.packed_images[2]), 0)
    assert_equal(len(plan.packed_images[1]), 256 + 17 * 17)
    assert_equal(len(plan.packed_images[3]), 256 + 17 * 17)


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
    var descriptors = packed.descriptors.unsafe_ptr().unsafe_mut_cast[False]()
    var words = packed.words.unsafe_ptr().unsafe_mut_cast[False]()
    for image in range(len(surfaces.image_textures)):
        var view = _GpuImageTextureView.load(UInt32(image), descriptors, words)
        ref source = surfaces.image_textures[image]
        for pixel in range(source.width * source.height):
            var actual = view.texel(pixel)
            for channel in range(3):
                assert_equal(
                    bitcast[.uint32](actual[channel]),
                    bitcast[.uint32](source.pixels[3 * pixel + channel]),
                )


def test_paletted_nearest_bilinear_and_environment_addressing_match_source() raises:
    var surfaces = make_textures()
    var packed = pack_texture_tables(surfaces)
    var descriptors = packed.descriptors.unsafe_ptr().unsafe_mut_cast[False]()
    var words = packed.words.unsafe_ptr().unsafe_mut_cast[False]()
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
            for channel in range(3):
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
