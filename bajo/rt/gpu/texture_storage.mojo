"""Lossless GPU image packing with exact Float32 palettes and raw fallback."""

from std.memory import bitcast
from bajo.rt.material_types import SurfaceStore
from bajo.rt.material_storage import TextureDescLayout


@fieldwise_init
struct PackedTextureTables:
    var descriptors: List[UInt32]
    var words: List[Float32]


@fieldwise_init
struct TextureUploadPlan:
    var descriptors: List[UInt32]
    # Raw images stay in SurfaceStore; only compressed images own new words.
    var packed_images: List[List[Float32]]
    var word_count: Int


def plan_texture_upload(surfaces: SurfaceStore) raises -> TextureUploadPlan:
    """Store three 8-bit palette indices per texel when it saves memory.

    A palette contains the original Float32 bit patterns, not quantized values.
    Images with more than 256 distinct channel values remain Float32 RGB.
    The unused top byte of each packed texel stays zero, avoiding NaN payloads
    in the Float32 buffer that carries these words to the device.
    """
    var descriptors = List[UInt32](
        length=len(surfaces.image_textures) * TextureDescLayout.STRIDE,
        fill=0,
    )
    var packed_images = List[List[Float32]](
        capacity=len(surfaces.image_textures)
    )
    var word_count = 0
    for index in range(len(surfaces.image_textures)):
        ref image = surfaces.image_textures[index]
        var offset = word_count
        var words = List[Float32]()
        var pixel_count = image.width * image.height
        var format = TextureDescLayout.RGB_F32
        if pixel_count > 128:
            var palette = Dict[UInt32, UInt32]()
            words = List[Float32](capacity=min(256 + pixel_count, 1024))
            words.resize(256, fill=0.0)
            var encoded = True
            for pixel in range(pixel_count):
                var packed = UInt32(0)
                comptime for channel in range(3):
                    var value = image.pixels[3 * pixel + channel]
                    var bits = bitcast[.uint32](value)
                    var code: UInt32
                    if case .Some(existing) = palette.get(bits):
                        code = existing
                    else:
                        if len(palette) == 256:
                            encoded = False
                            break
                        code = UInt32(len(palette))
                        palette[bits] = code
                        words[Int(code)] = value
                    packed |= code << UInt32(8 * channel)
                if not encoded:
                    break
                words.append(bitcast[.float32](packed))
            if encoded:
                format = TextureDescLayout.PALETTE_RGB8
            else:
                words = List[Float32]()
        if format == TextureDescLayout.RGB_F32:
            word_count += 3 * pixel_count
        else:
            word_count += len(words)
        packed_images.append(words^)
        if word_count >= Int(UInt32.MAX):
            raise Error("GPU texture storage exceeds 32-bit word offsets")
        var base = index * TextureDescLayout.STRIDE
        descriptors[base + TextureDescLayout.PIXEL_OFFSET] = UInt32(offset)
        descriptors[base + TextureDescLayout.WIDTH] = UInt32(image.width)
        descriptors[base + TextureDescLayout.HEIGHT] = UInt32(image.height)
        descriptors[base + TextureDescLayout.FORMAT] = format
    return TextureUploadPlan(descriptors^, packed_images^, word_count)


def pack_texture_tables(surfaces: SurfaceStore) raises -> PackedTextureTables:
    """Materialize a plan for host-side inspection and reference checks."""
    var plan = plan_texture_upload(surfaces)
    var words = List[Float32](capacity=plan.word_count)
    for index in range(len(surfaces.image_textures)):
        if (
            plan.descriptors[
                index * TextureDescLayout.STRIDE + TextureDescLayout.FORMAT
            ]
            == TextureDescLayout.RGB_F32
        ):
            words.extend(ImmSpan(surfaces.image_textures[index].pixels))
        else:
            words.extend(ImmSpan(plan.packed_images[index]))
    return PackedTextureTables(plan.descriptors.copy(), words^)
