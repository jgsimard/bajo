"""Small libpng-backed RGB texture decoder for PBRT assets."""

from std.ffi import OwnedDLHandle
from std.math import pow

from bajo.parser.obj.mmap import MMap
from bajo.rt.types import ImageTexture


@always_inline
def _srgb_to_linear(value: Float32) -> Float32:
    if value <= 0.04045:
        return value / 12.92
    return pow((value + 0.055) / 1.055, Float32(2.4))


def parse_png[
    origin: ImmOrigin
](bytes: Span[UInt8, origin]) raises -> ImageTexture:
    if len(bytes) == 0:
        raise Error("PNG image is empty")

    # png_image is a stable simplified-API struct. libpng 1.6's ABI uses 104
    # bytes on our supported linux-64 target; the UInt32 fields below are
    # version, width, height, format, flags, and colormap_entries.
    var image = List[UInt8](length=104, fill=0)
    var words = image.unsafe_ptr().unsafe_bitcast[UInt32]()
    words[unsafe_offset=2] = UInt32(1)  # PNG_IMAGE_VERSION
    var lib = OwnedDLHandle("libpng16.so.16")
    var ok = lib.call["png_image_begin_read_from_memory", Int32](
        image.unsafe_ptr(), bytes.unsafe_ptr(), UInt(len(bytes))
    )
    if ok == 0:
        lib.call["png_image_free"](image.unsafe_ptr())
        raise Error("libpng could not read image header")

    var width = Int(words[unsafe_offset=3])
    var height = Int(words[unsafe_offset=4])
    if width <= 0 or height <= 0:
        lib.call["png_image_free"](image.unsafe_ptr())
        raise Error("PNG dimensions must be positive")

    words[unsafe_offset=5] = UInt32(2)  # PNG_FORMAT_RGB
    var encoded = List[UInt8](length=width * height * 3, fill=0)
    ok = lib.call["png_image_finish_read", Int32](
        image.unsafe_ptr(),
        Int(0),
        encoded.unsafe_ptr(),
        Int32(0),
        Int(0),
    )
    _ = len(bytes)  # Keep memory-mapped input alive through finish_read.
    lib.call["png_image_free"](image.unsafe_ptr())
    if ok == 0:
        raise Error("libpng could not decode image pixels")

    var pixels = List[Float32](capacity=len(encoded))
    for channel in encoded:
        pixels.append(_srgb_to_linear(Float32(channel) / 255.0))
    return ImageTexture(width, height, pixels^)


def read_png(path: String) raises -> ImageTexture:
    var mapped = MMap[ImmutAnyOrigin](path)
    var texture = parse_png(mapped.as_bytes_span())
    _ = mapped.byte_length()
    return texture^
