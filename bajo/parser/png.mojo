"""Small libpng-backed RGB texture decoder for PBRT assets."""

from max.algorithm import parallelize
from std.ffi import OwnedDLHandle
from std.math import ceildiv, pow
from std.sys import num_logical_cores, simd_width_of

from bajo.parser.obj.mmap import MMap
from bajo.rt.types import ImageTexture


comptime _SRGB_SIMD_WIDTH = simd_width_of[Float32]()
comptime _SRGB_CHANNELS_PER_TASK = 4 * 1024 * 1024


@always_inline
def _srgb_to_linear(value: Float32) -> Float32:
    if value <= 0.04045:
        return value / 12.92
    return pow((value + 0.055) / 1.055, Float32(2.4))


def _srgb_u8_table() -> List[Float32]:
    """Evaluate the existing scalar transfer function for every u8 value."""
    var table = List[Float32](length=256, fill=0.0)
    for value in range(256):
        table[value] = _srgb_to_linear(Float32(value) / 255.0)
    return table^


def _linearize_srgb_u8[
    channels_per_task: Int = _SRGB_CHANNELS_PER_TASK
](encoded: List[UInt8]) -> List[Float32]:
    """Convert exact u8 sRGB values with threaded native-width SIMD gathers."""
    var table = _srgb_u8_table()
    var pixels = List[Float32](length=len(encoded), fill=0.0)
    var vector_end = len(encoded) - len(encoded) % _SRGB_SIMD_WIDTH
    var vector_count = vector_end // _SRGB_SIMD_WIDTH
    comptime assert channels_per_task >= _SRGB_SIMD_WIDTH
    var vectors_per_task = channels_per_task // _SRGB_SIMD_WIDTH
    var task_count = ceildiv(vector_count, vectors_per_task)

    def worker(task_idx: Int) {imm, mut pixels}:
        var vector_begin = task_idx * vectors_per_task
        var vector_stop = min(vector_begin + vectors_per_task, vector_count)
        var encoded_ptr = encoded.unsafe_ptr()
        var table_ptr = table.unsafe_ptr()
        var pixels_ptr = pixels.unsafe_ptr()
        for vector_idx in range(vector_begin, vector_stop):
            var base = vector_idx * _SRGB_SIMD_WIDTH
            var indices = (
                encoded_ptr.unsafe_offset(base)
                .unsafe_load[width=_SRGB_SIMD_WIDTH]()
                .cast[.int32]()
            )
            var linear = table_ptr.unsafe_gather(indices)
            pixels_ptr.unsafe_offset(base).unsafe_store[width=_SRGB_SIMD_WIDTH](
                linear
            )

    if task_count > 1:
        parallelize(worker, task_count, min(num_logical_cores(), task_count))
    elif task_count == 1:
        worker(0)

    for index in range(vector_end, len(encoded)):
        pixels[index] = table[Int(encoded[index])]
    return pixels^


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

    var pixels = _linearize_srgb_u8(encoded)
    return ImageTexture(width, height, pixels^)


def read_png(path: String) raises -> ImageTexture:
    var mapped = MMap(path)
    return parse_png(mapped.as_bytes_span())
