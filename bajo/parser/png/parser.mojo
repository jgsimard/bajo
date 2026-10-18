"""SIMD RGB PNG loading, with libpng for other formats and color transforms."""

from max.algorithm import parallelize
from std.ffi import OwnedDLHandle
from std.math import ceildiv, pow
from std.sys import num_logical_cores, simd_width_of

from bajo.parser.mmap import MMap
from bajo.parser.checksum import crc32
from bajo.parser.deflate import inflate_zlib
from bajo.parser.png.filters import _unfilter_rgb
from bajo.rt.material_types import ImageTexture


comptime _SRGB_SIMD_WIDTH = simd_width_of[Float32]()
comptime _SRGB_CHANNELS_PER_TASK = 4 * 1024 * 1024


@always_inline
def _srgb_to_linear(value: Float32) -> Float32:
    if value <= 0.04045:
        return value / 12.92
    return pow((value + 0.055) / 1.055, Float32(2.4))


def _srgb_u8_table() -> List[Float32]:
    """Evaluate the existing scalar transfer function for every u8 value."""
    return List[Float32](
        length=256,
        fill_with=lambda (value: Int) -> Float32: _srgb_to_linear(
            Float32(value) / 255.0
        ),
    )


def _linearize_srgb_u8[
    channels_per_task: Int = _SRGB_CHANNELS_PER_TASK
](encoded: List[UInt8]) -> List[Float32]:
    """Convert exact u8 sRGB values with threaded native-width SIMD gathers."""
    var table = _srgb_u8_table()
    var pixels = List[Float32]()
    pixels.resize(unsafe_uninit_length=len(encoded))
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


@always_inline
def _png_u32(bytes: ImmSpan[UInt8, _], offset: Int) -> UInt32:
    return (
        (UInt32(bytes[offset]) << 24)
        | (UInt32(bytes[offset + 1]) << 16)
        | (UInt32(bytes[offset + 2]) << 8)
        | UInt32(bytes[offset + 3])
    )


def _linearize_png_rows(
    raw: List[UInt8], width: Int, height: Int
) -> List[Float32]:
    var table = _srgb_u8_table()
    var pixels = List[Float32]()
    var row_bytes = width * 3
    pixels.resize(unsafe_uninit_length=row_bytes * height)
    var tasks = max(1, ceildiv(row_bytes * height, _SRGB_CHANNELS_PER_TASK))

    def convert(task: Int) {imm, mut pixels}:
        for y in range(height * task // tasks, height * (task + 1) // tasks):
            var source = raw.unsafe_ptr().unsafe_offset(y * (row_bytes + 1) + 1)
            var target = pixels.unsafe_ptr().unsafe_offset(y * row_bytes)
            var end = row_bytes // _SRGB_SIMD_WIDTH * _SRGB_SIMD_WIDTH
            for x in range(0, end, _SRGB_SIMD_WIDTH):
                var indices = (
                    source.unsafe_offset(x)
                    .unsafe_load[width=_SRGB_SIMD_WIDTH]()
                    .cast[.int32]()
                )
                target.unsafe_offset(x).unsafe_store[width=_SRGB_SIMD_WIDTH](
                    table.unsafe_ptr().unsafe_gather(indices)
                )
            for x in range(end, row_bytes):
                target[unsafe_offset=x] = table[Int(source[unsafe_offset=x])]

    if tasks > 1:
        parallelize(convert, tasks, min(num_logical_cores(), tasks))
    else:
        convert(0)
    return pixels^


def _inflate_png(input: ImmSpan[UInt8, _], output: MutSpan[UInt8, _]) raises:
    if inflate_zlib(input, output) != len(output):
        raise Error("unexpected PNG scanline byte count")


def _try_parse_png_simd(
    bytes: ImmSpan[UInt8, _]
) raises -> Optional[ImageTexture]:
    # Only bypass libpng for opaque, noninterlaced 8-bit RGB. In particular,
    # alpha, palettes, gAMA/ICC profiles, and transparency retain its transforms.
    if len(bytes) < 33:
        return None
    if _png_u32(bytes, 0) != 0x89504E47 or _png_u32(bytes, 4) != 0x0D0A1A0A:
        return None
    if _png_u32(bytes, 8) != 13 or _png_u32(bytes, 12) != 0x49484452:
        return None
    if (
        bytes[24] != 8
        or bytes[25] != 2
        or bytes[26] != 0
        or bytes[27] != 0
        or bytes[28] != 0
    ):
        return None
    var width = Int(_png_u32(bytes, 16))
    var height = Int(_png_u32(bytes, 20))
    if width == 0 or height == 0 or width > 0x7FFFFFFF or height > 0x7FFFFFFF:
        raise Error("invalid PNG dimensions")
    var stride = width * 3 + 1
    if height > (Int.MAX - 1) // stride:
        raise Error("PNG dimensions overflow decoded storage")

    var ranges = List[Tuple[Int, Int]]()
    var compressed_size = 0
    var offset = 8
    var ended = False
    var data_ended = False
    while offset <= len(bytes) - 12:
        var size = Int(_png_u32(bytes, offset))
        if size > 0x7FFFFFFF or size > len(bytes) - offset - 12:
            raise Error("truncated PNG chunk")
        var tag = _png_u32(bytes, offset + 4)
        var begin = offset + 8
        var end = begin + size
        var crc = crc32(bytes[offset + 4 : end])
        if crc != _png_u32(bytes, end):
            raise Error("PNG chunk checksum mismatch")
        match tag:
            case 0x49484452:  # IHDR
                if offset != 8:
                    raise Error("duplicate PNG header")
            case 0x49444154:  # IDAT
                if data_ended:
                    raise Error("PNG IDAT chunks must be consecutive")
                ranges.append((begin, end))
                compressed_size += size
            case 0x49454E44:  # IEND
                if size != 0 or len(ranges) == 0:
                    raise Error("invalid PNG end chunk")
                ended = True
            case 0x73524742:  # sRGB: already matches the output transfer function
                if size != 1 or bytes[begin] > 3 or len(ranges) != 0:
                    return None
            case 0x70485973 | 0x74494D45 | 0x74455874 | 0x7A545874 | 0x69545874:
                # pHYs, tIME, tEXt, zTXt, iTXt do not alter RGB samples.
                pass
            case _:
                return None
        if tag != 0x49444154 and len(ranges) != 0:
            data_ended = True
        offset = end + 4
        if ended:
            break
    if not ended:
        raise Error("PNG is missing its end chunk")

    var raw = List[UInt8]()
    var decoded_size = stride * height
    raw.resize(unsafe_uninit_length=decoded_size + 1)
    raw[decoded_size] = 0  # Padding for the final overlapping SIMD pixel load.
    if len(ranges) > 1:
        var compressed = List[UInt8]()
        compressed.resize(unsafe_uninit_length=compressed_size)
        var write = 0
        for chunk in ranges:
            var count = chunk[1] - chunk[0]
            Span(compressed)[write : write + count].copy_from(
                bytes[chunk[0] : chunk[1]]
            )
            write += count
        _inflate_png(compressed, Span(raw)[:decoded_size])
    else:
        _inflate_png(
            bytes[ranges[0][0] : ranges[0][1]], Span(raw)[:decoded_size]
        )
    _unfilter_rgb(raw, width, height)
    return ImageTexture(width, height, _linearize_png_rows(raw, width, height))


def _parse_png_libpng[
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
    var encoded = List[UInt8]()
    encoded.resize(unsafe_uninit_length=width * height * 3)
    # With a null background libpng composites alpha onto the existing output
    # buffer. Supply black explicitly instead of reading uninitialized bytes.
    var background = List[UInt8](length=3, fill=0)
    ok = lib.call["png_image_finish_read", Int32](
        image.unsafe_ptr(),
        background.unsafe_ptr(),
        encoded.unsafe_ptr(),
        Int32(0),
        Int(0),
    )
    _ = len(background)
    _ = len(bytes)  # Keep memory-mapped input alive through finish_read.
    lib.call["png_image_free"](image.unsafe_ptr())
    if ok == 0:
        raise Error("libpng could not decode image pixels")

    var pixels = _linearize_srgb_u8(encoded)
    return ImageTexture(width, height, pixels^)


def parse_png[
    origin: ImmOrigin
](bytes: Span[UInt8, origin]) raises -> ImageTexture:
    var image = _try_parse_png_simd(bytes)
    if image:
        return image.take()
    return _parse_png_libpng(bytes)


def read_png(path: String) raises -> ImageTexture:
    var mapped = MMap(path)
    return parse_png(mapped.as_bytes_span())
