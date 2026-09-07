"""Minimal OpenEXR scanline decoder for linear RGB environment maps."""

from std.ffi import OwnedDLHandle
from std.memory import bitcast
from std.utils.numerics import isfinite

from bajo.parser.obj.mmap import MMap
from bajo.rt.types import ImageTexture


comptime _EXR_MAGIC = UInt32(20000630)
comptime _NO_COMPRESSION = UInt8(0)
comptime _ZIPS_COMPRESSION = UInt8(2)
comptime _ZIP_COMPRESSION = UInt8(3)
comptime _UINT_PIXEL = Int32(0)
comptime _HALF_PIXEL = Int32(1)
comptime _FLOAT_PIXEL = Int32(2)


struct _Cursor[origin: ImmOrigin]:
    var bytes: ImmSpan[UInt8, Self.origin]
    var pos: Int

    def __init__(out self, bytes: ImmSpan[UInt8, Self.origin]):
        self.bytes = bytes
        self.pos = 0

    def require(self, count: Int) raises:
        if count < 0 or self.pos > len(self.bytes) - count:
            raise Error("truncated OpenEXR file")

    def read_u8(mut self) raises -> UInt8:
        self.require(1)
        var value = self.bytes[self.pos]
        self.pos += 1
        return value

    def read_u32(mut self) raises -> UInt32:
        self.require(4)
        var value = UInt32(self.bytes[self.pos])
        value |= UInt32(self.bytes[self.pos + 1]) << UInt32(8)
        value |= UInt32(self.bytes[self.pos + 2]) << UInt32(16)
        value |= UInt32(self.bytes[self.pos + 3]) << UInt32(24)
        self.pos += 4
        return value

    def read_i32(mut self) raises -> Int32:
        return bitcast[.int32](self.read_u32())

    def read_u64(mut self) raises -> UInt64:
        self.require(8)
        var value = UInt64(0)
        for byte in range(8):
            value |= UInt64(self.bytes[self.pos + byte]) << UInt64(8 * byte)
        self.pos += 8
        return value

    def read_string(mut self) raises -> String:
        var start = self.pos
        while self.pos < len(self.bytes) and self.bytes[self.pos] != UInt8(0):
            self.pos += 1
        if self.pos >= len(self.bytes):
            raise Error("unterminated OpenEXR header string")
        var value = String(
            StringSpan[Self.origin](
                unsafe_from_utf8=self.bytes[start : self.pos]
            )
        )
        self.pos += 1
        return value^

    def skip(mut self, count: Int) raises:
        self.require(count)
        self.pos += count


@fieldwise_init
struct _Channel(Copyable):
    var name: String
    var pixel_type: Int32
    var byte_width: Int


struct _Header:
    var channels: List[_Channel]
    var compression: UInt8
    var min_x: Int
    var min_y: Int
    var max_x: Int
    var max_y: Int
    var data_offset: Int

    def __init__(out self):
        self.channels = List[_Channel]()
        self.compression = UInt8(255)
        self.min_x = 0
        self.min_y = 0
        self.max_x = -1
        self.max_y = -1
        self.data_offset = 0

    def width(self) -> Int:
        return self.max_x - self.min_x + 1

    def height(self) -> Int:
        return self.max_y - self.min_y + 1

    def scanlines_per_chunk(self) -> Int:
        if self.compression == _ZIP_COMPRESSION:
            return 16
        return 1


def _parse_channels(mut header: _Header, payload: ImmSpan[UInt8, _]) raises:
    var cursor = _Cursor(payload)
    while cursor.pos < len(payload):
        var name = cursor.read_string()
        if name.byte_length() == 0:
            if cursor.pos != len(payload):
                raise Error("unexpected data after OpenEXR channel list")
            return
        var pixel_type = cursor.read_i32()
        if (
            pixel_type != _UINT_PIXEL
            and pixel_type != _HALF_PIXEL
            and pixel_type != _FLOAT_PIXEL
        ):
            raise Error("unsupported OpenEXR channel pixel type")
        _ = cursor.read_u8()  # pLinear
        cursor.skip(3)  # reserved
        var x_sampling = cursor.read_i32()
        var y_sampling = cursor.read_i32()
        if x_sampling != 1 or y_sampling != 1:
            raise Error("subsampled OpenEXR channels are not supported")
        var byte_width = 2 if pixel_type == _HALF_PIXEL else 4
        header.channels.append(_Channel(name^, pixel_type, byte_width))
    raise Error("OpenEXR channel list is missing its terminator")


def _parse_header(bytes: ImmSpan[UInt8, _]) raises -> _Header:
    var cursor = _Cursor(bytes)
    if cursor.read_u32() != _EXR_MAGIC:
        raise Error("invalid OpenEXR magic number")
    var version = cursor.read_u32()
    if (version & UInt32(0xFF)) != UInt32(2):
        raise Error("only OpenEXR version 2 files are supported")
    if (version & UInt32(0xFFFFFF00)) != 0:
        raise Error(
            "multipart, tiled, and deep OpenEXR files are not supported"
        )

    var header = _Header()
    while True:
        var name = cursor.read_string()
        if name.byte_length() == 0:
            header.data_offset = cursor.pos
            break
        var attribute_type = cursor.read_string()
        var size = Int(cursor.read_u32())
        cursor.require(size)
        var payload = bytes[cursor.pos : cursor.pos + size]
        if name == "channels":
            if attribute_type != "chlist":
                raise Error("invalid OpenEXR channels attribute")
            _parse_channels(header, payload)
        elif name == "compression":
            if attribute_type != "compression" or size != 1:
                raise Error("invalid OpenEXR compression attribute")
            header.compression = payload[0]
        elif name == "dataWindow":
            if attribute_type != "box2i" or size != 16:
                raise Error("invalid OpenEXR dataWindow attribute")
            var window = _Cursor(payload)
            header.min_x = Int(window.read_i32())
            header.min_y = Int(window.read_i32())
            header.max_x = Int(window.read_i32())
            header.max_y = Int(window.read_i32())
        cursor.skip(size)

    if len(header.channels) == 0:
        raise Error("OpenEXR file has no channels")
    if header.width() <= 0 or header.height() <= 0:
        raise Error("OpenEXR data window must be non-empty")
    if (
        header.compression != _NO_COMPRESSION
        and header.compression != _ZIPS_COMPRESSION
        and header.compression != _ZIP_COMPRESSION
    ):
        raise Error("only uncompressed and ZIP OpenEXR files are supported")
    return header^


def _undo_zip_prediction(mut bytes: List[UInt8]):
    for index in range(1, len(bytes)):
        bytes[index] = UInt8(
            (Int(bytes[index - 1]) + Int(bytes[index]) - 128) & 255
        )


def _undo_zip_interleave(encoded: List[UInt8]) -> List[UInt8]:
    var decoded = List[UInt8](length=len(encoded), fill=0)
    var first = 0
    var second = (len(encoded) + 1) / 2
    for index in range(len(encoded)):
        if index % 2 == 0:
            decoded[index] = encoded[first]
            first += 1
        else:
            decoded[index] = encoded[second]
            second += 1
    return decoded^


def _decompress_zip(
    payload: ImmSpan[UInt8, _], decoded_size: Int
) raises -> List[UInt8]:
    var predicted = List[UInt8](length=decoded_size, fill=0)
    var result_size = List[UInt](length=1, fill=UInt(decoded_size))
    var lib = OwnedDLHandle("libz.so.1")
    var status = lib.call["uncompress", Int32](
        predicted.unsafe_ptr(),
        result_size.unsafe_ptr(),
        payload.unsafe_ptr(),
        UInt(len(payload)),
    )
    if status != 0 or Int(result_size[0]) != decoded_size:
        raise Error("zlib failed to decompress OpenEXR scanlines")
    _undo_zip_prediction(predicted)
    return _undo_zip_interleave(predicted)


@always_inline
def _half_to_float(bits: UInt16) -> Float32:
    var sign = UInt32(bits >> UInt16(15))
    var exponent = UInt32((bits >> UInt16(10)) & UInt16(31))
    var mantissa = UInt32(bits & UInt16(1023))
    if exponent == 0:
        if mantissa == 0:
            return bitcast[.float32](sign << UInt32(31))
        var value = Float32(mantissa) * Float32(5.960464477539063e-8)
        return -value if sign != 0 else value
    if exponent == 31:
        return bitcast[.float32](
            (sign << UInt32(31)) | UInt32(0x7F800000) | (mantissa << UInt32(13))
        )
    return bitcast[.float32](
        (sign << UInt32(31))
        | ((exponent + UInt32(112)) << UInt32(23))
        | (mantissa << UInt32(13))
    )


def _read_channel_value(
    bytes: ImmSpan[UInt8, _], mut offset: Int, pixel_type: Int32
) raises -> Float32:
    if pixel_type == _HALF_PIXEL:
        if offset > len(bytes) - 2:
            raise Error("truncated OpenEXR channel data")
        var bits = UInt16(bytes[offset]) | (
            UInt16(bytes[offset + 1]) << UInt16(8)
        )
        offset += 2
        return _half_to_float(bits)
    if offset > len(bytes) - 4:
        raise Error("truncated OpenEXR channel data")
    var bits = UInt32(bytes[offset])
    bits |= UInt32(bytes[offset + 1]) << UInt32(8)
    bits |= UInt32(bytes[offset + 2]) << UInt32(16)
    bits |= UInt32(bytes[offset + 3]) << UInt32(24)
    offset += 4
    if pixel_type == _UINT_PIXEL:
        return Float32(bits)
    return bitcast[.float32](bits)


def _decode_chunk(
    mut pixels: List[Float32],
    header: _Header,
    chunk_y: Int,
    chunk: ImmSpan[UInt8, _],
) raises:
    var width = header.width()
    var line_count = min(
        header.scanlines_per_chunk(), header.max_y - chunk_y + 1
    )
    if chunk_y < header.min_y or line_count <= 0:
        raise Error("OpenEXR scanline chunk is outside the data window")
    var bytes_per_line = 0
    for channel in header.channels:
        bytes_per_line += width * channel.byte_width
    if len(chunk) != bytes_per_line * line_count:
        raise Error("unexpected OpenEXR scanline chunk size")

    var channel_offset = 0
    for line in range(line_count):
        var pixel_y = chunk_y + line - header.min_y
        for channel in header.channels:
            var output_channel = -1
            if channel.name == "R":
                output_channel = 0
            elif channel.name == "G":
                output_channel = 1
            elif channel.name == "B":
                output_channel = 2
            for x in range(width):
                var value = _read_channel_value(
                    chunk, channel_offset, channel.pixel_type
                )
                if output_channel >= 0:
                    if not isfinite(value):
                        raise Error(
                            "OpenEXR environment contains a non-finite value"
                        )
                    pixels[3 * (pixel_y * width + x) + output_channel] = value


def parse_exr[
    origin: ImmOrigin
](bytes: ImmSpan[UInt8, origin]) raises -> ImageTexture:
    var header = _parse_header(bytes)
    var red = False
    var green = False
    var blue = False
    for channel in header.channels:
        red |= channel.name == "R"
        green |= channel.name == "G"
        blue |= channel.name == "B"
    if not red or not green or not blue:
        raise Error("OpenEXR environment requires R, G, and B channels")

    var chunk_count = (header.height() + header.scanlines_per_chunk() - 1) / (
        header.scanlines_per_chunk()
    )
    var table = _Cursor(bytes)
    table.pos = header.data_offset
    var offsets = List[UInt64](capacity=chunk_count)
    for _ in range(chunk_count):
        offsets.append(table.read_u64())

    var pixels = List[Float32](
        length=header.width() * header.height() * 3, fill=0.0
    )
    for chunk_idx in range(chunk_count):
        var chunk_offset = offsets[chunk_idx]
        if chunk_offset > UInt64(len(bytes) - 8):
            raise Error("OpenEXR chunk offset is out of range")
        var cursor = _Cursor(bytes)
        cursor.pos = Int(chunk_offset)
        var chunk_y = Int(cursor.read_i32())
        var packed_size = Int(cursor.read_u32())
        cursor.require(packed_size)
        var packed = bytes[cursor.pos : cursor.pos + packed_size]

        var line_count = min(
            header.scanlines_per_chunk(), header.max_y - chunk_y + 1
        )
        var bytes_per_line = 0
        for channel in header.channels:
            bytes_per_line += header.width() * channel.byte_width
        var decoded_size = bytes_per_line * line_count
        var decoded: List[UInt8]
        if header.compression == _NO_COMPRESSION:
            decoded = List[UInt8](capacity=packed_size)
            decoded.extend(packed)
        else:
            decoded = _decompress_zip(packed, decoded_size)
        _decode_chunk(pixels, header, chunk_y, decoded)

    return ImageTexture(header.width(), header.height(), pixels^)


def read_exr(path: String) raises -> ImageTexture:
    var mapped = MMap(path)
    return parse_exr(mapped.as_bytes_span())
