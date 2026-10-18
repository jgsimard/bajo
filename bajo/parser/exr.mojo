"""Minimal OpenEXR scanline decoder for linear RGB environment maps."""

from std.bit import byte_swap
from std.memory import bitcast
from max.algorithm import parallelize
from std.math import ceildiv
from std.sys import Endian, num_logical_cores, simd_width_of
from std.utils.numerics import isfinite

from bajo.parser.mmap import MMap
from bajo.parser.deflate import inflate_zlib
from bajo.rt.material_types import ImageTexture


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
        var value = (
            self.bytes.unsafe_ptr()
            .unsafe_offset(self.pos)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[alignment=1]()
        )
        comptime if Endian.native() == .big:
            value = byte_swap(value)
        self.pos += 8
        return value

    def read_string(mut self) raises -> StringSpan[Self.origin]:
        var start = self.pos
        while self.pos < len(self.bytes) and self.bytes[self.pos] != UInt8(0):
            self.pos += 1
        if self.pos >= len(self.bytes):
            raise Error("unterminated OpenEXR header string")
        var value = StringSpan[Self.origin](
            unsafe_from_utf8=self.bytes[start : self.pos]
        )
        self.pos += 1
        return value

    def skip(mut self, count: Int) raises:
        self.require(count)
        self.pos += count


@fieldwise_init
struct _OutputChannel(EnumLike, Equatable, TrivialRegisterPassable):
    var value: UInt8

    comptime RED = Self(0)
    comptime GREEN = Self(1)
    comptime BLUE = Self(2)
    comptime IGNORE = Self(3)
    comptime _enum_case_names = ParameterList.of[
        "RED".value, "GREEN".value, "BLUE".value, "IGNORE".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType, NoneType
    ].values

    def _get_enum_discriminant(self) -> Int:
        return Int(self.value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass


@fieldwise_init
struct _Channel(Copyable):
    var output_channel: _OutputChannel
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
        var output_channel: _OutputChannel
        match name:
            case "R":
                output_channel = .RED
            case "G":
                output_channel = .GREEN
            case "B":
                output_channel = .BLUE
            case _:
                # EXR permits arbitrary auxiliary channel names.
                output_channel = .IGNORE
        header.channels.append(_Channel(output_channel, pixel_type, byte_width))
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
        match name:
            case "channels":
                if attribute_type != ImmStringSpan("chlist"):
                    raise Error("invalid OpenEXR channels attribute")
                _parse_channels(header, payload)
            case "compression":
                if attribute_type != ImmStringSpan("compression") or size != 1:
                    raise Error("invalid OpenEXR compression attribute")
                header.compression = payload[0]
            case "dataWindow":
                if attribute_type != ImmStringSpan("box2i") or size != 16:
                    raise Error("invalid OpenEXR dataWindow attribute")
                var window = _Cursor(payload)
                header.min_x = Int(window.read_i32())
                header.min_y = Int(window.read_i32())
                header.max_x = Int(window.read_i32())
                header.max_y = Int(window.read_i32())
            case _:
                pass
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


def _undo_zip_prediction(bytes: MutSpan[UInt8, _]):
    if len(bytes) == 0:
        return
    # Decode the recurrence as an inclusive prefix sum modulo 256. Keep
    # 16-byte blocks so every shuffle fits one 128-bit register.
    var carry = bytes[0]
    var vector_end = 1 + (len(bytes) - 1) // 16 * 16
    comptime zero = SIMD[.uint8, 16](0)
    for index in range(1, vector_end, 16):
        var ptr = bytes.unsafe_ptr().unsafe_offset(index)
        var values = ptr.unsafe_load[width=16]() - SIMD[.uint8, 16](128)
        values += values.shuffle[
            16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        ](zero)
        values += values.shuffle[
            16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
        ](zero)
        values += values.shuffle[
            16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ](zero)
        values += values.shuffle[
            16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7
        ](zero)
        values += SIMD[.uint8, 16](carry)
        ptr.unsafe_store[width=16](values)
        carry = values[15]
    for index in range(vector_end, len(bytes)):
        carry = UInt8((Int(carry) + Int(bytes[index]) - 128) & 255)
        bytes[index] = carry


def _undo_zip_interleave_into(
    encoded: ImmSpan[UInt8, _], decoded: MutSpan[UInt8, _]
):
    var half = (len(encoded) + 1) // 2
    var pairs = len(encoded) // 2
    var vector_end = pairs - pairs % 16
    var source = encoded.unsafe_ptr()
    var destination = decoded.unsafe_ptr()
    for index in range(0, vector_end, 16):
        var first = source.unsafe_offset(index).unsafe_load[width=16]()
        var second = source.unsafe_offset(half + index).unsafe_load[width=16]()
        # Pack each pair in the little-endian order used by the decoder's
        # native unaligned loads.
        var values = first.cast[.uint16]() | (second.cast[.uint16]() << 8)
        destination.unsafe_offset(2 * index).unsafe_bitcast[
            UInt16
        ]().unsafe_store[width=16, alignment=1](values)
    for index in range(vector_end, pairs):
        decoded[2 * index] = encoded[index]
        decoded[2 * index + 1] = encoded[half + index]
    if len(encoded) % 2:
        decoded[len(encoded) - 1] = encoded[half - 1]


def _undo_zip_interleave(encoded: ImmSpan[UInt8, _]) -> List[UInt8]:
    var decoded = List[UInt8](unsafe_uninit_length=len(encoded))
    _undo_zip_interleave_into(encoded, decoded)
    return decoded^


def _decompress_zip(
    payload: ImmSpan[UInt8, _], decoded_size: Int
) raises -> List[UInt8]:
    var predicted = List[UInt8](length=decoded_size, fill=0)
    if inflate_zlib(payload, predicted) != decoded_size:
        raise Error("unexpected OpenEXR scanline byte count")
    _undo_zip_prediction(predicted)
    return _undo_zip_interleave(predicted)


@always_inline
def _half_to_float(bits: UInt16) -> Float32:
    return bitcast[.float16](bits).cast[.float32]()


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
            var output_channel = channel.output_channel
            for x in range(width):
                var value = _read_channel_value(
                    chunk, channel_offset, channel.pixel_type
                )
                if output_channel != _OutputChannel.IGNORE:
                    if not isfinite(value):
                        raise Error(
                            "OpenEXR environment contains a non-finite value"
                        )
                    pixels[
                        3 * (pixel_y * width + x) + Int(output_channel.value)
                    ] = value


def _parse_exr_serial[
    origin: ImmOrigin
](bytes: ImmSpan[UInt8, origin]) raises -> ImageTexture:
    var header = _parse_header(bytes)
    var red = False
    var green = False
    var blue = False
    for channel in header.channels:
        red |= channel.output_channel == _OutputChannel.RED
        green |= channel.output_channel == _OutputChannel.GREEN
        blue |= channel.output_channel == _OutputChannel.BLUE
    if not red or not green or not blue:
        raise Error("OpenEXR environment requires R, G, and B channels")

    var chunk_count = (header.height() + header.scanlines_per_chunk() - 1) / (
        header.scanlines_per_chunk()
    )
    var table = _Cursor(bytes)
    table.pos = header.data_offset
    table.require(8 * chunk_count)
    var offsets = List[UInt64](
        length=chunk_count,
        fill_with=lambda (index: Int) -> UInt64: (
            bytes.unsafe_ptr()
            .unsafe_offset(header.data_offset + 8 * index)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[alignment=1]()
        ),
    )

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


@fieldwise_init
struct _Chunk(TrivialRegisterPassable):
    var begin: Int
    var end: Int
    var row: Int
    var lines: Int
    var decoded_size: Int


def _plan_chunks(
    bytes: ImmSpan[UInt8, _], header: _Header
) raises -> List[_Chunk]:
    var count = ceildiv(header.height(), header.scanlines_per_chunk())
    var table = _Cursor(bytes)
    table.pos = header.data_offset
    table.require(8 * count)
    var chunks = List[_Chunk](capacity=count)
    var rows = List[UInt8](length=header.height(), fill=0)
    var bytes_per_line = 0
    for channel in header.channels:
        bytes_per_line += header.width() * channel.byte_width
    for _ in range(count):
        var offset = table.read_u64()
        if len(bytes) < 8 or offset > UInt64(len(bytes) - 8):
            raise Error("OpenEXR chunk offset is out of range")
        var cursor = _Cursor(bytes)
        cursor.pos = Int(offset)
        var y = Int(cursor.read_i32())
        var size = Int(cursor.read_u32())
        cursor.require(size)
        var row = y - header.min_y
        if row < 0 or row >= header.height():
            raise Error("OpenEXR scanline chunk is outside the data window")
        var lines = min(header.scanlines_per_chunk(), header.height() - row)
        for line in range(row, row + lines):
            if rows[line]:
                raise Error(
                    "overlapping OpenEXR chunks require serial decoding"
                )
            rows[line] = 1
        chunks.append(
            _Chunk(
                cursor.pos,
                cursor.pos + size,
                row,
                lines,
                bytes_per_line * lines,
            )
        )
    for row in rows:
        if not row:
            raise Error("incomplete OpenEXR chunks require serial decoding")
    return chunks^


def _decode_channel[
    pixel_type: Int
](
    bytes: ImmSpan[UInt8, _],
    output: MutSpan[Float32, _],
    lane: Int,
    width: Int,
) raises:
    comptime vector_width = simd_width_of[Float32]()
    var vector_end = width - width % vector_width
    var offsets = SIMD[.int32, vector_width](0)
    comptime for index in range(vector_width):
        offsets[index] = Int32(3 * index)
    for x in range(0, vector_end, vector_width):
        var values: SIMD[.float32, vector_width]
        comptime if pixel_type == 1:
            var bits = (
                bytes.unsafe_ptr()
                .unsafe_offset(2 * x)
                .unsafe_bitcast[UInt16]()
                .unsafe_load[width=vector_width, alignment=1]()
            )
            values = bitcast[.float16](bits).cast[.float32]()
        else:
            var bits = (
                bytes.unsafe_ptr()
                .unsafe_offset(4 * x)
                .unsafe_bitcast[UInt32]()
                .unsafe_load[width=vector_width, alignment=1]()
            )
            comptime if pixel_type == 0:
                values = bits.cast[.float32]()
            else:
                values = bitcast[.float32](bits)
        if not isfinite(values).reduce_and():
            raise Error("OpenEXR environment contains a non-finite value")
        output.unsafe_ptr().unsafe_offset(3 * x + lane).unsafe_scatter(
            offsets, values
        )
    for x in range(vector_end, width):
        var value: Float32
        comptime if pixel_type == 1:
            var bits = (
                bytes.unsafe_ptr()
                .unsafe_offset(2 * x)
                .unsafe_bitcast[UInt16]()
                .unsafe_load[alignment=1]()
            )
            value = _half_to_float(bits)
        else:
            var bits = (
                bytes.unsafe_ptr()
                .unsafe_offset(4 * x)
                .unsafe_bitcast[UInt32]()
                .unsafe_load[alignment=1]()
            )
            comptime if pixel_type == 0:
                value = Float32(bits)
            else:
                value = bitcast[.float32](bits)
        if not isfinite(value):
            raise Error("OpenEXR environment contains a non-finite value")
        output[3 * x + lane] = value


def _decode_scanlines(
    output: MutSpan[Float32, _],
    header: _Header,
    chunk: ImmSpan[UInt8, _],
    lines: Int,
) raises:
    var width = header.width()
    var offset = 0
    for line in range(lines):
        var pixels = output[3 * width * line : 3 * width * (line + 1)]
        for channel in header.channels:
            var size = width * channel.byte_width
            if channel.output_channel != _OutputChannel.IGNORE:
                var lane = Int(channel.output_channel.value)
                var values = chunk[offset : offset + size]
                match channel.pixel_type:
                    case 0:
                        _decode_channel[0](values, pixels, lane, width)
                    case 1:
                        _decode_channel[1](values, pixels, lane, width)
                    case 2:
                        _decode_channel[2](values, pixels, lane, width)
                    case _:
                        raise Error("unsupported OpenEXR channel pixel type")
            offset += size


def _inflate_chunk(
    payload: ImmSpan[UInt8, _],
    predicted: MutSpan[UInt8, _],
    decoded: MutSpan[UInt8, _],
) raises:
    if inflate_zlib(payload, predicted) != len(predicted):
        raise Error("unexpected OpenEXR scanline byte count")
    _undo_zip_prediction(predicted)
    _undo_zip_interleave_into(predicted, decoded)


def _parse_exr_bulk(
    bytes: ImmSpan[UInt8, _], workers: Int
) raises -> ImageTexture:
    var header = _parse_header(bytes)
    var rgb = UInt8(0)
    for channel in header.channels:
        match channel.output_channel:
            case .RED:
                rgb |= 1
            case .GREEN:
                rgb |= 2
            case .BLUE:
                rgb |= 4
            case .IGNORE:
                pass
    if rgb != 7:
        raise Error("OpenEXR environment requires R, G, and B channels")
    var chunks = _plan_chunks(bytes, header)
    # Every row and RGB channel is covered before uninitialized output is used.
    var pixels = List[Float32](
        unsafe_uninit_length=3 * header.width() * header.height()
    )
    var tasks = min(max(1, workers) * 2, len(chunks))
    var failed = List[UInt8](length=tasks, fill=0)

    def decode(task: Int) {imm, mut pixels, mut failed}:
        var begin = len(chunks) * task // tasks
        var end = len(chunks) * (task + 1) // tasks
        try:
            if header.compression == _NO_COMPRESSION:
                for i in range(begin, end):
                    ref chunk = chunks[i]
                    if chunk.end - chunk.begin != chunk.decoded_size:
                        raise Error("unexpected OpenEXR scanline chunk size")
                    _decode_scanlines(
                        Span(pixels)[
                            3
                            * header.width()
                            * chunk.row : 3
                            * header.width()
                            * (chunk.row + chunk.lines)
                        ],
                        header,
                        bytes[chunk.begin : chunk.end],
                        chunk.lines,
                    )
            else:
                # Reuse scratch allocations across each worker's ZIP chunks.
                var scratch_size = 0
                for i in range(begin, end):
                    scratch_size = max(scratch_size, chunks[i].decoded_size)
                var predicted = List[UInt8](unsafe_uninit_length=scratch_size)
                var decoded = List[UInt8](unsafe_uninit_length=scratch_size)
                for i in range(begin, end):
                    ref chunk = chunks[i]
                    var data = Span(decoded)[: chunk.decoded_size]
                    _inflate_chunk(
                        bytes[chunk.begin : chunk.end],
                        Span(predicted)[: chunk.decoded_size],
                        data,
                    )
                    _decode_scanlines(
                        Span(pixels)[
                            3
                            * header.width()
                            * chunk.row : 3
                            * header.width()
                            * (chunk.row + chunk.lines)
                        ],
                        header,
                        data,
                        chunk.lines,
                    )
        except:
            failed[task] = 1

    if tasks > 1 and workers > 1:
        parallelize(decode, tasks, min(workers, tasks))
    else:
        for task in range(tasks):
            decode(task)
    for failure in failed:
        if failure:
            raise Error("failed to decode OpenEXR chunks")
    return ImageTexture(header.width(), header.height(), pixels^)


def parse_exr[
    origin: ImmOrigin
](bytes: ImmSpan[UInt8, origin]) raises -> ImageTexture:
    if len(bytes) < 256 * 1024:
        return _parse_exr_serial(bytes)
    try:
        return _parse_exr_bulk(bytes, min(8, num_logical_cores()))
    except:
        # Also retains serial behavior for overlapping or incomplete chunks.
        return _parse_exr_serial(bytes)


def read_exr(path: String) raises -> ImageTexture:
    var mapped = MMap(path)
    return parse_exr(mapped.as_bytes_span())
