"""Buffer-based DEFLATE/zlib decompression implemented entirely in Mojo.

Adapted from libdeflate v1.25's word bitbuffer, two-level Huffman tables,
and overlap-aware match copies. See LICENSE.libdeflate in this directory.
The output span is a capacity bound; return values report consumed input and
written output. No speculative memory access extends outside either span.
"""

from std.memory import unsafe_memcpy
from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic

from bajo.parser.checksum import adler32


comptime _LENGTH_BASE = SIMD[.uint16, 32](
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    15,
    17,
    19,
    23,
    27,
    31,
    35,
    43,
    51,
    59,
    67,
    83,
    99,
    115,
    131,
    163,
    195,
    227,
    258,
    0,
    0,
    0,
)
comptime _LENGTH_EXTRA = SIMD[.uint8, 32](
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    0,
    0,
    0,
    0,
)
comptime _DIST_BASE = SIMD[.uint16, 32](
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    25,
    33,
    49,
    65,
    97,
    129,
    193,
    257,
    385,
    513,
    769,
    1025,
    1537,
    2049,
    3073,
    4097,
    6145,
    8193,
    12289,
    16385,
    24577,
    0,
    0,
)
comptime _DIST_EXTRA = SIMD[.uint8, 32](
    0,
    0,
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
    12,
    13,
    13,
    0,
    0,
)
comptime _PRECODE_ORDER = SIMD[.uint8, 32](
    16,
    17,
    18,
    0,
    8,
    7,
    9,
    6,
    10,
    5,
    11,
    4,
    12,
    3,
    13,
    2,
    14,
    1,
    15,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)


@always_inline
def _reverse_code(value: UInt32, length: Int) -> Int:
    var reversed = llvm_intrinsic[
        "llvm.bitreverse", UInt32, has_side_effect=False
    ](value)
    return Int(reversed >> UInt32(32 - length))


@always_inline
def _shift_right(value: UInt64, count: UInt64) -> UInt64:
    # The modulo-64 mask matches x86 variable-shift semantics. LLVM removes it
    # when selecting SHRX; a funnel-shift intrinsic instead selected slower SHRD.
    return value >> (count & UInt64(63))


@always_inline
def _extract_bits(value: UInt64, entry: UInt32) -> UInt64:
    comptime if CompilationTarget._has_feature["bmi2"]():
        # BZHI reads the low eight bits of the count: the combined codeword
        # and extra-bit length stored in a leaf entry.
        return llvm_intrinsic[
            "llvm.x86.bmi.bzhi.64", UInt64, has_side_effect=False
        ](value, UInt64(entry))
    else:
        return value & ((UInt64(1) << UInt64(entry & 31)) - 1)


struct _Bits[origin: ImmOrigin]:
    var input: ImmSpan[UInt8, Self.origin]
    var offset: Int
    var buffer: UInt64
    var count: Int

    def __init__(out self, input: ImmSpan[UInt8, Self.origin]):
        self.input = input
        self.offset = 0
        self.buffer = 0
        self.count = 0

    @always_inline
    def refill(mut self):
        if self.count >= 56:
            return
        if self.offset <= len(self.input) - 8:
            var word = (
                self.input.unsafe_ptr()
                .unsafe_offset(self.offset)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            self.buffer |= word << UInt64(self.count)
            var take = (63 - self.count) // 8
            self.offset += take
            self.count += take * 8
        else:
            while self.count <= 56 and self.offset < len(self.input):
                self.buffer |= UInt64(self.input[self.offset]) << UInt64(
                    self.count
                )
                self.offset += 1
                self.count += 8

    @always_inline
    def drop(mut self, count: Int) raises:
        if count > self.count:
            raise Error("truncated DEFLATE bitstream")
        self.buffer >>= UInt64(count)
        self.count -= count

    @always_inline
    def read(mut self, count: Int) raises -> UInt32:
        self.refill()
        var value = UInt32(self.buffer & ((UInt64(1) << UInt64(count)) - 1))
        self.drop(count)
        return value


@always_inline
def _entry(symbol: Int, length: Int, alphabet: Int) -> UInt32:
    if alphabet == 0:
        if symbol < 256:
            return (UInt32(symbol) << 16) | UInt32(0x8000) | UInt32(length)
        if symbol == 256:
            return UInt32(0x2000 | length)
        if symbol > 285:
            return UInt32(0x4000)
        var slot = symbol - 257
        return (
            (UInt32(_LENGTH_BASE[slot]) << 16)
            | (UInt32(length) << 8)
            | UInt32(length + Int(_LENGTH_EXTRA[slot]))
        )
    if alphabet == 1:
        if symbol > 29:
            return UInt32(0x4000)
        return (
            (UInt32(_DIST_BASE[symbol]) << 16)
            | (UInt32(length) << 8)
            | UInt32(length + Int(_DIST_EXTRA[symbol]))
        )
    return (UInt32(symbol) << 16) | UInt32(length)


struct _Huffman:
    var root: Int
    var paired: Bool
    var entries: List[UInt32]

    def __init__(out self):
        self.root = 1
        self.paired = False
        self.entries = List[UInt32]()

    def build(
        mut self, lengths: ImmSpan[UInt8, _], alphabet: Int, root_bits: Int = 12
    ) raises:
        var counts = SIMD[.int32, 16](0)
        var longest = 0
        var used = 0
        for length in lengths:
            if length > 15:
                raise Error("invalid DEFLATE Huffman code length")
            counts[Int(length)] += 1
            longest = max(longest, Int(length))
            used += Int(length != 0)
        var left = 1
        for length in range(1, 16):
            left = 2 * left - Int(counts[length])
            if left < 0:
                raise Error("oversubscribed DEFLATE Huffman code")
        if left != 0 and not (alphabet != -1 and used == 1 and longest == 1):
            if alphabet != 1 or used != 0:
                raise Error("incomplete DEFLATE Huffman code")
        # An eight-bit distance root allows byte extraction in the decode loop
        # even when the longest distance code is shorter (libdeflate's layout).
        self.root = min(7, max(1, longest))
        self.paired = False
        if alphabet == 1:
            self.root = 8
        elif alphabet == 0:
            var candidate = min(root_bits, 2 * max(1, longest))
            var literal_counts = SIMD[.uint32, 16](0)
            for i in range(min(256, len(lengths))):
                literal_counts[Int(lengths[i])] += 1
            var space = SIMD[.uint32, 16](0)
            for n in range(1, candidate + 1):
                space[n] = 2 * space[n - 1] + literal_counts[n]
            var pairs = UInt32(0)
            for n in range(1, candidate):
                pairs += literal_counts[n] * space[candidate - n]
            # Use the wider paired table only when at least a quarter of its
            # prefixes resolve two literals. Near-uniform EXR chunks keep the
            # smaller single-literal loop; fixed trees cannot form pairs here.
            self.paired = pairs >= UInt32(1 << candidate) // 4
            self.root = candidate if self.paired else min(
                min(11, root_bits), max(1, longest)
            )
        var root_size = 1 << self.root
        var sub_bits = List[UInt8](length=root_size, fill=0)
        var codes = List[Int](length=len(lengths), fill=0)
        var next = SIMD[.uint32, 16](0)
        var code = UInt32(0)
        for length in range(1, 16):
            code = (
                code + UInt32(0 if length == 1 else counts[length - 1])
            ) << 1
            next[length] = code
        for symbol in range(len(lengths)):
            var length = Int(lengths[symbol])
            if length == 0:
                continue
            codes[symbol] = _reverse_code(next[length], length)
            next[length] += 1
            if length > self.root:
                var prefix = codes[symbol] & (root_size - 1)
                sub_bits[prefix] = max(
                    sub_bits[prefix], UInt8(length - self.root)
                )
        var total = root_size
        for bits in sub_bits:
            if bits != 0:
                total += 1 << Int(bits)
        self.entries.clear()
        self.entries.resize(total, fill=UInt32(0))
        var offset = root_size
        for prefix in range(root_size):
            if sub_bits[prefix] != 0:
                self.entries[prefix] = (
                    (UInt32(offset) << 16)
                    | (UInt32(sub_bits[prefix]) << 8)
                    | UInt32(0x80)
                )
                offset += 1 << Int(sub_bits[prefix])
        for symbol in range(len(lengths)):
            var length = Int(lengths[symbol])
            if length == 0:
                continue
            var start = codes[symbol]
            var base = 0
            var table_size = root_size
            if length > self.root:
                var prefix = start & (root_size - 1)
                base = Int(self.entries[prefix] >> 16)
                table_size = 1 << Int(sub_bits[prefix])
                start >>= self.root
                length -= self.root
            var value = _entry(symbol, length, alphabet)
            for i in range(start, table_size, 1 << length):
                self.entries[base + i] = value

        if self.paired:
            # Repeated root slots provide spare prefix bits. If those bits
            # resolve another literal, decode both bytes with one table load.
            # Low five bits hold total consumption; bits 8..11 retain the
            # first code length for the checked single-symbol decoder. Bits
            # 15/14 mark a literal/pair; bits 16..31 hold the decoded bytes.
            for i in range(root_size):
                var first = self.entries[i]
                if first & 0x8000 == 0:
                    continue
                var n = Int(first & 31)
                var second = self.entries[i >> n]
                var m = Int((second >> 8) & 15) if second & 0x4000 else Int(
                    second & 31
                )
                if second & 0x8000 and n + m <= self.root:
                    self.entries[i] = (
                        (first & 0x00FF0000)
                        | ((second & 0x00FF0000) << 8)
                        | UInt32(0xC000 | (n << 8) | (n + m))
                    )

    @always_inline
    def decode[
        origin: ImmOrigin
    ](self, mut bits: _Bits[origin]) raises -> UInt32:
        bits.refill()
        var entry = self.entries[
            Int(bits.buffer & UInt64((1 << self.root) - 1))
        ]
        if entry & 0x80:
            bits.drop(self.root)
            var count = Int((entry >> 8) & 15)
            entry = self.entries[
                Int(entry >> 16) + Int(bits.buffer & UInt64((1 << count) - 1))
            ]
        if entry & 0xC000 == 0xC000:
            entry = (entry & 0x00FF0000) | UInt32(0x8000) | ((entry >> 8) & 15)
        var count = Int(entry & 31)
        if count == 0 or entry & 0x4000:
            raise Error("invalid DEFLATE Huffman symbol")
        var value = entry >> 16
        var code_bits = Int((entry >> 8) & 15)
        if code_bits != 0:
            value += UInt32(
                (bits.buffer & ((UInt64(1) << UInt64(count)) - 1))
                >> UInt64(code_bits)
            )
        bits.drop(count)
        return (value << 16) | (entry & 0xFFFF)

    @always_inline
    def decode_bulk(
        self, mut buffer: UInt64, mut available: UInt32, var entry: UInt32
    ) -> Tuple[UInt32, UInt32]:
        # The caller guarantees enough real buffered bits for a full symbol
        # and its extra bits. Table construction validates every lookup range.
        var table = self.entries.unsafe_ptr()
        if entry & 0x80:
            buffer >>= UInt64(self.root)
            available -= UInt32(self.root)
            var sub_bits = Int((entry >> 8) & 15)
            entry = table[
                unsafe_offset=Int(entry >> 16)
                + Int(buffer & ((UInt64(1) << UInt64(sub_bits)) - 1))
            ]
        var value = entry >> 16
        # Literal and EOB callers ignore this value. Length/distance entries
        # always encode their codeword length, including zero-extra-bit codes.
        value += UInt32(
            _shift_right(_extract_bits(buffer, entry), UInt64(entry >> 8))
        )
        buffer = _shift_right(buffer, UInt64(entry))
        # Only the low byte of 'available' is significant. Subtracting the full
        # entry saves a mask; the bulk-loop bit budget prevents byte underflow.
        available -= entry
        return (entry, value)


def _fixed(mut literals: _Huffman, mut distances: _Huffman) raises:
    var lengths = List[UInt8](length=288, fill=8)
    for i in range(144, 256):
        lengths[i] = 9
    for i in range(256, 280):
        lengths[i] = 7
    literals.build(lengths, 0, 9)
    var offsets = List[UInt8](length=32, fill=5)
    distances.build(offsets, 1)


def _dynamic[
    origin: ImmOrigin
](
    mut bits: _Bits[origin], mut literals: _Huffman, mut distances: _Huffman
) raises:
    var literal_count = 257 + Int(bits.read(5))
    var distance_count = 1 + Int(bits.read(5))
    var pre_count = 4 + Int(bits.read(4))
    if literal_count > 286:
        raise Error("invalid DEFLATE literal alphabet size")
    var pre_lengths = List[UInt8](length=19, fill=0)
    for i in range(pre_count):
        pre_lengths[Int(_PRECODE_ORDER[i])] = UInt8(bits.read(3))
    var precode = _Huffman()
    precode.build(pre_lengths, -1)
    var lengths = List[UInt8](capacity=literal_count + distance_count)
    while len(lengths) < literal_count + distance_count:
        var symbol = Int(precode.decode(bits) >> 16)
        if symbol < 16:
            lengths.append(UInt8(symbol))
            continue
        var value = UInt8(0)
        var repeat: Int
        if symbol == 16:
            if len(lengths) == 0:
                raise Error("DEFLATE repeat has no preceding length")
            value = lengths[len(lengths) - 1]
            repeat = 3 + Int(bits.read(2))
        elif symbol == 17:
            repeat = 3 + Int(bits.read(3))
        else:
            repeat = 11 + Int(bits.read(7))
        if repeat > literal_count + distance_count - len(lengths):
            raise Error("DEFLATE code length repeat exceeds alphabet")
        for _ in range(repeat):
            lengths.append(value)
    if lengths[256] == 0:
        raise Error("DEFLATE code is missing end of block")
    literals.build(ImmSpan(lengths)[:literal_count], 0)
    distances.build(ImmSpan(lengths)[literal_count:], 1)


def _copy_match(
    output: MutSpan[UInt8, _], offset: Int, distance: Int, length: Int
):
    var ptr = output.unsafe_ptr()
    var i = 0
    var copy_distance = distance
    if distance != 1:
        if distance < 16:
            # Seed one vector, then extend from a whole number of periods.
            # This makes vector loads nonoverlapping even for RGB strides.
            for j in range(min(16, length)):
                ptr[unsafe_offset=offset + j] = ptr[
                    unsafe_offset=offset + j - distance
                ]
                i += 1
            copy_distance = (16 + distance - 1) // distance * distance
        while i + 16 <= length:
            ptr.unsafe_offset(offset + i).unsafe_store[width=16](
                ptr.unsafe_offset(offset + i - copy_distance).unsafe_load[
                    width=16
                ]()
            )
            i += 16
    elif distance == 1:
        var value = SIMD[.uint8, 16](ptr[unsafe_offset=offset - 1])
        while i + 16 <= length:
            ptr.unsafe_offset(offset + i).unsafe_store[width=16](value)
            i += 16
    for j in range(i, length):
        ptr[unsafe_offset=offset + j] = ptr[unsafe_offset=offset + j - distance]


@always_inline
def _refill_bulk(
    input: ImmSpan[UInt8, _],
    mut offset: Int,
    mut buffer: UInt64,
    mut available: UInt32,
):
    var word = (
        input.unsafe_ptr()
        .unsafe_offset(offset)
        .unsafe_bitcast[UInt64]()
        .unsafe_load[alignment=1]()
    )
    buffer |= word << UInt64(available & 63)
    offset += 7 - Int((available >> 3) & 7)
    available |= UInt32(56)


@always_inline
def _copy_match_bulk(
    output: MutSpan[UInt8, _], offset: Int, distance: Int, length: Int
):
    # The bulk loop reserves 288 output bytes. Round only the final copy up
    # within that capacity; no source load exceeds already reconstructed data.
    var ptr = output.unsafe_ptr()
    var i = 0
    if distance >= 16:
        while i < length:
            ptr.unsafe_offset(offset + i).unsafe_store[width=16](
                ptr.unsafe_offset(offset + i - distance).unsafe_load[width=16]()
            )
            i += 16
    elif distance >= 8:
        while i < length:
            ptr.unsafe_offset(offset + i).unsafe_bitcast[UInt64]().unsafe_store[
                alignment=1
            ](
                ptr.unsafe_offset(offset + i - distance)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            i += 8
    elif distance == 1:
        var repeated = SIMD[.uint8, 16](ptr[unsafe_offset=offset - 1])
        while i < length:
            ptr.unsafe_offset(offset + i).unsafe_store[width=16](repeated)
            i += 16
    else:
        for j in range(min(16, length)):
            ptr[unsafe_offset=offset + j] = ptr[
                unsafe_offset=offset + j - distance
            ]
            i += 1
        var period = Int(
            (UInt64(0x1512141012101000) >> UInt64(8 * distance)) & 255
        )
        while i < length:
            ptr.unsafe_offset(offset + i).unsafe_store[width=16](
                ptr.unsafe_offset(offset + i - period).unsafe_load[width=16]()
            )
            i += 16


@always_inline
def _store_literals[
    paired: Bool
](output: MutSpan[UInt8, _], offset: Int, entry: UInt32) -> Int:
    comptime if paired:
        output.unsafe_ptr().unsafe_offset(offset).unsafe_bitcast[
            UInt16
        ]().unsafe_store[alignment=1](UInt16(entry >> 16))
        return 1 + Int((entry >> 14) & 1)
    else:
        output.unsafe_ptr()[unsafe_offset=offset] = UInt8(entry >> 16)
        return 1


@no_inline
def _inflate_bulk[
    literal_bits: Int, distance_bits: Int, paired: Bool, origin: ImmOrigin
](
    mut bits: _Bits[origin],
    output: MutSpan[UInt8, _],
    mut written: Int,
    literals: _Huffman,
    distances: _Huffman,
    window_size: Int,
) -> Int:
    # Keep error construction outside the hot loop: 0 needs the checked tail,
    # 1 ends the block, -1 rejects a symbol, and -2 rejects a history distance.
    var buffer = bits.buffer
    var available = UInt32(bits.count)
    var offset = bits.offset
    var position = written
    var status = 0
    var literal_table = literals.entries.unsafe_ptr()
    var literal_mask = UInt64(
        (1 << (literal_bits if literal_bits != 0 else literals.root)) - 1
    )
    var distance_table = distances.entries.unsafe_ptr()
    var distance_mask = UInt64(
        (1 << (distance_bits if distance_bits != 0 else distances.root)) - 1
    )
    var entry = UInt32(0)
    if offset <= len(bits.input) - 16 and position <= len(output) - 288:
        _refill_bulk(bits.input, offset, buffer, available)
        entry = literal_table[unsafe_offset=Int(buffer & literal_mask)]
    # Two word refills read at most 16 bytes. Three literal pairs plus a
    # 258-byte match and 15-byte rounding fit in the reserved 288 output bytes.
    # A single literal may also store its unused second byte within this guard.
    while offset <= len(bits.input) - 16 and position <= len(output) - 288:
        if entry & 0x8000:
            position += _store_literals[paired](output, position, entry)
            buffer = _shift_right(buffer, UInt64(entry))
            available -= entry
            entry = literal_table[unsafe_offset=Int(buffer & literal_mask)]
            if entry & 0x8000:
                position += _store_literals[paired](output, position, entry)
                buffer = _shift_right(buffer, UInt64(entry))
                available -= entry
                entry = literal_table[unsafe_offset=Int(buffer & literal_mask)]
                if entry & 0x8000:
                    position += _store_literals[paired](output, position, entry)
                    buffer = _shift_right(buffer, UInt64(entry))
                    available -= entry
                    entry = literal_table[
                        unsafe_offset=Int(buffer & literal_mask)
                    ]
                    if entry & 0x8000:
                        position += _store_literals[paired](
                            output, position, entry
                        )
                        buffer = _shift_right(buffer, UInt64(entry))
                        available -= entry
                        entry = literal_table[
                            unsafe_offset=Int(buffer & literal_mask)
                        ]
                        _refill_bulk(bits.input, offset, buffer, available)
                        continue
        var decoded = literals.decode_bulk(buffer, available, entry)
        entry = decoded[0]
        if (entry & 31) == 0:
            return -1
        if entry & 0x8000:
            position += _store_literals[paired](output, position, entry)
            entry = literal_table[unsafe_offset=Int(buffer & literal_mask)]
            _refill_bulk(bits.input, offset, buffer, available)
            continue
        if entry & 0x2000:
            status = 1
            break
        var length = Int(decoded[1])
        # At most 3*12 + 20 bits have been consumed from the 64 physically
        # prefetched bits, leaving the eight-bit distance prefix available.
        # Refill before consuming it if fewer than 28+12 counted bits remain.
        var distance_entry = distance_table[
            unsafe_offset=Int(buffer & distance_mask)
        ]
        if (available & 255) < 40:
            _refill_bulk(bits.input, offset, buffer, available)
        var decoded_distance = distances.decode_bulk(
            buffer, available, distance_entry
        )
        distance_entry = decoded_distance[0]
        if (distance_entry & 31) == 0:
            return -1
        var distance = Int(decoded_distance[1])
        if distance > position or distance > window_size:
            return -2
        entry = literal_table[unsafe_offset=Int(buffer & literal_mask)]
        _refill_bulk(bits.input, offset, buffer, available)
        _copy_match_bulk(output, position, distance, length)
        position += length
    bits.offset = offset
    bits.buffer = buffer
    bits.count = Int(available & 255)
    written = position
    return status


def inflate_raw(
    input: ImmSpan[UInt8, _],
    output: MutSpan[UInt8, _],
    window_size: Int = 32768,
) raises -> Tuple[Int, Int]:
    """Decode one raw stream; return (input bytes consumed, output bytes written).
    """
    var bits = _Bits(input)
    var literals = _Huffman()
    var distances = _Huffman()
    var written = 0
    var fixed = False
    while True:
        var final = bits.read(1) != 0
        var kind = bits.read(2)
        if kind == 0:
            bits.drop(bits.count % 8)
            var length = Int(bits.read(16))
            if (UInt32(length) ^ bits.read(16)) != 65535:
                raise Error("invalid DEFLATE stored block length")
            var offset = bits.offset - bits.count // 8
            if length > len(input) - offset or length > len(output) - written:
                raise Error("DEFLATE stored block exceeds input or output")
            unsafe_memcpy(
                dest=output.unsafe_ptr().unsafe_offset(written),
                src=input.unsafe_ptr().unsafe_offset(offset),
                count=length,
            )
            written += length
            bits.offset = offset + length
            bits.buffer = 0
            bits.count = 0
        elif kind == 1 or kind == 2:
            if kind == 1:
                if not fixed:
                    _fixed(literals, distances)
                fixed = True
            else:
                _dynamic(bits, literals, distances)
                fixed = False
            var status: Int
            if literals.paired:
                if literals.root == 12 and distances.root == 8:
                    status = _inflate_bulk[12, 8, True](
                        bits, output, written, literals, distances, window_size
                    )
                else:
                    status = _inflate_bulk[0, 0, True](
                        bits, output, written, literals, distances, window_size
                    )
            elif literals.root == 11 and distances.root == 8:
                status = _inflate_bulk[11, 8, False](
                    bits, output, written, literals, distances, window_size
                )
            elif literals.root == 9 and distances.root == 8:
                status = _inflate_bulk[9, 8, False](
                    bits, output, written, literals, distances, window_size
                )
            else:
                status = _inflate_bulk[0, 0, False](
                    bits, output, written, literals, distances, window_size
                )
            if status < 0:
                raise Error("invalid DEFLATE symbol or match distance")
            if status == 1:
                if final:
                    return (bits.offset - bits.count // 8, written)
                continue
            while True:
                # Decode four common direct-table literals per refill. All
                # codewords fit in the counted bits and all stores fit the
                # output span, so this path needs no speculative overread.
                bits.refill()
                if (
                    bits.count >= 4 * literals.root
                    and written <= len(output) - 4
                ):
                    var table = literals.entries.unsafe_ptr()
                    var mask = UInt64((1 << literals.root) - 1)
                    var begin = written
                    for _ in range(4):
                        var value = table[unsafe_offset=Int(bits.buffer & mask)]
                        if value & 0x8000 == 0:
                            break
                        output[written] = UInt8(value >> 16)
                        written += 1
                        var count = Int(
                            (value >> 8) & 15
                        ) if value & 0x4000 else Int(value & 31)
                        bits.buffer >>= UInt64(count)
                        bits.count -= count
                    if written != begin:
                        continue
                var entry = literals.decode(bits)
                if entry & 0x8000:
                    if written == len(output):
                        raise Error("DEFLATE output capacity exceeded")
                    output[written] = UInt8(entry >> 16)
                    written += 1
                elif entry & 0x2000:
                    break
                else:
                    var length = Int(entry >> 16)
                    var offset_entry = distances.decode(bits)
                    var distance = Int(offset_entry >> 16)
                    if distance > written or distance > window_size:
                        raise Error("DEFLATE match distance exceeds history")
                    if length > len(output) - written:
                        raise Error("DEFLATE match exceeds output capacity")
                    _copy_match(output, written, distance, length)
                    written += length
        else:
            raise Error("reserved DEFLATE block type")
        if final:
            return (bits.offset - bits.count // 8, written)


def inflate_zlib(
    input: ImmSpan[UInt8, _], output: MutSpan[UInt8, _]
) raises -> Int:
    """Decode one zlib stream and verify its header and Adler-32 checksum."""
    if len(input) < 6:
        raise Error("truncated zlib stream")
    var header = (UInt32(input[0]) << 8) | UInt32(input[1])
    if input[0] & 15 != 8 or input[0] >> 4 > 7 or header % 31 != 0:
        raise Error("invalid zlib header")
    if input[1] & 32:
        raise Error("zlib preset dictionaries are not supported")
    var result = inflate_raw(input[2:], output, 1 << (8 + Int(input[0] >> 4)))
    var end = 2 + result[0]
    if end > len(input) - 4:
        raise Error("truncated zlib checksum")
    var expected = (
        (UInt32(input[end]) << 24)
        | (UInt32(input[end + 1]) << 16)
        | (UInt32(input[end + 2]) << 8)
        | UInt32(input[end + 3])
    )
    if adler32(ImmSpan(output)[: result[1]]) != expected:
        raise Error("zlib Adler-32 checksum mismatch")
    return result[1]
