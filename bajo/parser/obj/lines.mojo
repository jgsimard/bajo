"""Scan each SIMD byte block once and reuse its newline mask across records."""

from std.bit import count_trailing_zeros
from std.memory import pack_bits
from std.sys import simd_width_of


struct _ObjLines[origin: ImmOrigin]:
    var bytes: ImmSpan[UInt8, Self.origin]
    var end: Int
    var scan: Int
    var block: Int
    var newlines: UInt64

    @always_inline
    def __init__(
        out self, bytes: ImmSpan[UInt8, Self.origin], begin: Int, end: Int
    ):
        self.bytes = bytes
        self.end = end
        self.scan = begin
        self.block = begin
        self.newlines = 0

    @always_inline
    def next_end(mut self) -> Int:
        comptime width = min(64, simd_width_of[UInt8]())
        while self.newlines == 0:
            if self.end - self.scan < width:
                while self.scan < self.end:
                    var pos = self.scan
                    self.scan += 1
                    if self.bytes.unsafe_get(pos) == UInt8(10):
                        return pos
                return self.end
            self.block = self.scan
            var bytes = (
                self.bytes.unsafe_ptr()
                .unsafe_offset(self.scan)
                .unsafe_load[width=width, alignment=1]()
            )
            self.newlines = UInt64(pack_bits(bytes.eq(UInt8(10))))
            self.scan += width
        var end = self.block + Int(count_trailing_zeros(self.newlines))
        self.newlines &= self.newlines - 1
        return end
