from utils.index import Index
from sys import simd_width_of


fn _dtype[width: Int]() -> DType:
    if width <= 8:
        return DType.uint8
    elif width <= 16:
        return DType.uint16
    elif width <= 32:
        return DType.uint32
    else:
        return DType.uint64


struct BitField[width: Int]:
    comptime dtype = _dtype[Self.width]()
    comptime num_lanes = 1 if Self.width <= 64 else (Self.width + 63) // 64
    comptime storage_type = SIMD[Self.dtype, Self.num_lanes]

    var storage: Self.storage_type

    fn __init__(out self):
        comptime assert (
            Self.width <= simd_width_of[UInt64]() * 64
        ), "Too big for this CPU, max = " + String(simd_width_of[UInt64]() * 64)
        self.storage = Self.storage_type(0)

    @always_inline
    fn __getitem__(self, pos: Int) -> Bool:
        debug_assert["safe"](0 <= pos <= Self.width)

        @parameter
        if Self.width <= 64:
            var mask = Scalar[self.dtype](1) << Scalar[Self.dtype](pos)
            return (self.storage & mask) != 0
        else:
            var lane = pos // 64
            var bit_in_lane = Scalar[Self.dtype](pos % 64)
            var mask = Scalar[self.dtype](1) << bit_in_lane
            return (self.storage[lane] & mask) != 0

    @always_inline
    fn __setitem__(mut self, pos: Int, value: Bool):
        debug_assert["safe"](0 <= pos <= Self.width)
        var val_bit = Scalar[self.dtype](value)

        @parameter
        if Self.width <= 64:
            var mask = Scalar[self.dtype](1) << Scalar[Self.dtype](pos)
            self.storage = (self.storage & ~mask) | (
                val_bit << Scalar[Self.dtype](pos)
            )
        else:
            var lane = pos // 64
            var bit_in_lane = Scalar[Self.dtype](pos % 64)
            var mask = Scalar[self.dtype](1) << bit_in_lane
            self.storage = (self.storage[lane] & ~mask) | (
                val_bit << bit_in_lane
            )

    fn size_info(self):
        print("Width requested:", Self.width)
        print("Storage type:   ", self.dtype)
        print("Storage lanes:  ", self.num_lanes)
