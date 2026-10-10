"""Compile-time defaults for the spatial representation."""

from std.sys import simd_width_of

# Each node stores six Float32 coordinates. Widths 4/8 tied within 1% in
# the 256-item refit sweep on Ryzen 9700X; prefer the smaller packet and
# cap it to the target's native width. Callers can explicitly specialize.
comptime _NATIVE_BOUNDS_WIDTH = simd_width_of[DType.float32]()
comptime REFIT_WIDTH = 4 if _NATIVE_BOUNDS_WIDTH >= 4 else _NATIVE_BOUNDS_WIDTH

# A bounds reduction can process all six coordinates in one masked packet.
comptime BUILD_WIDTH = 8 if _NATIVE_BOUNDS_WIDTH >= 8 else _NATIVE_BOUNDS_WIDTH

# Large constructors avoid repeated median scans/partitions with stable Morton
# order and balanced emission. Small query workloads favor median tree quality.
comptime MORTON_BUILD_MIN_ITEMS = 16_384

# Lookahead beyond two nodes wastes work across rejected preorder subtrees.
comptime _NATIVE_QUERY_WIDTH = simd_width_of[DType.float64]()
comptime BATCH_WIDTH = 2 if _NATIVE_QUERY_WIDTH >= 2 else _NATIVE_QUERY_WIDTH

# Full ray counts amortize bounds work across more candidates. Width 4 won
# the 256-face 1/2/4/8 sweep; early-exit queries retain the smaller packet.
comptime RAY_COUNT_WIDTH = 4 if _NATIVE_QUERY_WIDTH >= 4 else _NATIVE_QUERY_WIDTH
