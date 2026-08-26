"""Compile-time modes for CPU triangle packet traversal."""


@fieldwise_init
struct CpuTraversalMode(Equatable, ImplicitlyCopyable):
    """Select the packet-dispatch mode, independent of packet sizes."""

    comptime FIXED_PACKET = Self(0)
    comptime AUTO_COHERENT = Self(1)
    comptime ADAPTIVE = Self(2)

    var value: Int

    def name(self) -> String:
        if self == Self.FIXED_PACKET:
            return "fixed-packet"
        if self == Self.AUTO_COHERENT:
            return "auto-coherent"
        if self == Self.ADAPTIVE:
            return "adaptive"
        return "unknown"
