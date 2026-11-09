"""Compile-time modes for CPU triangle packet traversal."""


from bajo.core.utils import enum_label


__enum CpuTraversalMode(Equatable, ImplicitlyCopyable):
    """Select the packet-dispatch mode, independent of packet sizes."""

    case FIXED_PACKET
    case AUTO_COHERENT
    case ADAPTIVE

    def name(self) -> String:
        return enum_label(self)
