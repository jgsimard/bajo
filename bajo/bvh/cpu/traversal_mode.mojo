"""Compile-time modes for CPU triangle packet traversal."""


@fieldwise_init
struct CpuTraversalMode(EnumLike, Equatable, ImplicitlyCopyable):
    """Select the packet-dispatch mode, independent of packet sizes."""

    comptime FIXED_PACKET = Self(0)
    comptime AUTO_COHERENT = Self(1)
    comptime ADAPTIVE = Self(2)
    comptime _enum_case_names = ParameterList.of[
        "FIXED_PACKET".value,
        "AUTO_COHERENT".value,
        "ADAPTIVE".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType
    ].values

    var value: Int

    def _get_enum_discriminant(self) -> Int:
        return self.value

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass

    def name(self) -> String:
        if self == Self.FIXED_PACKET:
            return "fixed-packet"
        if self == Self.AUTO_COHERENT:
            return "auto-coherent"
        if self == Self.ADAPTIVE:
            return "adaptive"
        return "unknown"
