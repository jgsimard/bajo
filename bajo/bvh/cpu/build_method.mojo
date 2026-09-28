"""Enum-like compile-time selector for CPU BVH topology builders."""


@fieldwise_init
struct CpuBvhBuildMethod(EnumLike, Equatable, ImplicitlyCopyable):
    """Typed CPU builder selector; SAH is the packed-BLAS default."""

    comptime MEDIAN = Self(0)
    comptime SAH = Self(1)
    comptime LBVH = Self(2)
    comptime HPLOC = Self(3)
    comptime _enum_case_names = ParameterList.of[
        "MEDIAN".value, "SAH".value, "LBVH".value, "HPLOC".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType, NoneType
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
        if self == Self.MEDIAN:
            return "median"
        if self == Self.SAH:
            return "sah"
        if self == Self.LBVH:
            return "lbvh"
        if self == Self.HPLOC:
            return "hploc"
        return "unknown"


struct CpuBvhBuildTraits[method: CpuBvhBuildMethod]:
    """Compile-time inputs required by one CPU topology builder."""

    comptime needs_root_bounds = (Self.method == .MEDIAN or Self.method == .SAH)
    comptime needs_centroid_bounds = Self.method != .MEDIAN
