"""Enum-like compile-time selector for CPU BVH topology builders."""


@fieldwise_init
struct CpuBvhBuildMethod(Equatable, ImplicitlyCopyable):
    """Typed CPU builder selector; SAH is the packed-BLAS default."""

    comptime MEDIAN = Self(0)
    comptime SAH = Self(1)
    comptime LBVH = Self(2)
    comptime HPLOC = Self(3)

    var value: Int

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
