"""Enum-like compile-time selector for CPU BVH topology builders."""


from bajo.core.utils import enum_label


__enum CpuBvhBuildMethod(Equatable, ImplicitlyCopyable):
    """Typed CPU builder selector; SAH is the packed-BLAS default."""

    case MEDIAN
    case SAH
    case LBVH
    case HPLOC

    def name(self) -> String:
        return enum_label(self)


struct CpuBvhBuildTraits[method: CpuBvhBuildMethod]:
    """Compile-time inputs required by one CPU topology builder."""

    comptime needs_root_bounds = (Self.method == .MEDIAN or Self.method == .SAH)
    comptime needs_centroid_bounds = Self.method != .MEDIAN
