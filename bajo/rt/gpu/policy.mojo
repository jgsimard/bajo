"""Compile-time scene-shape and BVH policies for GPU ray tracing."""

from bajo.bvh.gpu import GpuBvhBuildMethod, GpuBvhLayout


@fieldwise_init
struct GpuRtSceneKind(Equatable, ImplicitlyCopyable):
    """Enum-like geometry-presence mask used as one specialization value."""

    comptime SPHERES = Self(UInt8(1))
    comptime TRIANGLES = Self(UInt8(2))
    comptime SPHERES_TRIANGLES = Self(UInt8(3))
    comptime INSTANCES = Self(UInt8(4))
    comptime SPHERES_INSTANCES = Self(UInt8(5))
    comptime TRIANGLES_INSTANCES = Self(UInt8(6))
    comptime ALL = Self(UInt8(7))

    var bits: UInt8

    def has_spheres(self) -> Bool:
        return Bool(self.bits & UInt8(1))

    def has_triangles(self) -> Bool:
        return Bool(self.bits & UInt8(2))

    def has_instances(self) -> Bool:
        return Bool(self.bits & UInt8(4))

    def is_valid(self) -> Bool:
        return self.bits > 0 and self.bits <= UInt8(7)


@fieldwise_init
struct GpuRtBvhPolicy:
    """Compile-time BVH layout and builder selection."""

    var node_width: Int
    var leaf_width: Int
    var build_method: GpuBvhBuildMethod
    var layout: GpuBvhLayout


comptime GPU_RT_BVH_WIDE4_LBVH = GpuRtBvhPolicy(4, 4, .LBVH, .WIDE)
comptime GPU_RT_BVH_CWBVH8_HPLOC = GpuRtBvhPolicy(
    8, 4, .HPLOC, .CWBVH8
)
comptime GPU_RT_BVH_TLAS2_LBVH = GpuRtBvhPolicy(
    2, 1, .LBVH, .WIDE
)
