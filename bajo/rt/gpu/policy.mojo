"""Compile-time scene-shape and ready BVH formats for GPU ray tracing."""

from bajo.bvh.gpu import GpuBvhLayout


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
struct GpuRtBvhFormat:
    """Compile-time layout of a ready-to-traverse GPU BVH."""

    var node_width: Int
    var leaf_width: Int
    var layout: GpuBvhLayout


comptime GPU_RT_BVH_WIDE4 = GpuRtBvhFormat(4, 4, .WIDE)
comptime GPU_RT_BVH_CWBVH8 = GpuRtBvhFormat(8, 4, .CWBVH8)
comptime GPU_RT_BVH_TLAS2 = GpuRtBvhFormat(2, 1, .WIDE)
