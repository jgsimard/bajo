"""Compile-time scene shape and ready BVH formats for GPU ray tracing."""

from bajo.bvh.gpu import GpuBvhLayout


__enum GpuRtSceneKind(Equatable, ImplicitlyCopyable, Intable):
    """Geometry-presence mask used as one specialization value."""

    case SPHERES
    case TRIANGLES
    case SPHERES_TRIANGLES
    case INSTANCES
    case SPHERES_INSTANCES
    case TRIANGLES_INSTANCES
    case ALL

    def __init__[bits: Int](out self):
        """Construct a scene kind from a compile-time geometry mask."""
        comptime assert 1 <= bits <= 7
        self = (
            Self.SPHERES,
            Self.TRIANGLES,
            Self.SPHERES_TRIANGLES,
            Self.INSTANCES,
            Self.SPHERES_INSTANCES,
            Self.TRIANGLES_INSTANCES,
            Self.ALL,
        )[bits - 1]

    def __init__(out self, bits: Int) raises:
        """Construct a scene kind from a nonempty geometry mask."""
        match bits:
            case 1:
                self = Self.SPHERES
            case 2:
                self = Self.TRIANGLES
            case 3:
                self = Self.SPHERES_TRIANGLES
            case 4:
                self = Self.INSTANCES
            case 5:
                self = Self.SPHERES_INSTANCES
            case 6:
                self = Self.TRIANGLES_INSTANCES
            case 7:
                self = Self.ALL
            case _:
                raise Error(
                    "invalid GPU scene kind: expected a geometry mask from 1"
                    " to 7"
                )

    def __int__(self) -> Int:
        """Return the geometry mask used by external scene handles."""
        return self._get_enum_discriminant() + 1

    def has_spheres(self) -> Bool:
        return Bool(Int(self) & 1)

    def has_triangles(self) -> Bool:
        return Bool(Int(self) & 2)

    def has_instances(self) -> Bool:
        return Bool(Int(self) & 4)


@fieldwise_init
struct GpuRtBvhFormat:
    """Compile-time layout of a ready-to-traverse GPU BVH."""

    var node_width: Int
    var leaf_width: Int
    var layout: GpuBvhLayout


comptime GPU_RT_BVH_WIDE4 = GpuRtBvhFormat(4, 4, .WIDE)
comptime GPU_RT_BVH_WIDE2_LEAF4 = GpuRtBvhFormat(2, 4, .WIDE)
comptime GPU_RT_BVH_CWBVH8 = GpuRtBvhFormat(8, 4, .CWBVH8)
comptime GPU_RT_BVH_TLAS4 = GpuRtBvhFormat(4, 1, .WIDE)
