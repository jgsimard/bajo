from std.utils.numerics import max_finite, min_finite

comptime f32_max = max_finite[.float32]()
comptime f32_min = min_finite[.float32]()

comptime LBVH_SENTINEL = UInt32(0xFFFFFFFF)

# Shared CPU/GPU Morton refinement policy.
comptime MORTON_REFINEMENT_TILE_SIZE = 256
comptime MORTON_REFINEMENT_MIN_RUN = 8


__enum TraceMode(Equatable, ImplicitlyCopyable):
    case CLOSEST_HIT
    case ANY_HIT


comptime GPU_STACK_SIZE = 24
comptime GPU_REDUCE_THREADS = 4096

comptime EMPTY_LANE = UInt32(0xFFFFFFFF)
comptime MISS_PRIM = EMPTY_LANE
comptime MISS_INST = EMPTY_LANE

comptime CPU_STACK_SIZE = 64

comptime TRI_LEAF_VERTEX_STRIDE = 9
comptime TRI_LEAF_PACKED_STRIDE = 12
comptime CPU_TRI_LEAF_PACKED_STRIDE = 10
comptime SPHERE_LEAF_PACKED_STRIDE = 5


__enum PrimitiveKind(Equatable, ImplicitlyCopyable, Writable):
    case UNKNOWN
    case TRIANGLE
    case SPHERE
    case TRIANGLE_INSTANCE

    def __init__(out self, value: UInt32):
        """Decode a primitive tag; unrecognized tags map to UNKNOWN."""
        match Int(value):
            case 0:
                self = Self.TRIANGLE
            case 1:
                self = Self.SPHERE
            case 2:
                self = Self.TRIANGLE_INSTANCE
            case _:
                self = Self.UNKNOWN

    @always_inline
    def value(self) -> UInt32:
        match self:
            case .UNKNOWN:
                return UInt32.MAX
            case .TRIANGLE:
                return 0
            case .SPHERE:
                return 1
            case .TRIANGLE_INSTANCE:
                return 2


struct BinaryBvhNode:
    comptime META_STRIDE = 4
    comptime PARENT = 0
    comptime LEFT = 1
    comptime RIGHT = 2
    comptime FENCE = 3
    comptime BOUNDS_STRIDE = 12


struct WideNode:
    comptime CHILD_STRIDE = 7
    comptime MIN_X = 0
    comptime MIN_Y = 1
    comptime MIN_Z = 2
    comptime MAX_X = 3
    comptime MAX_Y = 4
    comptime MAX_Z = 5
    comptime META = 6


comptime GPU_BOUNDS_BVH_BLOCK_SIZE = 128
comptime BOUNDS_REDUCE_CHUNK = 256
comptime REDUCED_BOUNDS_STRIDE = 12
