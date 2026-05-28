from std.utils.numerics import max_finite, min_finite

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()

comptime LBVH_LEAF_FLAG = UInt32(0x80000000)
comptime LBVH_INDEX_MASK = UInt32(0x7FFFFFFF)
comptime LBVH_SENTINEL = UInt32(0xFFFFFFFF)


@fieldwise_init
struct TRACE(Equatable):
    comptime CLOSEST_HIT = Self(0)
    comptime ANY_HIT = Self(1)
    var v: Int


comptime GPU_STACK_SIZE = 64
comptime GPU_REDUCE_THREADS = 4096

comptime EMPTY_LANE = UInt32(0xFFFFFFFF)
comptime MISS_PRIM = EMPTY_LANE
comptime MISS_INST = EMPTY_LANE

comptime CPU_STACK_SIZE = 64

comptime BOUNDS_STRIDE = 6
comptime TRI_LEAF_VERTEX_STRIDE = 9
comptime SPHERE_STRIDE = 4
comptime BVH_BINS = 16


@fieldwise_init
struct Primitive(Equatable, TrivialRegisterPassable):
    comptime UNKNOWN = Self(-1)
    comptime TRIANGLE = Self(0)
    comptime SPHERE = Self(1)
    var v: Int32


comptime BINARY_BVH_NODE_META_STRIDE = 4
comptime BINARY_BVH_NODE_PARENT = 0
comptime BINARY_BVH_NODE_LEFT = 1
comptime BINARY_BVH_NODE_RIGHT = 2
comptime BINARY_BVH_NODE_FENCE = 3
comptime BINARY_BVH_NODE_BOUNDS_STRIDE = 12

comptime GPU_BOUNDS_BVH_BLOCK_SIZE = 128
comptime BOUNDS_REDUCE_CHUNK = 256
comptime REDUCED_BOUNDS_STRIDE = BOUNDS_STRIDE * 2
