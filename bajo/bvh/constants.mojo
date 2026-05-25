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


comptime GPU_TRAVERSAL_STACK_SIZE = 64
comptime GPU_REDUCE_THREADS = 4096

comptime EMPTY_LANE = UInt32(0xFFFFFFFF)
comptime MISS_PRIM = EMPTY_LANE
comptime MISS_INST = EMPTY_LANE

comptime CPU_TRAVERSAL_STACK_SIZE = 64

comptime BOUNDS_STRIDE = 6
comptime TRI_LEAF_VERTEX_STRIDE = 9
comptime SPHERE_STRIDE = 4
comptime BVH_BINS = 16
