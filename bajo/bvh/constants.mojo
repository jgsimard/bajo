comptime LBVH_LEAF_FLAG = UInt32(0x80000000)
comptime LBVH_INDEX_MASK = UInt32(0x7FFFFFFF)
comptime LBVH_SENTINEL = UInt32(0xFFFFFFFF)

comptime TRACE_PRIMARY_FULL = "primary_full"
comptime TRACE_PRIMARY_T = "primary_t"
comptime TRACE_SHADOW = "shadow"

comptime _gpu_inf_t = Float32(3.4028234663852886e38)

comptime GPU_TRAVERSAL_STACK_SIZE = 64
comptime GPU_REDUCE_THREADS = 4096

comptime EMPTY_LANE = UInt32(0xFFFFFFFF)
comptime MISS_PRIM = EMPTY_LANE
comptime MISS_INST = EMPTY_LANE

comptime CPU_TRAVERSAL_STACK_SIZE = 64
