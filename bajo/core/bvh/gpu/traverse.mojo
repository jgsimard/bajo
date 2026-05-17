from bajo.core.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _copy_f32_to_device,
    TRACE_PRIMARY_FULL,
    TRACE_SHADOW,
    TRACE_PRIMARY_T,
    _gpu_miss_prim,
    GPU_TRAVERSAL_STACK_SIZE,
    _wide_lane_base,
    EMPTY_LANE,
    _intersect_wide_node_bounds,
)
from bajo.core.bvh.types import Sphere, RayFlat, Hit, SphereLeafBlock


@always_inline
def trace_gpu_wide_ray[
    width: Int,
    mode: String,
    leaf_fn: def(
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[UInt32, MutAnyOrigin],
        UInt32,
        UInt32,
        RayFlat,
        mut Float32,
        mut Float32,
        mut Float32,
        mut UInt32,
    ) capturing -> Bool,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    ray: RayFlat,
) -> Hit:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var best_t = ray.t_max
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        var bounds_hit_mask = _intersect_wide_node_bounds[width](
            wide_bounds,
            current,
            ray,
            ray.t_max,
        )

        comptime for node_lane in range(width):
            var lane_base = _wide_lane_base[width](current, node_lane)
            var count = UInt32(wide_counts[lane_base])

            if count != EMPTY_LANE and bounds_hit_mask[node_lane]:
                var data = UInt32(wide_data[lane_base])

                if count == 0:
                    if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                        stack[stack_ptr] = data
                        stack_ptr += 1
                else:
                    var leaf_hit = leaf_fn(
                        leaf_spheres,
                        leaf_prims,
                        data,
                        count,
                        ray,
                        best_t,
                        best_u,
                        best_v,
                        best_prim,
                    )

                    comptime if mode == TRACE_SHADOW:
                        if leaf_hit:
                            return Hit(0.0, 0.0, 0.0, best_prim, UInt32(1))

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return Hit(best_t, best_u, best_v, best_prim, UInt32(0))
