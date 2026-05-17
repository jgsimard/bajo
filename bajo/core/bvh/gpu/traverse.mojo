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
    _gpu_inf_t,
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
        var node_t_max = best_t
        comptime if mode == TRACE_SHADOW:
            node_t_max = ray.t_max

        var bounds_hit = _intersect_wide_node_bounds[width](
            wide_bounds,
            current,
            ray,
            node_t_max,
        )

        var child_valid = InlineArray[Bool, width](fill=False)
        var child_data = InlineArray[UInt32, width](fill=0)
        var child_t = InlineArray[Float32, width](fill=0.0)

        comptime for node_lane in range(width):
            var lane_base = _wide_lane_base[width](current, node_lane)
            var count = UInt32(wide_counts[lane_base])

            if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                var data = UInt32(wide_data[lane_base])

                if count == 0:
                    child_valid[node_lane] = True
                    child_data[node_lane] = data
                    child_t[node_lane] = bounds_hit.tmin[node_lane]
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

        # Push internal children far-to-near.
        # Since stack is LIFO, nearest child is popped first.
        comptime for _ in range(width):
            var far_lane = -1
            var far_t = Float32(-1.0)

            comptime for lane in range(width):
                if child_valid[lane]:
                    var t = child_t[lane]
                    if t >= far_t:
                        far_t = t
                        far_lane = lane

            if far_lane != -1:
                child_valid[far_lane] = False

                comptime if mode != TRACE_SHADOW:
                    if far_t > best_t:
                        continue

                if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                    stack[stack_ptr] = child_data[far_lane]
                    stack_ptr += 1

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return Hit(best_t, best_u, best_v, best_prim, UInt32(0))
