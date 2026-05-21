from bajo.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _wide_lane_base,
    _intersect_wide_node_bounds,
)
from bajo.bvh.constants import (
    GPU_TRAVERSAL_STACK_SIZE,
    _gpu_inf_t,
    TRACE_CLOSEST_HIT,
    TRACE_ANY_HIT,
    EMPTY_LANE,
)
from bajo.bvh.types import Ray, Hit


def trace_bounds_bvh[
    width: Int,
    mode: String,
    leaf_fn: def(
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[UInt32, MutAnyOrigin],
        UInt32,
        UInt32,
        Ray,
        mut Hit,
    ) capturing -> Bool,
    lifo: Bool = True,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_data_f32: UnsafePointer[Float32, MutAnyOrigin],
    leaf_data_u32: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    ray: Ray,
) -> Hit:
    comptime assert mode in [TRACE_CLOSEST_HIT, TRACE_ANY_HIT]

    var hit = Hit.miss()
    hit.t = ray.t_max
    hit.prim = EMPTY_LANE

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](
        uninitialized=True
    )
    var stack_ptr = 0
    var current = root_idx

    while True:
        var node_t_max = hit.t
        comptime if mode == TRACE_ANY_HIT:
            node_t_max = ray.t_max

        var bounds_hit = _intersect_wide_node_bounds[width](
            wide_bounds,
            current,
            ray,
            node_t_max,
        )

        comptime if lifo:
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
                            leaf_data_f32,
                            leaf_data_u32,
                            data,
                            count,
                            ray,
                            hit,
                        )

                        comptime if mode == TRACE_ANY_HIT:
                            if leaf_hit:
                                return Hit.shadow_hit()

            # Push internal children far-to-near.
            # Since stack is LIFO, nearest child is popped first.
            comptime for _ in range(width):
                var far_lane = -1
                var far_t = -_gpu_inf_t

                comptime for lane in range(width):
                    if child_valid[lane]:
                        var t = child_t[lane]
                        if t >= far_t:
                            far_t = t
                            far_lane = lane

                if far_lane != -1:
                    child_valid[far_lane] = False

                    comptime if mode != TRACE_ANY_HIT:
                        if far_t > hit.t:
                            continue

                    if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                        stack[stack_ptr] = child_data[far_lane]
                        stack_ptr += 1
        else:
            # basically the same as the cpu version
            comptime for node_lane in range(width):
                var lane_base = _wide_lane_base[width](current, node_lane)
                var count = UInt32(wide_counts[lane_base])

                if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                    var data = UInt32(wide_data[lane_base])

                    if count == 0:
                        comptime if mode != TRACE_ANY_HIT:
                            if bounds_hit.tmin[node_lane] > hit.t:
                                continue

                        if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                            stack[stack_ptr] = data
                            stack_ptr += 1
                    else:
                        var leaf_hit = leaf_fn(
                            leaf_data_f32,
                            leaf_data_u32,
                            data,
                            count,
                            ray,
                            hit,
                        )

                        comptime if mode == TRACE_ANY_HIT:
                            if leaf_hit:
                                return Hit.shadow_hit()

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return hit
