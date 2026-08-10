from bajo.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _intersect_wide_node,
)
from bajo.bvh.gpu.wide_meta import _wide_meta_count, _wide_meta_data
from bajo.bvh.constants import (
    GPU_STACK_SIZE,
    f32_max,
    TRACE,
    EMPTY_LANE,
)
from bajo.bvh.types import Hit
from bajo.core import Frame, Rayf32


def _trace_bounds_bvh_distance_aware[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        Pointer[mut=False, Float32, _],
        UInt32,
        Rayf32[frame],
        mut Hit[frame],
    ) capturing -> Bool,
](
    wide_nodes: Pointer[mut=False, Float32, _],
    leaves: Pointer[mut=False, Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    var hit = Hit[frame].miss(ray.t_max)
    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_near = Array[Float32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = root_idx

    while True:
        var node_t_max = hit.t
        comptime if mode == TRACE.ANY_HIT:
            node_t_max = ray.t_max

        var node_hit = _intersect_wide_node[frame, width](
            wide_nodes, current, ray, node_t_max
        )
        var bounds_hit = node_hit.bounds_hit
        var child_valid = Array[Bool, width](fill=False)
        var child_data = Array[UInt32, width](fill=0)
        var child_t = Array[Float32, width](fill=0.0)

        comptime for node_lane in range(width):
            var meta = node_hit.meta[node_lane]
            var count = _wide_meta_count(meta)

            if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                var data = _wide_meta_data(meta)
                if count == 0:
                    child_valid[node_lane] = True
                    child_data[node_lane] = data
                    child_t[node_lane] = bounds_hit.t[node_lane]
                else:
                    var leaf_hit = leaf_fn(leaves, data, ray, hit)
                    comptime if mode == TRACE.ANY_HIT:
                        if leaf_hit:
                            return Hit[frame].shadow_hit()

        var nearest_lane = -1
        var nearest_t = f32_max
        comptime for lane in range(width):
            if child_valid[lane] and child_t[lane] < nearest_t:
                nearest_lane = lane
                nearest_t = child_t[lane]

        comptime for lane in range(width):
            if child_valid[lane] and lane != nearest_lane:
                comptime if mode == TRACE.CLOSEST_HIT:
                    if child_t[lane] > hit.t:
                        continue

                debug_assert["safe", _use_compiler_assume=True](
                    stack_ptr < GPU_STACK_SIZE,
                    "GPU BVH traversal stack overflow",
                )
                stack[stack_ptr] = child_data[lane]
                stack_near[stack_ptr] = child_t[lane]
                stack_ptr += 1

        if nearest_lane != -1:
            comptime if mode == TRACE.CLOSEST_HIT:
                if nearest_t > hit.t:
                    nearest_lane = -1

            if nearest_lane != -1:
                current = child_data[nearest_lane]
                continue

        comptime if mode == TRACE.CLOSEST_HIT:
            var found_pending = False
            while stack_ptr > 0:
                stack_ptr -= 1
                if stack_near[stack_ptr] <= hit.t:
                    current = stack[stack_ptr]
                    found_pending = True
                    break

            if not found_pending:
                break
        else:
            if stack_ptr == 0:
                break

            stack_ptr -= 1
            current = stack[stack_ptr]

    return hit


def trace_bounds_bvh[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        Pointer[mut=False, Float32, _],
        UInt32,
        Rayf32[frame],
        mut Hit[frame],
    ) capturing -> Bool,
    lifo: Bool = True,
    distance_aware: Bool = False,
](
    wide_nodes: Pointer[mut=False, Float32, _],
    leaves: Pointer[mut=False, Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    comptime if distance_aware:
        comptime assert lifo, "distance-aware GPU traversal requires LIFO"
        return _trace_bounds_bvh_distance_aware[frame, width, mode, leaf_fn](
            wide_nodes, leaves, root_idx, ray
        )

    var hit = Hit[frame].miss(ray.t_max)

    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = root_idx

    while True:
        var node_t_max = hit.t
        comptime if mode == TRACE.ANY_HIT:
            node_t_max = ray.t_max

        var node_hit = _intersect_wide_node[frame, width](
            wide_nodes,
            current,
            ray,
            node_t_max,
        )
        var bounds_hit = node_hit.bounds_hit

        comptime if lifo:
            var child_valid = Array[Bool, width](fill=False)
            var child_data = Array[UInt32, width](fill=0)
            var child_t = Array[Float32, width](fill=0.0)

            comptime for node_lane in range(width):
                var meta = node_hit.meta[node_lane]
                var count = _wide_meta_count(meta)

                if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                    var data = _wide_meta_data(meta)

                    if count == 0:
                        child_valid[node_lane] = True
                        child_data[node_lane] = data
                        child_t[node_lane] = bounds_hit.t[node_lane]
                    else:
                        var leaf_hit = leaf_fn(
                            leaves,
                            data,
                            ray,
                            hit,
                        )

                        comptime if mode == TRACE.ANY_HIT:
                            if leaf_hit:
                                return Hit[frame].shadow_hit()

            # Push all other children first and the nearest child last, so it
            # is popped first without fully sorting the remaining children.
            var nearest_lane = -1
            var nearest_t = f32_max
            comptime for lane in range(width):
                if child_valid[lane] and child_t[lane] < nearest_t:
                    nearest_lane = lane
                    nearest_t = child_t[lane]

            comptime for lane in range(width):
                if child_valid[lane] and lane != nearest_lane:
                    comptime if mode != TRACE.ANY_HIT:
                        if child_t[lane] > hit.t:
                            continue

                    debug_assert["safe", _use_compiler_assume=True](
                        stack_ptr < GPU_STACK_SIZE,
                        "GPU BVH traversal stack overflow",
                    )
                    stack[stack_ptr] = child_data[lane]
                    stack_ptr += 1

            if nearest_lane != -1:
                comptime if mode == TRACE.CLOSEST_HIT:
                    if nearest_t > hit.t:
                        nearest_lane = -1

                if nearest_lane != -1:
                    debug_assert["safe", _use_compiler_assume=True](
                        stack_ptr < GPU_STACK_SIZE,
                        "GPU BVH traversal stack overflow",
                    )
                    stack[stack_ptr] = child_data[nearest_lane]
                    stack_ptr += 1
        else:
            # basically the same as the cpu version
            comptime for node_lane in range(width):
                var meta = node_hit.meta[node_lane]
                var count = _wide_meta_count(meta)

                if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                    var data = _wide_meta_data(meta)

                    if count == 0:
                        comptime if mode != TRACE.ANY_HIT:
                            if bounds_hit.t[node_lane] > hit.t:
                                continue

                        debug_assert["safe", _use_compiler_assume=True](
                            stack_ptr < GPU_STACK_SIZE,
                            "GPU BVH traversal stack overflow",
                        )
                        stack[stack_ptr] = data
                        stack_ptr += 1
                    else:
                        var leaf_hit = leaf_fn(
                            leaves,
                            data,
                            ray,
                            hit,
                        )

                        comptime if mode == TRACE.ANY_HIT:
                            if leaf_hit:
                                return Hit[frame].shadow_hit()

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return hit
