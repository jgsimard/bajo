from bajo.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _wide_node_load_meta,
    _intersect_wide_node_bounds,
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


def trace_bounds_bvh[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        UnsafePointer[mut=False, Float32, _],
        UInt32,
        Rayf32[frame],
        mut Hit[frame],
    ) capturing -> Bool,
    lifo: Bool = True,
](
    wide_nodes: UnsafePointer[mut=False, Float32, _],
    leaves: UnsafePointer[mut=False, Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    var hit = Hit[frame].miss(ray.t_max)

    var stack = InlineArray[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = root_idx

    while True:
        var node_t_max = hit.t
        comptime if mode == TRACE.ANY_HIT:
            node_t_max = ray.t_max

        var bounds_hit = _intersect_wide_node_bounds[frame, width](
            wide_nodes,
            current,
            ray,
            node_t_max,
        )

        comptime if lifo:
            var child_valid = InlineArray[Bool, width](fill=False)
            var child_data = InlineArray[UInt32, width](fill=0)
            var child_t = InlineArray[Float32, width](fill=0.0)

            comptime for node_lane in range(width):
                var meta = _wide_node_load_meta[width](
                    wide_nodes,
                    current,
                    node_lane,
                )
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

            # Push internal children far-to-near.
            # Since stack is LIFO, nearest child is popped first.
            comptime for _ in range(width):
                var far_lane = -1
                var far_t = -f32_max

                comptime for lane in range(width):
                    if child_valid[lane]:
                        var t = child_t[lane]
                        if t >= far_t:
                            far_t = t
                            far_lane = lane

                if far_lane != -1:
                    child_valid[far_lane] = False

                    comptime if mode != TRACE.ANY_HIT:
                        if far_t > hit.t:
                            continue

                    debug_assert["safe"](
                        stack_ptr < GPU_STACK_SIZE,
                        "GPU BVH traversal stack overflow",
                    )
                    stack[stack_ptr] = child_data[far_lane]
                    stack_ptr += 1
        else:
            # basically the same as the cpu version
            comptime for node_lane in range(width):
                var meta = _wide_node_load_meta[width](
                    wide_nodes,
                    current,
                    node_lane,
                )
                var count = _wide_meta_count(meta)

                if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                    var data = _wide_meta_data(meta)

                    if count == 0:
                        comptime if mode != TRACE.ANY_HIT:
                            if bounds_hit.t[node_lane] > hit.t:
                                continue

                        debug_assert["safe"](
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
