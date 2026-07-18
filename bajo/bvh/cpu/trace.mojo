from bajo.bvh.types import Hit
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.core import Vec3, Point3, Frame, Rayf32
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.constants import EMPTY_LANE, CPU_STACK_SIZE, TRACE


def trace_bounds_bvh[
    frame: Frame,
    width: SIMDSize,
    mode: TRACE,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, width],
        Vec3[DType.float32, frame, width],
        UInt32,
        mut Hit[frame],
    ) capturing -> Bool,
](tree: BoundsBvh[frame, width], ray: Rayf32[frame]) -> Hit[frame]:
    debug_assert["safe"](len(tree.nodes) > 0)

    var hit = Hit[frame].miss(ray.t_max)

    var stack = InlineArray[UInt32, CPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var n_idx = UInt32(0)

    var O = ray.origin[width]()
    var D = ray.direction[width]()
    var rcp_d = ray.rcp_direction[width]()

    while True:
        ref node = tree.nodes[Int(n_idx)]

        var aabb_hit = intersect_ray_aabb_rcp(O, rcp_d, node.aabb, hit.t)
        var valid_lane = node.counts.ne(EMPTY_LANE)
        var mask = aabb_hit.mask & valid_lane

        if mask.reduce_or():
            for i in range(width):
                if mask[i]:
                    if node.counts[i] == 0:
                        debug_assert["safe"](
                            stack_ptr < CPU_STACK_SIZE,
                            "CPU BVH traversal stack overflow",
                        )
                        stack[stack_ptr] = node.data[i]
                        stack_ptr += 1
                    else:
                        if leaf_fn(
                            ray,
                            O,
                            D,
                            node.data[i],
                            hit,
                        ):
                            comptime if mode == TRACE.ANY_HIT:
                                return Hit[frame].shadow_hit()

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        n_idx = stack[stack_ptr]

    return hit
