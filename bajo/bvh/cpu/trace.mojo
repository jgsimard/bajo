from bajo.bvh.types import Ray, Hit
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.core.vec import Vec3
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.constants import EMPTY_LANE, CPU_STACK_SIZE, TRACE


def trace_bounds_bvh[
    width: SIMDSize,
    mode: TRACE,
    leaf_fn: def(
        Ray,
        Vec3[DType.float32, width],
        Vec3[DType.float32, width],
        UInt32,
        mut Hit,
    ) capturing -> Bool,
](tree: BoundsBvh[width], ray: Ray) -> Hit:
    debug_assert["safe"](len(tree.nodes) > 0)

    var hit = Hit.miss(ray.t_max)

    var stack = InlineArray[UInt32, CPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var n_idx = UInt32(0)

    var O = ray.simd_origin[width]()
    var D = ray.simd_direction[width]()
    var rcp_d = ray.simd_rcp_direction[width]()

    while True:
        ref node = tree.nodes[Int(n_idx)]

        var aabb_hit = intersect_ray_aabb_rcp(O, rcp_d, node.aabb, hit.t)
        var valid_lane = node.counts.ne(EMPTY_LANE)
        var mask = aabb_hit.mask & valid_lane

        if mask.reduce_or():
            for i in range(width):
                if mask[i]:
                    if node.counts[i] == 0:
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
                                return Hit.shadow_hit()

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        n_idx = stack[stack_ptr]

    return hit
