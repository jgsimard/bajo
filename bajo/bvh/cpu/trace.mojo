from bajo.bvh.types import Ray, Hit
from bajo.core.intersect import intersect_ray_aabb
from bajo.core.vec import Vec3
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.constants import (
    EMPTY_LANE,
    CPU_TRAVERSAL_STACK_SIZE,
    TRACE_ANY_HIT,
    TRACE_CLOSEST_HIT,
)


def trace_bounds_bvh[
    width: Int,
    mode: String,
    leaf_fn: def(Ray, UInt32, UInt32, mut Hit) capturing -> Bool,
](tree: BoundsBvh[width], ray: Ray) -> Hit:
    comptime assert mode in [TRACE_CLOSEST_HIT, TRACE_ANY_HIT]

    if len(tree.nodes) == 0:
        return Hit.miss()

    var out_hit = Hit.miss()
    out_hit.t = ray.t_max

    var stack = InlineArray[UInt32, CPU_TRAVERSAL_STACK_SIZE](
        uninitialized=True
    )
    var stack_ptr = 0
    var n_idx = UInt32(0)

    var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
    var rD = Vec3[DType.float32, width](ray.rd.x, ray.rd.y, ray.rd.z)

    while True:
        ref node = tree.nodes[Int(n_idx)]

        var hit = intersect_ray_aabb(O, rD, node.aabb, out_hit.t)
        var valid_lane = ~node.counts.eq(EMPTY_LANE)
        var mask = hit.mask & valid_lane

        if mask.reduce_or():
            for i in range(width):
                if mask[i]:
                    if node.counts[i] == 0:
                        stack[stack_ptr] = node.data[i]
                        stack_ptr += 1
                    else:
                        if leaf_fn(
                            ray,
                            node.data[i],
                            node.counts[i],
                            out_hit,
                        ):
                            comptime if mode == TRACE_ANY_HIT:
                                return Hit.shadow_hit()

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        n_idx = stack[stack_ptr]

    return out_hit
