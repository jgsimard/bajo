from bajo.bvh.types import Ray, Hit
from bajo.core.intersect import intersect_ray_aabb
from bajo.core.vec import Vec3
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.constants import EMPTY_LANE, CPU_TRAVERSAL_STACK_SIZE, TRACE


def trace_bounds_bvh[
    width: Int,
    mode: TRACE,
    leaf_fn: def(Ray, UInt32, UInt32, mut Hit) capturing -> Bool,
](tree: BoundsBvh[width], ray: Ray) -> Hit:
    if len(tree.nodes) == 0:
        return Hit.miss()

    var hit = Hit.miss(ray.t_max)

    var stack = InlineArray[UInt32, CPU_TRAVERSAL_STACK_SIZE](
        uninitialized=True
    )
    var stack_ptr = 0
    var n_idx = UInt32(0)

    var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
    var rD = 1.0 / Vec3[DType.float32, width](ray.d.x, ray.d.y, ray.d.z)

    while True:
        ref node = tree.nodes[Int(n_idx)]

        var aabb_hit = intersect_ray_aabb(O, rD, node.aabb, hit.t)
        var valid_lane = ~node.counts.eq(EMPTY_LANE)
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
                            node.data[i],
                            node.counts[i],
                            hit,
                        ):
                            comptime if mode == TRACE.ANY_HIT:
                                return Hit.shadow_hit()

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        n_idx = stack[stack_ptr]

    return hit
