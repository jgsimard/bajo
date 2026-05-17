from bajo.core.bvh.types import Ray
from bajo.core.intersect import intersect_ray_aabb
from bajo.core.vec import Vec3
from bajo.core.bvh.cpu.bounds_bvh import BoundsBvh, EMPTY_LANE


@always_inline
def traverse_wide_ray_bvh[
    width: Int,
    is_occlusion: Bool,
    leaf_fn: def(mut Ray, UInt32, UInt32) capturing -> Bool,
](tree: BoundsBvh[width], mut ray: Ray) -> Bool:
    if len(tree.nodes) == 0:
        return False

    var stack = InlineArray[UInt32, 64](fill=0)
    var s_ptr = 0
    var n_idx = UInt32(0)

    var O = Vec3[DType.float32, width](ray.O.x, ray.O.y, ray.O.z)
    var rD = Vec3[DType.float32, width](ray.rD.x, ray.rD.y, ray.rD.z)

    while True:
        ref node = tree.nodes[Int(n_idx)]

        var hit = intersect_ray_aabb(O, rD, node.aabb, ray.hit.t)
        var valid_lane = ~node.counts.eq(EMPTY_LANE)
        var mask = hit.mask & valid_lane

        if mask.reduce_or():
            for i in range(width):
                if mask[i]:
                    if node.counts[i] == 0:
                        stack[s_ptr] = node.data[i]
                        s_ptr += 1
                    else:
                        if leaf_fn(ray, node.data[i], node.counts[i]):
                            comptime if is_occlusion:
                                return True

        if s_ptr == 0:
            break

        s_ptr -= 1
        n_idx = stack[s_ptr]

    return False
