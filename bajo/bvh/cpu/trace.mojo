from std.bit import count_trailing_zeros
from std.memory import pack_bits

from bajo.bvh.types import Hit
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.core import Vec3, Point3, Frame, Rayf32, dot
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.constants import EMPTY_LANE, CPU_STACK_SIZE, TRACE


def trace_bounds_bvh[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, width],
        Vec3[DType.float32, frame, width],
        SIMD[DType.float32, width],
        SIMD[DType.float32, width],
        UInt32,
        mut Hit[frame],
    ) capturing -> Bool,
    precompute_ray_coefficients: Bool = False,
](tree: BoundsBvh[frame, width], ray: Rayf32[frame]) -> Hit[frame]:
    debug_assert["safe", _use_compiler_assume=True](len(tree.nodes) > 0)

    var hit = Hit[frame].miss(ray.t_max)

    # avoid bounds checks in hot loop
    var stack = Array[UInt32, CPU_STACK_SIZE](uninitialized=True)
    var stack_near = Array[Float32, CPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var n_idx = UInt32(0)

    var O = ray.origin[width]()
    var D = ray.direction[width]()
    var rcp_d = ray.rcp_direction[width]()
    var ray_a = SIMD[DType.float32, width](0.0)
    var ray_inv_a = SIMD[DType.float32, width](0.0)
    comptime if precompute_ray_coefficients:
        ray_a = dot(D, D)
        ray_inv_a = 1.0 / ray_a
    var nodes = Span(tree.nodes)

    while True:
        ref node = nodes.unsafe_get(Int(n_idx))

        var aabb_hit = intersect_ray_aabb_rcp(O, rcp_d, node.aabb, hit.t)
        var valid_lane = node.counts.ne(EMPTY_LANE)
        var mask = aabb_hit.mask & valid_lane

        comptime if mode == TRACE.CLOSEST_HIT:
            var children_begin = stack_ptr

            # keep the nearest child in registers
            # only deferred siblings go onto the stack
            var has_nearest = False
            var nearest_idx = UInt32(0)
            var nearest_t = Float32(0.0)

            var bits = pack_bits(mask)

            while bits != 0:
                var i = Int(count_trailing_zeros(bits))
                bits &= bits - 1

                if node.counts[i] == 0:
                    var child_idx = node.data[i]
                    var child_t = aabb_hit.t[i]

                    if not has_nearest:
                        nearest_idx = child_idx
                        nearest_t = child_t
                        has_nearest = True

                    elif child_t < nearest_t:
                        # previous nearest becomes deferred
                        debug_assert["safe", _use_compiler_assume=True](
                            stack_ptr < CPU_STACK_SIZE,
                            "CPU BVH traversal stack overflow",
                        )
                        stack.unsafe_get(stack_ptr) = nearest_idx
                        stack_near.unsafe_get(stack_ptr) = nearest_t
                        stack_ptr += 1

                        nearest_idx = child_idx
                        nearest_t = child_t

                    else:
                        debug_assert["safe", _use_compiler_assume=True](
                            stack_ptr < CPU_STACK_SIZE,
                            "CPU BVH traversal stack overflow",
                        )
                        stack.unsafe_get(stack_ptr) = child_idx
                        stack_near.unsafe_get(stack_ptr) = child_t
                        stack_ptr += 1

                else:
                    _ = leaf_fn(
                        ray,
                        O,
                        D,
                        ray_a,
                        ray_inv_a,
                        node.data[i],
                        hit,
                    )

            if has_nearest:
                if nearest_t <= hit.t:
                    # sort only deferred siblings
                    # they remain far-to-near, so LIFO pops the nearest remaining sibling
                    for slot in range(children_begin + 1, stack_ptr):
                        var deferred_node = stack.unsafe_get(slot)
                        var deferred_t = stack_near.unsafe_get(slot)
                        var insert_slot = slot

                        while (
                            insert_slot > children_begin
                            and stack_near.unsafe_get(insert_slot - 1)
                            < deferred_t
                        ):
                            stack.unsafe_get(insert_slot) = stack.unsafe_get(
                                insert_slot - 1
                            )
                            stack_near.unsafe_get(
                                insert_slot
                            ) = stack_near.unsafe_get(insert_slot - 1)
                            insert_slot -= 1

                        stack.unsafe_get(insert_slot) = deferred_node
                        stack_near.unsafe_get(insert_slot) = deferred_t

                    n_idx = nearest_idx
                    continue

                # a leaf in this node tightened hit.t below the nearest internal child
                # because nearest_t is the minimum, every newly deferred sibling can also be discarded at once
                stack_ptr = children_begin

            var found_pending = False
            while stack_ptr > 0:
                stack_ptr -= 1

                if stack_near.unsafe_get(stack_ptr) <= hit.t:
                    n_idx = stack.unsafe_get(stack_ptr)
                    found_pending = True
                    break

            if not found_pending:
                break

        else:
            # keep the child that will be visited immediately out of the stack
            var has_next = False
            var next_idx = UInt32(0)

            var bits = pack_bits(mask)

            while bits != 0:
                var i = Int(count_trailing_zeros(bits))
                bits &= bits - 1

                if node.counts[i] == 0:
                    if has_next:
                        debug_assert["safe", _use_compiler_assume=True](
                            stack_ptr < CPU_STACK_SIZE,
                            "CPU BVH traversal stack overflow",
                        )
                        stack.unsafe_get(stack_ptr) = next_idx
                        stack_ptr += 1

                    next_idx = node.data[i]
                    has_next = True

                else:
                    if leaf_fn(
                        ray,
                        O,
                        D,
                        ray_a,
                        ray_inv_a,
                        node.data[i],
                        hit,
                    ):
                        return Hit[frame].shadow_hit()

            if has_next:
                n_idx = next_idx
                continue

            if stack_ptr == 0:
                break

            stack_ptr -= 1
            n_idx = stack.unsafe_get(stack_ptr)
    return hit
