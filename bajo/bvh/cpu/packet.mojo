"""Shared-stack SIMD ray-packet traversal for CPU wide BVHs."""

from std.bit import pop_count
from std.math import fma
from std.memory import pack_bits

from bajo.bvh.constants import EMPTY_LANE, f32_max
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.types import Hit
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.core import Frame, Point3, Ray, Vec3
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.core.utils import fmin, fmax


comptime CPU_PACKET_STACK_SIZE = 256


@always_inline
def trace_packet_stack_bounds_bvh[
    frame: Frame,
    bounds_width: SIMDLength,
    length: SIMDLength,
    LeafFn: def(
        SIMD[DType.bool, length],
        UInt32,
        mut Hit[frame, length],
    ),
    HybridFn: def(
        SIMD[DType.bool, length],
        UInt32,
        mut Hit[frame, length],
    ),
    hybrid_threshold: Int = 0,
    root_scalar_max_tasks: Int = 0,
    common_octant_fma: Bool = False,
    positive_x: Bool = True,
    positive_y: Bool = True,
    positive_z: Bool = True,
    hybrid_leaves: Bool = False,
](
    tree: BoundsBvh[frame, bounds_width],
    rays: Ray[DType.float32, frame, length],
    valid: SIMD[DType.bool, length],
    mut hit: Hit[frame, length],
    ref leaf_fn: LeafFn,
    ref hybrid_fn: HybridFn,
):
    """Traverse one wide hierarchy with a shared stack and per-task ray mask."""
    if len(tree.nodes) == 0:
        return

    if not valid.reduce_or():
        return

    var reciprocal_direction = rays.reciprocal_direction()
    var origin_rcp_direction = Vec3[DType.float32, frame, length](0.0)
    comptime if common_octant_fma:
        origin_rcp_direction = Vec3[DType.float32, frame, length](
            rays.o.x * reciprocal_direction.x,
            rays.o.y * reciprocal_direction.y,
            rays.o.z * reciprocal_direction.z,
        )
    var stack_refs = Array[UInt32, CPU_PACKET_STACK_SIZE](uninitialized=True)
    var stack_masks = Array[SIMD[DType.bool, length], CPU_PACKET_STACK_SIZE](
        uninitialized=True
    )
    var stack_priorities = Array[Float32, CPU_PACKET_STACK_SIZE](
        uninitialized=True
    )
    var stack_ptr = 1
    stack_refs.unsafe_get(0) = UInt32(0)
    stack_masks.unsafe_get(0) = valid
    stack_priorities.unsafe_get(0) = 0.0

    @always_inline
    def push_task(
        child_ref: UInt32,
        child_mask: SIMD[DType.bool, length],
        child_near: SIMD[DType.float32, length],
    ) {
        imm,
        mut stack_ptr,
        mut stack_refs,
        mut stack_masks,
        mut stack_priorities,
    }:
        debug_assert["safe", _use_compiler_assume=True](
            stack_ptr < CPU_PACKET_STACK_SIZE,
            "CPU packet BVH traversal stack overflow",
        )
        var priority = child_mask.select(child_near, f32_max).reduce_min()
        var insert_idx = stack_ptr
        while insert_idx > 0:
            var previous_idx = insert_idx - 1
            if stack_priorities.unsafe_get(previous_idx) >= priority:
                break
            stack_refs.unsafe_get(insert_idx) = stack_refs.unsafe_get(
                previous_idx
            )
            stack_masks.unsafe_get(insert_idx) = stack_masks.unsafe_get(
                previous_idx
            )
            stack_priorities.unsafe_get(
                insert_idx
            ) = stack_priorities.unsafe_get(previous_idx)
            insert_idx = previous_idx
        stack_refs.unsafe_get(insert_idx) = child_ref
        stack_masks.unsafe_get(insert_idx) = child_mask
        stack_priorities.unsafe_get(insert_idx) = priority
        stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        var child_ref = stack_refs.unsafe_get(stack_ptr)
        var active = stack_masks.unsafe_get(stack_ptr)
        active &= SIMD[DType.float32, length](
            stack_priorities.unsafe_get(stack_ptr)
        ).le(hit.t)
        if not active.reduce_or():
            continue

        comptime if hybrid_threshold > 0:
            if (hybrid_leaves or not is_leaf_ref(child_ref)) and stack_ptr >= 4:
                var active_count = pop_count(Int(pack_bits(active)))
                if active_count <= hybrid_threshold:
                    hybrid_fn(active, child_ref, hit)
                    continue

        if is_leaf_ref(child_ref):
            leaf_fn(active, decode_ref_index(child_ref), hit)
            continue

        ref node = tree.nodes.unsafe_get(Int(child_ref))
        comptime for child_lane in range(bounds_width):
            var next_ref = node.data[child_lane]
            if next_ref != EMPTY_LANE:
                var bounds_mask: SIMD[DType.bool, length]
                var bounds_t: SIMD[DType.float32, length]
                comptime if common_octant_fma:
                    var near_x = node.aabb._min.x[child_lane]
                    var far_x = node.aabb._max.x[child_lane]
                    var near_y = node.aabb._min.y[child_lane]
                    var far_y = node.aabb._max.y[child_lane]
                    var near_z = node.aabb._min.z[child_lane]
                    var far_z = node.aabb._max.z[child_lane]
                    comptime if not positive_x:
                        near_x, far_x = far_x, near_x
                    comptime if not positive_y:
                        near_y, far_y = far_y, near_y
                    comptime if not positive_z:
                        near_z, far_z = far_z, near_z

                    var tx_near = fma(
                        SIMD[DType.float32, length](near_x),
                        reciprocal_direction.x,
                        -origin_rcp_direction.x,
                    )
                    var tx_far = fma(
                        SIMD[DType.float32, length](far_x),
                        reciprocal_direction.x,
                        -origin_rcp_direction.x,
                    )
                    var ty_near = fma(
                        SIMD[DType.float32, length](near_y),
                        reciprocal_direction.y,
                        -origin_rcp_direction.y,
                    )
                    var ty_far = fma(
                        SIMD[DType.float32, length](far_y),
                        reciprocal_direction.y,
                        -origin_rcp_direction.y,
                    )
                    var tz_near = fma(
                        SIMD[DType.float32, length](near_z),
                        reciprocal_direction.z,
                        -origin_rcp_direction.z,
                    )
                    var tz_far = fma(
                        SIMD[DType.float32, length](far_z),
                        reciprocal_direction.z,
                        -origin_rcp_direction.z,
                    )
                    bounds_t = fmax(fmax(tx_near, ty_near), fmax(tz_near, 0.0))
                    var bounds_far = fmin(
                        fmin(tx_far, ty_far), fmin(tz_far, hit.t)
                    )
                    bounds_mask = bounds_t.le(bounds_far)
                else:
                    var bmin = Point3[DType.float32, frame, length](
                        node.aabb._min.x[child_lane],
                        node.aabb._min.y[child_lane],
                        node.aabb._min.z[child_lane],
                    )
                    var bmax = Point3[DType.float32, frame, length](
                        node.aabb._max.x[child_lane],
                        node.aabb._max.y[child_lane],
                        node.aabb._max.z[child_lane],
                    )
                    var bounds_hit = intersect_ray_aabb_rcp(
                        rays.o, reciprocal_direction, bmin, bmax, hit.t
                    )
                    bounds_mask = bounds_hit.mask
                    bounds_t = bounds_hit.t
                var next_mask = active & bounds_mask
                if next_mask.reduce_or():
                    push_task(next_ref, next_mask, bounds_t)

        comptime if root_scalar_max_tasks > 0:
            if (
                child_ref == 0
                and stack_ptr > 0
                and stack_ptr <= root_scalar_max_tasks
            ):
                hybrid_fn(valid, UInt32(0), hit)
                stack_ptr = 0
