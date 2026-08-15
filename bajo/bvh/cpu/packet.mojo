"""Shared-stack SIMD ray-packet traversal for CPU wide BVHs."""

from bajo.bvh.constants import EMPTY_LANE, f32_max
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.core import Frame, Point3, Vec3
from bajo.core.intersect import intersect_ray_aabb_rcp


comptime CPU_PACKET_STACK_SIZE = 256


@fieldwise_init
struct RayPacket[frame: Frame, lanes: SIMDLength](Copyable):
    var o: Point3[DType.float32, Self.frame, Self.lanes]
    var d: Vec3[DType.float32, Self.frame, Self.lanes]
    var t_min: SIMD[DType.float32, Self.lanes]
    var t_max: SIMD[DType.float32, Self.lanes]


struct PacketHit[frame: Frame, lanes: SIMDLength](Copyable):
    var u: SIMD[DType.float32, Self.lanes]
    var v: SIMD[DType.float32, Self.lanes]
    var prim: SIMD[DType.uint32, Self.lanes]
    var inst: SIMD[DType.uint32, Self.lanes]
    var normal: Vec3[DType.float32, Self.frame, Self.lanes]
    var t: SIMD[DType.float32, Self.lanes]

    def __init__(out self, t_max: SIMD[DType.float32, Self.lanes]):
        self.u = 0.0
        self.v = 0.0
        self.prim = EMPTY_LANE
        self.inst = EMPTY_LANE
        self.normal = Vec3[DType.float32, Self.frame, Self.lanes](0.0)
        self.t = t_max

    @always_inline
    def hit_mask(self) -> SIMD[DType.bool, Self.lanes]:
        return self.prim.ne(EMPTY_LANE) & self.t.lt(f32_max)


@always_inline
def _packet_rcp_direction[
    frame: Frame, lanes: SIMDLength
](direction: Vec3[DType.float32, frame, lanes]) -> Vec3[
    DType.float32, frame, lanes
]:
    var epsilon = SIMD[DType.float32, lanes](1.0e-9)
    var large = SIMD[DType.float32, lanes](1.0e9)
    var one = SIMD[DType.float32, lanes](1.0)
    var mx = abs(direction.x).gt(epsilon)
    var my = abs(direction.y).gt(epsilon)
    var mz = abs(direction.z).gt(epsilon)
    var sx = direction.x.lt(0.0).select(-large, large)
    var sy = direction.y.lt(0.0).select(-large, large)
    var sz = direction.z.lt(0.0).select(-large, large)
    return Vec3[DType.float32, frame, lanes](
        mx.select(one / mx.select(direction.x, one), sx),
        my.select(one / my.select(direction.y, one), sy),
        mz.select(one / mz.select(direction.z, one), sz),
    )


def trace_packet_bounds_bvh[
    frame: Frame,
    bounds_width: SIMDLength,
    ray_lanes: SIMDLength,
    LeafFn: def(
        SIMD[DType.bool, ray_lanes],
        UInt32,
        mut PacketHit[frame, ray_lanes],
    ),
](
    tree: BoundsBvh[frame, bounds_width],
    rays: RayPacket[frame, ray_lanes],
    valid: SIMD[DType.bool, ray_lanes],
    mut hit: PacketHit[frame, ray_lanes],
    ref leaf_fn: LeafFn,
):
    """Traverse one wide hierarchy with a shared stack and per-task ray mask."""
    if len(tree.nodes) == 0:
        return

    if not valid.reduce_or():
        return

    var reciprocal_direction = _packet_rcp_direction(rays.d)
    var stack_refs = Array[UInt32, CPU_PACKET_STACK_SIZE](uninitialized=True)
    var stack_masks = Array[SIMD[DType.bool, ray_lanes], CPU_PACKET_STACK_SIZE](
        uninitialized=True
    )
    var stack_near = Array[
        SIMD[DType.float32, ray_lanes], CPU_PACKET_STACK_SIZE
    ](uninitialized=True)
    var stack_priorities = Array[Float32, CPU_PACKET_STACK_SIZE](
        uninitialized=True
    )
    var stack_ptr = 1
    stack_refs.unsafe_get(0) = UInt32(0)
    stack_masks.unsafe_get(0) = valid
    stack_near.unsafe_get(0) = 0.0
    stack_priorities.unsafe_get(0) = 0.0

    @always_inline
    def push_task(
        child_ref: UInt32,
        child_mask: SIMD[DType.bool, ray_lanes],
        child_near: SIMD[DType.float32, ray_lanes],
    ) {
        imm,
        mut stack_ptr,
        mut stack_refs,
        mut stack_masks,
        mut stack_near,
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
            stack_near.unsafe_get(insert_idx) = stack_near.unsafe_get(
                previous_idx
            )
            stack_priorities.unsafe_get(
                insert_idx
            ) = stack_priorities.unsafe_get(previous_idx)
            insert_idx = previous_idx
        stack_refs.unsafe_get(insert_idx) = child_ref
        stack_masks.unsafe_get(insert_idx) = child_mask
        stack_near.unsafe_get(insert_idx) = child_near
        stack_priorities.unsafe_get(insert_idx) = priority
        stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        var child_ref = stack_refs.unsafe_get(stack_ptr)
        var active = stack_masks.unsafe_get(stack_ptr) & (
            stack_near.unsafe_get(stack_ptr).le(hit.t)
        )
        if not active.reduce_or():
            continue

        if is_leaf_ref(child_ref):
            leaf_fn(active, decode_ref_index(child_ref), hit)
            continue

        ref node = tree.nodes.unsafe_get(Int(child_ref))
        comptime for child_lane in range(bounds_width):
            var next_ref = node.data[child_lane]
            if next_ref != EMPTY_LANE:
                var bmin = Point3[DType.float32, frame, ray_lanes](
                    node.aabb._min.x[child_lane],
                    node.aabb._min.y[child_lane],
                    node.aabb._min.z[child_lane],
                )
                var bmax = Point3[DType.float32, frame, ray_lanes](
                    node.aabb._max.x[child_lane],
                    node.aabb._max.y[child_lane],
                    node.aabb._max.z[child_lane],
                )
                var bounds_hit = intersect_ray_aabb_rcp(
                    rays.o, reciprocal_direction, bmin, bmax, hit.t
                )
                var next_mask = active & bounds_hit.mask
                if next_mask.reduce_or():
                    push_task(next_ref, next_mask, bounds_hit.t)
