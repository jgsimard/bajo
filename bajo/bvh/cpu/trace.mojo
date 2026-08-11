from std.bit import count_trailing_zeros
from std.memory import bitcast, pack_bits
from std.sys.intrinsics import llvm_intrinsic
from std.sys import size_of
from std.sys.info import CompilationTarget

from bajo.bvh.types import Hit
from bajo.core.intersect import (
    RayDistanceHit,
    intersect_ray_aabb_octant_fma,
)
from bajo.core import (
    AxisAlignedBoundingBox,
    Vec3,
    Point3,
    Frame,
    Rayf32,
    dot,
)
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    WideBvhNode,
    BVH_LEAF_REF_BIT,
    BVH_REF_INDEX_MASK,
)
from bajo.bvh.constants import EMPTY_LANE, CPU_STACK_SIZE, TRACE


@fieldwise_init
struct CpuBvhTraversalStats(Copyable):
    """Counters collected by an explicitly instrumented CPU BVH trace."""

    var rays: Int
    var internal_nodes: Int
    var nodes_with_hits: Int
    var aabb_packet_lanes: Int
    var active_child_lanes: Int
    var aabb_hit_lanes: Int
    var leaf_blocks: Int
    var primitive_packet_lanes: Int
    var valid_primitives: Int
    var primitive_hit_candidates: Int
    var closer_hit_updates: Int
    var stack_pushes: Int
    var stack_insertion_shifts: Int
    var stack_pops: Int
    var stack_pruned_tasks: Int
    var max_stack_depth: Int
    var any_hit_early_exits: Int

    def __init__(out self):
        self.rays = 0
        self.internal_nodes = 0
        self.nodes_with_hits = 0
        self.aabb_packet_lanes = 0
        self.active_child_lanes = 0
        self.aabb_hit_lanes = 0
        self.leaf_blocks = 0
        self.primitive_packet_lanes = 0
        self.valid_primitives = 0
        self.primitive_hit_candidates = 0
        self.closer_hit_updates = 0
        self.stack_pushes = 0
        self.stack_insertion_shifts = 0
        self.stack_pops = 0
        self.stack_pruned_tasks = 0
        self.max_stack_depth = 0
        self.any_hit_early_exits = 0


@always_inline
def _count_true_lanes[width: SIMDLength](mask: SIMD[DType.bool, width]) -> Int:
    var count = 0
    comptime for lane in range(width):
        if mask[lane]:
            count += 1
    return count


@always_inline
def _pack_pending_task(child_ref: UInt32, child_t: Float32) -> UInt64:
    # AABB entry distances are non-negative, so their Float32 bit patterns
    # preserve numeric ordering when placed in the high half of the task.
    var t_bits = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](child_t))[0]
    return (UInt64(t_bits) << 32) | UInt64(child_ref)


@always_inline
def _pending_task_ref(task: UInt64) -> UInt32:
    return UInt32(task)


@always_inline
def _pending_task_t(task: UInt64) -> Float32:
    var bits = SIMD[DType.uint32, 1](UInt32(task >> 32))
    return bitcast[DType.float32, 1](bits)[0]


@always_inline
def _extract_u32_lane[
    width: SIMDLength
](values: SIMD[DType.uint32, width], lane: Int) -> UInt32:
    """Extract a dynamic SIMD lane without forcing a vector spill."""
    comptime assert width in [2, 4, 8, 16]

    comptime if width == 16 and CompilationTarget.has_avx512f():
        var indices = SIMD[DType.uint32, width](UInt32(lane))
        return llvm_intrinsic[
            "llvm.x86.avx512.permvar.si.512",
            SIMD[DType.uint32, width],
            has_side_effect=False,
        ](values, indices)[0]
    elif width == 8 and CompilationTarget.has_avx2():
        var indices = SIMD[DType.uint32, width](UInt32(lane))
        return llvm_intrinsic[
            "llvm.x86.avx2.permd",
            SIMD[DType.uint32, width],
            has_side_effect=False,
        ](values, indices)[0]
    elif width == 4 and CompilationTarget.has_avx():
        var indices = SIMD[DType.int32, width](Int32(lane))
        var floats = bitcast[DType.float32, width](values)
        var permuted = llvm_intrinsic[
            "llvm.x86.avx.vpermilvar.ps",
            SIMD[DType.float32, width],
            has_side_effect=False,
        ](floats, indices)
        return bitcast[DType.uint32, width](permuted)[0]
    # elif width == 2:
    #     if lane == 0:
    #         return values[0]
    #     return values[1]
    else:
        return values[lane]


@always_inline
def _extract_f32_lane[
    width: SIMDLength
](values: SIMD[DType.float32, width], lane: Int) -> Float32:
    var bits = bitcast[DType.uint32, width](values)
    var extracted = _extract_u32_lane(bits, lane)
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](extracted))[0]


def _trace_bounds_bvh_impl[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE,
    collect_stats: Bool,
    leaf_uses_rcp_direction: Bool,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, leaf_width],
        Vec3[DType.float32, frame, leaf_width],
        SIMD[DType.float32, leaf_width],
        SIMD[DType.float32, leaf_width],
        UInt32,
        mut CpuBvhTraversalStats,
        mut Hit[frame],
    ) capturing -> Bool,
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
](
    tree: BoundsBvh[frame, bounds_width],
    ray: Rayf32[frame],
    ray_a: SIMD[DType.float32, leaf_width],
    ray_inv_a: SIMD[DType.float32, leaf_width],
    mut stats: CpuBvhTraversalStats,
) -> Hit[frame]:
    debug_assert["safe", _use_compiler_assume=True](len(tree.nodes) > 0)

    var hit = Hit[frame].miss(ray.t_max)

    var stack_ptr = 0

    var bounds_O = ray.origin[bounds_width]()
    var rcp_d = ray.rcp_direction[bounds_width]()
    var origin_rcp_d = Vec3[DType.float32, frame, bounds_width](
        bounds_O.x * rcp_d.x,
        bounds_O.y * rcp_d.y,
        bounds_O.z * rcp_d.z,
    )
    var leaf_O = ray.origin[leaf_width]()
    var leaf_D = ray.direction[leaf_width]()
    comptime if leaf_uses_rcp_direction:
        # `rcp_d` is splatted from one scalar ray, so lane zero can be
        # rebroadcast at a different leaf width without another reciprocal.
        leaf_D = Vec3[DType.float32, frame, leaf_width](
            rcp_d.x[0], rcp_d.y[0], rcp_d.z[0]
        )
    var nodes = Span(tree.nodes)

    @always_inline
    def intersect_node(
        aabb: AxisAlignedBoundingBox[DType.float32, frame, bounds_width]
    ) capturing -> RayDistanceHit[DType.float32, bounds_width]:
        return intersect_ray_aabb_octant_fma[
            positive_x=positive_x,
            positive_y=positive_y,
            positive_z=positive_z,
        ](origin_rcp_d, rcp_d, aabb, hit.t)

    comptime if mode == TRACE.CLOSEST_HIT:
        # stack entries are tagged child references, so they can represent either an internal node or a packed leaf
        var ordered_stack = Array[UInt64, CPU_STACK_SIZE](uninitialized=True)

        @always_inline
        def push_pending(child_ref: UInt32, child_t: Float32) capturing:
            debug_assert["safe", _use_compiler_assume=True](
                stack_ptr < CPU_STACK_SIZE,
                "CPU BVH traversal stack overflow",
            )

            # Keep pending tasks far-to-near so the nearest global task is
            # always the final entry and can be popped in O(1).
            var task = _pack_pending_task(child_ref, child_t)
            var insert_idx = stack_ptr
            while insert_idx > 0:
                var previous_idx = insert_idx - 1
                var previous = ordered_stack.unsafe_get(previous_idx)
                if previous >= task:
                    break
                ordered_stack.unsafe_get(insert_idx) = previous
                insert_idx = previous_idx

            ordered_stack.unsafe_get(insert_idx) = task

            stack_ptr += 1
            comptime if collect_stats:
                stats.stack_pushes += 1
                stats.stack_insertion_shifts += stack_ptr - insert_idx - 1
                if stack_ptr > stats.max_stack_depth:
                    stats.max_stack_depth = stack_ptr

        # root is an internal-node reference with index zero
        var current_ref = UInt32(0)

        while True:
            if (current_ref & BVH_LEAF_REF_BIT) != 0:
                # leaves are deferred exactly like internal nodes
                # why: nearby internal subtree run before a distant triangle block
                comptime if collect_stats:
                    stats.leaf_blocks += 1
                _ = leaf_fn(
                    ray,
                    leaf_O,
                    leaf_D,
                    ray_a,
                    ray_inv_a,
                    current_ref & BVH_REF_INDEX_MASK,
                    stats,
                    hit,
                )

            else:
                comptime if collect_stats:
                    stats.internal_nodes += 1
                    stats.aabb_packet_lanes += Int(bounds_width)
                ref node = nodes.unsafe_get(Int(current_ref))
                comptime assert size_of[
                    WideBvhNode[frame, bounds_width]
                ]() == 7 * 4 * Int(bounds_width)
                var node_data_ptr = (
                    nodes.unsafe_ptr()
                    .unsafe_offset(Int(current_ref))
                    .unsafe_bitcast[UInt32]()
                    .unsafe_offset(6 * Int(bounds_width))
                )

                var aabb_hit = intersect_node(node.aabb)
                var mask = aabb_hit.mask
                comptime if bounds_width != 16:
                    mask &= node.data.ne(EMPTY_LANE)

                comptime if collect_stats:
                    var active_mask = node.data.ne(EMPTY_LANE)
                    stats.active_child_lanes += _count_true_lanes(active_mask)
                    var hit_lanes = _count_true_lanes(mask)
                    stats.aabb_hit_lanes += hit_lanes
                    if hit_lanes > 0:
                        stats.nodes_with_hits += 1

                # internal children and leaves compete equally for the nearest task
                # keep that task in registers and defer the others
                var has_nearest = False
                var nearest_ref = UInt32(0)
                var nearest_t = Float32(0.0)

                def visit_closest_child(
                    child_ref: UInt32, child_t: Float32
                ) capturing:
                    # hit.t can only become tighter after a deferred leaf is evaluated
                    if child_t > hit.t:
                        return

                    if not has_nearest:
                        nearest_ref = child_ref
                        nearest_t = child_t
                        has_nearest = True
                        return

                    if child_t < nearest_t:
                        push_pending(nearest_ref, nearest_t)
                        nearest_ref = child_ref
                        nearest_t = child_t
                    else:
                        push_pending(child_ref, child_t)

                def visit_closest_lane(i: Int) capturing:
                    visit_closest_child(node.data[i], aabb_hit.t[i])

                comptime if bounds_width == 16:
                    # BVH16 benefits from consuming only set mask bits: see benchmarks
                    var bits = UInt32(pack_bits(mask)) & (
                        tree.child_masks.unsafe_get(Int(current_ref))
                    )

                    while bits != 0:
                        var lane = Int(count_trailing_zeros(bits))
                        bits &= bits - 1
                        visit_closest_child(
                            node_data_ptr[unsafe_offset=lane],
                            _extract_f32_lane(aabb_hit.t, lane),
                        )

                else:
                    # for BVH2/4/8, fully unrolled checks are faster: see benchmarks
                    if mask.reduce_or():
                        comptime for lane in range(bounds_width):
                            if mask[lane]:
                                visit_closest_lane(lane)

                if has_nearest:
                    if stack_ptr > 0:
                        var pending_idx = stack_ptr - 1
                        var pending = ordered_stack.unsafe_get(pending_idx)
                        if _pending_task_t(pending) < nearest_t:
                            stack_ptr = pending_idx
                            comptime if collect_stats:
                                stats.stack_pops += 1
                            push_pending(nearest_ref, nearest_t)
                            current_ref = _pending_task_ref(pending)
                            continue

                    current_ref = nearest_ref
                    continue

            # current leaf finished, or the current node had no children
            # skip every deferred task that is now farther than the best hit
            var found_pending = False
            if stack_ptr > 0:
                var next_idx = stack_ptr - 1
                var task = ordered_stack.unsafe_get(next_idx)
                if _pending_task_t(task) <= hit.t:
                    stack_ptr = next_idx
                    comptime if collect_stats:
                        stats.stack_pops += 1
                    current_ref = _pending_task_ref(task)
                    found_pending = True
                else:
                    # The nearest pending task is already too far, so all
                    # other sorted tasks can be discarded at once.
                    comptime if collect_stats:
                        stats.stack_pruned_tasks += stack_ptr
                    stack_ptr = 0

            if not found_pending:
                break

    else:
        # ANY_HIT : leaf-first behavior
        var stack = Array[UInt32, CPU_STACK_SIZE](uninitialized=True)
        var n_idx = UInt32(0)

        while True:
            comptime if collect_stats:
                stats.internal_nodes += 1
                stats.aabb_packet_lanes += Int(bounds_width)
            ref node = nodes.unsafe_get(Int(n_idx))
            comptime assert size_of[
                WideBvhNode[frame, bounds_width]
            ]() == 7 * 4 * Int(bounds_width)
            var node_data_ptr = (
                nodes.unsafe_ptr()
                .unsafe_offset(Int(n_idx))
                .unsafe_bitcast[UInt32]()
                .unsafe_offset(6 * Int(bounds_width))
            )

            var aabb_hit = intersect_node(node.aabb)
            var mask = aabb_hit.mask
            comptime if bounds_width != 16:
                mask &= node.data.ne(EMPTY_LANE)

            comptime if collect_stats:
                var active_mask = node.data.ne(EMPTY_LANE)
                stats.active_child_lanes += _count_true_lanes(active_mask)
                var hit_lanes = _count_true_lanes(mask)
                stats.aabb_hit_lanes += hit_lanes
                if hit_lanes > 0:
                    stats.nodes_with_hits += 1

            # keep the internal child visited immediately out of the stack
            var has_next = False
            var next_idx = UInt32(0)

            @always_inline
            def visit_any_child(child_ref: UInt32) capturing -> Bool:
                if (child_ref & BVH_LEAF_REF_BIT) != 0:
                    comptime if collect_stats:
                        stats.leaf_blocks += 1
                    var leaf_hit = leaf_fn(
                        ray,
                        leaf_O,
                        leaf_D,
                        ray_a,
                        ray_inv_a,
                        child_ref & BVH_REF_INDEX_MASK,
                        stats,
                        hit,
                    )
                    comptime if collect_stats:
                        if leaf_hit:
                            stats.any_hit_early_exits += 1
                    return leaf_hit

                if has_next:
                    debug_assert["safe", _use_compiler_assume=True](
                        stack_ptr < CPU_STACK_SIZE,
                        "CPU BVH traversal stack overflow",
                    )
                    stack.unsafe_get(stack_ptr) = next_idx
                    stack_ptr += 1
                    comptime if collect_stats:
                        stats.stack_pushes += 1
                        if stack_ptr > stats.max_stack_depth:
                            stats.max_stack_depth = stack_ptr

                next_idx = child_ref
                has_next = True
                return False

            comptime if bounds_width == 16:
                # BVH16: consume only active mask bits.
                var bits = UInt32(pack_bits(mask)) & (
                    tree.child_masks.unsafe_get(Int(n_idx))
                )

                while bits != 0:
                    var lane = Int(count_trailing_zeros(bits))
                    bits &= bits - 1

                    if visit_any_child(node_data_ptr[unsafe_offset=lane]):
                        return Hit[frame].shadow_hit()

            else:
                # BVH2/4/8: fully unrolled lane checks are faster
                if mask.reduce_or():
                    comptime for lane in range(bounds_width):
                        if mask[lane]:
                            if visit_any_child(node.data[lane]):
                                return Hit[frame].shadow_hit()

            if has_next:
                n_idx = next_idx
                continue

            if stack_ptr == 0:
                break

            stack_ptr -= 1
            comptime if collect_stats:
                stats.stack_pops += 1
            n_idx = stack.unsafe_get(stack_ptr)
    return hit


def _trace_bounds_bvh_octant[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE,
    collect_stats: Bool,
    leaf_uses_rcp_direction: Bool,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, leaf_width],
        Vec3[DType.float32, frame, leaf_width],
        SIMD[DType.float32, leaf_width],
        SIMD[DType.float32, leaf_width],
        UInt32,
        mut CpuBvhTraversalStats,
        mut Hit[frame],
    ) capturing -> Bool,
](
    tree: BoundsBvh[frame, bounds_width],
    ray: Rayf32[frame],
    ray_a: SIMD[DType.float32, leaf_width],
    ray_inv_a: SIMD[DType.float32, leaf_width],
    mut stats: CpuBvhTraversalStats,
) -> Hit[frame]:
    @always_inline
    def trace_octant[
        positive_x: Bool, positive_y: Bool, positive_z: Bool
    ]() capturing -> Hit[frame]:
        return _trace_bounds_bvh_impl[
            frame=frame,
            bounds_width=bounds_width,
            leaf_width=leaf_width,
            mode=mode,
            collect_stats=collect_stats,
            leaf_uses_rcp_direction=leaf_uses_rcp_direction,
            leaf_fn=leaf_fn,
            positive_x=positive_x,
            positive_y=positive_y,
            positive_z=positive_z,
        ](tree, ray, ray_a, ray_inv_a, stats)

    var positive_x = ray.d.x >= 0.0
    var positive_y = ray.d.y >= 0.0
    var positive_z = ray.d.z >= 0.0

    if positive_x:
        if positive_y:
            if positive_z:
                return trace_octant[
                    positive_x=True, positive_y=True, positive_z=True
                ]()
            return trace_octant[
                positive_x=True, positive_y=True, positive_z=False
            ]()
        if positive_z:
            return trace_octant[
                positive_x=True, positive_y=False, positive_z=True
            ]()
        return trace_octant[
            positive_x=True, positive_y=False, positive_z=False
        ]()

    if positive_y:
        if positive_z:
            return trace_octant[
                positive_x=False, positive_y=True, positive_z=True
            ]()
        return trace_octant[
            positive_x=False, positive_y=True, positive_z=False
        ]()
    if positive_z:
        return trace_octant[
            positive_x=False, positive_y=False, positive_z=True
        ]()
    return trace_octant[positive_x=False, positive_y=False, positive_z=False]()


def trace_bounds_bvh[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, leaf_width],
        Vec3[DType.float32, frame, leaf_width],
        SIMD[DType.float32, leaf_width],
        SIMD[DType.float32, leaf_width],
        UInt32,
        mut Hit[frame],
    ) capturing -> Bool,
](tree: BoundsBvh[frame, bounds_width], ray: Rayf32[frame]) -> Hit[frame]:
    @always_inline
    def unmeasured_leaf_fn(
        ray: Rayf32[frame],
        O: Point3[DType.float32, frame, leaf_width],
        D: Vec3[DType.float32, frame, leaf_width],
        ray_a: SIMD[DType.float32, leaf_width],
        ray_inv_a: SIMD[DType.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut _stats: CpuBvhTraversalStats,
        mut hit: Hit[frame],
    ) capturing -> Bool:
        return leaf_fn(ray, O, D, ray_a, ray_inv_a, leaf_block_idx, hit)

    var zero = SIMD[DType.float32, leaf_width](0.0)
    var unused_stats = CpuBvhTraversalStats()
    return _trace_bounds_bvh_octant[
        frame=frame,
        bounds_width=bounds_width,
        leaf_width=leaf_width,
        mode=mode,
        collect_stats=False,
        leaf_uses_rcp_direction=False,
        leaf_fn=unmeasured_leaf_fn,
    ](tree, ray, zero, zero, unused_stats)


def trace_bounds_bvh_measured[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, leaf_width],
        Vec3[DType.float32, frame, leaf_width],
        SIMD[DType.float32, leaf_width],
        SIMD[DType.float32, leaf_width],
        UInt32,
        mut CpuBvhTraversalStats,
        mut Hit[frame],
    ) capturing -> Bool,
](
    tree: BoundsBvh[frame, bounds_width],
    ray: Rayf32[frame],
    mut stats: CpuBvhTraversalStats,
) -> Hit[frame]:
    """Trace while accumulating diagnostic counters in `stats`.

    This is intentionally a separate entry point so production traversal is
    compiled with every counter update removed.
    """
    stats.rays += 1
    var zero = SIMD[DType.float32, leaf_width](0.0)
    return _trace_bounds_bvh_octant[
        frame=frame,
        bounds_width=bounds_width,
        leaf_width=leaf_width,
        mode=mode,
        collect_stats=True,
        leaf_uses_rcp_direction=False,
        leaf_fn=leaf_fn,
    ](tree, ray, zero, zero, stats)


def trace_bounds_bvh_leaf_rcp[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, leaf_width],
        Vec3[DType.float32, frame, leaf_width],
        SIMD[DType.float32, leaf_width],
        SIMD[DType.float32, leaf_width],
        UInt32,
        mut Hit[frame],
    ) capturing -> Bool,
](tree: BoundsBvh[frame, bounds_width], ray: Rayf32[frame]) -> Hit[frame]:
    """Trace with reciprocal ray direction in the leaf direction argument."""

    @always_inline
    def unmeasured_leaf_fn(
        ray: Rayf32[frame],
        O: Point3[DType.float32, frame, leaf_width],
        D: Vec3[DType.float32, frame, leaf_width],
        ray_a: SIMD[DType.float32, leaf_width],
        ray_inv_a: SIMD[DType.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut _stats: CpuBvhTraversalStats,
        mut hit: Hit[frame],
    ) capturing -> Bool:
        return leaf_fn(ray, O, D, ray_a, ray_inv_a, leaf_block_idx, hit)

    var zero = SIMD[DType.float32, leaf_width](0.0)
    var unused_stats = CpuBvhTraversalStats()
    return _trace_bounds_bvh_octant[
        frame=frame,
        bounds_width=bounds_width,
        leaf_width=leaf_width,
        mode=mode,
        collect_stats=False,
        leaf_uses_rcp_direction=True,
        leaf_fn=unmeasured_leaf_fn,
    ](tree, ray, zero, zero, unused_stats)


def trace_sphere_bounds_bvh[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE,
    leaf_fn: def(
        Rayf32[frame],
        Point3[DType.float32, frame, leaf_width],
        Vec3[DType.float32, frame, leaf_width],
        SIMD[DType.float32, leaf_width],
        SIMD[DType.float32, leaf_width],
        UInt32,
        mut Hit[frame],
    ) capturing -> Bool,
](tree: BoundsBvh[frame, bounds_width], ray: Rayf32[frame]) -> Hit[frame]:
    @always_inline
    def unmeasured_leaf_fn(
        ray: Rayf32[frame],
        O: Point3[DType.float32, frame, leaf_width],
        D: Vec3[DType.float32, frame, leaf_width],
        ray_a: SIMD[DType.float32, leaf_width],
        ray_inv_a: SIMD[DType.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut _stats: CpuBvhTraversalStats,
        mut hit: Hit[frame],
    ) capturing -> Bool:
        return leaf_fn(ray, O, D, ray_a, ray_inv_a, leaf_block_idx, hit)

    var leaf_D = ray.direction[leaf_width]()
    var ray_a = dot(leaf_D, leaf_D)
    var ray_inv_a = 1.0 / ray_a
    var unused_stats = CpuBvhTraversalStats()
    return _trace_bounds_bvh_octant[
        frame=frame,
        bounds_width=bounds_width,
        leaf_width=leaf_width,
        mode=mode,
        collect_stats=False,
        leaf_uses_rcp_direction=False,
        leaf_fn=unmeasured_leaf_fn,
    ](tree, ray, ray_a, ray_inv_a, unused_stats)
