from bajo.bvh.gpu.wide_layout import (
    WideNodeIntersection,
    _intersect_wide_node_precomputed,
    _intersect_wide_node_precomputed_octant,
)
from bajo.bvh.wide_meta import (
    _pack_wide_meta,
    _wide_meta_count,
    _wide_meta_data,
)
from bajo.bvh.constants import (
    GPU_STACK_SIZE,
    f32_max,
    TRACE,
    EMPTY_LANE,
)
from bajo.bvh.types import Hit
from bajo.core import Frame, Point3, Rayf32, Vec3


comptime GpuLeafFn[frame: Frame] = def(
    ImmPointer[Float32, _],
    UInt32,
    UInt32,
    Rayf32[frame],
    mut Hit[frame],
) thin -> Bool


comptime GpuLeafStateFn[frame: Frame, LeafState: AnyType] = def(
    LeafState,
    UInt32,
    UInt32,
    Rayf32[frame],
    mut Hit[frame],
) thin -> Bool


@fieldwise_init
struct GpuPrimitiveLeafState(TrivialRegisterPassable):
    var leaves: Pointer[Float32, ImmUntrackedOrigin]


@always_inline
def _dispatch_primitive_leaf[
    frame: Frame,
    mode: TRACE,
    leaf_fn: GpuLeafFn[frame],
](
    state: GpuPrimitiveLeafState,
    data: UInt32,
    count: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    var found = leaf_fn(state.leaves, data, count, ray, hit)
    comptime if mode == TRACE.ANY_HIT:
        if found:
            hit = Hit[frame].shadow_hit()
    return found


@fieldwise_init
struct GpuTraversalAlgorithm(Equatable):
    """Compile-time traversal selector, independent of build topology."""

    comptime STANDARD = Self(0)
    comptime UNIFIED_TASKS = Self(1)
    var value: Int


@fieldwise_init
struct GpuTraversalStats(TrivialRegisterPassable, Writable):
    comptime NODE_VISITS = 0
    comptime INTERNAL_CHILD_HITS = 1
    comptime LEAF_BLOCKS = 2
    comptime PRIMITIVE_TESTS = 3
    comptime MAX_STACK_DEPTH = 4
    comptime STRIDE = 5

    var node_visits: UInt32
    var internal_child_hits: UInt32
    var leaf_blocks: UInt32
    var primitive_tests: UInt32
    var max_stack_depth: UInt32

    @staticmethod
    def zero() -> Self:
        return Self(0, 0, 0, 0, 0)

    def store(
        self,
        dst: MutPointer[UInt32, _],
        ray_idx: Int,
    ):
        var base = ray_idx * Self.STRIDE
        dst[unsafe_offset=base + Self.NODE_VISITS] = self.node_visits
        dst[
            unsafe_offset=base + Self.INTERNAL_CHILD_HITS
        ] = self.internal_child_hits
        dst[unsafe_offset=base + Self.LEAF_BLOCKS] = self.leaf_blocks
        dst[unsafe_offset=base + Self.PRIMITIVE_TESTS] = self.primitive_tests
        dst[unsafe_offset=base + Self.MAX_STACK_DEPTH] = self.max_stack_depth


@fieldwise_init
struct GpuTraceResult[frame: Frame](TrivialRegisterPassable):
    var hit: Hit[Self.frame]
    var stats: GpuTraversalStats


@always_inline
def _intersect_trace_node_precomputed[
    frame: Frame,
    width: SIMDLength,
](
    wide_nodes: ImmPointer[Float32, _],
    node_idx: UInt32,
    bounds_origin: Point3[DType.float32, frame, width],
    rcp_direction: Vec3[DType.float32, frame, width],
    t_max: Float32,
) -> WideNodeIntersection[width]:
    return _intersect_wide_node_precomputed[frame, width](
        wide_nodes, node_idx, bounds_origin, rcp_direction, t_max
    )


@always_inline
def _intersect_trace_node_precomputed_octant[
    frame: Frame,
    width: SIMDLength,
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
](
    wide_nodes: ImmPointer[Float32, _],
    node_idx: UInt32,
    origin_rcp_direction: Vec3[DType.float32, frame, width],
    rcp_direction: Vec3[DType.float32, frame, width],
    t_max: Float32,
) -> WideNodeIntersection[width]:
    return _intersect_wide_node_precomputed_octant[
        frame, width, positive_x, positive_y, positive_z
    ](wide_nodes, node_idx, origin_rcp_direction, rcp_direction, t_max)


def _trace_bounds_bvh_distance_aware[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: GpuLeafFn[frame],
    collect_stats: Bool,
](
    wide_nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> GpuTraceResult[frame]:
    var hit = Hit[frame].miss(ray.t_max)
    var stats = GpuTraversalStats.zero()
    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_near = Array[Float32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = root_idx
    var bounds_origin = ray.origin[width]()
    var rcp_direction = ray.rcp_direction[width]()
    while True:
        comptime if collect_stats:
            stats.node_visits += 1
        var node_t_max = hit.t
        comptime if mode == TRACE.ANY_HIT:
            node_t_max = ray.t_max

        var node_hit = _intersect_trace_node_precomputed[frame, width](
            wide_nodes, current, bounds_origin, rcp_direction, node_t_max
        )
        var bounds_hit = node_hit.bounds_hit
        var child_valid = Array[Bool, width](fill=False)
        var child_data = Array[UInt32, width](fill=0)
        var child_t = Array[Float32, width](fill=0.0)

        comptime for node_lane in range(width):
            var meta = node_hit.meta[node_lane]
            var count = _wide_meta_count(meta)

            if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                var data = _wide_meta_data(meta)
                if count == 0:
                    comptime if collect_stats:
                        stats.internal_child_hits += 1
                    child_valid[node_lane] = True
                    child_data[node_lane] = data
                    child_t[node_lane] = bounds_hit.t[node_lane]
                else:
                    comptime if collect_stats:
                        stats.leaf_blocks += 1
                        stats.primitive_tests += count
                    var leaf_hit = leaf_fn(leaves, data, count, ray, hit)
                    comptime if mode == TRACE.ANY_HIT:
                        if leaf_hit:
                            return GpuTraceResult[frame](
                                Hit[frame].shadow_hit(), stats
                            )

        var nearest_lane = -1
        var nearest_t = f32_max
        comptime for lane in range(width):
            if child_valid[lane] and child_t[lane] < nearest_t:
                nearest_lane = lane
                nearest_t = child_t[lane]

        comptime for lane in range(width):
            if child_valid[lane] and lane != nearest_lane:
                comptime if mode == TRACE.CLOSEST_HIT:
                    if child_t[lane] > hit.t:
                        continue

                debug_assert["safe", _use_compiler_assume=True](
                    stack_ptr < GPU_STACK_SIZE,
                    "GPU BVH traversal stack overflow",
                )
                stack[stack_ptr] = child_data[lane]
                stack_near[stack_ptr] = child_t[lane]
                stack_ptr += 1
                comptime if collect_stats:
                    if UInt32(stack_ptr) > stats.max_stack_depth:
                        stats.max_stack_depth = UInt32(stack_ptr)

        if nearest_lane != -1:
            comptime if mode == TRACE.CLOSEST_HIT:
                if nearest_t > hit.t:
                    nearest_lane = -1

            if nearest_lane != -1:
                current = child_data[nearest_lane]
                continue

        comptime if mode == TRACE.CLOSEST_HIT:
            var found_pending = False
            while stack_ptr > 0:
                stack_ptr -= 1
                if stack_near[stack_ptr] <= hit.t:
                    current = stack[stack_ptr]
                    found_pending = True
                    break

            if not found_pending:
                break
        else:
            if stack_ptr == 0:
                break

            stack_ptr -= 1
            current = stack[stack_ptr]

    return GpuTraceResult[frame](hit, stats)


def _trace_bounds_bvh_unified_closest_impl[
    frame: Frame,
    width: SIMDLength,
    leaf_fn: GpuLeafFn[frame],
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
](
    wide_nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Near-first traversal where leaf and internal children are both tasks."""
    var hit = Hit[frame].miss(ray.t_max)
    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_near = Array[Float32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current_meta = _pack_wide_meta(root_idx, UInt32(0))
    var current_near = Float32(0.0)
    var bounds_origin = ray.origin[width]()
    var rcp_direction = ray.rcp_direction[width]()
    var origin_rcp_direction = Vec3[DType.float32, frame, width](
        bounds_origin.x * rcp_direction.x,
        bounds_origin.y * rcp_direction.y,
        bounds_origin.z * rcp_direction.z,
    )
    while True:
        if current_near > hit.t:
            var found_pending = False
            while stack_ptr > 0:
                stack_ptr -= 1
                if stack_near[stack_ptr] <= hit.t:
                    current_meta = stack[stack_ptr]
                    current_near = stack_near[stack_ptr]
                    found_pending = True
                    break
            if not found_pending:
                break

        var current_count = _wide_meta_count(current_meta)
        if current_count != 0:
            _ = leaf_fn(
                leaves,
                _wide_meta_data(current_meta),
                current_count,
                ray,
                hit,
            )

            var found_pending = False
            while stack_ptr > 0:
                stack_ptr -= 1
                if stack_near[stack_ptr] <= hit.t:
                    current_meta = stack[stack_ptr]
                    current_near = stack_near[stack_ptr]
                    found_pending = True
                    break
            if not found_pending:
                break
            continue

        var node_hit = _intersect_trace_node_precomputed_octant[
            frame, width, positive_x, positive_y, positive_z
        ](
            wide_nodes,
            _wide_meta_data(current_meta),
            origin_rcp_direction,
            rcp_direction,
            hit.t,
        )
        var child_valid = Array[Bool, width](fill=False)
        var child_meta = Array[UInt32, width](fill=0)
        var child_t = Array[Float32, width](fill=0.0)

        comptime for lane in range(width):
            var meta = node_hit.meta[lane]
            if (
                _wide_meta_count(meta) != EMPTY_LANE
                and node_hit.bounds_hit.mask[lane]
            ):
                child_valid[lane] = True
                child_meta[lane] = meta
                child_t[lane] = node_hit.bounds_hit.t[lane]

        var nearest_lane = -1
        var nearest_t = f32_max
        comptime for lane in range(width):
            if child_valid[lane] and child_t[lane] < nearest_t:
                nearest_lane = lane
                nearest_t = child_t[lane]

        comptime for lane in range(width):
            if (
                child_valid[lane]
                and lane != nearest_lane
                and child_t[lane] <= hit.t
            ):
                debug_assert["safe", _use_compiler_assume=True](
                    stack_ptr < GPU_STACK_SIZE,
                    "GPU unified BVH traversal stack overflow",
                )
                stack[stack_ptr] = child_meta[lane]
                stack_near[stack_ptr] = child_t[lane]
                stack_ptr += 1

        if nearest_lane != -1 and nearest_t <= hit.t:
            current_meta = child_meta[nearest_lane]
            current_near = nearest_t
            continue

        var found_pending = False
        while stack_ptr > 0:
            stack_ptr -= 1
            if stack_near[stack_ptr] <= hit.t:
                current_meta = stack[stack_ptr]
                current_near = stack_near[stack_ptr]
                found_pending = True
                break
        if not found_pending:
            break

    return hit


def trace_bounds_bvh_unified_closest[
    frame: Frame,
    width: SIMDLength,
    leaf_fn: GpuLeafFn[frame],
](
    wide_nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Dispatch unified traversal to one compile-time ray octant."""
    var positive_x = ray.d.x >= 0.0
    var positive_y = ray.d.y >= 0.0
    var positive_z = ray.d.z >= 0.0

    if positive_x:
        if positive_y:
            if positive_z:
                return _trace_bounds_bvh_unified_closest_impl[
                    frame, width, leaf_fn, True, True, True
                ](wide_nodes, leaves, root_idx, ray)
            return _trace_bounds_bvh_unified_closest_impl[
                frame, width, leaf_fn, True, True, False
            ](wide_nodes, leaves, root_idx, ray)
        if positive_z:
            return _trace_bounds_bvh_unified_closest_impl[
                frame, width, leaf_fn, True, False, True
            ](wide_nodes, leaves, root_idx, ray)
        return _trace_bounds_bvh_unified_closest_impl[
            frame, width, leaf_fn, True, False, False
        ](wide_nodes, leaves, root_idx, ray)

    if positive_y:
        if positive_z:
            return _trace_bounds_bvh_unified_closest_impl[
                frame, width, leaf_fn, False, True, True
            ](wide_nodes, leaves, root_idx, ray)
        return _trace_bounds_bvh_unified_closest_impl[
            frame, width, leaf_fn, False, True, False
        ](wide_nodes, leaves, root_idx, ray)
    if positive_z:
        return _trace_bounds_bvh_unified_closest_impl[
            frame, width, leaf_fn, False, False, True
        ](wide_nodes, leaves, root_idx, ray)
    return _trace_bounds_bvh_unified_closest_impl[
        frame, width, leaf_fn, False, False, False
    ](wide_nodes, leaves, root_idx, ray)


def _trace_bounds_bvh_with_counters[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: GpuLeafFn[frame],
    collect_stats: Bool,
    lifo: Bool = True,
    distance_aware: Bool = False,
](
    wide_nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> GpuTraceResult[frame]:
    comptime if distance_aware:
        comptime assert lifo, "distance-aware GPU traversal requires LIFO"
        return _trace_bounds_bvh_distance_aware[
            frame, width, mode, leaf_fn, collect_stats
        ](wide_nodes, leaves, root_idx, ray)

    var hit = Hit[frame].miss(ray.t_max)
    var stats = GpuTraversalStats.zero()

    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = root_idx
    var bounds_origin = ray.origin[width]()
    var rcp_direction = ray.rcp_direction[width]()

    while True:
        comptime if collect_stats:
            stats.node_visits += 1
        var node_t_max = hit.t
        comptime if mode == TRACE.ANY_HIT:
            node_t_max = ray.t_max

        var node_hit = _intersect_trace_node_precomputed[frame, width](
            wide_nodes, current, bounds_origin, rcp_direction, node_t_max
        )
        var bounds_hit = node_hit.bounds_hit

        comptime if lifo:
            var child_valid = Array[Bool, width](fill=False)
            var child_data = Array[UInt32, width](fill=0)
            var child_t = Array[Float32, width](fill=0.0)

            comptime for node_lane in range(width):
                var meta = node_hit.meta[node_lane]
                var count = _wide_meta_count(meta)

                if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                    var data = _wide_meta_data(meta)

                    if count == 0:
                        comptime if collect_stats:
                            stats.internal_child_hits += 1
                        child_valid[node_lane] = True
                        child_data[node_lane] = data
                        child_t[node_lane] = bounds_hit.t[node_lane]
                    else:
                        comptime if collect_stats:
                            stats.leaf_blocks += 1
                            stats.primitive_tests += count
                        var leaf_hit = leaf_fn(
                            leaves,
                            data,
                            count,
                            ray,
                            hit,
                        )

                        comptime if mode == TRACE.ANY_HIT:
                            if leaf_hit:
                                return GpuTraceResult[frame](
                                    Hit[frame].shadow_hit(), stats
                                )

            # Keep the nearest child as the direct continuation. Only deferred
            # siblings consume stack entries.
            var nearest_lane = -1
            var nearest_t = f32_max
            comptime for lane in range(width):
                if child_valid[lane] and child_t[lane] < nearest_t:
                    nearest_lane = lane
                    nearest_t = child_t[lane]

            comptime for lane in range(width):
                if child_valid[lane] and lane != nearest_lane:
                    comptime if mode != TRACE.ANY_HIT:
                        if child_t[lane] > hit.t:
                            continue

                    debug_assert["safe", _use_compiler_assume=True](
                        stack_ptr < GPU_STACK_SIZE,
                        "GPU BVH traversal stack overflow",
                    )
                    stack[stack_ptr] = child_data[lane]
                    stack_ptr += 1
                    comptime if collect_stats:
                        if UInt32(stack_ptr) > stats.max_stack_depth:
                            stats.max_stack_depth = UInt32(stack_ptr)

            if nearest_lane != -1:
                comptime if mode == TRACE.CLOSEST_HIT:
                    if nearest_t > hit.t:
                        nearest_lane = -1

                if nearest_lane != -1:
                    current = child_data[nearest_lane]
                    continue
        else:
            # basically the same as the cpu version
            comptime for node_lane in range(width):
                var meta = node_hit.meta[node_lane]
                var count = _wide_meta_count(meta)

                if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                    var data = _wide_meta_data(meta)

                    if count == 0:
                        comptime if collect_stats:
                            stats.internal_child_hits += 1
                        comptime if mode != TRACE.ANY_HIT:
                            if bounds_hit.t[node_lane] > hit.t:
                                continue

                        debug_assert["safe", _use_compiler_assume=True](
                            stack_ptr < GPU_STACK_SIZE,
                            "GPU BVH traversal stack overflow",
                        )
                        stack[stack_ptr] = data
                        stack_ptr += 1
                        comptime if collect_stats:
                            if UInt32(stack_ptr) > stats.max_stack_depth:
                                stats.max_stack_depth = UInt32(stack_ptr)
                    else:
                        comptime if collect_stats:
                            stats.leaf_blocks += 1
                            stats.primitive_tests += count
                        var leaf_hit = leaf_fn(
                            leaves,
                            data,
                            count,
                            ray,
                            hit,
                        )

                        comptime if mode == TRACE.ANY_HIT:
                            if leaf_hit:
                                return GpuTraceResult[frame](
                                    Hit[frame].shadow_hit(), stats
                                )

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return GpuTraceResult[frame](hit, stats)


@always_inline
def _trace_bounds_bvh_state_bvh2[
    frame: Frame,
    LeafState: AnyType,
    leaf_fn: GpuLeafStateFn[frame, LeafState],
](
    wide_nodes: ImmPointer[Float32, _],
    leaf_state: LeafState,
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Two-child closest-hit loop without generic width-N child arrays."""
    var hit = Hit[frame].miss(ray.t_max)
    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = root_idx
    var bounds_origin = ray.origin[2]()
    var rcp_direction = ray.rcp_direction[2]()

    while True:
        var node_hit = _intersect_trace_node_precomputed[frame, 2](
            wide_nodes,
            current,
            bounds_origin,
            rcp_direction,
            hit.t,
        )

        var child0_valid = False
        var child1_valid = False
        var child0_data = UInt32(0)
        var child1_data = UInt32(0)
        var child0_t = Float32(0.0)
        var child1_t = Float32(0.0)

        comptime for lane in range(2):
            var meta = node_hit.meta[lane]
            var count = _wide_meta_count(meta)
            if count != EMPTY_LANE and node_hit.bounds_hit.mask[lane]:
                var data = _wide_meta_data(meta)
                if count == 0:
                    if lane == 0:
                        child0_valid = True
                        child0_data = data
                        child0_t = node_hit.bounds_hit.t[lane]
                    else:
                        child1_valid = True
                        child1_data = data
                        child1_t = node_hit.bounds_hit.t[lane]
                else:
                    _ = leaf_fn(leaf_state, data, count, ray, hit)

        var nearest_lane = -1
        var nearest_t = f32_max
        if child0_valid:
            nearest_lane = 0
            nearest_t = child0_t
        if child1_valid and (nearest_lane == -1 or child1_t < nearest_t):
            nearest_lane = 1
            nearest_t = child1_t

        if child0_valid and nearest_lane != 0 and child0_t <= hit.t:
            debug_assert["safe", _use_compiler_assume=True](
                stack_ptr < GPU_STACK_SIZE,
                "GPU BVH traversal stack overflow",
            )
            stack[stack_ptr] = child0_data
            stack_ptr += 1
        if child1_valid and nearest_lane != 1 and child1_t <= hit.t:
            debug_assert["safe", _use_compiler_assume=True](
                stack_ptr < GPU_STACK_SIZE,
                "GPU BVH traversal stack overflow",
            )
            stack[stack_ptr] = child1_data
            stack_ptr += 1

        if nearest_lane != -1 and nearest_t <= hit.t:
            if nearest_lane == 0:
                current = child0_data
            else:
                current = child1_data
            continue

        if stack_ptr == 0:
            break
        stack_ptr -= 1
        current = stack[stack_ptr]

    return hit


def trace_bounds_bvh_state[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    LeafState: AnyType,
    leaf_fn: GpuLeafStateFn[frame, LeafState],
    compact_bvh2: Bool = False,
](
    wide_nodes: ImmPointer[Float32, _],
    leaf_state: LeafState,
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Shared near-first BVH loop for primitive and instance leaf state."""
    comptime if compact_bvh2:
        comptime assert width == 2 and mode == TRACE.CLOSEST_HIT
        return _trace_bounds_bvh_state_bvh2[frame, LeafState, leaf_fn](
            wide_nodes, leaf_state, root_idx, ray
        )

    var hit = Hit[frame].miss(ray.t_max)
    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = root_idx
    var bounds_origin = ray.origin[width]()
    var rcp_direction = ray.rcp_direction[width]()

    while True:
        var node_t_max = hit.t
        comptime if mode == TRACE.ANY_HIT:
            node_t_max = ray.t_max
        var node_hit = _intersect_trace_node_precomputed[frame, width](
            wide_nodes,
            current,
            bounds_origin,
            rcp_direction,
            node_t_max,
        )

        # Any-hit queries only need to find one leaf.  Keep this path
        # explicitly unordered: closest-hit's nearest-child reduction and
        # child-distance arrays are unnecessary work here.  The comptime
        # split preserves the generic leaf callback and the shared stack
        # while specializing the generated kernel for the query mode.
        comptime if mode == TRACE.ANY_HIT:
            comptime for node_lane in range(width):
                var meta = node_hit.meta[node_lane]
                var count = _wide_meta_count(meta)
                if count != EMPTY_LANE and node_hit.bounds_hit.mask[node_lane]:
                    var data = _wide_meta_data(meta)
                    if count == 0:
                        debug_assert["safe", _use_compiler_assume=True](
                            stack_ptr < GPU_STACK_SIZE,
                            "GPU BVH traversal stack overflow",
                        )
                        stack[stack_ptr] = data
                        stack_ptr += 1
                    else:
                        var leaf_hit = leaf_fn(
                            leaf_state, data, count, ray, hit
                        )
                        if leaf_hit:
                            return hit

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            current = stack[stack_ptr]
            continue

        var child_valid = Array[Bool, width](fill=False)
        var child_data = Array[UInt32, width](fill=0)
        var child_t = Array[Float32, width](fill=0.0)

        comptime for node_lane in range(width):
            var meta = node_hit.meta[node_lane]
            var count = _wide_meta_count(meta)
            if count != EMPTY_LANE and node_hit.bounds_hit.mask[node_lane]:
                var data = _wide_meta_data(meta)
                if count == 0:
                    child_valid[node_lane] = True
                    child_data[node_lane] = data
                    child_t[node_lane] = node_hit.bounds_hit.t[node_lane]
                else:
                    var leaf_hit = leaf_fn(leaf_state, data, count, ray, hit)
                    comptime if mode == TRACE.ANY_HIT:
                        if leaf_hit:
                            return hit

        var nearest_lane = -1
        var nearest_t = f32_max
        comptime for lane in range(width):
            if child_valid[lane] and child_t[lane] < nearest_t:
                nearest_lane = lane
                nearest_t = child_t[lane]

        comptime for lane in range(width):
            if child_valid[lane] and lane != nearest_lane:
                comptime if mode == TRACE.CLOSEST_HIT:
                    if child_t[lane] > hit.t:
                        continue
                debug_assert["safe", _use_compiler_assume=True](
                    stack_ptr < GPU_STACK_SIZE,
                    "GPU BVH traversal stack overflow",
                )
                stack[stack_ptr] = child_data[lane]
                stack_ptr += 1

        if nearest_lane != -1:
            comptime if mode == TRACE.CLOSEST_HIT:
                if nearest_t > hit.t:
                    nearest_lane = -1
            if nearest_lane != -1:
                current = child_data[nearest_lane]
                continue

        if stack_ptr == 0:
            break
        stack_ptr -= 1
        current = stack[stack_ptr]

    return hit


def trace_bounds_bvh[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: GpuLeafFn[frame],
    lifo: Bool = True,
    distance_aware: Bool = False,
    compact_bvh2: Bool = False,
](
    wide_nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    comptime if lifo and not distance_aware:
        return trace_bounds_bvh_state[
            frame,
            width,
            mode,
            GpuPrimitiveLeafState,
            _dispatch_primitive_leaf[frame, mode, leaf_fn],
            compact_bvh2,
        ](
            wide_nodes,
            GpuPrimitiveLeafState(
                leaves.unsafe_origin_cast[ImmUntrackedOrigin]()
            ),
            root_idx,
            ray,
        )
    return _trace_bounds_bvh_with_counters[
        frame, width, mode, leaf_fn, False, lifo, distance_aware
    ](wide_nodes, leaves, root_idx, ray).hit


def trace_bounds_bvh_with_stats[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    leaf_fn: GpuLeafFn[frame],
    lifo: Bool = True,
    distance_aware: Bool = False,
](
    wide_nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> GpuTraceResult[frame]:
    return _trace_bounds_bvh_with_counters[
        frame, width, mode, leaf_fn, True, lifo, distance_aware
    ](wide_nodes, leaves, root_idx, ray)
