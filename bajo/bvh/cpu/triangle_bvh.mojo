from max.algorithm import parallelize
from std.bit import count_trailing_zeros
from std.memory import pack_bits
from std.sys import size_of

from bajo.core import (
    Vec3,
    Vec3f32,
    Normal3f32,
    AABB,
    Point3,
    Point3f32,
    Frame,
    cross,
    normalize,
    Rayf32,
    Ray,
)
from bajo.bvh.constants import EMPTY_LANE, TraceMode, TRI_LEAF_PACKED_STRIDE
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BinaryBoundsBvh,
    WideBvhNode,
    WideLeafRange,
    _checked_typed_leaf_range,
)
from bajo.bvh.types import Hit, TriangleLeafBlock
from bajo.core.intersect import (
    intersect_ray_tri_edges,
    intersect_ray_tri_edges_scaled,
)
from bajo.bvh.cpu.trace import _extract_f32_lane
from bajo.bvh.cpu.packet import (
    trace_packet_stack_bounds_bvh,
)
from bajo.bvh.cpu.parallel import _worker_count


comptime PARALLEL_TRIANGLE_BUILD_MIN_ITEMS = 4096
comptime HYBRID_TRIANGLE_MIN_NODES = 1024
comptime ROOT_SCALAR_TRIANGLE_MIN_NODES = 2048


@always_inline
def _make_triangle_bounds_item[
    frame: Frame
](vertices: ImmSpan[Point3f32[frame], _], i: Int) -> BoundsItem[frame]:
    debug_assert["safe", _use_compiler_assume=True](i * 3 + 2 <= len(vertices))

    ref v0 = vertices.unsafe_get(i * 3 + 0)
    ref v1 = vertices.unsafe_get(i * 3 + 1)
    ref v2 = vertices.unsafe_get(i * 3 + 2)
    var bounds = AABB[frame](v0, v1, v2)
    return BoundsItem(bounds, UInt32(i))


@always_inline
def _trace_triangle_leaf_block[
    frame: Frame,
    width: SIMDLength,
    mode: TraceMode,
    packed_layout: Bool = False,
](
    ray: Rayf32[frame],
    O: Point3[DType.float32, frame, width],
    D: Vec3[.float32, frame, width],
    block: TriangleLeafBlock[frame, width],
    block_ptr: ImmPointer[Float32, _],
    mut hit: Hit[frame],
) -> Bool:
    """Shared proven CPU SIMD triangle-packet intersection kernel."""
    comptime omit_leaf_validity_mask = width == 16

    comptime if mode == .ANY_HIT:
        var tri_hit = intersect_ray_tri_edges(
            O,
            D,
            block.v0,
            block.e1,
            block.e2,
            hit.t,
            ray.t_min,
        )
        comptime if omit_leaf_validity_mask:
            return tri_hit.mask.reduce_or()
        else:
            var valid_lane = block.prim_indices.ne(EMPTY_LANE)
            return (tri_hit.mask & valid_lane).reduce_or()

    var scaled_hit = intersect_ray_tri_edges_scaled(
        O,
        D,
        block.v0,
        block.e1,
        block.e2,
        hit.t,
        ray.t_min,
    )
    var hit_mask = scaled_hit.mask
    comptime if not omit_leaf_validity_mask:
        hit_mask &= block.prim_indices.ne(EMPTY_LANE)

    if not hit_mask.reduce_or():
        return False

    comptime if mode == .CLOSEST_HIT:
        # Compare t_scaled / abs_det ratios without division, then calculate
        # one reciprocal for the winning scalar lane.
        var bits = pack_bits(hit_mask)
        var lane = Int(count_trailing_zeros(bits))
        bits &= bits - 1

        comptime if width == 16:
            comptime if packed_layout:
                comptime assert TRI_LEAF_PACKED_STRIDE == 12
            else:
                comptime assert size_of[
                    TriangleLeafBlock[frame, width]
                ]() == 10 * 4 * Int(width)
            var best_t_scaled = _extract_f32_lane(scaled_hit.t_scaled, lane)
            var best_abs_det = _extract_f32_lane(scaled_hit.abs_det, lane)

            while bits != 0:
                var candidate = Int(count_trailing_zeros(bits))
                bits &= bits - 1
                var candidate_t_scaled = _extract_f32_lane(
                    scaled_hit.t_scaled, candidate
                )
                var candidate_abs_det = _extract_f32_lane(
                    scaled_hit.abs_det, candidate
                )

                if (
                    candidate_t_scaled * best_abs_det
                    < best_t_scaled * candidate_abs_det
                ):
                    lane = candidate
                    best_t_scaled = candidate_t_scaled
                    best_abs_det = candidate_abs_det

            var inv_det = 1.0 / best_abs_det
            hit.t = best_t_scaled * inv_det
            hit.u = _extract_f32_lane(scaled_hit.u_scaled, lane) * inv_det
            hit.v = _extract_f32_lane(scaled_hit.v_scaled, lane) * inv_det

            var prim_field = 9
            var e1_field = 3
            var e2_field = 6
            comptime if packed_layout:
                prim_field = 3
                e1_field = 4
                e2_field = 8
            var block_u32 = block_ptr.unsafe_bitcast[UInt32]()
            hit.prim = block_u32[unsafe_offset=prim_field * Int(width) + lane]
            hit.inst = EMPTY_LANE
            var e1 = Vec3f32[frame](
                block_ptr[unsafe_offset=(e1_field + 0) * Int(width) + lane],
                block_ptr[unsafe_offset=(e1_field + 1) * Int(width) + lane],
                block_ptr[unsafe_offset=(e1_field + 2) * Int(width) + lane],
            )
            var e2 = Vec3f32[frame](
                block_ptr[unsafe_offset=(e2_field + 0) * Int(width) + lane],
                block_ptr[unsafe_offset=(e2_field + 1) * Int(width) + lane],
                block_ptr[unsafe_offset=(e2_field + 2) * Int(width) + lane],
            )
            var geometric_normal = cross(e1, e2)
            hit.normal = Normal3f32[frame](
                geometric_normal.x,
                geometric_normal.y,
                geometric_normal.z,
            )
            return True

        while bits != 0:
            var candidate = Int(count_trailing_zeros(bits))
            bits &= bits - 1

            if (
                scaled_hit.t_scaled[candidate] * scaled_hit.abs_det[lane]
                < scaled_hit.t_scaled[lane] * scaled_hit.abs_det[candidate]
            ):
                lane = candidate

        var inv_det = 1.0 / scaled_hit.abs_det[lane]

        hit.t = scaled_hit.t_scaled[lane] * inv_det
        hit.u = scaled_hit.u_scaled[lane] * inv_det
        hit.v = scaled_hit.v_scaled[lane] * inv_det
        hit.prim = block.prim_indices[lane]
        hit.inst = EMPTY_LANE

        var e1 = Vec3f32[frame](
            block.e1.x[lane],
            block.e1.y[lane],
            block.e1.z[lane],
        )
        var e2 = Vec3f32[frame](
            block.e2.x[lane],
            block.e2.y[lane],
            block.e2.z[lane],
        )

        var geometric_normal = cross(e1, e2)

        hit.normal = Normal3f32[frame](
            geometric_normal.x,
            geometric_normal.y,
            geometric_normal.z,
        )

    return True


@always_inline
def _trace_triangle_packet_primitive[
    frame: Frame,
    length: SIMDLength,
](
    rays: Ray[.float32, frame, length],
    active: SIMD[.bool, length],
    prim_idx: UInt32,
    v0_scalar: Point3f32[frame],
    e1_scalar: Vec3f32[frame],
    e2_scalar: Vec3f32[frame],
    mut packet_hit: Hit[frame, length],
):
    var v0 = Point3[DType.float32, frame, length](
        v0_scalar.x, v0_scalar.y, v0_scalar.z
    )
    var e1 = Vec3[.float32, frame, length](
        e1_scalar.x, e1_scalar.y, e1_scalar.z
    )
    var e2 = Vec3[.float32, frame, length](
        e2_scalar.x, e2_scalar.y, e2_scalar.z
    )
    var candidate = intersect_ray_tri_edges_scaled(
        rays.o,
        rays.d,
        v0,
        e1,
        e2,
        packet_hit.t,
        rays.t_min,
    )
    var closer = active & candidate.mask
    if closer.reduce_or():
        var safe_det = closer.select(candidate.abs_det, Float32(1.0))
        var inv_det = Float32(1.0) / safe_det
        var candidate_t = candidate.t_scaled * inv_det
        packet_hit.t = closer.select(candidate_t, packet_hit.t)
        packet_hit.u = closer.select(candidate.u_scaled * inv_det, packet_hit.u)
        packet_hit.v = closer.select(candidate.v_scaled * inv_det, packet_hit.v)
        packet_hit.prim = closer.select(
            SIMD[DType.uint32, length](prim_idx), packet_hit.prim
        )
        packet_hit.inst = closer.select(
            SIMD[DType.uint32, length](EMPTY_LANE), packet_hit.inst
        )
        var geometric_normal = normalize(cross(e1_scalar, e2_scalar))
        packet_hit.normal.x = closer.select(
            geometric_normal.x, packet_hit.normal.x
        )
        packet_hit.normal.y = closer.select(
            geometric_normal.y, packet_hit.normal.y
        )
        packet_hit.normal.z = closer.select(
            geometric_normal.z, packet_hit.normal.z
        )


@always_inline
def _trace_triangle_packet_policy[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    common_octant_fma: Bool,
    packed_meta: Bool,
    LeafFn: def(SIMD[.bool, length], UInt32, mut Hit[frame, length]),
    HybridFn: def(SIMD[.bool, length], UInt32, mut Hit[frame, length]),
    PrefetchFn: def(UInt32),
](
    nodes: ImmSpan[WideBvhNode[frame, bounds_width], _],
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length],
    ref leaf_fn: LeafFn,
    ref hybrid_fn: HybridFn,
    ref prefetch_fn: PrefetchFn,
) -> Hit[frame, length]:
    """Apply the production packet policy to typed or packed node spans."""
    var hit = Hit[frame, length].miss(rays.t_max)

    @always_inline
    def run_packet[
        use_octant_fma: Bool,
        positive_x: Bool = True,
        positive_y: Bool = True,
        positive_z: Bool = True,
    ]() {imm, mut hit}:
        @always_inline
        def run_kernel[
            hybrid_threshold: Int,
            root_scalar_max_tasks: Int,
            use_frustum: Bool = False,
            prefetch_tasks: Bool = False,
        ]() {imm, mut hit}:
            trace_packet_stack_bounds_bvh[
                frame=frame,
                bounds_width=bounds_width,
                length=length,
                hybrid_threshold=hybrid_threshold,
                root_scalar_max_tasks=root_scalar_max_tasks,
                common_octant_fma=use_octant_fma,
                positive_x=positive_x,
                positive_y=positive_y,
                positive_z=positive_z,
                hybrid_leaves=common_octant_fma,
                coherent_frustum=use_frustum,
                prefetch_tasks=prefetch_tasks,
                packed_meta=packed_meta,
            ](
                nodes,
                rays,
                valid,
                hit,
                leaf_fn,
                hybrid_fn,
                prefetch_fn,
            )

        comptime if bounds_width == 16 and leaf_width == 16:
            if len(nodes) >= HYBRID_TRIANGLE_MIN_NODES:
                comptime if length == 4:
                    if len(nodes) >= ROOT_SCALAR_TRIANGLE_MIN_NODES:
                        run_kernel[3, 4]()
                    else:
                        run_kernel[3, 0]()
                    return
                elif length == 8:
                    comptime if use_octant_fma:
                        if len(nodes) >= ROOT_SCALAR_TRIANGLE_MIN_NODES:
                            run_kernel[7, 0, True, True]()
                        else:
                            run_kernel[7, 0]()
                    else:
                        run_kernel[7, 0]()
                    return
                elif length == 16:
                    comptime if use_octant_fma:
                        if len(nodes) >= ROOT_SCALAR_TRIANGLE_MIN_NODES:
                            run_kernel[8, 0, True, True]()
                        else:
                            run_kernel[8, 0]()
                    else:
                        run_kernel[8, 0]()
                    return
        run_kernel[0, 0]()

    comptime if common_octant_fma:
        debug_assert["safe", _use_compiler_assume=True](
            valid[0], "common-octant packet traversal requires lane zero"
        )
        var positive_x = rays.d.x[0] >= 0.0
        var positive_y = rays.d.y[0] >= 0.0
        var positive_z = rays.d.z[0] >= 0.0
        if positive_x:
            if positive_y:
                if positive_z:
                    run_packet[True, True, True, True]()
                else:
                    run_packet[True, True, True, False]()
            else:
                if positive_z:
                    run_packet[True, True, False, True]()
                else:
                    run_packet[True, True, False, False]()
        else:
            if positive_y:
                if positive_z:
                    run_packet[True, False, True, True]()
                else:
                    run_packet[True, False, True, False]()
            else:
                if positive_z:
                    run_packet[True, False, False, True]()
                else:
                    run_packet[True, False, False, False]()
        return hit

    run_packet[False]()
    return hit


struct _TriangleBuild[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength = bounds_width,
](Copyable):
    """Private typed build result consumed immediately by `CpuBlasSet` packing.

    Binary-to-wide collapse packs triangle leaves in the same pass. Tagged
    leaf references therefore point directly at TriangleLeafBlock entries:

        node.data[lane] == EMPTY_LANE -> unused lane
        is_leaf_ref(node.data[lane])  -> TriangleLeafBlock index
        otherwise                     -> internal node index
    """

    var tree: BoundsBvh[Self.frame, Self.bounds_width]
    var leaf_blocks: List[TriangleLeafBlock[Self.frame, Self.leaf_width]]
    var tri_count: Int

    def __init__[
        method: CpuBvhBuildMethod = .MEDIAN
    ](out self, vertices: ImmSpan[Point3f32[Self.frame], _]):
        self.tri_count = len(vertices) / 3
        self.leaf_blocks = List[
            TriangleLeafBlock[Self.frame, Self.leaf_width]
        ]()
        var tri_count = self.tri_count

        var items = List[BoundsItem[Self.frame]](capacity=tri_count)
        var root_bounds = AABB[Self.frame].invalid()
        var centroid_bounds = AABB[Self.frame].invalid()

        def append_items(
            mut items: List[BoundsItem[Self.frame]],
            mut root_bounds: AABB[Self.frame],
            mut centroid_bounds: AABB[Self.frame],
        ) {imm}:
            for i in range(tri_count):
                var item = _make_triangle_bounds_item(vertices, i)
                comptime if (
                    method != .LBVH
                    and method != .HPLOC
                ):
                    root_bounds.grow(item.bounds)
                comptime if method != .MEDIAN:
                    centroid_bounds.grow(item.bounds.centroid())
                items.append(item)

        if tri_count >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS:
            items.resize(unsafe_uninit_length=tri_count)
            var worker_count = _worker_count(tri_count)
            var root_partials = List[AABB[Self.frame]]()
            var centroid_partials = List[AABB[Self.frame]]()
            comptime if (
                method != .LBVH
                and method != .HPLOC
            ):
                root_partials = List[AABB[Self.frame]](capacity=worker_count)
                root_partials.resize(unsafe_uninit_length=worker_count)
            comptime if method != .MEDIAN:
                centroid_partials = List[AABB[Self.frame]](
                    capacity=worker_count
                )
                centroid_partials.resize(unsafe_uninit_length=worker_count)

            def item_chunk_worker(
                task_idx: Int,
            ) {imm, mut items, mut root_partials, mut centroid_partials}:
                var first = tri_count * task_idx // worker_count
                var end = tri_count * (task_idx + 1) // worker_count
                var chunk_bounds = AABB[Self.frame].invalid()
                var chunk_centroid_bounds = AABB[Self.frame].invalid()
                for i in range(first, end):
                    var item = _make_triangle_bounds_item(vertices, i)
                    comptime if (
                        method != .LBVH
                        and method != .HPLOC
                    ):
                        chunk_bounds.grow(item.bounds)
                    comptime if method != .MEDIAN:
                        chunk_centroid_bounds.grow(item.bounds.centroid())
                    items[i] = item
                comptime if (
                    method != .LBVH
                    and method != .HPLOC
                ):
                    root_partials[task_idx] = chunk_bounds
                comptime if method != .MEDIAN:
                    centroid_partials[task_idx] = chunk_centroid_bounds

            parallelize(item_chunk_worker, worker_count, worker_count)
            for worker_idx in range(worker_count):
                comptime if (
                    method != .LBVH
                    and method != .HPLOC
                ):
                    root_bounds.grow(root_partials[worker_idx])
                comptime if method != .MEDIAN:
                    centroid_bounds.grow(centroid_partials[worker_idx])
        else:
            append_items(items, root_bounds, centroid_bounds)

        var builder = BinaryBoundsBvh[Self.frame, Int(Self.leaf_width), method](
            items^, root_bounds, centroid_bounds
        )

        var leaf_blocks = List[TriangleLeafBlock[Self.frame, Self.leaf_width]](
            capacity=(Int(builder.nodes_used) + 1) // 2
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(vertices) == self.tri_count * 3,
            "triangle vertex count changed while packing leaves",
        )

        @always_inline
        def fill_leaf_block(
            first_item: UInt32,
            item_count: UInt32,
            mut block: TriangleLeafBlock[Self.frame, Self.leaf_width],
        ) {imm}:
            var first, count = _checked_typed_leaf_range[Self.leaf_width](
                first_item, item_count, len(builder.item_indices)
            )

            for k in range(count):
                var item_ref = Int(builder.item_indices.unsafe_get(first + k))
                # Typed items are created in primitive order, so the builder
                # item index is already the final primitive payload.
                var prim_idx = UInt32(item_ref)
                var base = Int(prim_idx) * 3

                ref p0 = vertices[base + 0]
                ref p1 = vertices[base + 1]
                ref p2 = vertices[base + 2]

                block.v0.x[k] = p0.x
                block.v0.y[k] = p0.y
                block.v0.z[k] = p0.z

                block.e1.x[k] = p1.x - p0.x
                block.e1.y[k] = p1.y - p0.y
                block.e1.z[k] = p1.z - p0.z

                block.e2.x[k] = p2.x - p0.x
                block.e2.y[k] = p2.y - p0.y
                block.e2.z[k] = p2.z - p0.z

                block.prim_indices[k] = prim_idx

        @always_inline
        def pack_leaf(
            first_item: UInt32, item_count: UInt32
        ) {imm, mut leaf_blocks} -> UInt32:
            var block_idx = UInt32(len(leaf_blocks))
            var block = TriangleLeafBlock[Self.frame, Self.leaf_width]()
            fill_leaf_block(first_item, item_count, block)
            leaf_blocks.append(block^)
            return block_idx

        if self.tri_count >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS:
            var leaf_ranges = List[WideLeafRange]()

            @always_inline
            def record_leaf_range(
                first_item: UInt32, item_count: UInt32
            ) {imm, mut leaf_ranges} -> UInt32:
                var block_idx = UInt32(len(leaf_ranges))
                leaf_ranges.append(WideLeafRange(first_item, item_count))
                return block_idx

            self.tree = BoundsBvh[Self.frame, Self.bounds_width](
                builder, record_leaf_range
            )
            leaf_blocks.resize(unsafe_uninit_length=len(leaf_ranges))

            var task_count = len(leaf_ranges)
            var worker_count = _worker_count(task_count)

            def pack_worker(block_idx: Int) {imm, mut leaf_blocks}:
                ref leaf_range = leaf_ranges[block_idx]
                var block = TriangleLeafBlock[Self.frame, Self.leaf_width]()
                fill_leaf_block(
                    leaf_range.first_item, leaf_range.item_count, block
                )
                leaf_blocks[block_idx] = block^

            parallelize(pack_worker, task_count, worker_count)
        else:
            self.tree = BoundsBvh[Self.frame, Self.bounds_width](
                builder, pack_leaf
            )

        self.leaf_blocks = leaf_blocks^
