from max.algorithm import parallelize
from std.bit import count_trailing_zeros
from std.memory import pack_bits
from std.sys import size_of
from std.sys.intrinsics import prefetch

from bajo.core import (
    GeoKind,
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
from bajo.bvh.constants import EMPTY_LANE, TRACE
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BinaryBoundsBvh,
    WideLeafRange,
    _checked_typed_leaf_range,
)
from bajo.bvh.types import Hit, TriangleLeafBlock, TypedBvh
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.core.intersect import (
    intersect_ray_tri_edges,
    intersect_ray_tri_edges_scaled,
)
from bajo.bvh.cpu.trace import (
    CpuBvhTraversalStats,
    _count_true_lanes,
    _extract_f32_lane,
    _extract_u32_lane,
    trace_bounds_bvh,
    trace_bounds_bvh_from_ref,
    trace_bounds_bvh_measured,
)
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


struct TriangleBvh[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength = bounds_width,
](Copyable, TypedBvh):
    comptime bvh_frame: Frame = Self.frame

    """Triangle BVH with independent bounds and triangle packet widths.

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
        split_method: String = "median"
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
                comptime if split_method != "lbvh" and split_method != "hploc":
                    root_bounds.grow(item.bounds)
                comptime if split_method != "median":
                    centroid_bounds.grow(item.bounds.centroid())
                items.append(item)

        if tri_count >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS:
            items.resize(unsafe_uninit_length=tri_count)
            var worker_count = _worker_count(tri_count)
            var root_partials = List[AABB[Self.frame]]()
            var centroid_partials = List[AABB[Self.frame]]()
            comptime if split_method != "lbvh" and split_method != "hploc":
                root_partials = List[AABB[Self.frame]](capacity=worker_count)
                root_partials.resize(unsafe_uninit_length=worker_count)
            comptime if split_method != "median":
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
                    comptime if split_method != "lbvh" and split_method != "hploc":
                        chunk_bounds.grow(item.bounds)
                    comptime if split_method != "median":
                        chunk_centroid_bounds.grow(item.bounds.centroid())
                    items[i] = item
                comptime if split_method != "lbvh" and split_method != "hploc":
                    root_partials[task_idx] = chunk_bounds
                comptime if split_method != "median":
                    centroid_partials[task_idx] = chunk_centroid_bounds

            parallelize(item_chunk_worker, worker_count, worker_count)
            for worker_idx in range(worker_count):
                comptime if split_method != "lbvh" and split_method != "hploc":
                    root_bounds.grow(root_partials[worker_idx])
                comptime if split_method != "median":
                    centroid_bounds.grow(centroid_partials[worker_idx])
        else:
            append_items(items, root_bounds, centroid_bounds)

        var builder = BinaryBoundsBvh[
            Self.frame, Int(Self.leaf_width), split_method
        ](items^, root_bounds, centroid_bounds)

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

    def bounds(self) -> AABB[Self.frame]:
        return self.tree.root_bounds()

    @always_inline
    def trace[
        mode: TRACE, length: SIMDLength
    ](
        self,
        rays: Ray[DType.float32, Self.bvh_frame, length],
        valid: SIMD[DType.bool, length] = SIMD[DType.bool, length](fill=True),
    ) -> Hit[Self.bvh_frame, length]:
        comptime if length == 1:
            if not valid[0]:
                return Hit[Self.bvh_frame, length].miss(rays.t_max)
            var ray = Rayf32[Self.bvh_frame](
                Point3f32[Self.bvh_frame](
                    rays.o.x[0], rays.o.y[0], rays.o.z[0]
                ),
                Vec3[DType.float32, Self.bvh_frame](
                    rays.d.x[0], rays.d.y[0], rays.d.z[0]
                ),
                rays.t_min[0],
                rays.t_max[0],
            )
            var scalar_hit = self._trace_ordered[mode](ray)
            var result = Hit[Self.bvh_frame, length].miss(rays.t_max)
            result.u[0] = scalar_hit.u[0]
            result.v[0] = scalar_hit.v[0]
            result.prim[0] = scalar_hit.prim[0]
            result.inst[0] = scalar_hit.inst[0]
            result.normal.x[0] = scalar_hit.normal.x[0]
            result.normal.y[0] = scalar_hit.normal.y[0]
            result.normal.z[0] = scalar_hit.normal.z[0]
            result.t[0] = scalar_hit.t[0]
            return result
        else:
            comptime assert mode == TRACE.CLOSEST_HIT
            return self._trace_shared_stack[False](rays, valid)

    @always_inline
    def _trace_packet_scalar[
        length: SIMDLength
    ](
        self,
        rays: Ray[DType.float32, Self.bvh_frame, length],
        valid: SIMD[DType.bool, length],
    ) -> Hit[Self.bvh_frame, length]:
        comptime assert length > 1
        var result = Hit[Self.bvh_frame, length].miss(rays.t_max)
        comptime for lane in range(length):
            if valid[lane]:
                var ray = Rayf32[Self.bvh_frame](
                    [rays.o.x[lane], rays.o.y[lane], rays.o.z[lane]],
                    [rays.d.x[lane], rays.d.y[lane], rays.d.z[lane]],
                    rays.t_min[lane],
                    rays.t_max[lane],
                )
                var lane_hit = self._trace_ordered[TRACE.CLOSEST_HIT](ray)
                result.u[lane] = lane_hit.u[0]
                result.v[lane] = lane_hit.v[0]
                result.prim[lane] = lane_hit.prim[0]
                result.inst[lane] = lane_hit.inst[0]
                result.normal.x[lane] = lane_hit.normal.x[0]
                result.normal.y[lane] = lane_hit.normal.y[0]
                result.normal.z[lane] = lane_hit.normal.z[0]
                result.t[lane] = lane_hit.t[0]
        return result

    @always_inline
    def trace_packet_common_octant[
        length: SIMDLength
    ](
        self,
        rays: Ray[DType.float32, Self.bvh_frame, length],
        valid: SIMD[DType.bool, length],
    ) -> Hit[Self.bvh_frame, length]:
        """Fast path requiring valid rays to share lane zero's octant."""
        comptime assert length > 1
        return self._trace_shared_stack[True](rays, valid)

    @always_inline
    def _trace_ordered[
        mode: TRACE,
    ](self, ray: Rayf32[Self.bvh_frame]) -> Hit[Self.bvh_frame]:
        var unused_stats = CpuBvhTraversalStats()
        return self._trace[mode, collect_stats=False](ray, unused_stats)

    @always_inline
    def _trace_shared_stack[
        common_octant_fma: Bool,
        length: SIMDLength,
    ](
        self,
        rays: Ray[DType.float32, Self.bvh_frame, length],
        valid: SIMD[DType.bool, length],
    ) -> Hit[Self.bvh_frame, length]:
        """Trace a coherent SIMD ray packet with one shared hierarchy stack."""
        var hit = Hit[Self.bvh_frame, length].miss(rays.t_max)

        def leaf_fn(
            active: SIMD[DType.bool, length],
            leaf_block_idx: UInt32,
            mut packet_hit: Hit[Self.bvh_frame, length],
        ) {imm}:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            comptime for prim_lane in range(Self.leaf_width):
                var prim_idx = block.prim_indices[prim_lane]
                if prim_idx != EMPTY_LANE:
                    var v0 = Point3[DType.float32, Self.bvh_frame, length](
                        block.v0.x[prim_lane],
                        block.v0.y[prim_lane],
                        block.v0.z[prim_lane],
                    )
                    var e1 = Vec3[DType.float32, Self.bvh_frame, length](
                        block.e1.x[prim_lane],
                        block.e1.y[prim_lane],
                        block.e1.z[prim_lane],
                    )
                    var e2 = Vec3[DType.float32, Self.bvh_frame, length](
                        block.e2.x[prim_lane],
                        block.e2.y[prim_lane],
                        block.e2.z[prim_lane],
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
                        var safe_det = closer.select(
                            candidate.abs_det, Float32(1.0)
                        )
                        var inv_det = Float32(1.0) / safe_det
                        var candidate_t = candidate.t_scaled * inv_det
                        packet_hit.t = closer.select(candidate_t, packet_hit.t)
                        packet_hit.u = closer.select(
                            candidate.u_scaled * inv_det, packet_hit.u
                        )
                        packet_hit.v = closer.select(
                            candidate.v_scaled * inv_det, packet_hit.v
                        )
                        packet_hit.prim = closer.select(
                            SIMD[DType.uint32, length](prim_idx),
                            packet_hit.prim,
                        )
                        packet_hit.inst = closer.select(
                            SIMD[DType.uint32, length](EMPTY_LANE),
                            packet_hit.inst,
                        )
                        var geometric_normal = normalize(cross(e1, e2))
                        packet_hit.normal.x = closer.select(
                            geometric_normal.x, packet_hit.normal.x
                        )
                        packet_hit.normal.y = closer.select(
                            geometric_normal.y, packet_hit.normal.y
                        )
                        packet_hit.normal.z = closer.select(
                            geometric_normal.z, packet_hit.normal.z
                        )

        var unused_stats = CpuBvhTraversalStats()

        def hybrid_fn(
            active: SIMD[DType.bool, length],
            child_ref: UInt32,
            mut packet_hit: Hit[Self.bvh_frame, length],
        ) {imm, mut unused_stats}:
            comptime if length == 4:
                if child_ref == 0:
                    packet_hit = self._trace_packet_scalar(rays, active)
                    return
            var bits = UInt32(pack_bits(active))
            while bits != 0:
                var lane = Int(count_trailing_zeros(bits))
                bits &= bits - 1
                var ray = Rayf32[Self.bvh_frame](
                    Point3f32[Self.bvh_frame](
                        _extract_f32_lane(rays.o.x, lane),
                        _extract_f32_lane(rays.o.y, lane),
                        _extract_f32_lane(rays.o.z, lane),
                    ),
                    Vec3f32[Self.bvh_frame](
                        _extract_f32_lane(rays.d.x, lane),
                        _extract_f32_lane(rays.d.y, lane),
                        _extract_f32_lane(rays.d.z, lane),
                    ),
                    _extract_f32_lane(rays.t_min, lane),
                    _extract_f32_lane(rays.t_max, lane),
                )
                var initial_hit = Hit[Self.bvh_frame](
                    _extract_f32_lane(packet_hit.u, lane),
                    _extract_f32_lane(packet_hit.v, lane),
                    _extract_u32_lane(packet_hit.prim, lane),
                    _extract_u32_lane(packet_hit.inst, lane),
                    Normal3f32[Self.bvh_frame](
                        _extract_f32_lane(packet_hit.normal.x, lane),
                        _extract_f32_lane(packet_hit.normal.y, lane),
                        _extract_f32_lane(packet_hit.normal.z, lane),
                    ),
                    _extract_f32_lane(packet_hit.t, lane),
                )
                var scalar_hit = self._trace_from_ref[
                    TRACE.CLOSEST_HIT,
                    False,
                ](ray, child_ref, initial_hit, unused_stats)
                packet_hit.u[lane] = scalar_hit.u[0]
                packet_hit.v[lane] = scalar_hit.v[0]
                packet_hit.prim[lane] = scalar_hit.prim[0]
                packet_hit.inst[lane] = scalar_hit.inst[0]
                packet_hit.normal.x[lane] = scalar_hit.normal.x[0]
                packet_hit.normal.y[lane] = scalar_hit.normal.y[0]
                packet_hit.normal.z[lane] = scalar_hit.normal.z[0]
                packet_hit.t[lane] = scalar_hit.t[0]

        @always_inline
        def prefetch_fn(child_ref: UInt32) {imm}:
            if is_leaf_ref(child_ref):
                var leaf_ptr = self.leaf_blocks.unsafe_ptr().unsafe_offset(
                    Int(decode_ref_index(child_ref))
                )
                prefetch(leaf_ptr.unsafe_bitcast[UInt8]())
            else:
                var node_ptr = self.tree.nodes.unsafe_ptr().unsafe_offset(
                    Int(child_ref)
                )
                prefetch(node_ptr.unsafe_bitcast[UInt8]())

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
                    frame=Self.frame,
                    bounds_width=Self.bounds_width,
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
                ](
                    self.tree,
                    rays,
                    valid,
                    hit,
                    leaf_fn,
                    hybrid_fn,
                    prefetch_fn,
                )

            comptime if Self.bounds_width == 16 and Self.leaf_width == 16:
                if len(self.tree.nodes) >= HYBRID_TRIANGLE_MIN_NODES:
                    comptime if length == 4:
                        if (
                            len(self.tree.nodes)
                            >= ROOT_SCALAR_TRIANGLE_MIN_NODES
                        ):
                            run_kernel[3, 4]()
                        else:
                            run_kernel[3, 0]()
                        return
                    elif length == 8:
                        comptime if use_octant_fma:
                            if (
                                len(self.tree.nodes)
                                >= ROOT_SCALAR_TRIANGLE_MIN_NODES
                            ):
                                run_kernel[7, 0, True, True]()
                            else:
                                run_kernel[7, 0]()
                        else:
                            run_kernel[7, 0]()
                        return
                    elif length == 16:
                        comptime if use_octant_fma:
                            if (
                                len(self.tree.nodes)
                                >= ROOT_SCALAR_TRIANGLE_MIN_NODES
                            ):
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

    def trace_with_stats[
        mode: TRACE
    ](
        self,
        ray: Rayf32[Self.bvh_frame],
        mut stats: CpuBvhTraversalStats,
    ) -> Hit[Self.bvh_frame]:
        """Trace one ray and accumulate traversal diagnostics in `stats`."""
        return self._trace[mode, collect_stats=True](ray, stats)

    def _trace[
        mode: TRACE,
        collect_stats: Bool,
    ](
        self,
        ray: Rayf32[Self.bvh_frame],
        mut stats: CpuBvhTraversalStats,
    ) -> Hit[Self.bvh_frame]:
        return self._trace_from_ref[
            mode,
            collect_stats,
        ](
            ray,
            UInt32(0),
            Hit[Self.bvh_frame].miss(ray.t_max),
            stats,
        )

    def _trace_from_ref[
        mode: TRACE,
        collect_stats: Bool,
    ](
        self,
        ray: Rayf32[Self.bvh_frame],
        initial_ref: UInt32,
        initial_hit: Hit[Self.bvh_frame],
        mut stats: CpuBvhTraversalStats,
    ) -> Hit[Self.bvh_frame]:
        # Zero-filled unused triangles are degenerate. Omitting their explicit
        # validity load is benchmark-positive only for full-width packets.
        comptime omit_leaf_validity_mask = Self.leaf_width == 16

        def leaf_fn(
            ray: Rayf32[Self.bvh_frame],
            O: Point3[DType.float32, Self.bvh_frame, Self.leaf_width],
            D: Vec3[DType.float32, Self.bvh_frame, Self.leaf_width],
            _ray_a: SIMD[DType.float32, Self.leaf_width],
            _ray_inv_a: SIMD[DType.float32, Self.leaf_width],
            leaf_block_idx: UInt32,
            mut leaf_stats: CpuBvhTraversalStats,
            mut hit: Hit[Self.bvh_frame],
        ) {imm} -> Bool:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            comptime if collect_stats:
                leaf_stats.primitive_packet_lanes += Int(Self.leaf_width)
                leaf_stats.valid_primitives += _count_true_lanes(
                    block.prim_indices.ne(EMPTY_LANE)
                )

            comptime if mode == TRACE.ANY_HIT:
                var tri_hit = intersect_ray_tri_edges(
                    O,
                    D,
                    block.v0,
                    block.e1,
                    block.e2,
                    hit.t,
                    ray.t_min,
                )
                comptime if collect_stats:
                    leaf_stats.primitive_hit_candidates += _count_true_lanes(
                        tri_hit.mask & block.prim_indices.ne(EMPTY_LANE)
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

            comptime if collect_stats:
                leaf_stats.primitive_hit_candidates += _count_true_lanes(
                    hit_mask & block.prim_indices.ne(EMPTY_LANE)
                )

            if not hit_mask.reduce_or():
                return False

            comptime if mode == TRACE.CLOSEST_HIT:
                comptime if collect_stats:
                    leaf_stats.closer_hit_updates += 1
                # Compare t_scaled / abs_det ratios without division, then
                # calculate one reciprocal for the winning scalar lane.
                var bits = pack_bits(hit_mask)
                var lane = Int(count_trailing_zeros(bits))
                bits &= bits - 1

                comptime if Self.leaf_width == 16:
                    comptime assert size_of[
                        TriangleLeafBlock[Self.bvh_frame, Self.leaf_width]
                    ]() == 10 * 4 * Int(Self.leaf_width)
                    var best_t_scaled = _extract_f32_lane(
                        scaled_hit.t_scaled, lane
                    )
                    var best_abs_det = _extract_f32_lane(
                        scaled_hit.abs_det, lane
                    )

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
                    hit.u = (
                        _extract_f32_lane(scaled_hit.u_scaled, lane) * inv_det
                    )
                    hit.v = (
                        _extract_f32_lane(scaled_hit.v_scaled, lane) * inv_det
                    )

                    # Triangle packet fields already live in memory. Load the
                    # selected scalars directly rather than first loading and
                    # spilling their complete SIMD vectors.
                    var block_ptr = (
                        self.leaf_blocks.unsafe_ptr()
                        .unsafe_offset(Int(leaf_block_idx))
                        .unsafe_bitcast[Float32]()
                    )
                    var block_u32 = block_ptr.unsafe_bitcast[UInt32]()
                    var lane_offset = lane
                    hit.prim = block_u32[
                        unsafe_offset=9 * Int(Self.leaf_width) + lane_offset
                    ]
                    hit.inst = EMPTY_LANE
                    var e1 = Vec3f32[Self.bvh_frame](
                        block_ptr[
                            unsafe_offset=3 * Int(Self.leaf_width) + lane_offset
                        ],
                        block_ptr[
                            unsafe_offset=4 * Int(Self.leaf_width) + lane_offset
                        ],
                        block_ptr[
                            unsafe_offset=5 * Int(Self.leaf_width) + lane_offset
                        ],
                    )
                    var e2 = Vec3f32[Self.bvh_frame](
                        block_ptr[
                            unsafe_offset=6 * Int(Self.leaf_width) + lane_offset
                        ],
                        block_ptr[
                            unsafe_offset=7 * Int(Self.leaf_width) + lane_offset
                        ],
                        block_ptr[
                            unsafe_offset=8 * Int(Self.leaf_width) + lane_offset
                        ],
                    )
                    var geometric_normal = cross(e1, e2)
                    hit.normal = Normal3f32[Self.bvh_frame](
                        geometric_normal.x,
                        geometric_normal.y,
                        geometric_normal.z,
                    )
                    return True

                while bits != 0:
                    var candidate = Int(count_trailing_zeros(bits))
                    bits &= bits - 1

                    if (
                        scaled_hit.t_scaled[candidate]
                        * scaled_hit.abs_det[lane]
                        < scaled_hit.t_scaled[lane]
                        * scaled_hit.abs_det[candidate]
                    ):
                        lane = candidate

                var inv_det = 1.0 / scaled_hit.abs_det[lane]

                hit.t = scaled_hit.t_scaled[lane] * inv_det
                hit.u = scaled_hit.u_scaled[lane] * inv_det
                hit.v = scaled_hit.v_scaled[lane] * inv_det
                hit.prim = block.prim_indices[lane]
                hit.inst = EMPTY_LANE

                var e1 = Vec3f32[Self.bvh_frame](
                    block.e1.x[lane],
                    block.e1.y[lane],
                    block.e1.z[lane],
                )
                var e2 = Vec3f32[Self.bvh_frame](
                    block.e2.x[lane],
                    block.e2.y[lane],
                    block.e2.z[lane],
                )

                var geometric_normal = cross(e1, e2)

                hit.normal = Normal3f32[Self.bvh_frame](
                    geometric_normal.x,
                    geometric_normal.y,
                    geometric_normal.z,
                )

            return True

        @always_inline
        def unmeasured_leaf_fn(
            ray: Rayf32[Self.bvh_frame],
            O: Point3[DType.float32, Self.bvh_frame, Self.leaf_width],
            D: Vec3[DType.float32, Self.bvh_frame, Self.leaf_width],
            ray_a: SIMD[DType.float32, Self.leaf_width],
            ray_inv_a: SIMD[DType.float32, Self.leaf_width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Self.bvh_frame],
        ) {imm, mut stats} -> Bool:
            return leaf_fn(
                ray,
                O,
                D,
                ray_a,
                ray_inv_a,
                leaf_block_idx,
                stats,
                hit,
            )

        var hit: Hit[Self.bvh_frame]
        comptime if collect_stats:
            debug_assert["safe", _use_compiler_assume=True](
                initial_ref == 0 and initial_hit.prim == EMPTY_LANE,
                "measured traversal must start at the root with a miss",
            )
            hit = trace_bounds_bvh_measured[
                frame=Self.frame,
                bounds_width=Self.bounds_width,
                leaf_width=Self.leaf_width,
                mode=mode,
            ](self.tree, ray, stats, leaf_fn)
        else:
            comptime if mode == TRACE.CLOSEST_HIT:
                hit = trace_bounds_bvh_from_ref[
                    frame=Self.frame,
                    bounds_width=Self.bounds_width,
                    leaf_width=Self.leaf_width,
                    single_child_fast_path=True,
                    terminal_mask_fast_path=True,
                ](
                    self.tree,
                    ray,
                    initial_ref,
                    initial_hit,
                    unmeasured_leaf_fn,
                )
            else:
                debug_assert["safe", _use_compiler_assume=True](
                    initial_ref == 0 and initial_hit.prim == EMPTY_LANE,
                    "any-hit traversal must start at the root with a miss",
                )
                hit = trace_bounds_bvh[
                    frame=Self.frame,
                    bounds_width=Self.bounds_width,
                    leaf_width=Self.leaf_width,
                    mode=mode,
                ](self.tree, ray, unmeasured_leaf_fn)

        comptime if mode == TRACE.CLOSEST_HIT:
            if hit.is_hit() and (
                hit.prim[0] != initial_hit.prim[0]
                or hit.t[0] != initial_hit.t[0]
            ):
                var geometric_normal = Vec3f32[Self.bvh_frame](
                    hit.normal.x,
                    hit.normal.y,
                    hit.normal.z,
                )
                var unit_normal = normalize(geometric_normal)
                hit.normal = Normal3f32[Self.bvh_frame](
                    unit_normal.x,
                    unit_normal.y,
                    unit_normal.z,
                )

        return hit
