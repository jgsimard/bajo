"""Diagnostic sweep of alternative CPU BVH leaf layouts."""

from std.bit import count_trailing_zeros
from std.math import abs, round
from std.memory import pack_bits
from std.sys import size_of
from std.time import perf_counter_ns

from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.trace import (
    CpuBvhTraversalStats,
    _count_true_lanes,
    _extract_f32_lane,
    trace_bounds_bvh,
)
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit, TriangleLeafBlock
from bajo.core import (
    Frame,
    Normal3f32,
    Point3,
    Point3f32,
    Rayf32,
    Vec3,
    Vec3f32,
    cross,
    normalize,
)
from bajo.core.intersect import (
    RayTriScaledHit,
    intersect_ray_tri_edges,
    intersect_ray_tri_edges_scaled,
)
from bajo.core.utils import ns_to_mrays_per_s
from bajo.obj.pack import pack_obj_triangles
from bench.bvh.fixtures import (
    make_camera_rays_and_params,
    make_grid_triangles,
    make_hit_and_miss_rays,
    permute_rays,
    select_and_repeat_hit_rays,
)
from bench.timing import ratio


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime RAY_WIDTH = 512
comptime RAY_HEIGHT = 288
comptime TIMING_REPEATS = 4
comptime WIDTH = 16

comptime PACKED = 0
comptime SPLIT_HOT_COLD = 1
comptime DENSE_SOA = 2
comptime HALF_EDGES = 3


@fieldwise_init
struct TimingResult(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


@fieldwise_init
struct TriangleGeometryBlock(Copyable):
    """Nine hot Float32 vectors, without the cold primitive-id vector."""

    var v0: Point3[DType.float32, Frame.WORLD, WIDTH]
    var e1: Vec3[DType.float32, Frame.WORLD, WIDTH]
    var e2: Vec3[DType.float32, Frame.WORLD, WIDTH]


struct SplitTriangleLeaves(Copyable):
    """Block-padded geometry and IDs in independent streams."""

    var geometry: List[TriangleGeometryBlock]
    var prim_indices: List[SIMD[DType.uint32, WIDTH]]

    def __init__(out self, bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH]):
        self.geometry = List[TriangleGeometryBlock](
            capacity=len(bvh.leaf_blocks)
        )
        self.prim_indices = List[SIMD[DType.uint32, WIDTH]](
            capacity=len(bvh.leaf_blocks)
        )
        for ref block in bvh.leaf_blocks:
            self.geometry.append(
                TriangleGeometryBlock(
                    block.v0.copy(), block.e1.copy(), block.e2.copy()
                )
            )
            self.prim_indices.append(block.prim_indices.copy())

    def bytes(self) -> Int:
        return (
            len(self.geometry) * size_of[TriangleGeometryBlock]()
            + len(self.prim_indices) * size_of[SIMD[DType.uint32, WIDTH]]()
        )


@fieldwise_init
struct HalfEdgeTriangleBlock(Copyable):
    """Float32 anchor, Float16 edges, and UInt32 primitive IDs."""

    var v0: Point3[DType.float32, Frame.WORLD, WIDTH]
    var e1: Vec3[DType.float16, Frame.WORLD, WIDTH]
    var e2: Vec3[DType.float16, Frame.WORLD, WIDTH]
    var prim_indices: SIMD[DType.uint32, WIDTH]


struct HalfEdgeTriangleLeaves(Copyable):
    var blocks: List[HalfEdgeTriangleBlock]

    def __init__(out self, bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH]):
        self.blocks = List[HalfEdgeTriangleBlock](capacity=len(bvh.leaf_blocks))
        for ref block in bvh.leaf_blocks:
            self.blocks.append(
                HalfEdgeTriangleBlock(
                    block.v0.copy(),
                    Vec3[DType.float16, Frame.WORLD, WIDTH](
                        block.e1.x.cast[DType.float16](),
                        block.e1.y.cast[DType.float16](),
                        block.e1.z.cast[DType.float16](),
                    ),
                    Vec3[DType.float16, Frame.WORLD, WIDTH](
                        block.e2.x.cast[DType.float16](),
                        block.e2.y.cast[DType.float16](),
                        block.e2.z.cast[DType.float16](),
                    ),
                    block.prim_indices.copy(),
                )
            )

    def bytes(self) -> Int:
        return len(self.blocks) * size_of[HalfEdgeTriangleBlock]()


struct DenseTriangleLeaves(Copyable):
    """Globally dense SoA geometry with one offset/count per BVH leaf."""

    var v0x: List[Float32]
    var v0y: List[Float32]
    var v0z: List[Float32]
    var e1x: List[Float32]
    var e1y: List[Float32]
    var e1z: List[Float32]
    var e2x: List[Float32]
    var e2y: List[Float32]
    var e2z: List[Float32]
    var prim_indices: List[UInt32]
    var first: List[UInt32]
    var count: List[UInt32]
    var valid_primitives: Int

    def __init__(out self, bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH]):
        var capacity = bvh.tri_count + WIDTH
        self.v0x = List[Float32](capacity=capacity)
        self.v0y = List[Float32](capacity=capacity)
        self.v0z = List[Float32](capacity=capacity)
        self.e1x = List[Float32](capacity=capacity)
        self.e1y = List[Float32](capacity=capacity)
        self.e1z = List[Float32](capacity=capacity)
        self.e2x = List[Float32](capacity=capacity)
        self.e2y = List[Float32](capacity=capacity)
        self.e2z = List[Float32](capacity=capacity)
        self.prim_indices = List[UInt32](capacity=capacity)
        self.first = List[UInt32](capacity=len(bvh.leaf_blocks))
        self.count = List[UInt32](capacity=len(bvh.leaf_blocks))
        self.valid_primitives = 0

        for ref block in bvh.leaf_blocks:
            self.first.append(UInt32(len(self.v0x)))
            var leaf_count = 0
            comptime for lane in range(WIDTH):
                if block.prim_indices[lane] != EMPTY_LANE:
                    self.v0x.append(block.v0.x[lane])
                    self.v0y.append(block.v0.y[lane])
                    self.v0z.append(block.v0.z[lane])
                    self.e1x.append(block.e1.x[lane])
                    self.e1y.append(block.e1.y[lane])
                    self.e1z.append(block.e1.z[lane])
                    self.e2x.append(block.e2.x[lane])
                    self.e2y.append(block.e2.y[lane])
                    self.e2z.append(block.e2.z[lane])
                    self.prim_indices.append(block.prim_indices[lane])
                    leaf_count += 1
            self.count.append(UInt32(leaf_count))
            self.valid_primitives += leaf_count

        # Every dense leaf performs a full SIMD load. Tail padding makes the
        # final load valid; lanes beyond the leaf count are masked out.
        for _ in range(WIDTH - 1):
            self.v0x.append(0.0)
            self.v0y.append(0.0)
            self.v0z.append(0.0)
            self.e1x.append(0.0)
            self.e1y.append(0.0)
            self.e1z.append(0.0)
            self.e2x.append(0.0)
            self.e2y.append(0.0)
            self.e2z.append(0.0)
            self.prim_indices.append(EMPTY_LANE)

    def bytes(self) -> Int:
        return (
            9 * len(self.v0x) * size_of[Float32]()
            + len(self.prim_indices) * size_of[UInt32]()
            + len(self.first) * size_of[UInt32]()
            + len(self.count) * size_of[UInt32]()
        )


def _finish_hit(
    scaled_hit: RayTriScaledHit[DType.float32, WIDTH],
    hit_mask: SIMD[DType.bool, WIDTH],
    prim_indices: SIMD[DType.uint32, WIDTH],
    e1: Vec3[DType.float32, Frame.WORLD, WIDTH],
    e2: Vec3[DType.float32, Frame.WORLD, WIDTH],
    mut hit: Hit[Frame.WORLD],
) -> Bool:
    if not hit_mask.reduce_or():
        return False

    var bits = pack_bits(hit_mask)
    var lane = Int(count_trailing_zeros(bits))
    bits &= bits - 1
    var best_t_scaled = _extract_f32_lane(scaled_hit.t_scaled, lane)
    var best_abs_det = _extract_f32_lane(scaled_hit.abs_det, lane)

    while bits != 0:
        var candidate = Int(count_trailing_zeros(bits))
        bits &= bits - 1
        var candidate_t_scaled = _extract_f32_lane(
            scaled_hit.t_scaled, candidate
        )
        var candidate_abs_det = _extract_f32_lane(scaled_hit.abs_det, candidate)
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
    hit.prim = prim_indices[lane]
    hit.inst = EMPTY_LANE

    var selected_e1 = Vec3f32[Frame.WORLD](
        _extract_f32_lane(e1.x, lane),
        _extract_f32_lane(e1.y, lane),
        _extract_f32_lane(e1.z, lane),
    )
    var selected_e2 = Vec3f32[Frame.WORLD](
        _extract_f32_lane(e2.x, lane),
        _extract_f32_lane(e2.y, lane),
        _extract_f32_lane(e2.z, lane),
    )
    var geometric_normal = cross(selected_e1, selected_e2)
    hit.normal = Normal3f32[Frame.WORLD](
        geometric_normal.x, geometric_normal.y, geometric_normal.z
    )
    return True


def _normalize_hit(mut hit: Hit[Frame.WORLD]) -> Hit[Frame.WORLD]:
    if hit.is_hit():
        var geometric_normal = Vec3f32[Frame.WORLD](
            hit.normal.x, hit.normal.y, hit.normal.z
        )
        var unit_normal = normalize(geometric_normal)
        hit.normal = Normal3f32[Frame.WORLD](
            unit_normal.x, unit_normal.y, unit_normal.z
        )
    return hit


def trace_packed[
    mode: TRACE
](bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH], ray: Rayf32[Frame.WORLD]) -> Hit[
    Frame.WORLD
]:
    def leaf_fn(
        ray: Rayf32[Frame.WORLD],
        O: Point3[DType.float32, Frame.WORLD, WIDTH],
        D: Vec3[DType.float32, Frame.WORLD, WIDTH],
        _ray_a: SIMD[DType.float32, WIDTH],
        _ray_inv_a: SIMD[DType.float32, WIDTH],
        leaf_idx: UInt32,
        mut hit: Hit[Frame.WORLD],
    ) {imm} -> Bool:
        ref block = bvh.leaf_blocks.unsafe_get(Int(leaf_idx))
        comptime if mode == TRACE.ANY_HIT:
            var candidate = intersect_ray_tri_edges(
                O, D, block.v0, block.e1, block.e2, hit.t, ray.t_min
            )
            return candidate.mask.reduce_or()
        else:
            var candidate = intersect_ray_tri_edges_scaled(
                O, D, block.v0, block.e1, block.e2, hit.t, ray.t_min
            )
            return _finish_hit(
                candidate,
                candidate.mask,
                block.prim_indices,
                block.e1,
                block.e2,
                hit,
            )

    var hit = trace_bounds_bvh[
        frame=Frame.WORLD,
        bounds_width=WIDTH,
        leaf_width=WIDTH,
        mode=mode,
    ](bvh.tree, ray, leaf_fn)
    return _normalize_hit(hit)


def trace_split[
    mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    leaves: SplitTriangleLeaves,
    ray: Rayf32[Frame.WORLD],
) -> Hit[Frame.WORLD]:
    def leaf_fn(
        ray: Rayf32[Frame.WORLD],
        O: Point3[DType.float32, Frame.WORLD, WIDTH],
        D: Vec3[DType.float32, Frame.WORLD, WIDTH],
        _ray_a: SIMD[DType.float32, WIDTH],
        _ray_inv_a: SIMD[DType.float32, WIDTH],
        leaf_idx: UInt32,
        mut hit: Hit[Frame.WORLD],
    ) {imm} -> Bool:
        ref block = leaves.geometry.unsafe_get(Int(leaf_idx))
        comptime if mode == TRACE.ANY_HIT:
            var candidate = intersect_ray_tri_edges(
                O, D, block.v0, block.e1, block.e2, hit.t, ray.t_min
            )
            return candidate.mask.reduce_or()
        else:
            var candidate = intersect_ray_tri_edges_scaled(
                O, D, block.v0, block.e1, block.e2, hit.t, ray.t_min
            )
            return _finish_hit(
                candidate,
                candidate.mask,
                leaves.prim_indices.unsafe_get(Int(leaf_idx)),
                block.e1,
                block.e2,
                hit,
            )

    var hit = trace_bounds_bvh[
        frame=Frame.WORLD,
        bounds_width=WIDTH,
        leaf_width=WIDTH,
        mode=mode,
    ](bvh.tree, ray, leaf_fn)
    return _normalize_hit(hit)


def trace_dense[
    mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    leaves: DenseTriangleLeaves,
    ray: Rayf32[Frame.WORLD],
) -> Hit[Frame.WORLD]:
    def leaf_fn(
        ray: Rayf32[Frame.WORLD],
        O: Point3[DType.float32, Frame.WORLD, WIDTH],
        D: Vec3[DType.float32, Frame.WORLD, WIDTH],
        _ray_a: SIMD[DType.float32, WIDTH],
        _ray_inv_a: SIMD[DType.float32, WIDTH],
        leaf_idx: UInt32,
        mut hit: Hit[Frame.WORLD],
    ) {imm} -> Bool:
        var idx = Int(leaf_idx)
        var first = Int(leaves.first.unsafe_get(idx))
        var count = Int(leaves.count.unsafe_get(idx))
        var v0 = Point3[DType.float32, Frame.WORLD, WIDTH](
            leaves.v0x.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
            leaves.v0y.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
            leaves.v0z.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
        )
        var e1 = Vec3[DType.float32, Frame.WORLD, WIDTH](
            leaves.e1x.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
            leaves.e1y.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
            leaves.e1z.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
        )
        var e2 = Vec3[DType.float32, Frame.WORLD, WIDTH](
            leaves.e2x.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
            leaves.e2y.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
            leaves.e2z.unsafe_ptr()
            .unsafe_offset(first)
            .unsafe_load[width=WIDTH](),
        )
        var valid = SIMD[DType.bool, WIDTH](fill=False)
        comptime for lane in range(WIDTH):
            valid[lane] = lane < count

        comptime if mode == TRACE.ANY_HIT:
            var candidate = intersect_ray_tri_edges(
                O, D, v0, e1, e2, hit.t, ray.t_min
            )
            return (candidate.mask & valid).reduce_or()
        else:
            var candidate = intersect_ray_tri_edges_scaled(
                O, D, v0, e1, e2, hit.t, ray.t_min
            )
            var prim_indices = (
                leaves.prim_indices.unsafe_ptr()
                .unsafe_offset(first)
                .unsafe_load[width=WIDTH]()
            )
            return _finish_hit(
                candidate,
                candidate.mask & valid,
                prim_indices,
                e1,
                e2,
                hit,
            )

    var hit = trace_bounds_bvh[
        frame=Frame.WORLD,
        bounds_width=WIDTH,
        leaf_width=WIDTH,
        mode=mode,
    ](bvh.tree, ray, leaf_fn)
    return _normalize_hit(hit)


def trace_half_edges[
    mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    leaves: HalfEdgeTriangleLeaves,
    ray: Rayf32[Frame.WORLD],
) -> Hit[Frame.WORLD]:
    def leaf_fn(
        ray: Rayf32[Frame.WORLD],
        O: Point3[DType.float32, Frame.WORLD, WIDTH],
        D: Vec3[DType.float32, Frame.WORLD, WIDTH],
        _ray_a: SIMD[DType.float32, WIDTH],
        _ray_inv_a: SIMD[DType.float32, WIDTH],
        leaf_idx: UInt32,
        mut hit: Hit[Frame.WORLD],
    ) {imm} -> Bool:
        ref block = leaves.blocks.unsafe_get(Int(leaf_idx))
        var e1 = Vec3[DType.float32, Frame.WORLD, WIDTH](
            block.e1.x.cast[DType.float32](),
            block.e1.y.cast[DType.float32](),
            block.e1.z.cast[DType.float32](),
        )
        var e2 = Vec3[DType.float32, Frame.WORLD, WIDTH](
            block.e2.x.cast[DType.float32](),
            block.e2.y.cast[DType.float32](),
            block.e2.z.cast[DType.float32](),
        )
        comptime if mode == TRACE.ANY_HIT:
            var candidate = intersect_ray_tri_edges(
                O, D, block.v0, e1, e2, hit.t, ray.t_min
            )
            return candidate.mask.reduce_or()
        else:
            var candidate = intersect_ray_tri_edges_scaled(
                O, D, block.v0, e1, e2, hit.t, ray.t_min
            )
            return _finish_hit(
                candidate,
                candidate.mask,
                block.prim_indices,
                e1,
                e2,
                hit,
            )

    var hit = trace_bounds_bvh[
        frame=Frame.WORLD,
        bounds_width=WIDTH,
        leaf_width=WIDTH,
        mode=mode,
    ](bvh.tree, ray, leaf_fn)
    return _normalize_hit(hit)


def trace_layout[
    layout: Int, mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    split: SplitTriangleLeaves,
    dense: DenseTriangleLeaves,
    half: HalfEdgeTriangleLeaves,
    ray: Rayf32[Frame.WORLD],
) -> Hit[Frame.WORLD]:
    comptime if layout == PACKED:
        return trace_packed[mode](bvh, ray)
    elif layout == SPLIT_HOT_COLD:
        return trace_split[mode](bvh, split, ray)
    elif layout == DENSE_SOA:
        return trace_dense[mode](bvh, dense, ray)
    else:
        return trace_half_edges[mode](bvh, half, ray)


def trace_layout_rays[
    layout: Int, mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    split: SplitTriangleLeaves,
    dense: DenseTriangleLeaves,
    half: HalfEdgeTriangleLeaves,
    rays: List[Rayf32[Frame.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit = trace_layout[layout, mode](bvh, split, dense, half, ray)
        comptime if mode == TRACE.CLOSEST_HIT:
            if hit.is_hit():
                checksum += Float64(hit.t) + Float64(hit.prim)
                hits += 1
        else:
            if hit.is_occluded():
                checksum += 1.0
                hits += 1
    return (checksum, hits)


def time_layout[
    layout: Int, mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    split: SplitTriangleLeaves,
    dense: DenseTriangleLeaves,
    half: HalfEdgeTriangleLeaves,
    rays: List[Rayf32[Frame.WORLD]],
) -> TimingResult:
    var summary = trace_layout_rays[layout, mode](bvh, split, dense, half, rays)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var start = perf_counter_ns()
        summary = trace_layout_rays[layout, mode](bvh, split, dense, half, rays)
        var elapsed = Int(perf_counter_ns() - start)
        if elapsed < best_ns:
            best_ns = elapsed
    return TimingResult(best_ns, summary[0], summary[1])


def trace_production[
    leaf_width: SIMDLength, mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit = bvh.trace[mode](ray)
        comptime if mode == TRACE.CLOSEST_HIT:
            if hit.is_hit():
                checksum += Float64(hit.t) + Float64(hit.prim)
                hits += 1
        else:
            if hit.is_occluded():
                checksum += 1.0
                hits += 1
    return (checksum, hits)


def time_production[
    leaf_width: SIMDLength, mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> TimingResult:
    var summary = trace_production[leaf_width, mode](bvh, rays)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var start = perf_counter_ns()
        summary = trace_production[leaf_width, mode](bvh, rays)
        var elapsed = Int(perf_counter_ns() - start)
        if elapsed < best_ns:
            best_ns = elapsed
    return TimingResult(best_ns, summary[0], summary[1])


def collect_production_stats[
    leaf_width: SIMDLength, mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> CpuBvhTraversalStats:
    var stats = CpuBvhTraversalStats()
    for ray in rays:
        _ = bvh.trace_with_stats[mode](ray, stats)
    return stats^


def print_work(label: String, stats: CpuBvhTraversalStats) raises:
    print(
        t"    {label}:"
        t" nodes/ray={round(ratio(stats.internal_nodes, stats.rays), 3)},"
        t" leaves/ray={round(ratio(stats.leaf_blocks, stats.rays), 3)},"
        t" valid"
        t" tris/ray={round(ratio(stats.valid_primitives, stats.rays), 3)},"
        t" visited"
        t" occupancy={round(100.0 * ratio(stats.valid_primitives, stats.primitive_packet_lanes), 2)}%"
    )


def print_layout_case[
    mode: TRACE
](
    label: String,
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    split: SplitTriangleLeaves,
    dense: DenseTriangleLeaves,
    half: HalfEdgeTriangleLeaves,
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var baseline = time_layout[PACKED, mode](bvh, split, dense, half, rays)
    var split_result = time_layout[SPLIT_HOT_COLD, mode](
        bvh, split, dense, half, rays
    )
    var dense_result = time_layout[DENSE_SOA, mode](
        bvh, split, dense, half, rays
    )
    var half_result = time_layout[HALF_EDGES, mode](
        bvh, split, dense, half, rays
    )
    print(t"\n{label}")
    print(
        t"  packed:"
        t" {round(ns_to_mrays_per_s(baseline.ns, len(rays)), 3)} MRay/s (1.0x)"
    )
    print(
        t"  split hot/cold:"
        t" {round(ns_to_mrays_per_s(split_result.ns, len(rays)), 3)} MRay/s"
        t" ({round(Float64(baseline.ns) / Float64(split_result.ns), 3)}x)"
    )
    print(
        t"  dense SoA:"
        t" {round(ns_to_mrays_per_s(dense_result.ns, len(rays)), 3)} MRay/s"
        t" ({round(Float64(baseline.ns) / Float64(dense_result.ns), 3)}x)"
    )
    print(
        t"  Float16 edges:"
        t" {round(ns_to_mrays_per_s(half_result.ns, len(rays)), 3)} MRay/s"
        t" ({round(Float64(baseline.ns) / Float64(half_result.ns), 3)}x)"
    )
    print(
        t"  split checksum/hits delta:"
        t" {round(split_result.checksum - baseline.checksum, 6)} /"
        t" {split_result.hits - baseline.hits}"
    )
    print(
        t"  dense checksum/hits delta:"
        t" {round(dense_result.checksum - baseline.checksum, 6)} /"
        t" {dense_result.hits - baseline.hits}"
    )
    print(
        t"  Float16 checksum/hits delta:"
        t" {round(half_result.checksum - baseline.checksum, 6)} /"
        t" {half_result.hits - baseline.hits}"
    )


def validate_candidate[
    layout: Int, mode: TRACE
](
    label: String,
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    split: SplitTriangleLeaves,
    dense: DenseTriangleLeaves,
    half: HalfEdgeTriangleLeaves,
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var classification_mismatches = 0
    var primitive_mismatches = 0
    var distance_mismatches = 0
    var max_relative_t_error = Float32(0.0)
    for ray in rays:
        var baseline = trace_layout[PACKED, mode](bvh, split, dense, half, ray)
        var candidate = trace_layout[layout, mode](bvh, split, dense, half, ray)
        comptime if mode == TRACE.ANY_HIT:
            if baseline.is_occluded() != candidate.is_occluded():
                classification_mismatches += 1
        else:
            if baseline.is_hit() != candidate.is_hit():
                classification_mismatches += 1
            elif baseline.is_hit():
                if baseline.prim != candidate.prim:
                    primitive_mismatches += 1
                var scale = abs(baseline.t)
                if scale < 1.0:
                    scale = 1.0
                var relative_error = abs(candidate.t - baseline.t) / scale
                if relative_error > max_relative_t_error:
                    max_relative_t_error = relative_error
                if relative_error > 1.0e-4:
                    distance_mismatches += 1
    print(
        t"  {label}: class={classification_mismatches},"
        t" prim={primitive_mismatches}, t>1e-4={distance_mismatches},"
        t" max-relative-t={max_relative_t_error}"
    )


def validate_layouts[
    mode: TRACE
](
    label: String,
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    split: SplitTriangleLeaves,
    dense: DenseTriangleLeaves,
    half: HalfEdgeTriangleLeaves,
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    print(t"\n{label} validation ({len(rays)} rays)")
    validate_candidate[SPLIT_HOT_COLD, mode](
        "split hot/cold", bvh, split, dense, half, rays
    )
    validate_candidate[DENSE_SOA, mode](
        "dense SoA", bvh, split, dense, half, rays
    )
    validate_candidate[HALF_EDGES, mode](
        "Float16 edges", bvh, split, dense, half, rays
    )


def static_occupancy[
    leaf_width: SIMDLength
](bvh: TriangleBvh[Frame.WORLD, WIDTH, leaf_width]) raises:
    var valid = 0
    var histogram = List[Int](length=Int(leaf_width) + 1, fill=0)
    for ref block in bvh.leaf_blocks:
        var count = _count_true_lanes(block.prim_indices.ne(EMPTY_LANE))
        valid += count
        histogram[count] += 1
    var lanes = len(bvh.leaf_blocks) * Int(leaf_width)
    print(
        t"  leaf{Int(leaf_width)}: {len(bvh.leaf_blocks)} blocks,"
        t" {round(100.0 * Float64(valid) / Float64(lanes), 2)}% occupied,"
        t" {lanes * 40} bytes"
    )
    print("    block counts by occupancy:", histogram)


def print_width_case[
    mode: TRACE
](
    label: String,
    bvh4: TriangleBvh[Frame.WORLD, WIDTH, 4],
    bvh8: TriangleBvh[Frame.WORLD, WIDTH, 8],
    bvh16: TriangleBvh[Frame.WORLD, WIDTH, 16],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var result4 = time_production[4, mode](bvh4, rays)
    var result8 = time_production[8, mode](bvh8, rays)
    var result16 = time_production[16, mode](bvh16, rays)
    print(t"\n{label}")
    print(
        t"  leaf4: {round(ns_to_mrays_per_s(result4.ns, len(rays)), 3)} MRay/s"
    )
    print(
        t"  leaf8: {round(ns_to_mrays_per_s(result8.ns, len(rays)), 3)} MRay/s"
    )
    print(
        t"  leaf16:"
        t" {round(ns_to_mrays_per_s(result16.ns, len(rays)), 3)} MRay/s"
    )
    var stats4 = collect_production_stats[4, mode](bvh4, rays)
    var stats8 = collect_production_stats[8, mode](bvh8, rays)
    var stats16 = collect_production_stats[16, mode](bvh16, rays)
    print_work("leaf4", stats4)
    print_work("leaf8", stats8)
    print_work("leaf16", stats16)


def main() raises:
    print("CPU triangle leaf layout and occupancy investigation")
    print("BVH16 / SAH / best of four / full traversal")
    print(
        t"sizes: packed leaf16={size_of[TriangleLeafBlock[Frame.WORLD, 16]]()}"
        t", hot geometry={size_of[TriangleGeometryBlock]()}"
    )

    var dragon_vertices = pack_obj_triangles[Frame.WORLD](OBJ_PATH)
    var dragon_bounds = compute_bounds(dragon_vertices)
    var camera = make_camera_rays_and_params(
        dragon_bounds, RAY_WIDTH, RAY_HEIGHT, 1, 0.2
    )
    var dragon_rays = camera[0].copy()
    var dragon4 = TriangleBvh[Frame.WORLD, WIDTH, 4].__init__["sah"](
        dragon_vertices
    )
    var dragon8 = TriangleBvh[Frame.WORLD, WIDTH, 8].__init__["sah"](
        dragon_vertices
    )
    var dragon16 = TriangleBvh[Frame.WORLD, WIDTH, 16].__init__["sah"](
        dragon_vertices
    )
    var hit_rays = select_and_repeat_hit_rays(dragon16, dragon_rays)
    var hit_permuted = permute_rays(hit_rays)

    print("\nDragon static occupancy")
    static_occupancy[4](dragon4)
    static_occupancy[8](dragon8)
    static_occupancy[16](dragon16)

    print("\nProduction packet-width comparison")
    print_width_case[TRACE.CLOSEST_HIT](
        "Dragon natural closest", dragon4, dragon8, dragon16, dragon_rays
    )
    print_width_case[TRACE.CLOSEST_HIT](
        "Dragon high-hit coherent closest", dragon4, dragon8, dragon16, hit_rays
    )
    print_width_case[TRACE.CLOSEST_HIT](
        "Dragon high-hit permuted closest",
        dragon4,
        dragon8,
        dragon16,
        hit_permuted,
    )
    print_width_case[TRACE.ANY_HIT](
        "Dragon high-hit coherent any", dragon4, dragon8, dragon16, hit_rays
    )
    print_width_case[TRACE.ANY_HIT](
        "Dragon high-hit permuted any", dragon4, dragon8, dragon16, hit_permuted
    )

    var split = SplitTriangleLeaves(dragon16)
    var dense = DenseTriangleLeaves(dragon16)
    var half = HalfEdgeTriangleLeaves(dragon16)
    var packed_bytes = (
        len(dragon16.leaf_blocks)
        * size_of[TriangleLeafBlock[Frame.WORLD, WIDTH]]()
    )
    print("\nDragon leaf storage")
    print(t"  packed: {packed_bytes} bytes")
    print(t"  split hot/cold: {split.bytes()} bytes")
    print(t"  dense SoA: {dense.bytes()} bytes")
    print(t"  Float16 edges: {half.bytes()} bytes")

    print("\nLeaf-layout comparison (identical hierarchy and leaf membership)")
    print_layout_case[TRACE.CLOSEST_HIT](
        "Dragon natural closest", dragon16, split, dense, half, dragon_rays
    )
    print_layout_case[TRACE.CLOSEST_HIT](
        "Dragon high-hit coherent closest",
        dragon16,
        split,
        dense,
        half,
        hit_rays,
    )
    print_layout_case[TRACE.CLOSEST_HIT](
        "Dragon high-hit permuted closest",
        dragon16,
        split,
        dense,
        half,
        hit_permuted,
    )
    print_layout_case[TRACE.ANY_HIT](
        "Dragon high-hit coherent any", dragon16, split, dense, half, hit_rays
    )
    print_layout_case[TRACE.ANY_HIT](
        "Dragon high-hit permuted any",
        dragon16,
        split,
        dense,
        half,
        hit_permuted,
    )
    validate_layouts[TRACE.CLOSEST_HIT](
        "Dragon natural closest", dragon16, split, dense, half, dragon_rays
    )
    validate_layouts[TRACE.CLOSEST_HIT](
        "Dragon high-hit permuted closest",
        dragon16,
        split,
        dense,
        half,
        hit_permuted,
    )
    validate_layouts[TRACE.ANY_HIT](
        "Dragon high-hit permuted any",
        dragon16,
        split,
        dense,
        half,
        hit_permuted,
    )

    var grid_vertices = make_grid_triangles()
    var grid_rays = make_hit_and_miss_rays()
    var grid4 = TriangleBvh[Frame.WORLD, WIDTH, 4].__init__["sah"](
        grid_vertices
    )
    var grid8 = TriangleBvh[Frame.WORLD, WIDTH, 8].__init__["sah"](
        grid_vertices
    )
    var grid16 = TriangleBvh[Frame.WORLD, WIDTH, 16].__init__["sah"](
        grid_vertices
    )
    print("\nGrid static occupancy")
    static_occupancy[4](grid4)
    static_occupancy[8](grid8)
    static_occupancy[16](grid16)
    print_width_case[TRACE.CLOSEST_HIT](
        "Grid closest", grid4, grid8, grid16, grid_rays
    )
    var grid_split = SplitTriangleLeaves(grid16)
    var grid_dense = DenseTriangleLeaves(grid16)
    var grid_half = HalfEdgeTriangleLeaves(grid16)
    print_layout_case[TRACE.CLOSEST_HIT](
        "Grid closest", grid16, grid_split, grid_dense, grid_half, grid_rays
    )
    validate_layouts[TRACE.CLOSEST_HIT](
        "Grid closest",
        grid16,
        grid_split,
        grid_dense,
        grid_half,
        grid_rays,
    )
