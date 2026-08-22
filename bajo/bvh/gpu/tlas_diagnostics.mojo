"""Diagnostic-only counters for nested GPU TLAS to CWBVH8 traversal."""

from std.bit import count_leading_zeros, pop_count
from std.math import ceildiv, min
from std.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import (
    EMPTY_LANE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    GPU_STACK_SIZE,
    Primitive,
    TRACE,
    f32_max,
)
from bajo.bvh.gpu.camera_launch import _camera_ray, _store_camera_hit
from bajo.bvh.gpu.compressed_bounds_bvh import _intersect_cwbvh8_node_tasks
from bajo.bvh.gpu.trace import (
    GpuTraceResult,
    GpuTraversalStats,
    _intersect_trace_node_precomputed,
)
from bajo.bvh.gpu.triangle_bvh import _intersect_cwbvh_triangle
from bajo.bvh.gpu.tlas import GpuTriangleTlas
from bajo.bvh.wide_meta import _wide_meta_count, _wide_meta_data
from bajo.bvh.tlas_common import (
    finalize_tlas_hit_normal,
    promote_tlas_local_hit,
)
from bajo.bvh.gpu.blas_storage import GpuBlasSet, GpuBvhLayout
from bajo.bvh.types import BlasDesc, Hit
from bajo.core import Affine3f32, Frame, Rayf32


@fieldwise_init
struct GpuTlasTraversalStats(TrivialRegisterPassable, Writable):
    comptime TLAS_NODE_VISITS = 0
    comptime TLAS_LEAVES = 1
    comptime BLAS_DISPATCHES = 2
    comptime BLAS_HITS = 3
    comptime BLAS_MISSES = 4
    comptime BLAS_NODE_VISITS = 5
    comptime BLAS_LEAVES = 6
    comptime BLAS_PRIMITIVES = 7
    comptime HIT_REPLACEMENTS = 8
    comptime TLAS_MAX_STACK = 9
    comptime BLAS_MAX_STACK = 10
    comptime WINNING_BLAS = 11
    comptime STRIDE = 12

    var tlas_node_visits: UInt32
    var tlas_leaves: UInt32
    var blas_dispatches: UInt32
    var blas_hits: UInt32
    var blas_misses: UInt32
    var blas_node_visits: UInt32
    var blas_leaves: UInt32
    var blas_primitives: UInt32
    var hit_replacements: UInt32
    var tlas_max_stack: UInt32
    var blas_max_stack: UInt32
    var winning_blas: UInt32

    @staticmethod
    def zero() -> Self:
        return Self(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, EMPTY_LANE)

    def store(
        self,
        dst: MutPointer[UInt32, _],
        ray_idx: Int,
    ):
        var base = ray_idx * Self.STRIDE
        dst[unsafe_offset=base + Self.TLAS_NODE_VISITS] = self.tlas_node_visits
        dst[unsafe_offset=base + Self.TLAS_LEAVES] = self.tlas_leaves
        dst[unsafe_offset=base + Self.BLAS_DISPATCHES] = self.blas_dispatches
        dst[unsafe_offset=base + Self.BLAS_HITS] = self.blas_hits
        dst[unsafe_offset=base + Self.BLAS_MISSES] = self.blas_misses
        dst[unsafe_offset=base + Self.BLAS_NODE_VISITS] = self.blas_node_visits
        dst[unsafe_offset=base + Self.BLAS_LEAVES] = self.blas_leaves
        dst[unsafe_offset=base + Self.BLAS_PRIMITIVES] = self.blas_primitives
        dst[unsafe_offset=base + Self.HIT_REPLACEMENTS] = self.hit_replacements
        dst[unsafe_offset=base + Self.TLAS_MAX_STACK] = self.tlas_max_stack
        dst[unsafe_offset=base + Self.BLAS_MAX_STACK] = self.blas_max_stack
        dst[unsafe_offset=base + Self.WINNING_BLAS] = self.winning_blas


@fieldwise_init
struct GpuTlasStatsSummary(Copyable, Writable):
    var rays: UInt64
    var tlas_node_visits: UInt64
    var tlas_leaves: UInt64
    var blas_dispatches: UInt64
    var blas_hits: UInt64
    var blas_misses: UInt64
    var blas_node_visits: UInt64
    var blas_leaves: UInt64
    var blas_primitives: UInt64
    var hit_replacements: UInt64
    var tlas_max_stack: UInt32
    var blas_max_stack: UInt32
    var winner_lanes: UInt64
    var winner_lanes_sharing_blas: UInt64


@always_inline
def _trace_cwbvh8_with_stats(
    nodes: ImmPointer[Float32, _],
    triangles: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[Frame.LOCAL],
) -> GpuTraceResult[Frame.LOCAL]:
    var hit = Hit[Frame.LOCAL].miss(ray.t_max)
    var stats = GpuTraversalStats.zero()
    var stack_base = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_mask = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0

    var sign_code = UInt32(0)
    if ray.d.x < 0.0:
        sign_code |= UInt32(4)
    if ray.d.y < 0.0:
        sign_code |= UInt32(2)
    if ray.d.z < 0.0:
        sign_code |= UInt32(1)
    var octant_inverse = UInt32(7) - sign_code
    var ray_rcp = ray.rcp_direction[1]()

    var node_group_base = root_idx
    var node_group_mask = UInt32(1) << UInt32(31)
    var triangle_group_base: UInt32
    var triangle_group_mask: UInt32

    while True:
        if node_group_mask > UInt32(0x00FFFFFF):
            var group_imask = node_group_mask
            var child_bit = 31 - Int(count_leading_zeros(node_group_mask))
            node_group_mask &= ~(UInt32(1) << UInt32(child_bit))
            if node_group_mask > UInt32(0x00FFFFFF):
                stack_base[stack_ptr] = node_group_base
                stack_mask[stack_ptr] = node_group_mask
                stack_ptr += 1
                if UInt32(stack_ptr) > stats.max_stack_depth:
                    stats.max_stack_depth = UInt32(stack_ptr)

            var slot = UInt32(child_bit - 24) ^ octant_inverse
            var slots_before = (UInt32(1) << slot) - UInt32(1)
            var relative = UInt32(pop_count(group_imask & slots_before))
            var node_idx = node_group_base + relative
            stats.node_visits += 1
            var tasks = _intersect_cwbvh8_node_tasks[Frame.LOCAL](
                nodes,
                node_idx,
                ray,
                ray_rcp.x,
                ray_rcp.y,
                ray_rcp.z,
                hit.t,
                octant_inverse,
            )
            node_group_base = tasks.child_base
            node_group_mask = tasks.node_group_mask
            triangle_group_base = tasks.triangle_base
            triangle_group_mask = tasks.triangle_group_mask
        else:
            triangle_group_base = node_group_base
            triangle_group_mask = node_group_mask
            node_group_mask = UInt32(0)

        if triangle_group_mask != 0:
            stats.leaf_blocks += 1
        while triangle_group_mask != 0:
            var triangle_bit = 31 - Int(
                count_leading_zeros(triangle_group_mask)
            )
            triangle_group_mask &= ~(UInt32(1) << UInt32(triangle_bit))
            stats.primitive_tests += 1
            _ = _intersect_cwbvh_triangle[Frame.LOCAL, TRACE.CLOSEST_HIT](
                triangles,
                triangle_group_base + UInt32(triangle_bit),
                ray,
                hit,
            )

        if node_group_mask <= UInt32(0x00FFFFFF):
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_group_base = stack_base[stack_ptr]
            node_group_mask = stack_mask[stack_ptr]

    return GpuTraceResult[Frame.LOCAL](hit, stats)


@always_inline
def _trace_tlas2_leaf1_cwbvh8_with_stats(
    tlas_nodes: ImmPointer[Float32, _],
    tlas_leaf_instances: ImmPointer[UInt32, _],
    inst_inv_transform: ImmPointer[Float32, _],
    inst_blas_indices: ImmPointer[UInt32, _],
    blas_descs: ImmPointer[UInt32, _],
    blas_nodes: ImmPointer[Float32, _],
    blas_leaves: ImmPointer[Float32, _],
    instance_count: Int,
    tlas_root: UInt32,
    ray: Rayf32[Frame.WORLD],
) -> Tuple[Hit[Frame.WORLD], GpuTlasTraversalStats]:
    var hit = Hit[Frame.WORLD].miss(ray.t_max)
    var stats = GpuTlasTraversalStats.zero()
    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = tlas_root
    var bounds_origin = ray.origin[2]()
    var rcp_direction = ray.rcp_direction[2]()
    var inverse_span = Span(
        unsafe_ptr=inst_inv_transform,
        length=instance_count * Affine3f32.STRIDE,
    )

    while True:
        stats.tlas_node_visits += 1
        var node_hit = _intersect_trace_node_precomputed[Frame.WORLD, 2](
            tlas_nodes,
            current,
            bounds_origin,
            rcp_direction,
            hit.t,
        )
        var child_valid = Array[Bool, 2](fill=False)
        var child_data = Array[UInt32, 2](fill=0)
        var child_t = Array[Float32, 2](fill=f32_max)

        comptime for lane in range(2):
            var meta = node_hit.meta[lane]
            var count = _wide_meta_count(meta)
            if count != EMPTY_LANE and node_hit.bounds_hit.mask[lane]:
                var data = _wide_meta_data(meta)
                if count == 0:
                    child_valid[lane] = True
                    child_data[lane] = data
                    child_t[lane] = node_hit.bounds_hit.t[lane]
                else:
                    stats.tlas_leaves += 1
                    var inst_idx = UInt32(
                        tlas_leaf_instances[unsafe_offset=Int(data)]
                    )
                    if inst_idx != EMPTY_LANE:
                        stats.blas_dispatches += 1
                        var blas_idx = UInt32(
                            inst_blas_indices[unsafe_offset=Int(inst_idx)]
                        )
                        var blas_desc = BlasDesc.load(blas_descs, blas_idx)
                        var inverse = Affine3f32[Frame.WORLD, Frame.LOCAL].load(
                            inverse_span,
                            Int(inst_idx) * Affine3f32.STRIDE,
                        )
                        var local_ray = inverse.ray(ray, hit.t)
                        var local_nodes = blas_nodes.unsafe_offset(
                            Int(blas_desc.node_f32_base)
                        )
                        var local_leaves = blas_leaves.unsafe_offset(
                            Int(blas_desc.leaf_f32_base)
                        )
                        var local_root = blas_desc.root_idx
                        var local = _trace_cwbvh8_with_stats(
                            local_nodes,
                            local_leaves,
                            local_root,
                            local_ray,
                        )
                        stats.blas_node_visits += local.stats.node_visits
                        stats.blas_leaves += local.stats.leaf_blocks
                        stats.blas_primitives += local.stats.primitive_tests
                        if local.stats.max_stack_depth > stats.blas_max_stack:
                            stats.blas_max_stack = local.stats.max_stack_depth
                        if local.hit.is_hit():
                            stats.blas_hits += 1
                        else:
                            stats.blas_misses += 1
                        if promote_tlas_local_hit(local.hit, inst_idx, hit):
                            stats.hit_replacements += 1
                            stats.winning_blas = blas_idx

        var nearest_lane = -1
        var nearest_t = f32_max
        comptime for lane in range(2):
            if child_valid[lane] and child_t[lane] < nearest_t:
                nearest_lane = lane
                nearest_t = child_t[lane]

        comptime for lane in range(2):
            if (
                child_valid[lane]
                and lane != nearest_lane
                and child_t[lane] <= hit.t
            ):
                stack[stack_ptr] = child_data[lane]
                stack_ptr += 1
                if UInt32(stack_ptr) > stats.tlas_max_stack:
                    stats.tlas_max_stack = UInt32(stack_ptr)

        if nearest_lane != -1 and nearest_t <= hit.t:
            current = child_data[nearest_lane]
            continue
        if stack_ptr == 0:
            break
        stack_ptr -= 1
        current = stack[stack_ptr]

    if hit.is_hit():
        var inverse = Affine3f32[Frame.WORLD, Frame.LOCAL].load(
            inverse_span,
            Int(hit.inst) * Affine3f32.STRIDE,
        )
        finalize_tlas_hit_normal(hit, inverse)
    return (hit, stats)


def trace_tlas_camera_diagnostic_kernel(
    tlas_nodes: Pointer[Float32, ImmutAnyOrigin],
    tlas_leaf_instances: Pointer[UInt32, ImmutAnyOrigin],
    inst_inv_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: Pointer[UInt32, ImmutAnyOrigin],
    blas_descs: Pointer[UInt32, ImmutAnyOrigin],
    blas_nodes: Pointer[Float32, ImmutAnyOrigin],
    blas_leaves: Pointer[Float32, ImmutAnyOrigin],
    tlas_root: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    stats_out: Pointer[UInt32, MutAnyOrigin],
    instance_count: Int32,
    ray_count: Int32,
    width: Int32,
    height: Int32,
    inv_height: Float32,
):
    var idx = global_idx.x
    if idx >= Int(ray_count):
        return
    var ray = _camera_ray(
        camera_params,
        Int(ray_count),
        idx,
        Int(width),
        Int(height),
        inv_height,
    )
    var result = _trace_tlas2_leaf1_cwbvh8_with_stats(
        tlas_nodes,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_nodes,
        blas_leaves,
        Int(instance_count),
        tlas_root,
        ray,
    )
    _store_camera_hit(result[0], hits, Int(ray_count), idx)
    result[1].store(stats_out, idx)


def launch_triangle_tlas_camera_diagnostics[
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
](
    ctx: DeviceContext,
    tlas: GpuTriangleTlas[
        2,
        blas_node_width,
        1,
        blas_leaf_width,
        GpuBvhLayout.CWBVH8,
    ],
    blases: GpuBlasSet[
        Primitive.TRIANGLE,
        GpuBvhLayout.CWBVH8,
        blas_node_width,
        blas_leaf_width,
    ],
    camera_params: DeviceBuffer[DType.float32],
    hits: DeviceBuffer[DType.float32],
    stats: DeviceBuffer[DType.uint32],
    ray_count: Int,
    width: Int,
    height: Int,
) raises:
    debug_assert["safe", _use_compiler_assume=True](
        len(stats) >= ray_count * GpuTlasTraversalStats.STRIDE,
        "TLAS diagnostic stats buffer is too short",
    )
    ctx.enqueue_function[trace_tlas_camera_diagnostic_kernel](
        tlas.core.tree.wide_nodes,
        tlas.core.tree.leaf_block_indices,
        tlas.core.inst_inv_transform,
        tlas.core.inst_blas_indices,
        blases.descs,
        blases.nodes,
        blases.leaves,
        tlas.core.tree.root_idx,
        camera_params,
        hits,
        stats,
        Int32(tlas.core.inst_count),
        Int32(ray_count),
        Int32(width),
        Int32(height),
        Float32(1.0) / Float32(height),
        grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )


def summarize_tlas_diagnostics(
    stats: DeviceBuffer[DType.uint32],
    ray_count: Int,
) raises -> GpuTlasStatsSummary:
    var out = GpuTlasStatsSummary(
        UInt64(ray_count), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    )
    with stats.map_to_host() as mapped:
        for ray_idx in range(ray_count):
            var base = ray_idx * GpuTlasTraversalStats.STRIDE
            out.tlas_node_visits += UInt64(
                mapped[base + GpuTlasTraversalStats.TLAS_NODE_VISITS]
            )
            out.tlas_leaves += UInt64(
                mapped[base + GpuTlasTraversalStats.TLAS_LEAVES]
            )
            out.blas_dispatches += UInt64(
                mapped[base + GpuTlasTraversalStats.BLAS_DISPATCHES]
            )
            out.blas_hits += UInt64(
                mapped[base + GpuTlasTraversalStats.BLAS_HITS]
            )
            out.blas_misses += UInt64(
                mapped[base + GpuTlasTraversalStats.BLAS_MISSES]
            )
            out.blas_node_visits += UInt64(
                mapped[base + GpuTlasTraversalStats.BLAS_NODE_VISITS]
            )
            out.blas_leaves += UInt64(
                mapped[base + GpuTlasTraversalStats.BLAS_LEAVES]
            )
            out.blas_primitives += UInt64(
                mapped[base + GpuTlasTraversalStats.BLAS_PRIMITIVES]
            )
            out.hit_replacements += UInt64(
                mapped[base + GpuTlasTraversalStats.HIT_REPLACEMENTS]
            )
            var tlas_stack = mapped[base + GpuTlasTraversalStats.TLAS_MAX_STACK]
            var blas_stack = mapped[base + GpuTlasTraversalStats.BLAS_MAX_STACK]
            if tlas_stack > out.tlas_max_stack:
                out.tlas_max_stack = tlas_stack
            if blas_stack > out.blas_max_stack:
                out.blas_max_stack = blas_stack

        for warp_base in range(0, ray_count, 32):
            var warp_end = min(warp_base + 32, ray_count)
            for lane in range(warp_base, warp_end):
                var winner = mapped[
                    lane * GpuTlasTraversalStats.STRIDE
                    + GpuTlasTraversalStats.WINNING_BLAS
                ]
                if winner == EMPTY_LANE:
                    continue
                out.winner_lanes += 1
                var shared = False
                for other in range(warp_base, warp_end):
                    if (
                        other != lane
                        and mapped[
                            other * GpuTlasTraversalStats.STRIDE
                            + GpuTlasTraversalStats.WINNING_BLAS
                        ]
                        == winner
                    ):
                        shared = True
                        break
                if shared:
                    out.winner_lanes_sharing_blas += 1
    return out^
