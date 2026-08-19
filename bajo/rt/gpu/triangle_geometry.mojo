"""Reusable GPU RT trace representation for world/local triangle geometry."""

from std.math import max
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_TRIANGLE_WORDS,
    build_cwbvh8_representation,
)
from bajo.bvh.gpu.triangle_bvh import (
    build_triangle_bvh,
    enqueue_build_triangle_wide,
)
from bajo.bvh.gpu.utils import upload_vertices
from bajo.core import Frame, Point3f32


struct GpuRtTriangleGeometry[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
]:
    """Owned trace buffers selected entirely by compile-time policy."""

    var nodes: DeviceBuffer[DType.float32]
    var leaves: DeviceBuffer[DType.float32]
    var root: UInt32

    def __init__(
        out self,
        mut ctx: DeviceContext,
        vertices: ImmSpan[Point3f32[Self.frame], _],
    ) raises:
        var tri_count = len(vertices) / 3
        debug_assert["safe", _use_compiler_assume=True](
            tri_count > 0 and len(vertices) % 3 == 0,
            "GPU RT triangle geometry requires complete triangles",
        )
        var device_vertices = upload_vertices(ctx, vertices)
        comptime if Self.compressed:
            comptime assert Self.node_width == 8 and Self.leaf_width == 4
            var pending = enqueue_build_triangle_wide[
                Self.frame, 8, 4, Self.build_method, 3, True
            ](ctx, device_vertices)
            ctx.synchronize()
            pending.finish_synchronized()
            self.nodes = ctx.enqueue_create_buffer[DType.float32](
                max(pending.tree.node_count, 1) * CWBVH_NODE_WORDS
            )
            self.leaves = ctx.enqueue_create_buffer[DType.float32](
                max(tri_count, 1) * CWBVH_TRIANGLE_WORDS
            )
            build_cwbvh8_representation[Self.leaf_width](
                ctx,
                pending.tree.wide_nodes,
                pending.tree.leaf_block_indices,
                pending.source_vertices,
                self.nodes,
                self.leaves,
                pending.tree.node_count,
                tri_count,
            )
            self.root = pending.tree.root_idx
        else:
            var bvh = build_triangle_bvh[
                Self.frame,
                Self.node_width,
                Self.leaf_width,
                Self.build_method,
            ](ctx, device_vertices)
            self.nodes = bvh.tree.wide_nodes.copy()
            self.leaves = bvh.leaf_vertices.copy()
            self.root = bvh.tree.root_idx
