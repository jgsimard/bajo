"""Reusable GPU RT trace representation for world/local triangle geometry."""

from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu import GpuBvhBuildMethod, build_triangle_blas_set
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
        var owned_vertices = List[Point3f32[Self.frame]](capacity=len(vertices))
        for vertex in vertices:
            owned_vertices.append(vertex)
        var packed = build_triangle_blas_set[
            Self.node_width,
            Self.leaf_width,
            Self.build_method,
            Self.compressed,
            Self.frame,
        ](ctx, [owned_vertices^])
        self.nodes = packed.nodes.copy()
        self.leaves = packed.leaves.copy()
        self.root = UInt32(0)
