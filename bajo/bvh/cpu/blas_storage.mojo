"""CPU-native ownership for packed BLAS storage."""

from bajo.bvh.constants import PrimitiveKind
from bajo.bvh.cpu.bounds_bvh import WideBvhNode
from bajo.bvh.types import BlasDesc
from bajo.core import Frame


@fieldwise_init
struct _CpuBlasView[origin: ImmOrigin, frame: Frame, node_width: SIMDLength](
    Copyable
):
    """Borrowed node and leaf storage for one resolved CPU BLAS."""

    var nodes: ImmSpan[WideBvhNode[Self.frame, Self.node_width], Self.origin]
    var leaves: ImmPointer[Float32, Self.origin]


@fieldwise_init
struct CpuBlasSet[
    kind: PrimitiveKind,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Own primitive-typed descriptor, node, and leaf arrays on the CPU."""

    var descs: List[UInt32]
    var nodes: List[Float32]
    var leaves: List[Float32]
    var blas_count: Int

    @always_inline
    def _view[
        frame: Frame
    ](self, desc: BlasDesc) -> _CpuBlasView[
        origin_of(self), frame, Self.node_width
    ]:
        """Decode a resolved descriptor while borrowing both owner buffers."""
        var nodes = (
            self.nodes.unsafe_ptr()
            .unsafe_offset(Int(desc.node_f32_base))
            .unsafe_bitcast[WideBvhNode[frame, Self.node_width]]()
            .unsafe_origin_cast[origin_of(self)]()
        )
        var leaves = (
            self.leaves.unsafe_ptr()
            .unsafe_offset(Int(desc.leaf_f32_base))
            .unsafe_origin_cast[origin_of(self)]()
        )
        return _CpuBlasView[origin_of(self), frame, Self.node_width](
            ImmSpan(unsafe_ptr=nodes, length=Int(desc.node_count)), leaves
        )
