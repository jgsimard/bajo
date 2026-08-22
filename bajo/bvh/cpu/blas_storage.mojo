"""CPU-native ownership for packed BLAS storage."""

from bajo.bvh.constants import Primitive


@fieldwise_init
struct CpuBlasSet[
    kind: Primitive,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Own primitive-typed descriptor, node, and leaf arrays on the CPU."""

    var descs: List[UInt32]
    var nodes: List[Float32]
    var leaves: List[Float32]
    var blas_count: Int
