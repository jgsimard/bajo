"""CPU-native ownership for packed BLAS storage."""


@fieldwise_init
struct CpuBlasSet[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Own descriptor, node, and leaf arrays in CPU-native lists."""

    var descs: List[UInt32]
    var nodes: List[Float32]
    var leaves: List[Float32]
    var blas_count: Int
