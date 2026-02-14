from layout import Layout, LayoutTensor, UNKNOWN_VALUE


fn create_vector[
    dtype: DType, layout: Layout
](
    m: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m),
        type_of(result.runtime_layout.stride)(1),
    )
    return {ptr, dynamic_layout}


fn create_tensor[
    dtype: DType, layout: Layout
](
    m: Int,
    n: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m, n),
        type_of(result.runtime_layout.stride)(1, m),
    )
    return {ptr, dynamic_layout}


fn create_tensor2[
    dtype: DType, layout: Layout
](m: Int, n: Int, out result: LayoutTensor[dtype, layout, MutExternalOrigin],):
    var ptr = alloc[Scalar[dtype]](m * n)
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m, n),
        type_of(result.runtime_layout.stride)(1, m),
    )
    return {ptr, dynamic_layout}
