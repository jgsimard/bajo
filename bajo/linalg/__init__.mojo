from layout import Layout, LayoutTensor, UNKNOWN_VALUE


def create_vector[
    dtype: DType, layout: Layout
](
    m: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m),
        type_of(result.runtime_layout.stride)(1),
    )
    return {ptr, dynamic_layout}


def create_tensor[
    dtype: DType, layout: Layout
](
    m: Int,
    n: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m, n),
        type_of(result.runtime_layout.stride)(1, m),
    )
    return {ptr, dynamic_layout}


def create_tensor2[
    dtype: DType, layout: Layout
](m: Int, n: Int, out result: LayoutTensor[dtype, layout, MutExternalOrigin],):
    ptr = alloc[Scalar[dtype]](m * n)
    dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m, n),
        type_of(result.runtime_layout.stride)(1, m),
    )
    return {ptr, dynamic_layout}
