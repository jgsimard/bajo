from std.gpu.host import DeviceContext, DeviceBuffer

from .bitonic_sort import bitonic_sort, bitonic_sort_basic
from .radix_sort import radix_sort


fn gpu_sort[
    dtype: DType, //, THREADS_PER_BLOCK: Int = 256
](
    ctx: DeviceContext,
    keys: DeviceBuffer[dtype],
    values: DeviceBuffer[dtype],
    size: Int,
) raises:
    if size <= 2**16:
        bitonic_sort(ctx, keys, values, size)
    else:
        radix_sort(ctx, keys, values, size)
