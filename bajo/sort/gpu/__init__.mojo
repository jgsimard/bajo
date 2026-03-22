from std.gpu.host import DeviceContext, DeviceBuffer

from .bitonic_sort import bitonic_sort, bitonic_sort_basic
from .radix_sort import device_radix_sort_pairs


def gpu_sort[
    dtype: DType, //, THREADS_PER_BLOCK: Int = 256
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[dtype],
    mut values: DeviceBuffer[dtype],
    size: Int,
) raises:
    if size <= 2**16:
        bitonic_sort(ctx, keys, values, size)
    else:
        device_radix_sort_pairs(ctx, keys, values, size)
