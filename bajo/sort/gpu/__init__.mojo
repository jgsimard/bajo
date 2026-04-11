from std.gpu.host import DeviceContext, DeviceBuffer

from .bitonic_sort import bitonic_sort_pairs
from .onesweep import onesweep_radix_sort_pairs


def gpu_sort_pairs[
    keys_dtype: DType, vals_dtype: DType
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    if size <= 2**14:
        bitonic_sort_pairs(ctx, keys, values, size)
    else:
        onesweep_radix_sort_pairs(ctx, keys, values, size)
