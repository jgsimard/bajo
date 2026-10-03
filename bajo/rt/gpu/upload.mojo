"""Nonempty device-table upload shared by prepared GPU RT resources."""

from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.bvh.gpu.utils import upload_list


def _upload_nonempty[
    dtype: DType
](
    mut ctx: DeviceContext, var values: List[Scalar[dtype]]
) raises -> DeviceBuffer[dtype]:
    if len(values) == 0:
        values.append(0)
    return upload_list(ctx, values)
