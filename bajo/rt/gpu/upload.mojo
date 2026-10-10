"""Nonempty device-table upload shared by prepared GPU RT resources."""

from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.gpu.buffers import upload_list


def _upload_nonempty[
    dtype: DType
](
    mut ctx: DeviceContext, values: ImmSpan[Scalar[dtype], _]
) raises -> DeviceBuffer[dtype]:
    if len(values) == 0:
        return upload_list(ctx, List[Scalar[dtype]](length=1, fill=0))
    return upload_list(ctx, values)
