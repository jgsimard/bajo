"""Device buffer uploads, retained-buffer copies, and device ABI views."""

from max.gpu.host import DeviceBuffer, DeviceContext


def _device_span[
    mut: Bool,
    dtype: DType,
](
    buffer: DeviceBuffer[dtype],
) -> Span[
    mut=mut, Scalar[dtype], AnyOrigin[mut=mut]
]:
    """Create a length-backed view at the device ABI boundary."""
    return Span(
        unsafe_ptr=buffer.unsafe_ptr()
        .unsafe_mut_cast[mut]()
        .unsafe_origin_cast[AnyOrigin[mut=mut]](),
        length=len(buffer),
    )


def upload_list[
    dtype: DType
](
    mut ctx: DeviceContext,
    a: ImmSpan[Scalar[dtype], _],
) raises -> DeviceBuffer[
    dtype
]:
    var h_a = ctx.enqueue_create_host_buffer[dtype](len(a))
    var d_a = ctx.enqueue_create_buffer[dtype](len(a))
    h_a.enqueue_copy_from(a)
    h_a.enqueue_copy_to(d_a)
    return d_a^


def copy_to_device[
    dtype: DType
](buffer: DeviceBuffer[dtype], values: ImmSpan[Scalar[dtype], _]) raises:
    """Copy into retained storage; the caller controls context synchronization.
    """
    if len(values) != len(buffer):
        raise Error("Device buffer copy requires matching lengths")
    with buffer.map_to_host() as data:
        for i in range(len(values)):
            data[i] = values[i]
