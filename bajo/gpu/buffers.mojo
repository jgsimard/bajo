"""Device buffer uploads, retained-buffer copies, and device ABI views."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import align_of, size_of
from bajo.core import Frame, Point3f32


@always_inline
def _packed_point3_span[
    frame: Frame, origin: ImmOrigin
](points: ImmSpan[Point3f32[frame], origin]) -> ImmSpan[Float32, origin]:
    """Borrow scalar XYZ words from tightly packed points, preserving bits."""
    comptime assert size_of[Point3f32[frame]]() == 3 * size_of[Float32]()
    comptime assert align_of[Point3f32[frame]]() >= align_of[Float32]()
    return ImmSpan(
        unsafe_ptr=points.unsafe_ptr().unsafe_bitcast[Float32](),
        length=len(points) * 3,
    )


def _copy_staging_slice[
    dtype: DType
](staging: MutSpan[UInt32, _], offset: Int, values: ImmSpan[Scalar[dtype], _]):
    """Copy four-byte scalars into a checked slice of host staging storage."""
    comptime assert size_of[Scalar[dtype]]() == size_of[UInt32]()
    comptime assert align_of[Scalar[dtype]]() <= align_of[UInt32]()
    var destination = staging[offset : offset + len(values)]
    # Word storage satisfies this scalar's size and alignment. Reinterpret
    # only after slicing the host view, preserving all payload bits.
    var typed_destination = MutSpan(
        unsafe_ptr=destination.unsafe_ptr().unsafe_bitcast[Scalar[dtype]](),
        length=len(destination),
    )
    typed_destination.copy_from(values)


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
        data.as_span().copy_from(values)
