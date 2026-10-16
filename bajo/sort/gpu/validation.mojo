"""Host-only validation before GPU sorting allocates or submits work."""

from std.math import ceildiv


def validate_sort_size(size: Int) raises:
    if size < 0 or size > Int(Int32.MAX):
        raise Error("sort size must fit a nonnegative Int32")


def require_sort_capacity(required: Int, available: Int, label: String) raises:
    if available < required:
        raise Error(label + " buffer is too short")


def validate_onesweep_size[
    partition_size: Int, passes: Int
](size: Int) raises -> Int:
    comptime assert partition_size > 0 and passes > 0
    validate_sort_size(size)
    # Two state bits share each UInt32 look-back entry with its prefix count.
    if size > 0x3FFFFFFF:
        raise Error("OneSweep size exceeds its 30-bit look-back count storage")
    var length = (ceildiv(size, partition_size) + 1) * 256 * passes
    if length > Int(Int32.MAX):
        raise Error("OneSweep histogram length exceeds Int32 storage")
    return length


def validate_bitonic_size(size: Int) raises:
    validate_sort_size(size)
    if size > 1 and not size.is_power_of_two():
        raise Error("bitonic sort requires a power-of-two size")
