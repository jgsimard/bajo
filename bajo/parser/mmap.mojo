"""Read-only file mapping shared by binary and text asset decoders."""

from std.ffi import external_call
from std.io.file_descriptor import RawFileDescriptor
from std.os import SEEK_END


struct MMap:
    comptime RawPointer = Pointer[UInt8, ImmUntrackedOrigin]
    var _data: Optional[Self.RawPointer]
    var _size: Int

    def __init__(out self, path: String) raises:
        var data = Optional[Self.RawPointer]()
        var size: Int

        with open(path, "r") as file:
            comptime PROT_READ = 1
            comptime MAP_PRIVATE = 2

            size = Int(file.seek(0, SEEK_END))
            if size != 0:
                var mapped = external_call["mmap", Self.RawPointer](
                    Int(0),  # addr: let the kernel choose
                    UInt(size),
                    Int32(PROT_READ),
                    Int32(MAP_PRIVATE),
                    RawFileDescriptor(file),
                    Int64(0),
                )
                # mmap returns MAP_FAILED ((void*)-1) on failure
                if Int(mapped) == -1:
                    raise Error("mmap failed for " + path)
                data = mapped

        self._data = data
        self._size = size

    def __deinit__(deinit self):
        if self._size != 0:
            _ = external_call["munmap", Int32](
                self._data.unsafe_value(), UInt(self._size)
            )

    def byte_length(self) -> Int:
        return self._size

    def as_bytes_span(self) -> Span[UInt8, origin_of(self)]:
        if self._size == 0:
            return {}
        var data = self._data.unsafe_value().unsafe_origin_cast[
            origin_of(self)
        ]()
        return {unsafe_ptr = data, length = self._size}

    def as_string_span(self) -> StringSpan[origin_of(self)]:
        return StringSpan(unsafe_from_utf8=self.as_bytes_span())
