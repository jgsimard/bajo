from std.ffi import external_call
from std.os import SEEK_END


struct MMap:
    comptime RawPointer = Pointer[UInt8, ImmUntrackedOrigin]
    comptime ptr = Optional[Self.RawPointer]
    var _data: Self.ptr
    var _size: Int

    def __init__(out self, path: String) raises:
        var data = Self.ptr()
        var size: Int

        with open(path, "r") as file:
            comptime PROT_READ = 1
            comptime MAP_PRIVATE = 2

            size = Int(file.seek(0, SEEK_END))
            if size != 0:
                data = external_call["mmap", Self.ptr](
                    Self.ptr(),  # addr: let the kernel choose
                    size,
                    PROT_READ,
                    MAP_PRIVATE,
                    file._get_raw_fd(),
                    0,  # offset
                )

        if size != 0 and not data:
            raise Error("mmap failed")
        self._data = data
        self._size = size

    def __deinit__(deinit self):
        if self._data:
            _ = external_call["munmap", Int](self._data, self._size)

    def byte_length(ref self) -> Int:
        return self._size

    def as_bytes_span(self) -> Span[UInt8, origin_of(self)]:
        comptime T = Span[UInt8, origin_of(self)]
        if self._size == 0:
            return T()
        var data = self._data.unsafe_value().unsafe_origin_cast[
            origin_of(self)
        ]()
        return T(unsafe_ptr=data, length=self._size)

    def as_string_span(self) -> StringSpan[origin_of(self)]:
        return StringSpan(unsafe_from_utf8=self.as_bytes_span())
