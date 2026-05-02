from std.ffi import external_call
from std.os import SEEK_END


struct MMap[mut: Bool, //, origin: Origin[mut=mut]]:
    comptime ptr = Optional[UnsafePointer[UInt8, Self.origin]]
    var _data: Self.ptr
    var _size: Int

    def __init__(out self, path: String) raises:
        self._data = Self.ptr()
        self._size = 0

        with open(path, "r") as file:
            comptime PROT_READ = 1
            comptime MAP_PRIVATE = 2

            self._size = Int(file.seek(0, SEEK_END))
            if self._size == 0:
                return

            self._data = external_call["mmap", Self.ptr](
                Self.ptr(),  # addr: let the kernel choose
                self._size,
                PROT_READ,
                MAP_PRIVATE,
                file._get_raw_fd(),
                0,  # offset
            )

        if not self._data:
            raise Error("mmap failed")

    def __del__(deinit self):
        if self._data:
            _ = external_call["munmap", Int](self._data, self._size)

    def byte_length(ref self) -> Int:
        return self._size

    def as_string_slice(ref self) -> StringSlice[Self.origin]:
        if self._size == 0:
            return StringSlice[Self.origin]()
        return StringSlice[Self.origin](
            ptr=self._data.unsafe_value(), length=self._size
        )
