"""Linux eventfd wakeups, with polling fallback when eventfd is unavailable."""
from std.ffi import c_int, external_call
from std.memory import Pointer


def create_notification() -> c_int:
    # CLOEXEC keeps the descriptor private to the worker's owning process.
    return external_call["eventfd", c_int](UInt32(0), c_int(0x80000))


def _interrupted() -> Bool:
    return (
        external_call[
            "__errno_location", Pointer[c_int, MutUntrackedOrigin]
        ]()[]
        == 4
    )


def notify(fd: c_int):
    if fd < 0:
        return
    var value = UInt64(1)
    while external_call["write", Int](fd, Pointer(to=value), UInt(8)) < 0:
        if not _interrupted():
            return


def wait_notification(fd: c_int):
    if fd >= 0:
        var value = UInt64(0)
        while True:
            if external_call["read", Int](fd, Pointer(to=value), UInt(8)) == 8:
                return
            if not _interrupted():
                break
    _ = external_call["usleep", c_int](UInt32(1000))
