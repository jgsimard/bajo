@fieldwise_init
struct Frame:
    comptime WORLD = Self(0)
    comptime LOCAL = Self(1)

    var v: Int
