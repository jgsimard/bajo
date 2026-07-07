@fieldwise_init
struct Frame(Equatable, TrivialRegisterPassable):
    var v: Int

    comptime WORLD: Frame = Frame(0)
    comptime CAMERA: Frame = Frame(1)
    comptime OBJECT: Frame = Frame(2)
