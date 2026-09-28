@fieldwise_init
struct Frame(EnumLike, Equatable, TrivialRegisterPassable):
    var v: Int

    comptime WORLD: Frame = Frame(0)
    comptime CAMERA: Frame = Frame(1)
    comptime LOCAL: Frame = Frame(2)
    comptime _enum_case_names = ParameterList.of[
        "WORLD".value, "CAMERA".value, "LOCAL".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType
    ].values

    def _get_enum_discriminant(self) -> Int:
        return self.v

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass
