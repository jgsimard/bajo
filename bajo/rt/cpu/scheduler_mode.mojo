"""Compile-time CPU renderer scheduling modes."""


@fieldwise_init
struct CpuSchedulerMode(EnumLike, Equatable, ImplicitlyCopyable):
    """Select runtime-default or explicit worker partitioning."""

    comptime RUNTIME_DEFAULT = Self(0)
    comptime TASK_PARTITIONS = Self(1)
    comptime _enum_case_names = ParameterList.of[
        "RUNTIME_DEFAULT".value,
        "TASK_PARTITIONS".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType
    ].values
    comptime is_valid[mode: Self] = (
        mode.value == Self.RUNTIME_DEFAULT.value
        or mode.value == Self.TASK_PARTITIONS.value
    )

    var value: Int

    def _get_enum_discriminant(self) -> Int:
        return self.value

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass
