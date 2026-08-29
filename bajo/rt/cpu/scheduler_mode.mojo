"""Compile-time CPU renderer scheduling modes."""


@fieldwise_init
struct CpuSchedulerMode(Equatable, ImplicitlyCopyable):
    """Select how parallel render work is partitioned."""

    comptime RUNTIME_DEFAULT = Self(0)
    comptime LOGICAL_CORES = Self(1)
    comptime TASK_PARTITIONS = Self(2)
    comptime is_valid[mode: Self] = (
        mode.value == Self.RUNTIME_DEFAULT.value
        or mode.value == Self.LOGICAL_CORES.value
        or mode.value == Self.TASK_PARTITIONS.value
    )

    var value: Int
