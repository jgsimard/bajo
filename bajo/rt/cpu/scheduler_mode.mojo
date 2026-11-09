"""Compile-time CPU renderer scheduling modes."""


__enum CpuSchedulerMode(Equatable, ImplicitlyCopyable):
    """Select runtime-default or explicit worker partitioning."""

    case RUNTIME_DEFAULT
    case TASK_PARTITIONS
