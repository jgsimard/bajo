from std.math import clamp
from std.sys import num_performance_cores


@always_inline
def _worker_count(task_count: Int) -> Int:
    """Clamp CPU BVH work to the available performance cores."""
    debug_assert["safe", _use_compiler_assume=True](task_count > 0)
    return clamp(num_performance_cores(), 1, task_count)
