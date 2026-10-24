"""Progressive sample ranges, independent of the rendering backend."""

from std.math import min, max


def samples_for_batch(
    samples_per_batch: Int,
    completed: Int,
    total: Int,
    moving: Bool = False,
    moving_samples: Int = 1,
) -> Int:
    var remaining = max(0, total - completed)
    if samples_per_batch == -1:
        return remaining
    var size = samples_per_batch
    if moving:
        size = min(size, moving_samples)
    return min(size, remaining)


def smaller_batch(samples_per_batch: Int, total: Int) -> Int:
    return max(
        1, (total if samples_per_batch == -1 else samples_per_batch) // 2
    )


def larger_batch(samples_per_batch: Int, total: Int) -> Int:
    if samples_per_batch == -1 or samples_per_batch >= total:
        return -1
    return min(total, samples_per_batch * 2)
