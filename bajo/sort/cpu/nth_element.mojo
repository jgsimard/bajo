from std.math import min, max
from std.builtin.sort import (
    _insertion_sort,
    _sort3,
    insertion_sort_threshold,
    _heap_sort,
    _quicksort_partition_right,
)
from std.bit import log2_floor

# https://github.com/modular/modular/blob/main/mojo/stdlib/std/builtin/sort.mojo


def _estimate_max_iters(size: Int) -> Int:
    return 2 * log2_floor(size)


def nth_element[
    T: Copyable,
    origin: MutOrigin,
    //,
    small_threshold: Int = insertion_sort_threshold,
](mut span: Span[T, origin], n: Int, cmp_fn: Some[def(T, T) -> Bool]):
    """
    Rearranges elements in the span such that the element at index n
    is in the position it would be in a sorted sequence. All elements
    before are less, all after are more.

    Introselect has expected O(N) cost, with an O(N log N) worst-case
    heapsort fallback. small_threshold controls the insertion-sort cutoff.
    """
    comptime assert small_threshold >= 2
    var size = len(span)
    if size <= 1 or n < 0 or n >= size:
        return
    var k = n

    # If the partition is taking too long, we switch to Heapsort
    # to avoid the O(N^2) worst case of QuickSelect.
    var max_iters = _estimate_max_iters(size)

    while len(span) > small_threshold:
        if max_iters == 0:
            # Switch to Heapsort for the remaining span to guarantee O(N log N)
            # for this branch, bounding the overall worst case by O(N log N).
            _heap_sort(span, cmp_fn)
            return

        max_iters -= 1

        # pick pivot using Median-of-3 (same as Quicksort)
        _sort3(span, len(span) >> 1, 0, len(span) - 1, cmp_fn)

        # partition
        var pivot_pos = _quicksort_partition_right(span, cmp_fn)

        # narrow
        if pivot_pos == k:
            return
        elif k < pivot_pos:
            # Target is in the left partition
            span = span.unsafe_subspan(offset=0, length=pivot_pos)
        else:
            # Target is in the right partition
            # Adjust span and the relative target index k
            var offset = pivot_pos + 1
            span = span.unsafe_subspan(offset=offset, length=len(span) - offset)
            k -= offset

    _insertion_sort(span, cmp_fn)
