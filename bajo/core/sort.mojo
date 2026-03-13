from math import min, max
from memory import Span
from std.builtin.sort import (
    _insertion_sort,
    _sort3,
    insertion_sort_threshold,
    _heap_sort,
    _quicksort_partition_right,
)
from bit import count_leading_zeros
from sys.info import bit_width_of

# https://github.com/modular/modular/blob/main/mojo/stdlib/std/builtin/sort.mojo


@always_inline
def _estimate_max_iters(size: Int) -> Int:
    # Maximum iterations before switching to Heapsort (2 * log2(n))
    var log2: Int = (bit_width_of[DType.int]() - 1) ^ count_leading_zeros(
        size | 1
    )
    return log2 * 2


def nth_element[
    T: Copyable,
    origin: MutOrigin,
    //,
    cmp_fn: fn(T, T) capturing[_] -> Bool,
](mut span: Span[T, origin], n: Int):
    """
    Rearranges elements in the span such that the element at index n
    is in the position it would be in a sorted sequence. All elements
    before are less, all after are more.

    This is an IntroSelect implementation: O(N) average and worst-case.
    """
    var size = len(span)
    if size <= 1 or n < 0 or n >= size:
        return
    k = n

    # If the partition is taking too long, we switch to Heapsort
    # to avoid the O(N^2) worst case of QuickSelect.
    var max_iters = _estimate_max_iters(size)

    while len(span) > insertion_sort_threshold:
        if max_iters == 0:
            # Switch to Heapsort for the remaining span to guarantee O(N log N)
            # for this branch, which maintains the overall O(N) guarantee.
            _heap_sort[cmp_fn](span)
            return

        max_iters -= 1

        # pick pivot using Median-of-3 (same as Quicksort)
        _sort3[T, cmp_fn](span, len(span) >> 1, 0, len(span) - 1)

        # partition
        var pivot_pos = _quicksort_partition_right[cmp_fn](span)

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

    _insertion_sort[cmp_fn](span)
