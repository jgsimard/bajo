"""Shared benchmark timing summaries and small numeric helpers."""


@fieldwise_init
struct TimingSummary(Copyable):
    var median_ns: Int
    var min_ns: Int
    var max_ns: Int


def summarize_timings(mut values: List[Int]) -> TimingSummary:
    debug_assert["safe", _use_compiler_assume=True](
        len(values) > 0, "cannot summarize an empty timing sample"
    )
    sort(Span(values))
    var middle = (len(values) - 1) >> 1
    return TimingSummary(values[middle], values[0], values[len(values) - 1])


def ratio(numerator: Int, denominator: Int) -> Float64:
    if denominator == 0:
        return 0.0
    return Float64(numerator) / Float64(denominator)
