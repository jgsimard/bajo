from std.math import round

from bajo.core.utils import ns_to_ms
from bajo.bvh.gpu.utils import GpuBuildTimings


def gpu_build_other_ns(
    build_ns: Int,
    timings: GpuBuildTimings,
) -> Int:
    return build_ns - timings.total()


def _dashes(width: Int) -> String:
    var out = String()
    for _ in range(width):
        out += "-"
    return out^


def print_transposed_header(
    value_width: Int,
    *labels: String,
):
    comptime metric_width = 15

    var header = String("metric").ascii_ljust(metric_width)
    var rule = _dashes(metric_width)

    for label in labels:
        header += " "
        header += label.ascii_rjust(value_width)

        rule += " "
        rule += _dashes(value_width)

    print(header)
    print(rule)


def print_transposed_row[
    T: Writable
](name: String, value_width: Int, *values: T):
    comptime metric_width = 15
    var row = name.ascii_ljust(metric_width)
    for value in values:
        row += " "
        row += String(t"{value}").ascii_rjust(value_width)
    print(row)


def print_transposed_ms_row(
    name: String,
    v0_ns: Int,
    v1_ns: Int,
    v2_ns: Int,
    value_width: Int,
):
    print_transposed_row(
        String(t"{name} (ms)"),
        value_width,
        round(ns_to_ms(v0_ns), 3),
        round(ns_to_ms(v1_ns), 3),
        round(ns_to_ms(v2_ns), 3),
    )


def print_gpu_build_timing_rows(
    build0_ns: Int,
    timings0: GpuBuildTimings,
    build1_ns: Int,
    timings1: GpuBuildTimings,
    build2_ns: Int,
    timings2: GpuBuildTimings,
    include_pack: Bool,
    value_width: Int,
):
    print_transposed_ms_row(
        String("build tot"),
        build0_ns,
        build1_ns,
        build2_ns,
        value_width,
    )
    print_transposed_ms_row(
        String("- morton"),
        timings0.morton_ns,
        timings1.morton_ns,
        timings2.morton_ns,
        value_width,
    )
    print_transposed_ms_row(
        String("- sort"),
        timings0.sort_ns,
        timings1.sort_ns,
        timings2.sort_ns,
        value_width,
    )
    print_transposed_ms_row(
        String("- topology"),
        timings0.topology_ns,
        timings1.topology_ns,
        timings2.topology_ns,
        value_width,
    )
    print_transposed_ms_row(
        String("- refit"),
        timings0.refit_ns,
        timings1.refit_ns,
        timings2.refit_ns,
        value_width,
    )
    print_transposed_ms_row(
        String("- collapse"),
        timings0.collapse_ns,
        timings1.collapse_ns,
        timings2.collapse_ns,
        value_width,
    )

    if include_pack:
        print_transposed_ms_row(
            String("- pack"),
            timings0.leaf_pack_ns,
            timings1.leaf_pack_ns,
            timings2.leaf_pack_ns,
            value_width,
        )

    print_transposed_ms_row(
        String("- other"),
        gpu_build_other_ns(build0_ns, timings0),
        gpu_build_other_ns(build1_ns, timings1),
        gpu_build_other_ns(build2_ns, timings2),
        value_width,
    )
