from std.math import round

from bajo.core.utils import ns_to_ms
from bajo.bvh.gpu.utils import GpuBuildTimings


def gpu_build_other_ns(
    build_ns: Int,
    timings: GpuBuildTimings,
    pack_ns: Int,
) -> Int:
    return build_ns - timings.total() - pack_ns


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


def print_transposed_ms_row(
    name: String,
    v0_ns: Int,
    v1_ns: Int,
    v2_ns: Int,
    value_width: Int,
):
    var c0 = String(t"{name} (ms)").ascii_ljust(15)
    var c1 = String(t"{round(ns_to_ms(v0_ns), 3)}").ascii_rjust(value_width)
    var c2 = String(t"{round(ns_to_ms(v1_ns), 3)}").ascii_rjust(value_width)
    var c3 = String(t"{round(ns_to_ms(v2_ns), 3)}").ascii_rjust(value_width)

    print(t"{c0} {c1} {c2} {c3}")


def print_transposed_f64_row(
    name: String,
    v0: Float64,
    v1: Float64,
    v2: Float64,
    value_width: Int,
    digits: Int,
):
    var c0 = name.ascii_ljust(15)
    var c1 = String(t"{round(v0, digits)}").ascii_rjust(value_width)
    var c2 = String(t"{round(v1, digits)}").ascii_rjust(value_width)
    var c3 = String(t"{round(v2, digits)}").ascii_rjust(value_width)

    print(t"{c0} {c1} {c2} {c3}")


def print_transposed_row[
    T: Writable
](name: String, value_width: Int, *values: T,):
    comptime metric_width = 15
    var row = name.ascii_ljust(metric_width)
    for value in values:
        row += " "
        row += String(t"{value}").ascii_rjust(value_width)
    print(row)


def print_gpu_build_timing_rows(
    build0_ns: Int,
    timings0: GpuBuildTimings,
    pack0_ns: Int,
    build1_ns: Int,
    timings1: GpuBuildTimings,
    pack1_ns: Int,
    build2_ns: Int,
    timings2: GpuBuildTimings,
    pack2_ns: Int,
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
            pack0_ns,
            pack1_ns,
            pack2_ns,
            value_width,
        )

    print_transposed_ms_row(
        String("- other"),
        gpu_build_other_ns(build0_ns, timings0, pack0_ns),
        gpu_build_other_ns(build1_ns, timings1, pack1_ns),
        gpu_build_other_ns(build2_ns, timings2, pack2_ns),
        value_width,
    )
