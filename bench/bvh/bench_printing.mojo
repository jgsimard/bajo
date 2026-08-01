from std.math import round

from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.bvh.gpu.utils import GpuBuildTimings


@fieldwise_init
struct GpuBenchResult(Copyable, Writable):
    var label: String
    var build_ns: Int
    var timings: GpuBuildTimings
    var kernel_ns: Int
    var ray_count: Int
    var checksum: Float64
    var diff: Float64
    var hit_count: UInt32
    var reference_checksum: Float64
    var reference_hit_count: UInt32
    var hit_rel_eps: Float64

    def hit_diff(self) -> Int:
        return Int(self.hit_count) - Int(self.reference_hit_count)

    def rel_hit_diff(self) -> Float64:
        var hit_diff = self.hit_diff()
        var abs_hit_diff = hit_diff
        if abs_hit_diff < 0:
            abs_hit_diff = 0 - abs_hit_diff

        if self.reference_hit_count > 0:
            return Float64(abs_hit_diff) / Float64(self.reference_hit_count)

        if self.hit_count > 0:
            return 1.0

        return 0.0

    def status(self) -> String:
        comptime CHECKSUM_ABS_EPS = 1.0e-3
        comptime CHECKSUM_PER_HIT_EPS = 1.0e-6

        var per_hit_diff = Float64(0.0)
        if self.hit_count > 0:
            per_hit_diff = self.diff / Float64(self.hit_count)

        var hit_diff = Int(self.hit_count) - Int(self.reference_hit_count)
        var abs_hit_diff = hit_diff
        if abs_hit_diff < 0:
            abs_hit_diff = 0 - abs_hit_diff

        var hit_rel_diff = Float64(0.0)
        if self.reference_hit_count > 0:
            hit_rel_diff = Float64(abs_hit_diff) / Float64(
                self.reference_hit_count
            )
        else:
            if self.hit_count > 0:
                hit_rel_diff = 1.0

        if hit_rel_diff > self.hit_rel_eps:
            return String("CHECK")

        if hit_diff == 0 and (
            self.diff > CHECKSUM_ABS_EPS and per_hit_diff > CHECKSUM_PER_HIT_EPS
        ):
            return String("CHECK")

        return String("OK")


def _dashes(width: SIMDLength) -> String:
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
    value_width: Int,
    v0_ns: Int,
    v1_ns: Int,
    v2_ns: Int,
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
    value_width: Int,
):
    print_transposed_ms_row(
        String("build tot"),
        value_width,
        build0_ns,
        build1_ns,
        build2_ns,
    )
    print_transposed_ms_row(
        String("- morton"),
        value_width,
        timings0.morton_ns,
        timings1.morton_ns,
        timings2.morton_ns,
    )
    print_transposed_ms_row(
        String("- sort"),
        value_width,
        timings0.sort_ns,
        timings1.sort_ns,
        timings2.sort_ns,
    )
    print_transposed_ms_row(
        String("- topology"),
        value_width,
        timings0.topology_ns,
        timings1.topology_ns,
        timings2.topology_ns,
    )
    print_transposed_ms_row(
        String("- refit"),
        value_width,
        timings0.refit_ns,
        timings1.refit_ns,
        timings2.refit_ns,
    )
    print_transposed_ms_row(
        String("- collapse"),
        value_width,
        timings0.collapse_ns,
        timings1.collapse_ns,
        timings2.collapse_ns,
    )

    print_transposed_ms_row(
        String("- pack"),
        value_width,
        timings0.leaf_pack_ns,
        timings1.leaf_pack_ns,
        timings2.leaf_pack_ns,
    )

    print_transposed_ms_row(
        String("- other"),
        value_width,
        build0_ns - timings0.total(),
        build1_ns - timings1.total(),
        build2_ns - timings2.total(),
    )


def _print_gpu_result_trace_rows(
    row0: GpuBenchResult,
    row1: GpuBenchResult,
    row2: GpuBenchResult,
    value_width: Int,
):
    print_gpu_build_timing_rows(
        row0.build_ns,
        row0.timings,
        row1.build_ns,
        row1.timings,
        row2.build_ns,
        row2.timings,
        value_width,
    )

    print_transposed_ms_row(
        String("camera"),
        value_width,
        row0.kernel_ns,
        row1.kernel_ns,
        row2.kernel_ns,
    )
    print_transposed_row(
        String("MRay/s"),
        value_width,
        round(ns_to_mrays_per_s(row0.kernel_ns, row0.ray_count), 3),
        round(ns_to_mrays_per_s(row1.kernel_ns, row1.ray_count), 3),
        round(ns_to_mrays_per_s(row2.kernel_ns, row2.ray_count), 3),
    )
    print_transposed_row(
        String("hits"),
        value_width,
        row0.hit_count,
        row1.hit_count,
        row2.hit_count,
    )
    print_transposed_row(
        String("checksum"),
        value_width,
        round(row0.checksum, 3),
        round(row1.checksum, 3),
        round(row2.checksum, 3),
    )


def _print_gpu_result_validation_rows(
    row0: GpuBenchResult,
    row1: GpuBenchResult,
    row2: GpuBenchResult,
    value_width: Int,
):
    print_transposed_row(
        String("diff"),
        value_width,
        round(row0.diff, 6),
        round(row1.diff, 6),
        round(row2.diff, 6),
    )
    print_transposed_row(
        String("dhit"),
        value_width,
        row0.hit_diff(),
        row1.hit_diff(),
        row2.hit_diff(),
    )
    print_transposed_row(
        String("rel_dhit"),
        value_width,
        round(row0.rel_hit_diff(), 6),
        round(row1.rel_hit_diff(), 6),
        round(row2.rel_hit_diff(), 6),
    )
    print_transposed_row(
        String("status"),
        value_width,
        row0.status(),
        row1.status(),
        row2.status(),
    )
