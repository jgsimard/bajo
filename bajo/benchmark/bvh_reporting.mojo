"""Shared formatting for BVH benchmark results."""

from std.collections import StringDict

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
        var abs_hit_diff = abs(self.hit_diff())

        if self.reference_hit_count > 0:
            return Float64(abs_hit_diff) / Float64(self.reference_hit_count)

        if self.hit_count > 0:
            return 1.0

        return 0.0

    def status(self) -> String:
        comptime CHECKSUM_ABS_EPS = 1.0e-3
        comptime CHECKSUM_PER_HIT_EPS = 1.0e-6
        comptime CHECKSUM_REL_EPS = 1.0e-7

        var per_hit_diff = Float64(0.0)
        if self.hit_count > 0:
            per_hit_diff = self.diff / Float64(self.hit_count)

        var reference_magnitude = abs(self.reference_checksum)
        var relative_checksum_diff = self.diff
        if reference_magnitude > 1.0:
            relative_checksum_diff /= reference_magnitude

        var hit_diff = self.hit_diff()
        var abs_hit_diff = abs(hit_diff)

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
            self.diff > CHECKSUM_ABS_EPS
            and per_hit_diff > CHECKSUM_PER_HIT_EPS
            and relative_checksum_diff > CHECKSUM_REL_EPS
        ):
            return String("CHECK")

        if hit_diff != 0:
            return String("TOL")

        return String("OK")


struct TablePrinter:
    var columns: StringDict[Int]

    def __init__(out self, var **columns: Int):
        self.columns = columns^

    def header(self):
        var header = String()
        var separator = String()
        var first = True

        for column in self.columns.items():
            if not first:
                header += " "
                separator += " "

            if first:
                header += column.key.ascii_ljust(column.value)
            else:
                header += column.key.ascii_rjust(column.value)

            separator += "-" * column.value
            first = False

        print(header)
        print(separator)

    def result_line(self, var **values: String) raises:
        var line = String()
        var first = True

        for column in self.columns.items():
            if not first:
                line += " "

            var value = values[column.key]

            if first:
                line += value.ascii_ljust(column.value)
            else:
                line += value.ascii_rjust(column.value)

            first = False

        print(line)
