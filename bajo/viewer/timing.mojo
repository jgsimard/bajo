"""Active-work accounting and sample/batch-aware viewer completion estimates."""
from std.math import round


def format_duration(ns: Int) -> String:
    if ns < 0:
        return String("--")
    var seconds = Float64(ns) / 1_000_000_000
    if seconds < 1:
        return String(t"{round(seconds * 1000, 2)} ms")
    if seconds < 60:
        return String(t"{round(seconds, 2)} s")
    var whole = Int(seconds)
    if whole < 3600:
        return String(t"{whole // 60}m {whole % 60}s")
    return String(t"{whole // 3600}h {(whole % 3600) // 60}m")


struct ViewerTimings:
    """Retain scene costs; reset rendering costs with the sample accumulation.

    Total means active measured work, including an in-flight request, rather
    than time spent idle in the window. Expected total includes sunk scene
    costs, remaining samples and one postprocess operation per remaining batch.
    """

    var asset_ns: Int
    var preparation_ns: Int
    var render_work_ns: Int
    var batch_ns: Int
    var postprocess_ns: Int
    var traced_ns: Int
    var traced_samples: Int
    var request_start_ns: Int
    var pending: Bool

    def __init__(out self, asset_ns: Int = 0, preparation_ns: Int = 0):
        self.asset_ns = asset_ns
        self.preparation_ns = preparation_ns
        self.render_work_ns = 0
        self.batch_ns = 0
        self.postprocess_ns = 0
        self.traced_ns = 0
        self.traced_samples = 0
        self.request_start_ns = 0
        self.pending = False

    def reset_render(mut self):
        self.render_work_ns = 0
        self.batch_ns = 0
        self.postprocess_ns = 0
        self.traced_ns = 0
        self.traced_samples = 0
        self.pending = False

    def begin_request(mut self, now_ns: Int):
        self.request_start_ns = now_ns
        self.pending = True

    def finish_request(
        mut self,
        preparation_ns: Int,
        batch_ns: Int,
        postprocess_ns: Int,
        work_ns: Int,
        samples: Int,
        current_generation: Bool = True,
    ):
        self.pending = False
        self.preparation_ns += max(preparation_ns, 0)
        if not current_generation:
            return
        self.render_work_ns += max(work_ns - preparation_ns, 0)
        self.postprocess_ns = max(postprocess_ns, 0)
        if samples > 0:
            self.batch_ns = max(batch_ns, 0)
            self.traced_ns += self.batch_ns
            self.traced_samples += samples

    def total_ns(self, now_ns: Int) -> Int:
        var total = self.asset_ns + self.preparation_ns + self.render_work_ns
        if self.pending:
            total += max(now_ns - self.request_start_ns, 0)
        return total

    def expected_total_ns(
        self, now_ns: Int, completed: Int, maximum: Int, batch_size: Int
    ) -> Int:
        var remaining = max(maximum - completed, 0)
        if remaining == 0 and completed > 0:
            return self.total_ns(now_ns)
        if self.traced_samples == 0 or self.traced_ns <= 0:
            return -1
        var size = remaining if batch_size == -1 else max(batch_size, 1)
        var batches = (remaining + size - 1) // size if remaining else 0
        var future = Int(
            Float64(self.traced_ns)
            * Float64(remaining)
            / Float64(self.traced_samples)
        )
        future += batches * self.postprocess_ns
        var expected = (
            self.asset_ns + self.preparation_ns + self.render_work_ns + future
        )
        return max(expected, self.total_ns(now_ns))

    def lines(
        self,
        width: Int,
        now_ns: Int,
        completed: Int,
        maximum: Int,
        batch_size: Int,
    ) -> List[String]:
        var estimate = self.expected_total_ns(
            now_ns, completed, maximum, batch_size
        )
        var labels: List[String] = [
            String(t"Load {format_duration(self.asset_ns)}"),
            String(t"Prep {format_duration(self.preparation_ns)}"),
            String(t"Batch {format_duration(self.batch_ns)}"),
            String(t"Total {format_duration(self.total_ns(now_ns))}"),
            String(t"Est. total {format_duration(estimate)}"),
            String(t"Post {format_duration(self.postprocess_ns)}"),
        ]
        var per_line = 6 if width >= 840 else (3 if width >= 450 else 2)
        var result = List[String]()
        for first in range(0, 6, per_line):
            var line = labels[first].copy()
            for index in range(first + 1, min(first + per_line, 6)):
                line += " | " + labels[index]
            result.append(line^)
        return result^
