"""Backend-neutral, owned display results with progressive sample counts."""
from bajo.rt.render_types import RenderTimings


struct PreviewResult:
    var rgba: List[UInt8]
    var timings: RenderTimings
    var samples: Int
    var batches: Int
    var postprocess_error: String
    var preparation_ns: Int
    var batch_ns: Int
    var postprocess_ns: Int

    def __init__(
        out self,
        var rgba: List[UInt8],
        timings: RenderTimings,
        samples: Int,
        batches: Int,
        postprocess_error: String = "",
        preparation_ns: Int = -1,
        batch_ns: Int = -1,
        postprocess_ns: Int = 0,
    ):
        self.rgba = rgba^
        self.timings = timings.copy()
        self.samples = samples
        self.batches = batches
        self.postprocess_error = postprocess_error.copy()
        self.preparation_ns = (
            timings.init_ns if preparation_ns < 0 else preparation_ns
        )
        self.batch_ns = timings.render_ns if batch_ns < 0 else batch_ns
        self.postprocess_ns = postprocess_ns
