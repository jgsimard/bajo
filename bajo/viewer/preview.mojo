"""Worker-owned CPU accumulation and optional spatial reconstruction."""
from layout import Coord, Idx, TileTensor, row_major
from bajo.core.camera import Camera
from bajo.core.color import Color, color_to_byte
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.cpu.denoise_guides import make_cpu_denoise_guides
from bajo.postprocess import (
    DenoiseGuides,
    AtrousWorkspace,
    DenoiseStatistics,
    DenoiseSampleBatch,
)
from bajo.postprocess import denoise as denoise_frame


struct CpuPreviewAccumulator:
    var width: Int
    var height: Int
    var generation: UInt64
    var samples: Int
    var batches: Int
    var sums: List[Color]
    var guides: Optional[DenoiseGuides]
    var workspace: AtrousWorkspace
    var statistics: Optional[DenoiseStatistics]
    var statistics_error: String

    def __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.generation = 0
        self.samples = 0
        self.batches = 0
        self.sums = List[Color](length=width * height, fill=Color(0))
        self.guides = {}
        self.workspace = {}
        self.statistics = {}
        self.statistics_error = ""

    def add(
        mut self,
        pixels: ImmSpan[Color, _],
        generation: UInt64,
        samples: Int,
        sample_batch: Optional[DenoiseSampleBatch] = {},
    ) raises:
        if len(pixels) != len(self.sums) or samples <= 0:
            raise Error(
                "CPU preview requires matching pixels and positive samples"
            )
        var reset = self.samples == 0 or self.generation != generation
        if reset:
            self.samples = 0
            self.batches = 0
            self.guides = {}
            self.statistics = {}
            self.statistics_error = ""
        for i in range(len(pixels)):
            var prior = Color(0) if reset else self.sums[i]
            self.sums[i] = prior + pixels[i] * Float32(samples)
        self.samples += samples
        self.batches += 1
        self.generation = generation
        try:
            guard case .Some(ref batch) = sample_batch;
              batch.samples_per_pixel == samples else:
                raise Error("complete camera-sample statistics are unavailable")
            if not self.statistics:
                self.statistics = DenoiseStatistics(self.width, self.height)
            self.statistics.value().add(batch)
        except error:
            self.statistics_error = String(error)

    def present(
        mut self, scene: CpuScene[16, 16], camera: Camera, denoise: Bool
    ) raises -> List[UInt8]:
        if self.samples == 0:
            raise Error("CPU preview has no accumulated samples")
        if denoise:
            if not self.guides:
                self.guides = make_cpu_denoise_guides(
                    scene, camera, self.width, self.height
                )
            guard not self.statistics_error;
              case .Some(ref statistics) = self.statistics;
              statistics.samples == self.samples else:
                raise Error(
                    "denoising requires complete sample statistics: "
                    + self.statistics_error
                )
            denoise_frame(statistics, self.guides.value(), self.workspace)
        var rgba = List[UInt8](length=self.width * self.height * 4, fill=255)
        var output = TileTensor(rgba, row_major(Coord(len(self.sums), Idx[4])))
        for i in range(len(self.sums)):
            var color = self.workspace.ping[i] if denoise else self.sums[
                i
            ] / Float32(self.samples)
            comptime for channel in range(3):
                output[i, channel] = color_to_byte(color[channel])
        return rgba^
