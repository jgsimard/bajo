"""Optional complete-sample sidecars; collecting these never changes transport."""

from bajo.core.color import Color


struct DenoiseComponent:
    comptime PASSTHROUGH = UInt32(0)
    comptime DIFFUSE = UInt32(1)
    comptime GLOSSY = UInt32(2)


struct DenoiseSampleBatch:
    var pixel_count: Int
    var samples_per_pixel: Int
    var colors: List[Color]
    var weights: List[Float32]
    var components: List[UInt32]

    def __init__(out self, pixel_count: Int, samples_per_pixel: Int):
        self.pixel_count = pixel_count
        self.samples_per_pixel = samples_per_pixel
        var count = pixel_count * samples_per_pixel
        self.colors = List[Color](length=count, fill=Color(0))
        self.weights = List[Float32](length=count, fill=1)
        self.components = List[UInt32](length=count, fill=0)
