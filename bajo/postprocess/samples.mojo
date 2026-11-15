"""Optional complete-sample sidecars; collecting these never changes transport."""

from bajo.core.color import Color


__enum DenoiseComponent(Equatable, ImplicitlyCopyable, Writable):
    case PASSTHROUGH
    case DIFFUSE
    case GLOSSY

    def __init__(out self, value: UInt32):
        """Decode a validated component from complete-sample storage."""
        debug_assert["safe"](value <= Self.GLOSSY.value())
        match Int(value):
            case 0:
                self = Self.PASSTHROUGH
            case 1:
                self = Self.DIFFUSE
            case _:
                self = Self.GLOSSY

    def value(self) -> UInt32:
        return UInt32(self._get_enum_discriminant())


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
        self.components = List[UInt32](
            length=count, fill=DenoiseComponent.PASSTHROUGH.value()
        )
