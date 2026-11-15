"""Shared host/device contracts for edge-avoiding à-trous filtering."""

from std.math import isfinite


struct GuideLayout:
    """Thirteen Float32 words per pixel; UInt32.MAX marks a background surface.
    """

    comptime STRIDE = 13
    comptime POSITION = 0
    comptime NORMAL = 3
    comptime ALBEDO = 6
    comptime FOOTPRINT = 9
    comptime DEPTH = 10
    comptime DEPTH_FWIDTH = 11
    comptime DIFFUSE = 12
    comptime MISS = UInt32.MAX


struct AtrousSettings(Copyable):
    var passes: Int
    var phi_color: Float32
    var phi_normal: Float32
    var firefly: Bool

    def __init__(
        out self,
        passes: Int = 4,
        phi_color: Float32 = 10,
        phi_normal: Float32 = 128,
        firefly: Bool = True,
    ):
        self.passes = passes
        self.phi_color = phi_color
        self.phi_normal = phi_normal
        self.firefly = firefly

    def validate(self) raises:
        if self.passes < 0 or self.passes > 8:
            raise Error("à-trous passes must be between 0 and 8")
        for value in [self.phi_color, self.phi_normal]:
            if not isfinite(value) or value <= 0:
                raise Error("SVGF weights must be finite and positive")


struct DenoiseGuides:
    """Owned surface guides, independent of scene and rendering backend."""

    var width: Int
    var height: Int
    var fields: List[Float32]
    var surfaces: List[UInt32]

    def __init__(out self, width: Int = 0, height: Int = 0) raises:
        if width < 0 or height < 0:
            raise Error("denoising dimensions cannot be negative")
        self.width = width
        self.height = height
        self.fields = List[Float32](
            length=width * height * GuideLayout.STRIDE, fill=0
        )
        self.surfaces = List[UInt32](
            length=width * height, fill=GuideLayout.MISS
        )

    def validate(self, width: Int, height: Int) raises:
        if (
            width <= 0
            or height <= 0
            or self.width != width
            or self.height != height
            or len(self.fields) != width * height * GuideLayout.STRIDE
            or len(self.surfaces) != width * height
        ):
            raise Error(
                "denoising requires matching image and guide dimensions"
            )
