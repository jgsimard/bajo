from bajo.bvh.camera import Camera
from bajo.rt.types import RenderSettings, CpuScene


struct PbrtScene:
    """A parsed PBRT scene ready for Bajo's CPU renderers."""

    var world: CpuScene[]
    var camera: Camera
    var settings: RenderSettings
    var max_depth: Int
    var integrator: String

    def __init__(
        out self,
        var world: CpuScene[],
        camera: Camera,
        settings: RenderSettings,
        max_depth: Int,
        integrator: String,
    ):
        self.world = world^
        self.camera = camera
        self.settings = settings.copy()
        self.max_depth = max_depth
        self.integrator = integrator
