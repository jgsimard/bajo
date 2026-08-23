"""Backend-neutral authored scene and render inputs."""

from bajo.bvh import Camera
from bajo.rt.types import Integrator, RenderSettings, SceneData


struct SceneDescription:
    """Complete frontend output from which either backend can prepare.

    `data` is the acceleration/material input, while `camera`, `settings`, and
    `integrator` remain render inputs. Preparing one backend never constructs
    or imports the other backend's owner.
    """

    var data: SceneData
    var camera: Camera
    var settings: RenderSettings
    var integrator: Integrator

    def __init__(
        out self,
        var data: SceneData,
        camera: Camera,
        settings: RenderSettings,
        integrator: Integrator,
    ):
        self.data = data^
        self.camera = camera
        self.settings = settings.copy()
        self.integrator = integrator

    def take_data(deinit self) -> SceneData:
        """Consume the description and return its neutral authoring data."""
        return self.data^
