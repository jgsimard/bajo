"""Camera, render settings, and scene ownership for native viewers."""

from std.math import cos, sin
from bajo.core.camera import Camera
from bajo.bvh.cpu import CpuBvhBuildMethod
from bajo.core import Point3W, Vec3W
from bajo.core.utils import degrees_to_radians
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.material_types import Integrator
from bajo.rt.render_types import RenderSettings
from bajo.rt.scene_data import SceneData
from bajo.rt.scene_description import SceneDescription


def viewer_camera(
    origin: Point3W,
    yaw_degrees: Float32,
    pitch_degrees: Float32,
    vfov: Float32,
) -> Camera:
    var yaw = degrees_to_radians(yaw_degrees)
    var pitch = degrees_to_radians(pitch_degrees)
    var cos_pitch = cos(pitch)
    var forward = Vec3W(
        sin(yaw) * cos_pitch,
        sin(pitch),
        -cos(yaw) * cos_pitch,
    )
    return Camera.from_vfov(
        origin,
        origin + forward,
        Vec3W(0.0, 1.0, 0.0),
        vfov,
        10.0,
        0.0,
    )


struct ViewerState(Copyable, Writable):
    """The camera and render inputs shared by native presentation backends."""

    var camera: Camera
    var settings: RenderSettings
    var integrator: Integrator

    def __init__(
        out self,
        camera: Camera,
        settings: RenderSettings,
        integrator: Integrator = .PATH,
    ):
        self.camera = camera
        self.settings = settings.copy()
        self.integrator = integrator


struct ViewerScene[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
]:
    """Own one scene snapshot, with optional lazy CPU acceleration.

    Raster upload and GPU preparation can borrow scene_data() before or after
    CPU preparation. Preparing the CPU backend moves the existing snapshot;
    it never clones geometry. Camera and settings changes do not rebuild it.
    """

    var state: ViewerState
    var _data: Optional[SceneData]
    var _cpu: Optional[CpuScene[Self.world_bvh_width, Self.instance_bvh_width]]
    var _cpu_build_method: Optional[CpuBvhBuildMethod]

    def __init__(out self, var description: SceneDescription):
        self.state = ViewerState(
            description.camera, description.settings, description.integrator
        )
        self._data = description^.take_data()
        self._cpu = Optional[
            CpuScene[Self.world_bvh_width, Self.instance_bvh_width]
        ]()
        self._cpu_build_method = Optional[CpuBvhBuildMethod]()

    def scene_data(
        self,
    ) -> ref[self._data.value(), self._cpu.value().scene_data()] SceneData:
        """Borrow the same immutable geometry regardless of preparation state.
        """
        if self._cpu:
            return self._cpu.value().scene_data()
        return self._data.value()

    def has_cpu_scene(self) -> Bool:
        return Bool(self._cpu)

    def prepare_cpu[build_method: CpuBvhBuildMethod = .SAH](mut self) raises:
        """Prepare once; changing the builder requires a newly loaded scene."""
        if self._cpu:
            if self._cpu_build_method.value() != build_method:
                raise Error(
                    "CPU scene is already prepared with another builder"
                )
            return
        self._cpu = CpuScene[
            Self.world_bvh_width, Self.instance_bvh_width
        ].__init__[build_method](self._data.take())
        self._cpu_build_method = build_method

    def cpu_scene(
        self,
    ) raises -> ref[self._cpu.value()] CpuScene[
        Self.world_bvh_width, Self.instance_bvh_width
    ]:
        """Borrow prepared acceleration without transferring ownership."""
        if not self._cpu:
            raise Error("CPU scene has not been prepared")
        return self._cpu.value()
