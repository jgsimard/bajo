"""Geometry-owned GPU emission lookup; scattering keeps its original surface."""

from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.core.color import Color
from bajo.rt.material_types import SurfaceId
from bajo.rt.scene_data import SceneData
from bajo.rt.gpu.views import GpuEmissionView, GpuRtSceneView, _immut
from bajo.rt.gpu.upload import _upload_nonempty


struct GpuEmissionTables:
    var keys_and_flags: DeviceBuffer[.uint32]
    var radiance: DeviceBuffer[.float32]
    var count: Int32

    def __init__(out self, mut ctx: DeviceContext, data: SceneData) raises:
        var keys = List[UInt32]()
        for entry in data.emission().records.items():
            keys.append(entry.key)
        sort(keys)
        if len(keys) > Int(Int32.MAX):
            raise Error("GPU emission table exceeds Int32 indexing")
        var keys_and_flags = List[UInt32](capacity=2 * len(keys))
        var radiance = List[Float32](capacity=3 * len(keys))
        for key in keys:
            ref emitter = data.emission().records[key]
            keys_and_flags.extend([key, UInt32(emitter.two_sided)])
            radiance.extend(
                [emitter.radiance.x, emitter.radiance.y, emitter.radiance.z]
            )
        self.keys_and_flags = _upload_nonempty(ctx, keys_and_flags)
        self.radiance = _upload_nonempty(ctx, radiance)
        self.count = Int32(len(keys))

    def view(self) -> GpuEmissionView:
        return GpuEmissionView(
            _immut(self.keys_and_flags), _immut(self.radiance), self.count
        )


@fieldwise_init
struct GpuAreaEmitter(TrivialRegisterPassable):
    var radiance: Color
    var two_sided: Bool


@always_inline
def gpu_emitter(
    scene: GpuRtSceneView, primitive: UInt32, surface: UInt32
) -> GpuAreaEmitter:
    var begin = 0
    var end = Int(scene.emission.count)
    while begin < end:
        var middle = (begin + end) // 2
        if scene.emission.keys_and_flags[unsafe_offset=2 * middle] < primitive:
            begin = middle + 1
        else:
            end = middle
    if (
        begin < Int(scene.emission.count)
        and scene.emission.keys_and_flags[unsafe_offset=2 * begin] == primitive
    ):
        return GpuAreaEmitter(
            Color(scene.emission.radiance.unsafe_offset(3 * begin)),
            scene.emission.keys_and_flags[unsafe_offset=2 * begin + 1] != 0,
        )
    if SurfaceId.kind_from_raw(surface) == .EMISSIVE:
        return GpuAreaEmitter(
            Color(
                scene.emissives.unsafe_offset(
                    3 * Int(SurfaceId.index_from_raw(surface))
                )
            ),
            False,
        )
    return GpuAreaEmitter(Color(0.0), False)
