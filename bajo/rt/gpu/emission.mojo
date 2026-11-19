"""Geometry-owned GPU emission lookup; scattering keeps its original surface."""

from layout import TileTensor, row_major
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


def gpu_emitter(
    scene: GpuRtSceneView, primitive: UInt32, surface: UInt32
) -> GpuAreaEmitter:
    var records = TileTensor(
        ptr=scene.emission.keys_and_flags,
        layout=row_major(Int(scene.emission.count), 2),
    )
    var radiances = TileTensor(
        ptr=scene.emission.radiance,
        layout=row_major(Int(scene.emission.count), 3),
    )
    var begin = 0
    var end = Int(scene.emission.count)
    while begin < end:
        var middle = (begin + end) // 2
        if records[middle, 0] < primitive:
            begin = middle + 1
        else:
            end = middle
    if begin < Int(scene.emission.count) and records[begin, 0] == primitive:
        return GpuAreaEmitter(
            Color(radiances.ptr_at_offset((begin, 0))),
            records[begin, 1] != 0,
        )
    if SurfaceId.kind_from_raw(surface) == .EMISSIVE:
        var index = Int(SurfaceId.index_from_raw(surface))
        var emissives = TileTensor(
            ptr=scene.emissives, layout=row_major(index + 1, 3)
        )
        return GpuAreaEmitter(
            Color(emissives.ptr_at_offset((index, 0))),
            False,
        )
    return GpuAreaEmitter(Color(0.0), False)
