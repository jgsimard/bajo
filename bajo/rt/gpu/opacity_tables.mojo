"""Compact scalar image and primitive-opacity tables for GPU hit filtering."""

from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.rt.gpu.upload import _upload_nonempty
from bajo.rt.gpu.views import GpuOpacityView, _immut
from bajo.rt.material_types import NO_TEXTURE
from bajo.rt.scene_data import SceneData


struct GpuOpacityTables:
    var keys: DeviceBuffer[.uint32]
    var descriptors: DeviceBuffer[.uint32]
    var parameters: DeviceBuffer[.float32]
    var pixels: DeviceBuffer[.float32]
    var count: Int32

    def __init__(out self, mut ctx: DeviceContext, data: SceneData) raises:
        var keys = List[UInt32]()
        for entry in data.opacity().records.items():
            keys.append(entry.key)
        sort(keys)
        var descriptors = List[UInt32](capacity=3 * len(keys))
        var parameters = List[Float32](capacity=3 * len(keys))
        var pixels = List[Float32]()
        var image_offsets = Dict[UInt32, UInt32]()
        for key in keys:
            ref opacity = data.opacity().records[key]
            var offset = NO_TEXTURE
            var width = UInt32(1)
            var height = UInt32(1)
            if opacity.texture_index != NO_TEXTURE:
                ref image = data.surfaces().image_textures[
                    Int(opacity.texture_index)
                ]
                width = UInt32(image.width)
                height = UInt32(image.height)
                if case .Some(stored) = image_offsets.get(
                    opacity.texture_index
                ):
                    offset = stored
                else:
                    if len(pixels) + image.width * image.height >= Int(
                        UInt32.MAX
                    ):
                        raise Error(
                            "GPU scalar opacity images exceed 32-bit offsets"
                        )
                    offset = UInt32(len(pixels))
                    image_offsets[opacity.texture_index] = offset
                    if image.has_scalar_alpha:
                        pixels.extend(ImmSpan(image.alpha))
                    else:
                        for pixel in range(image.width * image.height):
                            var base = 3 * pixel
                            pixels.append(
                                (
                                    image.pixels[base]
                                    + image.pixels[base + 1]
                                    + image.pixels[base + 2]
                                )
                                / 3.0
                            )
            descriptors.extend([offset, width, height])
            parameters.extend([opacity.scale, opacity.u_scale, opacity.v_scale])
        self.count = Int32(len(keys))
        self.keys = _upload_nonempty(ctx, keys)
        self.descriptors = _upload_nonempty(ctx, descriptors)
        self.parameters = _upload_nonempty(ctx, parameters)
        self.pixels = _upload_nonempty(ctx, pixels)

    def view(self) -> GpuOpacityView:
        return GpuOpacityView(
            _immut(self.keys),
            _immut(self.descriptors),
            _immut(self.parameters),
            _immut(self.pixels),
            self.count,
        )
