"""Deterministic GPU surface guides with no radiance transport or readback."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from bajo.core.camera import Camera
from bajo.core.color import Color
from bajo.rt.denoise_guides import (
    denoise_guide_ray,
    store_denoise_guide,
    denoise_component,
)
from bajo.rt.gpu.config import GpuRtSceneKind, _GpuRtKernelFormats
from bajo.rt.gpu.views import GpuRtSceneView
from bajo.rt.gpu.spectral_trace import spectral_trace_surface
from bajo.rt.gpu.textures import (
    _sample_gpu_lambertian,
    _sample_gpu_coated_albedo,
    _sample_gpu_image_texture,
)
from bajo.rt.material_types import NO_TEXTURE
from bajo.rt.material_storage import MetalLayout
from bajo.rt.render_types import SamplingConfig
from bajo.postprocess.gpu import GpuDenoiseGuides
from bajo.postprocess.types import GuideLayout
from bajo.postprocess.samples import DenoiseComponent
from bajo.rt.gpu.emission import gpu_emitter
from bajo.rt.texture_math import albedo_from_texture


def _guide_kernel[
    kind: GpuRtSceneKind, Formats: _GpuRtKernelFormats
](
    scene: GpuRtSceneView,
    camera_data: ImmPointer[Float32, ImmutAnyOrigin],
    fields: MutPointer[Float32, MutAnyOrigin],
    surfaces: MutPointer[UInt32, MutAnyOrigin],
    metal_textures: ImmPointer[UInt32, ImmutAnyOrigin],
    width: Int32,
    height: Int32,
):
    var i = global_idx.x
    if i >= Int(width) * Int(height):
        return
    var camera = Camera(ImmSpan(unsafe_ptr=camera_data, length=Camera.STRIDE))
    var ray = denoise_guide_ray(camera, i, Int(width), Int(height))
    var sampling = SamplingConfig(
        0, .INDEPENDENT, 1, 0, 1, UInt32(width), image_height=UInt32(height)
    )
    var hit, primitive = spectral_trace_surface[.NORMALS, kind, Formats](
        scene, ray, sampling, UInt32(i), 0, 1, 0.5
    )
    surfaces[unsafe_offset=i] = GuideLayout.MISS
    # Initialize misses too: guide buffers are reused and publicly inspectable.
    for field in range(GuideLayout.STRIDE):
        fields[unsafe_offset=i * GuideLayout.STRIDE + field] = 0
    if not hit.hit[0]:
        return
    var surface = hit.surface.value[0]
    var texture_descs = scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin]()
    var texture_pixels = scene.texture_pixels.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    var albedo = Color(1)
    if hit.surface.kind() == .LAMBERTIAN:
        albedo = _sample_gpu_lambertian(
            surface,
            hit.uv_u[0],
            hit.uv_v[0],
            scene.lambertians.unsafe_origin_cast[ImmutAnyOrigin](),
            scene.lambertian_texture_indices.unsafe_origin_cast[
                ImmutAnyOrigin
            ](),
            texture_descs,
            texture_pixels,
        )
    elif hit.surface.kind() == .COATED_DIFFUSE:
        albedo = _sample_gpu_coated_albedo(
            surface,
            hit.uv_u[0],
            hit.uv_v[0],
            scene.coated_diffuses.unsafe_origin_cast[ImmutAnyOrigin](),
            scene.coated_diffuse_texture_indices.unsafe_origin_cast[
                ImmutAnyOrigin
            ](),
            texture_descs,
            texture_pixels,
        )
    elif hit.surface.kind() == .METAL:
        var index = Int(hit.surface.index())
        albedo = Color(scene.metals.unsafe_offset(index * MetalLayout.STRIDE))
        var texture = metal_textures[unsafe_offset=index]
        if texture != NO_TEXTURE:
            albedo = albedo_from_texture(
                albedo,
                _sample_gpu_image_texture(
                    texture,
                    hit.uv_u[0],
                    hit.uv_v[0],
                    texture_descs,
                    texture_pixels,
                ),
            )
    surfaces[unsafe_offset=i] = surface
    var emitter = gpu_emitter(scene, primitive, surface)
    fields[
        unsafe_offset=i * GuideLayout.STRIDE + GuideLayout.DIFFUSE
    ] = Float32(
        UInt32(
            denoise_component(
                hit.surface.kind(), emitter.radiance.max_component() > 0
            )
            == DenoiseComponent.DIFFUSE
        )
    )
    store_denoise_guide(
        fields,
        i,
        ray.at(hit.t),
        hit.normal,
        albedo,
        camera,
        Int(height),
        hit.geometric_normal,
    )


def enqueue_gpu_denoise_guides[
    kind: GpuRtSceneKind, Formats: _GpuRtKernelFormats
](
    ctx: DeviceContext,
    scene: GpuRtSceneView,
    camera: DeviceBuffer[.float32],
    metal_textures: DeviceBuffer[.uint32],
    mut guides: GpuDenoiseGuides,
) raises:
    ctx.enqueue_function[_guide_kernel[kind, Formats]](
        scene,
        camera,
        guides.fields,
        guides.surfaces,
        metal_textures,
        Int32(guides.width),
        Int32(guides.height),
        grid_dim=ceildiv(guides.width * guides.height, 128),
        block_dim=128,
    )
