"""Lazy guide generation using the renderer's existing surface queries."""

from max.algorithm import parallelize
from bajo.core.camera import Camera
from bajo.core.color import Color
from bajo.rt.material_types import MaterialKind
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.denoise_guides import denoise_guide_ray, store_denoise_guide
from bajo.postprocess.types import DenoiseGuides, GuideLayout


def make_cpu_denoise_guides[
    world_width: SIMDLength, instance_width: SIMDLength
](
    scene: CpuScene[world_width, instance_width],
    camera: Camera,
    width: Int,
    height: Int,
) raises -> DenoiseGuides:
    if width <= 0 or height <= 0:
        raise Error("denoising dimensions must be positive")
    var guides = DenoiseGuides(width, height)
    var fields = guides.fields.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var surfaces = guides.surfaces.unsafe_ptr()

    def trace_pixel(i: Int) {imm}:
        var ray = denoise_guide_ray(camera, i, width, height)
        var hit = scene.trace_surface(ray)
        if not hit.hit[0]:
            return
        ref materials = scene.scene_data().surfaces()
        var surface = materials.resolve_mix(
            hit.surface, 0.5, hit.uv_u[0], hit.uv_v[0]
        )
        var normal = materials.shading_normal(
            surface,
            hit.normal,
            hit.uv_u,
            hit.uv_v,
            dpdu=hit.dpdu,
            front_face=hit.front_face,
        )
        var albedo = Color(1)
        if surface.kind() != .DIELECTRIC and surface.kind() != .EMISSIVE:
            albedo = materials.sample_albedo(surface, hit.uv_u, hit.uv_v)
        surfaces[unsafe_offset=i] = surface.value[0]
        fields[
            unsafe_offset=i * GuideLayout.STRIDE + GuideLayout.DIFFUSE
        ] = Float32(
            UInt32(
                surface.kind() == .LAMBERTIAN
                and not (hit.emission.max_component()[0] > 0)
            )
        )
        store_denoise_guide(
            fields,
            i,
            ray.at(hit.t),
            normal,
            albedo,
            camera,
            height,
            hit.geometric_normal,
        )

    parallelize(trace_pixel, width * height)
    return guides^
