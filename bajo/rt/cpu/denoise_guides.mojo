"""Lazy guide generation using the renderer's existing surface queries."""

from layout import TileTensor, Coord, Idx, row_major
from max.algorithm import parallelize
from std.math import iota, ceildiv
from std.sys import simd_width_of
from bajo.core import Vec3
from bajo.rt.material_types import MaterialKind, SURFACE_INDEX_BITS
from bajo.core.camera import Camera
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.denoise_guides import (
    denoise_guide_ray,
    store_denoise_guide,
)
from bajo.postprocess.types import DenoiseGuides, GuideLayout


def make_cpu_denoise_guides[
    world_width: SIMDLength,
    instance_width: SIMDLength,
    packet_width: SIMDLength = simd_width_of[Float32](),
](
    scene: CpuScene[world_width, instance_width],
    camera: Camera,
    width: Int,
    height: Int,
) raises -> DenoiseGuides:
    if width <= 0 or height <= 0:
        raise Error("denoising dimensions must be positive")
    var guides = DenoiseGuides(width, height)
    var fields = TileTensor(
        ptr=guides.fields.unsafe_ptr(),
        layout=row_major(Coord(width * height, Idx[GuideLayout.STRIDE])),
    )
    var surfaces = guides.surfaces.unsafe_ptr()

    var count = width * height

    def trace_packet(block: Int) {imm}:
        var indices = iota[.int, packet_width]() + block * Int(packet_width)
        var valid = indices.lt(count)
        var safe_indices = valid.select(indices, 0)
        var ray = denoise_guide_ray(camera, safe_indices, width, height)
        var hit = scene.trace_surface(ray, valid)
        var active = valid & hit.hit
        if not active.reduce_or():
            return
        ref materials = scene.scene_data().surfaces()
        var surface = materials.resolve_mix_packet(
            hit.surface,
            SIMD[.float32, packet_width](0.5),
            hit.uv_u,
            hit.uv_v,
            active,
        )
        var normal = materials.shading_normal(
            surface,
            hit.normal,
            hit.uv_u,
            hit.uv_v,
            active,
            hit.dpdu,
            hit.front_face,
        )
        var kinds = surface.value >> SURFACE_INDEX_BITS
        var colored = (
            active
            & kinds.ne(MaterialKind.DIELECTRIC.value())
            & kinds.ne(MaterialKind.EMISSIVE.value())
        )
        var albedo = Vec3.select(
            colored,
            materials.sample_albedo(surface, hit.uv_u, hit.uv_v, colored),
            Vec3[.float32, .WORLD, packet_width](1),
        )
        var diffuse = kinds.eq(
            MaterialKind.LAMBERTIAN.value()
        ) & ~hit.emission.max_component().gt(0)
        for lane in range(Int(packet_width)):
            if active[lane]:
                var i = safe_indices[lane]
                surfaces[unsafe_offset=i] = surface.value[lane]
                fields[i, GuideLayout.DIFFUSE] = diffuse[lane].cast[.float32]()
        store_denoise_guide(
            fields,
            safe_indices,
            ray.at(hit.t),
            normal,
            albedo,
            camera,
            height,
            hit.geometric_normal,
            active,
        )

    parallelize(trace_packet, ceildiv(count, Int(packet_width)))
    return guides^
