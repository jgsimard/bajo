"""Shared deterministic guide-ray construction and surface encoding."""

from layout import TileTensor
from bajo.core import Camera, Point3, Vec3, Ray, dot
from bajo.core.color import Color
from bajo.core.numerics import FLOAT32_GAMMA
from bajo.rt.rays import make_camera_ray_from_samples
from bajo.postprocess.types import GuideLayout
from std.math import abs
from bajo.rt.material_types import MaterialKind
from bajo.postprocess.samples import DenoiseComponent


def denoise_guide_ray[
    length: SIMDLength = 1
](camera: Camera, index: SIMD[.int, length], width: Int, height: Int) -> Ray[
    .float32, .WORLD, length
]:
    return make_camera_ray_from_samples(
        camera,
        (index % width).cast[.float32](),
        (index // width).cast[.float32](),
        Float32(width),
        Float32(height),
        Float32(0.5),
        Float32(0.5),
        Float32(0),
        Float32(0),
    )


def store_denoise_guide[
    origin: MutOrigin, //, length: SIMDLength = 1
](
    fields: TileTensor[.float32, _, origin],
    index: SIMD[.int, length],
    position: Point3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    camera: Camera,
    height: Int,
    geometric_normal: Vec3[.float32, .WORLD, length],
    active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
):
    comptime assert fields.flat_rank == 2
    var displacement = position - camera.origin.broadcast[length]()
    var forward = camera.forward.broadcast[length]()
    var depth = abs(dot(displacement, forward))
    var footprint = max(2 * depth * camera.fov_scale / Float32(height), 1e-6)
    var denominator = max(
        abs(dot(geometric_normal, displacement)), Float32(1e-8)
    )
    var derivative = (
        2
        * depth
        * depth
        * camera.fov_scale
        / Float32(height)
        * (
            abs(dot(geometric_normal, camera.right.broadcast[length]()))
            + abs(dot(geometric_normal, camera.up.broadcast[length]()))
        )
        / denominator
    )
    # Bound ray/hit roundoff before applying the reference SVGF weights.
    comptime gamma8 = FLOAT32_GAMMA[8, .float32]
    var magnitude = depth
    comptime for c in range(3):
        magnitude += (abs(position[c]) + abs(camera.origin[c])) * abs(
            camera.forward[c]
        )
    derivative = max(derivative, 2 * gamma8 * magnitude)
    for lane in range(Int(length)):
        if not active[lane]:
            continue
        var pixel = index[lane]
        comptime for channel in range(3):
            fields[pixel, channel] = position[channel][lane]
            fields[pixel, GuideLayout.NORMAL + channel] = normal[channel][lane]
            fields[pixel, GuideLayout.ALBEDO + channel] = albedo[channel][lane]
        fields[pixel, GuideLayout.FOOTPRINT] = footprint[lane]
        fields[pixel, GuideLayout.DEPTH] = depth[lane]
        fields[pixel, GuideLayout.DEPTH_FWIDTH] = derivative[lane]


def denoise_component(
    kind: MaterialKind, emissive: Bool = False
) -> DenoiseComponent:
    if not emissive:
        if kind == .LAMBERTIAN:
            return DenoiseComponent.DIFFUSE
        if kind == .METAL or kind == .COATED_DIFFUSE:
            return DenoiseComponent.GLOSSY
    return DenoiseComponent.PASSTHROUGH
