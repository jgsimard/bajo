"""Shared deterministic guide-ray construction and surface encoding."""

from bajo.core import Camera, Point3f32, Vec3f32, Rayf32, dot
from bajo.core.color import Color
from bajo.core.numerics import FLOAT32_GAMMA
from bajo.rt.rays import make_camera_ray_from_samples
from bajo.postprocess.types import GuideLayout
from std.math import abs
from bajo.rt.material_types import MaterialKind
from bajo.postprocess.samples import DenoiseComponent


def denoise_guide_ray(
    camera: Camera, index: Int, width: Int, height: Int
) -> Rayf32[.WORLD]:
    return make_camera_ray_from_samples(
        camera,
        Float32(index % width),
        Float32(index // width),
        Float32(width),
        Float32(height),
        Float32(0.5),
        Float32(0.5),
        Float32(0),
        Float32(0),
    )


def store_denoise_guide(
    fields: MutPointer[Float32, _],
    index: Int,
    position: Point3f32[.WORLD],
    normal: Vec3f32[.WORLD],
    albedo: Color,
    camera: Camera,
    height: Int,
    geometric_normal: Vec3f32[.WORLD],
):
    var first = index * GuideLayout.STRIDE
    comptime for channel in range(3):
        fields[unsafe_offset=first + channel] = position[channel]
        fields[unsafe_offset=first + GuideLayout.NORMAL + channel] = normal[
            channel
        ]
        fields[unsafe_offset=first + GuideLayout.ALBEDO + channel] = albedo[
            channel
        ]
    fields[unsafe_offset=first + GuideLayout.FOOTPRINT] = max(
        2
        * abs(dot(position - camera.origin, camera.forward))
        * camera.fov_scale
        / Float32(height),
        1e-6,
    )

    var depth = abs(dot(position - camera.origin, camera.forward))
    fields[unsafe_offset=first + GuideLayout.DEPTH] = depth
    # Analytic planar depth fwidth, in raster pixels, as required by SVGF.
    var denominator = max(
        abs(dot(geometric_normal, position - camera.origin)), Float32(1e-8)
    )
    fields[unsafe_offset=first + GuideLayout.DEPTH_FWIDTH] = (
        2
        * depth
        * depth
        * camera.fov_scale
        / Float32(height)
        * (
            abs(dot(geometric_normal, camera.right))
            + abs(dot(geometric_normal, camera.up))
        )
        / denominator
    )
    # SVGF's raster G-buffer does not have ray/triangle hit-point roundoff.
    # Bound that Float32 uncertainty before passing a depth derivative to the
    # unchanged reference weights. Otherwise a frontoparallel plane with zero
    # analytic derivative rejects neighbors differing by only one depth ULP.
    comptime gamma8 = FLOAT32_GAMMA[8, .float32]
    var magnitude = depth
    comptime for c in range(3):
        magnitude += (abs(position[c]) + abs(camera.origin[c])) * abs(
            camera.forward[c]
        )
    fields[unsafe_offset=first + GuideLayout.DEPTH_FWIDTH] = max(
        fields[unsafe_offset=first + GuideLayout.DEPTH_FWIDTH],
        2 * gamma8 * magnitude,
    )


def denoise_component(kind: MaterialKind, emissive: Bool = False) -> UInt32:
    if not emissive:
        if kind == .LAMBERTIAN:
            return DenoiseComponent.DIFFUSE
        if kind == .METAL or kind == .COATED_DIFFUSE:
            return DenoiseComponent.GLOSSY
    return DenoiseComponent.PASSTHROUGH
