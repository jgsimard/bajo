"""Procedural viewer scenes that expose current light-transport limits."""

from bajo.core import Point3W
from bajo.rt import (
    CPU_SCENE_DEFAULT_CONFIG,
    Color,
    CpuScene,
    CpuSceneConfig,
    SceneBuilder,
    SurfaceId,
)


def _add_box(
    mut builder: SceneBuilder,
    minimum: Point3W,
    maximum: Point3W,
    surface: SurfaceId[1],
):
    var x0 = minimum.x
    var y0 = minimum.y
    var z0 = minimum.z
    var x1 = maximum.x
    var y1 = maximum.y
    var z1 = maximum.z
    builder.add_quad(
        Point3W(x0, y0, z0),
        Point3W(x1, y0, z0),
        Point3W(x1, y0, z1),
        Point3W(x0, y0, z1),
        surface,
    )
    builder.add_quad(
        Point3W(x0, y1, z1),
        Point3W(x1, y1, z1),
        Point3W(x1, y1, z0),
        Point3W(x0, y1, z0),
        surface,
    )
    builder.add_quad(
        Point3W(x0, y0, z1),
        Point3W(x1, y0, z1),
        Point3W(x1, y1, z1),
        Point3W(x0, y1, z1),
        surface,
    )
    builder.add_quad(
        Point3W(x1, y0, z0),
        Point3W(x0, y0, z0),
        Point3W(x0, y1, z0),
        Point3W(x1, y1, z0),
        surface,
    )
    builder.add_quad(
        Point3W(x0, y0, z0),
        Point3W(x0, y0, z1),
        Point3W(x0, y1, z1),
        Point3W(x0, y1, z0),
        surface,
    )
    builder.add_quad(
        Point3W(x1, y0, z1),
        Point3W(x1, y0, z0),
        Point3W(x1, y1, z0),
        Point3W(x1, y1, z1),
        surface,
    )


def _add_room(
    mut builder: SceneBuilder,
    half_width: Float32,
    height: Float32,
    front: Float32,
    back: Float32,
    floor_surface: SurfaceId[1],
    wall_surface: SurfaceId[1],
    left_surface: SurfaceId[1],
    right_surface: SurfaceId[1],
):
    builder.add_quad(
        Point3W(-half_width, 0.0, front),
        Point3W(half_width, 0.0, front),
        Point3W(half_width, 0.0, back),
        Point3W(-half_width, 0.0, back),
        floor_surface,
    )
    builder.add_quad(
        Point3W(-half_width, height, back),
        Point3W(half_width, height, back),
        Point3W(half_width, height, front),
        Point3W(-half_width, height, front),
        wall_surface,
    )
    builder.add_quad(
        Point3W(-half_width, 0.0, back),
        Point3W(half_width, 0.0, back),
        Point3W(half_width, height, back),
        Point3W(-half_width, height, back),
        wall_surface,
    )
    builder.add_quad(
        Point3W(-half_width, 0.0, front),
        Point3W(-half_width, 0.0, back),
        Point3W(-half_width, height, back),
        Point3W(-half_width, height, front),
        left_surface,
    )
    builder.add_quad(
        Point3W(half_width, 0.0, back),
        Point3W(half_width, 0.0, front),
        Point3W(half_width, height, front),
        Point3W(half_width, height, back),
        right_surface,
    )


def make_many_lights_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    config: CpuSceneConfig = CPU_SCENE_DEFAULT_CONFIG,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    """Dense mixed-material field illuminated by 96 small colored lights."""
    var builder = SceneBuilder()
    var neutral = builder.add_lambertian(Color(0.52, 0.54, 0.58))
    var red = builder.add_lambertian(Color(0.68, 0.08, 0.04))
    var green = builder.add_lambertian(Color(0.06, 0.55, 0.16))
    var blue = builder.add_lambertian(Color(0.05, 0.18, 0.70))
    var rough_metal = builder.add_metal(Color(0.78, 0.72, 0.58), 0.28)
    var glossy_metal = builder.add_metal(Color(0.84, 0.87, 0.92), 0.035)
    var glass = builder.add_dielectric(1.52)
    var warm_light = builder.add_emissive(Color(70.0, 18.0, 5.0))
    var cool_light = builder.add_emissive(Color(7.0, 24.0, 75.0))
    var green_light = builder.add_emissive(Color(8.0, 62.0, 15.0))

    _add_room(
        builder,
        8.0,
        6.0,
        3.0,
        -12.0,
        neutral,
        neutral,
        red,
        blue,
    )

    # Small lights are deliberately hard for plain path tracing to discover.
    for row in range(8):
        for column in range(12):
            var light_surface = warm_light.copy()
            var kind = (row + 2 * column) % 3
            if kind == 1:
                light_surface = cool_light.copy()
            elif kind == 2:
                light_surface = green_light.copy()
            builder.add_sphere(
                Point3W(
                    -6.6 + 1.2 * Float32(column),
                    5.62,
                    1.4 - 1.55 * Float32(row),
                ),
                0.075,
                light_surface,
            )

    # 432 objects make traversal/build costs visible without external meshes.
    for row in range(18):
        for column in range(24):
            var surface = neutral.copy()
            var kind = (column + 3 * row) % 7
            if kind == 1:
                surface = red.copy()
            elif kind == 2:
                surface = green.copy()
            elif kind == 3:
                surface = blue.copy()
            elif kind == 4:
                surface = rough_metal.copy()
            elif kind == 5:
                surface = glossy_metal.copy()
            elif kind == 6:
                surface = glass.copy()
            var radius = Float32(0.18)
            if (row + column) % 11 == 0:
                radius = 0.27
            builder.add_sphere(
                Point3W(
                    -6.55 + 0.57 * Float32(column),
                    radius,
                    1.0 - 0.59 * Float32(row),
                ),
                radius,
                surface,
            )

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width].__init__[config](
        scene^
    )


def make_indirect_hall_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    config: CpuSceneConfig = CPU_SCENE_DEFAULT_CONFIG,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    """Alternating baffles hide the emitters behind several diffuse bounces."""
    var builder = SceneBuilder()
    var white = builder.add_lambertian(Color(0.72, 0.72, 0.72))
    var red = builder.add_lambertian(Color(0.72, 0.055, 0.035))
    var green = builder.add_lambertian(Color(0.06, 0.58, 0.12))
    var blue = builder.add_lambertian(Color(0.04, 0.16, 0.68))
    var gold = builder.add_metal(Color(0.82, 0.67, 0.32), 0.16)
    var warm_light = builder.add_emissive(Color(55.0, 18.0, 4.0))
    var cool_light = builder.add_emissive(Color(4.0, 18.0, 55.0))

    _add_room(builder, 5.0, 5.0, 2.0, -18.0, white, white, red, green)

    # These slabs form an alternating labyrinth. The back emitters have no
    # direct line of sight to most of the foreground visible to the camera.
    _add_box(builder, Point3W(-5.0, 0.0, -2.9), Point3W(1.45, 4.55, -2.5), blue)
    _add_box(builder, Point3W(-1.45, 0.0, -6.8), Point3W(5.0, 4.55, -6.4), red)
    _add_box(
        builder, Point3W(-5.0, 0.0, -10.7), Point3W(1.45, 4.55, -10.3), green
    )
    _add_box(
        builder, Point3W(-1.45, 0.0, -14.6), Point3W(5.0, 4.55, -14.2), blue
    )

    for row in range(4):
        var z = -1.0 - 3.9 * Float32(row)
        var x = Float32(3.25)
        if row % 2 == 1:
            x = -3.25
        builder.add_sphere(Point3W(x, 0.72, z), 0.72, gold)
        builder.add_sphere(Point3W(-0.45 * x, 0.42, z - 1.2), 0.42, white)

    # Camera-facing emitters at the end of the hall.
    builder.add_quad(
        Point3W(-4.3, 0.55, -17.92),
        Point3W(-0.35, 0.55, -17.92),
        Point3W(-0.35, 4.45, -17.92),
        Point3W(-4.3, 4.45, -17.92),
        warm_light,
    )
    builder.add_quad(
        Point3W(0.35, 0.55, -17.91),
        Point3W(4.3, 0.55, -17.91),
        Point3W(4.3, 4.45, -17.91),
        Point3W(0.35, 4.45, -17.91),
        cool_light,
    )

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width].__init__[config](
        scene^
    )


def make_specular_transport_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    config: CpuSceneConfig = CPU_SCENE_DEFAULT_CONFIG,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    """Glass shells, glossy reflectors, and tiny emitters stress path sampling.
    """
    var builder = SceneBuilder()
    var white = builder.add_lambertian(Color(0.70, 0.70, 0.70))
    var dark = builder.add_lambertian(Color(0.08, 0.09, 0.11))
    var mirror = builder.add_metal(Color(0.94, 0.95, 0.98), 0.008)
    var brushed = builder.add_metal(Color(0.82, 0.56, 0.22), 0.12)
    var glass = builder.add_dielectric(1.52)
    var diamond = builder.add_dielectric(2.42)
    var point_light = builder.add_emissive(Color(650.0, 500.0, 330.0))
    var strip_light = builder.add_emissive(Color(9.0, 18.0, 48.0))

    _add_room(builder, 6.0, 5.5, 2.5, -12.0, white, dark, mirror, mirror)

    # A bright strip seen through multiple dielectric interfaces.
    builder.add_quad(
        Point3W(-4.8, 1.0, -11.92),
        Point3W(4.8, 1.0, -11.92),
        Point3W(4.8, 4.7, -11.92),
        Point3W(-4.8, 4.7, -11.92),
        strip_light,
    )
    builder.add_sphere(Point3W(0.0, 4.75, -4.8), 0.085, point_light)
    builder.add_sphere(Point3W(-3.9, 3.9, -7.8), 0.07, point_light)
    builder.add_sphere(Point3W(3.8, 3.5, -9.2), 0.06, point_light)

    # Concentric positive/negative radii make hollow dielectric shells.
    builder.add_sphere(Point3W(-2.15, 1.35, -4.6), 1.35, glass)
    builder.add_sphere(Point3W(-2.15, 1.35, -4.6), -1.08, glass)
    builder.add_sphere(Point3W(2.0, 1.55, -6.2), 1.55, diamond)
    builder.add_sphere(Point3W(2.0, 1.55, -6.2), -1.30, diamond)

    for row in range(4):
        for column in range(9):
            var surface = glass.copy()
            if (row + column) % 3 == 1:
                surface = mirror.copy()
            elif (row + column) % 3 == 2:
                surface = brushed.copy()
            var radius = 0.24 + 0.035 * Float32((row + column) % 3)
            builder.add_sphere(
                Point3W(
                    -4.35 + 1.08 * Float32(column),
                    radius,
                    -1.15 - 2.35 * Float32(row),
                ),
                radius,
                surface,
            )

    # Black blockers make the transmitted and reflected light paths obvious.
    _add_box(builder, Point3W(-0.5, 0.0, -2.5), Point3W(0.5, 2.3, -2.1), dark)
    _add_box(builder, Point3W(-4.7, 0.0, -8.9), Point3W(-3.2, 2.8, -8.5), dark)

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width].__init__[config](
        scene^
    )
