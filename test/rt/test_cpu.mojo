from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
    assert_false,
)

from bajo.bvh.camera import Camera
from bajo.bvh.types import Ray
from bajo.core import Vec3f32, assert_vec_equal, dot, length
from bajo.core.random import Rng
from bajo.rt import (
    Color,
    Material,
    RenderSettings,
    RtSphere,
    World,
    add_material,
    add_sphere,
    render,
)
from bajo.rt.cpu import (
    reflect,
    reflectance,
    scatter,
    scatter_dielectric,
    scatter_lambertian,
    scatter_metal,
)
from bajo.rt.types import HitRecord


def _front_hit() -> HitRecord:
    return HitRecord(
        Vec3f32(0.0, 0.0, -1.0),
        Vec3f32(0.0, 0.0, 1.0),
        UInt32(0),
        1.0,
        True,
    )


def test_reflect_and_reflectance() raises:
    var reflected = reflect(Vec3f32(1.0, -1.0, 0.0), Vec3f32(0.0, 1.0, 0.0))
    assert_vec_equal(reflected, Vec3f32(1.0, 1.0, 0.0))

    assert_almost_equal(reflectance(1.0, 1.5), 0.04, atol=1e-5)


def test_lambertian_scatter_is_explicit() raises:
    var rng = Rng(seed=1, id=0)
    var material = Material.lambertian(Color(0.2, 0.4, 0.8))
    var hit = _front_hit()
    var incoming = Ray(Vec3f32(0.0), Vec3f32(0.0, 0.0, -1.0))

    var scattered = scatter_lambertian(material, hit, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.attenuation, material.albedo)
    assert_true(length(scattered.ray.d) > 0.0)

    var dispatched = scatter(material, incoming, hit, rng)
    assert_true(dispatched.ok)
    assert_vec_equal(dispatched.attenuation, material.albedo)


def test_metal_scatter_can_absorb() raises:
    var rng = Rng(seed=2, id=0)
    var material = Material.metal(Color(0.8, 0.7, 0.6), 0.0)
    var hit = _front_hit()
    var incoming = Ray(Vec3f32(0.0), Vec3f32(0.0, 0.0, -1.0))

    var scattered = scatter_metal(material, incoming, hit, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.attenuation, material.albedo)
    assert_true(dot(scattered.ray.d, hit.normal) > 0.0)

    var back_face_hit = HitRecord(
        hit.p,
        -hit.normal,
        hit.material_id,
        hit.t,
        False,
    )
    var absorbed = scatter_metal(material, incoming, back_face_hit, rng)
    assert_false(absorbed.ok)


def test_dielectric_scatter_is_explicit() raises:
    var rng = Rng(seed=3, id=0)
    var material = Material.dielectric(1.5)
    var hit = _front_hit()
    var incoming = Ray(Vec3f32(0.0), Vec3f32(0.0, 0.0, -1.0))

    var scattered = scatter_dielectric(material, incoming, hit, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.attenuation, Color(1.0))
    assert_true(length(scattered.ray.d) > 0.0)

    var dispatched = scatter(material, incoming, hit, rng)
    assert_true(dispatched.ok)
    assert_vec_equal(dispatched.attenuation, Color(1.0))


def test_world_hit_maps_material_and_normal() raises:
    var materials = List[Material]()
    var spheres = List[RtSphere]()
    var matte = add_material(materials, Material.lambertian(Color(0.5)))
    add_sphere(spheres, Vec3f32(0.0, 0.0, -1.0), 0.5, matte)
    var world = World(spheres^, materials^)

    var hit = (
        world.hit(Ray(Vec3f32(0.0), Vec3f32(0.0, 0.0, -1.0))).value().copy()
    )
    assert_equal(hit.material_id, matte)
    assert_true(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32(0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 0.5)


def test_render_settings_and_tiny_render() raises:
    var settings = RenderSettings(4, 2, 2, 2, UInt64(9))
    assert_equal(settings.image_width, 4)
    assert_equal(settings.image_height, 2)
    assert_equal(settings.rng_seed, 9)

    var materials = List[Material]()
    var spheres = List[RtSphere]()
    var matte = add_material(materials, Material.lambertian(Color(0.5)))
    add_sphere(spheres, Vec3f32(0.0, 0.0, -1.0), 0.5, matte)
    var world = World(spheres^, materials^)
    var camera = Camera.from_vfov(
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, -1.0),
        Vec3f32(0.0, 1.0, 0.0),
        90.0,
    )

    var result = render(settings, camera, world)
    assert_equal(len(result.pixels), 8)
    assert_equal(result.timings.pixel_count, 8)
    assert_equal(result.timings.sample_count, 16)
    assert_equal(result.timings.max_depth, 2)
    assert_true(result.timings.total_ns >= result.timings.render_ns)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
