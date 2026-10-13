"""Absorption transport and textured material parity for CPU/GPU adapters."""

from std.math import exp, sqrt
from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.bvh import Camera
from bajo.core import Point3f32, Vec3f32, dot
from bajo.core.random import Sampler
from bajo.rt import CpuScene, render_depth_first, render_wavefront
from bajo.rt.render_types import Color, RenderSettings
from bajo.rt.scene_data import SceneBuilder
from bajo.rt.media import SceneMedia, medium_transmittance, transmitted_medium
from bajo.rt.material_types import ImageTexture, NO_TEXTURE
from bajo.rt.material_storage import pack_coated_materials
from bajo.rt.shading import _sample_dielectric
from bajo.rt.gpu.materials import (
    _flatten_coated_diffuse_texture_indices,
    _flatten_texture_descs,
    _flatten_texture_pixels,
    _flatten_dielectrics,
)
from bajo.rt.gpu.textures import (
    _sample_gpu_coated_parameters,
    _gpu_dielectric_shading_normal,
)
from bajo.rt.gpu.material_layout import GpuDielectricLayout
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    WavePathFloatAbi,
    store_device_wave_path,
    load_device_wave_path,
)
from bajo.rt.gpu.wavefront_contract import (
    store_gpu_rt_path_values,
    load_gpu_rt_path,
)


def test_beer_lambert_and_boundary_selection() raises:
    var sigma = Color(0.01, 0.2, 0.5)
    var attenuation = medium_transmittance(sigma, 2)
    assert_almost_equal(attenuation.x[0], exp(Float32(-0.02)))
    assert_almost_equal(attenuation.y[0], exp(Float32(-0.4)))
    assert_almost_equal(attenuation.z[0], exp(Float32(-1)))
    assert_equal(medium_transmittance(Color(0), Float32(1e30)).x[0], Float32(1))
    var boundary = UInt32(1 | (2 << 16))
    assert_equal(transmitted_medium(boundary, True), UInt32(1))
    assert_equal(transmitted_medium(boundary, False), UInt32(2))
    # Total internal reflection must stay on the incident side.
    var sampled = _sample_dielectric(
        Vec3f32[.WORLD](0.95, 0, -0.3),
        Vec3f32[.WORLD](0, 0, 1),
        Float32(1.5),
        False,
        Float32(0.9),
    )
    assert_true(dot(sampled.direction, Vec3f32[.WORLD](0, 0, 1)) > 0)


def test_absorbing_sphere_packet_compaction_and_depth_first() raises:
    var builder = SceneBuilder()
    builder.set_environment(.uniform(Color(1)))
    var surface = builder.add_dielectric(1)
    builder.add_sphere(Point3f32[.WORLD](0), 1, surface)
    builder.media.absorption.append(Color(0.01, 0.2, 0.5))
    var medium = UInt32(1)
    SceneMedia.set_boundary(builder.media.spheres, 0, medium)
    var world = CpuScene[4, 4](builder^.finish())
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0, 0, 3),
        Point3f32[.WORLD](0),
        Vec3f32[.WORLD](0, 1, 0),
        0.01,
    )
    var settings = RenderSettings(3, 2, 17, 2026, 8, .HALTON)
    var scalar = render_depth_first[.PATH](settings, camera, world)
    var packet = render_wavefront[.PATH, 4](settings, camera, world)
    var expected = medium_transmittance(Color(0.01, 0.2, 0.5), 2)
    for i in range(len(scalar.pixels)):
        assert_almost_equal(packet.pixels[i].x[0], expected.x[0], atol=2e-4)
        assert_almost_equal(packet.pixels[i].y[0], expected.y[0], atol=2e-4)
        assert_almost_equal(packet.pixels[i].z[0], expected.z[0], atol=2e-4)
        assert_almost_equal(
            packet.pixels[i].x[0], scalar.pixels[i].x[0], atol=1e-5
        )
        assert_almost_equal(
            packet.pixels[i].y[0], scalar.pixels[i].y[0], atol=1e-5
        )
        assert_almost_equal(
            packet.pixels[i].z[0], scalar.pixels[i].z[0], atol=1e-5
        )


def test_reflective_absorbing_sphere_energy() raises:
    var builder = SceneBuilder()
    builder.set_environment(.uniform(Color(1)))
    var surface = builder.add_dielectric(3)
    builder.add_sphere(Point3f32[.WORLD](0), 1, surface)
    builder.media.absorption.append(Color(0.5))
    var medium = UInt32(1)
    SceneMedia.set_boundary(builder.media.spheres, 0, medium)
    var world = CpuScene[4, 4](builder^.finish())
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0, 0, 3),
        Point3f32[.WORLD](0),
        Vec3f32[.WORLD](0, 1, 0),
        0.01,
    )
    var settings = RenderSettings(1, 1, 8192, 2026, 100, .HALTON)
    var image = render_wavefront[.PATH, 4](settings, camera, world)
    var t = exp(Float32(-1))
    var expected = Float32(0.25) + Float32(0.75 * 0.75) * t / (
        1 - Float32(0.25) * t
    )
    assert_almost_equal(image.pixels[0].x[0], expected, atol=0.01)


def test_gpu_texture_and_bump_adapters_match_cpu() raises:
    var builder = SceneBuilder()
    var texture = builder.surfaces.add_image_texture(
        ImageTexture(2, 1, [0, 0, 0, 1, 1, 1])
    )
    var conductor = builder.surfaces.add_conductor(Color(0.2), Color(3), 0)
    ref coat = builder.surfaces.coated_diffuses[0]
    coat.roughness_texture_index = texture
    coat.roughness_scale = 0.1
    var glass = builder.add_dielectric(3)
    ref dielectric = builder.surfaces.dielectrics[0]
    dielectric.displacement_texture_index = texture
    dielectric.displacement_scale = 0.25
    builder.add_sphere(Point3f32[.WORLD](0), 1, glass)
    var world = builder^.finish()
    var records = pack_coated_materials[False](world.surfaces())
    var indices = _flatten_coated_diffuse_texture_indices(world)
    var descs = _flatten_texture_descs(world)
    var pixels = _flatten_texture_pixels(world)
    var data = _flatten_dielectrics(world)
    for u in [0.1, 0.6]:
        var gpu = _sample_gpu_coated_parameters(
            records.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            0,
            Float32(u),
            0.5,
            indices.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            descs.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            pixels.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
        )
        var cpu = world.surfaces().sample_conductor_roughness(
            conductor, Float32(u), 0.5
        )
        assert_almost_equal(gpu.conductor_roughness, cpu)
    var normal = Vec3f32[.WORLD](0, 0, 1)
    var cpu = world.surfaces().shading_normal(glass, normal, 0.25, 0.5)
    var gpu = _gpu_dielectric_shading_normal(
        glass.value[0],
        normal,
        0.25,
        0.5,
        data.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        descs.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        pixels.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
    )
    assert_almost_equal(cpu.x[0], gpu.x[0])
    assert_almost_equal(cpu.y[0], gpu.y[0])
    assert_almost_equal(cpu.z[0], gpu.z[0])
    assert_equal(
        bitcast[.uint32](data[GpuDielectricLayout.DISPLACEMENT_TEXTURE]),
        texture,
    )


def test_gpu_medium_queue_roundtrip() raises:
    var ids = List[UInt32](length=2, fill=0)
    var fields = List[Float32](length=2 * WavePathFloatAbi.PLANES, fill=0)
    var path = DeviceWavePath(
        1, 0, 0, 0, 0.001, 0, 0, -1, 100, 0.8, 0.7, 0.6, 0.25, False, 2
    )
    store_device_wave_path(
        path,
        ids.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        fields.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        2,
        0,
    )
    var loaded = load_device_wave_path(
        ids.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        fields.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        2,
        0,
    )
    assert_equal(loaded.medium_id, UInt32(2))
    store_gpu_rt_path_values[.MIS](
        path,
        ids.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        fields.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        2,
        1,
    )
    loaded = load_gpu_rt_path[.MIS](
        ids.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        fields.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        2,
        1,
    )
    assert_equal(loaded.medium_id, UInt32(2))
    assert_equal(loaded.tx, Float32(0.8))
    assert_equal(loaded.delta, False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
