"""Asset-backed Bistro ingestion checks; this does not verify full rendering.

Run from the repository root after downloading the Bistro assets:
    pixi run mojo -I . scripts/check_bistro_load.mojo [assets/bistro]
"""

from std.sys.arg import argv
from std.os.path import join
from std.testing import assert_equal, assert_true
from bajo.parser.pbrt import read_pbrt
from bajo.rt.material_types import NO_TEXTURE, PRIMITIVE_INDEX_BITS
from bajo.bvh.constants import PrimitiveKind


def main() raises:
    var args = argv()
    var directory = String(args[1]) if len(args) > 1 else String(
        "assets/bistro"
    )
    for name in [
        "bistro_cafe.pbrt",
        "bistro_boulangerie.pbrt",
        "bistro_vespa.pbrt",
    ]:
        var scene = read_pbrt(join(directory, name))
        ref data = scene.data
        assert_equal(len(data.triangle_meshes()), 1591)
        assert_equal(len(data.triangle_instances()), 1591)
        assert_equal(len(data.surfaces().normal_maps), 131)
        assert_equal(len(data.surfaces().image_textures), 263)
        assert_equal(len(data.surfaces().optical_spectra), 2)
        var reflectance_materials = 0
        for material in data.surfaces().coated_diffuses:
            if material.conductor_reflectance:
                reflectance_materials += 1
                assert_true(material.texture_index != NO_TEXTURE)
        assert_equal(reflectance_materials, 1)
        assert_equal(len(data.opacity().records), 243)
        assert_equal(len(data.emission().records), 102)
        assert_equal(len(data.surfaces().emissives), 0)
        for light in data.lights().records:
            assert_true(light.surface.kind() != .EMISSIVE)
        var triangles = 0
        for mesh in data.triangle_meshes():
            triangles += len(mesh) // 3
        for entry in data.opacity().records.items():
            assert_equal(entry.value.scale, Float32(1.0))
            assert_true(entry.value.texture_index != NO_TEXTURE)
            assert_equal(entry.value.u_scale, Float32(1.0))
            assert_equal(entry.value.v_scale, Float32(1.0))
            assert_true(
                entry.key >> PRIMITIVE_INDEX_BITS
                == PrimitiveKind.TRIANGLE_INSTANCE.value
            )
        assert_equal(scene.settings.image_width, 1920)
        assert_equal(scene.settings.image_height, 1080)
        assert_equal(scene.settings.samples_per_pixel, 256)
        assert_equal(scene.settings.max_depth, 5)
        assert_equal(scene.settings.film.max_component_value, Float32(20))
        assert_equal(scene.settings.filter.kind, UInt32(1))
        assert_equal(scene.settings.filter.x_radius, Float32(1.5))
        assert_equal(scene.settings.filter.y_radius, Float32(1.5))
        assert_equal(scene.settings.filter.sigma, Float32(0.5))
        assert_true(scene.integrator == .MIS)
        print(
            name,
            "triangles",
            triangles,
            "lights",
            len(data.lights().records),
            "opacity bindings",
            len(data.opacity().records),
        )
