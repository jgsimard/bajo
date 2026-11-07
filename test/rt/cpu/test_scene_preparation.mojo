"""SIMD preparation preserves scalar geometry, ordering and rejection."""
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.utils.numerics import isfinite
from bajo.core import AABB, Affine3f32, Point3f32, Vec3f32
from bajo.core.geometry import triangle_area, triangle_is_valid
from bajo.rt import Color, SceneBuilder
from bajo.rt.lighting_types import LightStore, LightRecord, _light_importance
from bajo.rt.material_types import SurfaceId, PrimitiveId, NO_TEXTURE
from bajo.rt.emission import AreaEmitter
from bajo.rt.opacity import Opacity
from bajo.bvh.constants import PrimitiveKind
from bajo.rt.scene_preparation import (
    _validate_mesh_bounds,
    _append_instance_lights,
)


def _vertices(count: Int, pattern: Int) -> List[Point3f32[.LOCAL]]:
    var result = List[Point3f32[.LOCAL]]()
    for i in range(count):
        for corner in range(3):
            var z = bitcast[.float32](
                UInt32(0x80000000 if (i + corner) % 2 else 0)
                | UInt32((i + corner) % 7 if pattern == 1 else 0)
            )
            if pattern == 2:
                z = Float32((i + corner) % 7) * 0.03125
            elif pattern == 3:
                z = Float32(-1e-20) * Float32(i + corner + 1)
            result.append(
                Point3f32[.LOCAL](
                    Float32(i) * 0.125 + Float32(1 if corner == 1 else 0),
                    Float32(1 if corner == 2 else 0),
                    z,
                )
            )
    return result^


def _scalar_bounds(
    vertices: ImmSpan[Point3f32[.LOCAL], _]
) raises -> AABB[.LOCAL]:
    var result = AABB[.LOCAL].invalid()
    for i in range(len(vertices) // 3):
        if not triangle_is_valid(
            vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]
        ):
            raise Error(
                "triangle mesh vertices must be finite and non-degenerate"
            )
        result.grow(vertices[3 * i])
        result.grow(vertices[3 * i + 1])
        result.grow(vertices[3 * i + 2])
    return result


def _check_bounds(vertices: ImmSpan[Point3f32[.LOCAL], _]) raises:
    var old = AABB[.LOCAL].invalid()
    var new = AABB[.LOCAL].invalid()
    var old_error = String("")
    var new_error = String("")
    try:
        old = _scalar_bounds(vertices)
    except error:
        old_error = String(error)
    try:
        new = _validate_mesh_bounds(vertices)
    except error:
        new_error = String(error)
    assert_equal(old_error, new_error)
    if old_error == "":
        comptime for axis in range(3):
            assert_equal(
                bitcast[.uint32](old._min[axis]),
                bitcast[.uint32](new._min[axis]),
            )
            assert_equal(
                bitcast[.uint32](old._max[axis]),
                bitcast[.uint32](new._max[axis]),
            )


def test_mesh_validation_preserves_bounds_bits_at_simd_tails() raises:
    for count in range(1, 66):
        for pattern in range(4):
            _check_bounds(_vertices(count, pattern))


def test_mesh_validation_rejects_bad_lanes_and_preserves_error_priority() raises:
    var vertices = _vertices(35, 2)
    for bad in range(len(vertices)):
        var saved = vertices[bad]
        for bits in [UInt32(0x7FC00000), 0x7F800000, 0xFF800000, 0x7F7FFFFF]:
            comptime for axis in range(3):
                vertices[bad] = saved
                vertices[bad][axis] = bitcast[.float32](bits)
                _check_bounds(vertices)
        vertices[bad] = saved
    for bad in range(35):
        var saved = vertices[3 * bad + 1]
        vertices[3 * bad + 1] = vertices[3 * bad]
        _check_bounds(vertices)
        vertices[3 * bad + 1] = saved
    # Attribute validation must still precede geometric validation.
    var builder = SceneBuilder()
    var surface = builder.add_lambertian(Color(0.5))
    vertices[0].x = bitcast[.float32](UInt32(0x7FC00000))
    var mesh = builder.add_owned_triangle_mesh(
        vertices^, [Float32(0)], List[Float32]()
    )
    builder.add_triangle_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].identity(),
        AABB[.LOCAL].invalid(),
        surface,
    )
    var message = String("")
    try:
        _ = builder^.finish()
    except error:
        message = String(error)
    assert_equal(
        message, String("triangle mesh normals must contain xyz per vertex")
    )


def _scalar_lights(
    vertices: ImmSpan[Point3f32[.LOCAL], _],
    transform: Affine3f32[.LOCAL, .WORLD],
    radiance: Color,
    mut output: LightStore,
) raises:
    var surface = SurfaceId(.EMISSIVE, UInt32(0))
    var reverse = transform.reverses_orientation()[0]
    for i in range(len(vertices) // 3):
        var p0 = transform.point(vertices[3 * i])
        var p1 = transform.point(vertices[3 * i + 1])
        var p2 = transform.point(vertices[3 * i + 2])
        if reverse:
            swap(p1, p2)
        var weight = triangle_area(p0, p1, p2) * _light_importance(radiance)
        if not isfinite(weight):
            raise Error("triangle instance light weight must be finite")
        if weight > 0:
            output.append(
                LightRecord.triangle(
                    PrimitiveId(PrimitiveKind.TRIANGLE_INSTANCE, UInt32(3)),
                    surface,
                    weight,
                    p0,
                    p1,
                    p2,
                )
            )
            if not isfinite(output.total_weight):
                raise Error("total light weight must be finite")


def _check_lights(
    vertices: ImmSpan[Point3f32[.LOCAL], _],
    transform: Affine3f32[.LOCAL, .WORLD],
    radiance: Color,
    prefix: Float32,
) raises:
    var old = LightStore()
    var new = LightStore()
    var surface = SurfaceId(.EMISSIVE, UInt32(0))
    var initial = LightRecord.sphere(
        PrimitiveId(PrimitiveKind.SPHERE, UInt32(7)),
        surface,
        prefix,
        Point3f32[.WORLD](0),
        1,
    )
    old.append(initial.copy())
    new.append(initial.copy())
    var old_error = String("")
    var new_error = String("")
    try:
        _scalar_lights(vertices, transform, radiance, old)
    except error:
        old_error = String(error)
    try:
        _append_instance_lights(vertices, transform, surface, 3, radiance, new)
    except error:
        new_error = String(error)
    assert_equal(old_error, new_error)
    assert_equal(
        bitcast[.uint32](old.total_weight), bitcast[.uint32](new.total_weight)
    )
    assert_equal(len(old.records), len(new.records))
    for i in range(len(old.records)):
        ref a = old.records[i]
        ref b = new.records[i]
        assert_equal(a.primitive.value, b.primitive.value)
        assert_equal(a.surface.value, b.surface.value)
        assert_equal(bitcast[.uint32](a.weight), bitcast[.uint32](b.weight))
        comptime for axis in range(3):
            assert_equal(
                bitcast[.uint32](a.p0[axis]), bitcast[.uint32](b.p0[axis])
            )
            assert_equal(
                bitcast[.uint32](a.p1[axis]), bitcast[.uint32](b.p1[axis])
            )
            assert_equal(
                bitcast[.uint32](a.p2[axis]), bitcast[.uint32](b.p2[axis])
            )
    if old_error == "":
        old.build_alias_table()
        new.build_alias_table()
        assert_equal(old.alias_indices, new.alias_indices)
        for i in range(len(old.records)):
            assert_equal(
                bitcast[.uint32](old.alias_probabilities[i]),
                bitcast[.uint32](new.alias_probabilities[i]),
            )


def test_emissive_triangle_packets_preserve_records_aliases_and_failures() raises:
    for count in range(1, 34):
        for pattern in range(3):
            var vertices = _vertices(count, pattern)
            for reflected in [False, True]:
                var transform = Affine3f32[.LOCAL, .WORLD].from_scale(
                    Vec3f32[.LOCAL](
                        Float32(-1.125 if reflected else 2.125), 3.25, 1.5
                    )
                )
                transform.tx = 0.17
                transform.m01 = 0.125
                _check_lights(vertices, transform, Color(2, 3, 4), 0.03125)
    var vertices = _vertices(35, 2)
    var identity = Affine3f32[.LOCAL, .WORLD].identity()
    # A zero-area world-space emitter is omitted; finite geometry can also
    # overflow either its per-light weight or the accumulated scene weight.
    _check_lights(vertices, Affine3f32[.LOCAL, .WORLD](0), Color(2), 1)
    _check_lights(vertices, identity, Color(1e38), Float32(3.4e38))
    _check_lights(vertices, identity, Color(3.4e38), 1)
    for bad in range(35):
        var saved = vertices[3 * bad]
        vertices[3 * bad].x = bitcast[.float32](UInt32(0x7F7FFFFF))
        _check_lights(vertices, identity, Color(2, 3, 4), 1)
        vertices[3 * bad] = saved


def test_instance_light_preparation_retains_uvs_emission_and_opacity_on_both_paths() raises:
    # Two large meshes cross main's parallel threshold; the small case uses
    # its serial SIMD path. Neither path may drop independent emission or UVs.
    for triangles in [3, 2053]:
        var builder = SceneBuilder()
        var surface = builder.add_lambertian(Color(0.7))
        var texcoords = List[Float32]()
        for _ in range(triangles):
            texcoords.extend([Float32(0.125), 0.25, 0.75, 0.25, 0.5, 0.875])
        var mesh = builder.add_owned_triangle_mesh(
            _vertices(triangles, 2), List[Float32](), texcoords^
        )
        for instance in range(2):
            var transform = Affine3f32[.LOCAL, .WORLD].from_scale(
                Vec3f32[.LOCAL](Float32(-1 if instance else 2), 3, 1)
            )
            builder.add_triangle_instance(
                mesh, transform, AABB[.LOCAL].invalid(), surface
            )
            var key = PrimitiveId(.TRIANGLE_INSTANCE, UInt32(instance)).value
            builder.emission.records[key] = AreaEmitter(
                Color(Float32(2 + instance), 3, 4), instance == 1
            )
            builder.opacity.records[key] = Opacity(0.5, NO_TEXTURE, 1, 1)
        var data = builder^.finish()
        var parallel_records = data._prepare_instance_light_records()
        assert_equal(len(data.lights().records), 2 * triangles)
        assert_equal(len(parallel_records), 2 * triangles)
        for instance in range(2):
            ref transform = data.triangle_instances()[instance].transform
            var radiance = Color(Float32(2 + instance), 3, 4)
            for triangle in range(triangles):
                var base = 3 * triangle
                ref vertices = data.triangle_meshes()[Int(mesh)]
                var p0 = transform.point(vertices[base])
                var p1 = transform.point(vertices[base + 1])
                var p2 = transform.point(vertices[base + 2])
                if instance == 1:
                    swap(p1, p2)
                var expected_weight = (
                    triangle_area(p0, p1, p2)
                    * _light_importance(radiance)
                    * Float32(2 if instance else 1)
                )
                for path in range(2):
                    var record = (
                        data.lights()
                        .records[instance * triangles + triangle]
                        .copy() if path
                        == 0 else parallel_records[
                            instance * triangles + triangle
                        ].copy()
                    )
                    assert_equal(
                        record.primitive.value,
                        PrimitiveId(.TRIANGLE_INSTANCE, UInt32(instance)).value,
                    )
                    assert_equal(record.flags, UInt32(instance))
                    assert_almost_equal(
                        record.weight, expected_weight, atol=1e-5
                    )
                    comptime for axis in range(3):
                        assert_equal(record.radiance[axis], radiance[axis])
                        assert_equal(record.p0[axis], p0[axis])
                        assert_equal(record.p1[axis], p1[axis])
                        assert_equal(record.p2[axis], p2[axis])
                    assert_equal(record.uv0_u, Float32(0.125))
                    assert_equal(record.uv0_v, Float32(0.25))
                    assert_equal(
                        record.uv1_u, Float32(0.5 if instance else 0.75)
                    )
                    assert_equal(
                        record.uv1_v, Float32(0.875 if instance else 0.25)
                    )
                    assert_equal(
                        record.uv2_u, Float32(0.75 if instance else 0.5)
                    )
                    assert_equal(
                        record.uv2_v, Float32(0.25 if instance else 0.875)
                    )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
