"""Instance attribute interpolation across packet widths and inactive lanes."""

from std.testing import TestSuite, assert_almost_equal, assert_equal
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Point3f32, Point3, Ray, Rayf32, Vec3, Vec3f32
from bajo.rt import Color, CpuScene, SceneBuilder
from bajo.rt.scene_data import SceneData
from bajo.rt.cpu.scene import _triangle_instance_shading


def make_scene() raises -> SceneData:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    var bounds = compute_bounds(
        [
            Point3f32[.LOCAL](0.0, 0.0, 0.0),
            Point3f32[.LOCAL](3.0, 1.0, 0.0),
        ]
    )
    for mesh in range(4):
        var vertices = List[Point3f32[.LOCAL]]()
        var normals = List[Float32]()
        var uvs = List[Float32]()
        for triangle in range(2):
            for corner in range(3):
                vertices.append(
                    Point3f32[.LOCAL](
                        Float32(triangle * 2 + Int(corner == 1)),
                        Float32(Int(corner == 2)),
                        0.0,
                    )
                )
                if mesh & 1:
                    normals.extend(
                        [
                            Float32(corner) * 0.1,
                            Float32(2 - corner) * 0.2,
                            Float32(1.0) if triangle == 0 else Float32(-1.0),
                        ]
                    )
                if mesh & 2:
                    uvs.extend(
                        [
                            Float32(corner) * 0.3,
                            Float32(triangle + corner) * 0.4,
                        ]
                    )
        var transform = Affine3f32[.LOCAL, .WORLD].identity()
        transform.m00 = -1.5 if mesh % 2 == 0 else 0.7
        transform.m01 = 0.2
        transform.m11 = 1.3
        transform.m20 = 0.15
        transform.m22 = 0.8
        transform.tx = Float32(mesh) * 5.0
        _ = builder.add_owned_triangle_mesh_instance(
            vertices^, normals^, uvs^, transform, bounds, matte
        )
    # Same attribute buffers, different inverse transform.
    var transform = Affine3f32[.LOCAL, .WORLD].identity()
    transform.m00 = 0.9
    transform.m11 = 1.8
    transform.tx = 25.0
    builder.add_triangle_instance(UInt32(3), transform, bounds, matte)
    return builder^.finish()


def check_packets[length: SIMDLength](data: SceneData) raises:
    for pattern in range(7):
        var instances = SIMD[.uint32, length](UInt32.MAX)
        var triangles = instances
        var u = SIMD[.float32, length](0.0)
        var v = SIMD[.float32, length](0.3)
        var normal = Vec3[.float32, .WORLD, length](0.0, 0.0, 1.0)
        var active = SIMD[.bool, length](fill=True)
        comptime for lane in range(length):
            active[lane] = (
                pattern != 4
                and (pattern != 2 or lane % 3 == 1)
                and (pattern != 3 or lane == Int(length) - 1)
            )
            u[lane] = 0.2 + Float32(lane) * 0.01
            normal.z[lane] = -1.0 if lane % 2 else 1.0
            if active[lane]:
                instances[lane] = UInt32(lane % 5) if pattern == 1 else UInt32(
                    4 if pattern
                    == 2 else 1 if pattern
                    == 3 else 0 if pattern
                    == 5 else 2 if pattern
                    == 6 else 3
                )
                triangles[lane] = UInt32(lane % 2)
        var actual = _triangle_instance_shading(
            data, instances, triangles, u, v, normal, active
        )
        comptime for lane in range(length):
            if not active[lane]:
                assert_equal(actual[0].x[lane], normal.x[lane])
                assert_equal(actual[0].y[lane], normal.y[lane])
                assert_equal(actual[0].z[lane], normal.z[lane])
                assert_equal(actual[1][lane], Float32(0.0))
                assert_equal(actual[2][lane], Float32(0.0))
                continue
            var expected = _triangle_instance_shading(
                data,
                instances[lane],
                triangles[lane],
                u[lane],
                v[lane],
                Vec3[.float32, .WORLD](
                    normal.x[lane], normal.y[lane], normal.z[lane]
                ),
            )
            comptime for axis in range(3):
                assert_almost_equal(
                    actual[0][axis][lane], expected[0][axis], atol=1e-6
                )
            assert_almost_equal(actual[1][lane], expected[1], atol=1e-6)
            assert_almost_equal(actual[2][lane], expected[2], atol=1e-6)


def test_instance_shading_shared_and_mixed_instances_masks_and_attributes() raises:
    var data = make_scene()
    comptime for width in [1, 4, 8, 16]:
        check_packets[width](data)


def check_surface_packets[length: SIMDLength](world: CpuScene[16, 16]) raises:
    var rays = Ray[.float32, .WORLD, length](
        Point3[.float32, .WORLD, length](0.0),
        Vec3[.float32, .WORLD, length](0.0, 0.0, -1.0),
    )
    var valid = SIMD[.bool, length](fill=True)
    comptime for lane in range(length):
        ref instance = world.scene_data().triangle_instances()[lane % 5]
        var target = instance.transform.point(Point3f32[.LOCAL](0.2, 0.3, 0.0))
        rays.o.x[lane] = target.x
        rays.o.y[lane] = target.y
        rays.o.z[lane] = target.z + 3.0
        valid[lane] = lane % 3 != 0
    var actual = world.trace_surface(rays, valid)
    comptime for lane in range(length):
        if not valid[lane]:
            assert_equal(actual.hit[lane], False)
            continue
        var expected = world.trace_surface(
            Rayf32[.WORLD](
                Point3f32[.WORLD](
                    rays.o.x[lane], rays.o.y[lane], rays.o.z[lane]
                ),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
        assert_equal(actual.hit[lane], expected.hit)
        assert_equal(actual.surface.value[lane], expected.surface.value)
        assert_equal(actual.front_face[lane], expected.front_face)
        assert_almost_equal(actual.t[lane], expected.t, atol=1e-6)
        assert_almost_equal(actual.uv_u[lane], expected.uv_u, atol=1e-6)
        assert_almost_equal(actual.uv_v[lane], expected.uv_v, atol=1e-6)
        comptime for axis in range(3):
            assert_almost_equal(
                actual.normal[axis][lane], expected.normal[axis], atol=1e-6
            )
            assert_almost_equal(
                actual.geometric_normal[axis][lane],
                expected.geometric_normal[axis],
                atol=1e-6,
            )


def test_instance_surface_packets_match_scalar_with_mixed_mesh_attributes() raises:
    var world = CpuScene[16, 16](make_scene())
    comptime for width in [4, 8, 16]:
        check_surface_packets[width](world)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
