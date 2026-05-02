from std.memory import UnsafePointer
from std.testing import TestSuite, assert_almost_equal, assert_true
from std.utils.numerics import max_finite

from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.cpu.tlas import BvhInstance, Tlas
from bajo.core.bvh.types import Ray
from bajo.core.mat import Mat44f32, transform_point, transform_vector
from bajo.core.vec import Vec3f32, assert_vec_equal


comptime f32_max = max_finite[DType.float32]()


def _translation(tx: Float32, ty: Float32, tz: Float32) -> Mat44f32:
    # Row-major matrix, column-vector transform convention.
    return Mat44f32(
        1.0,
        0.0,
        0.0,
        tx,
        0.0,
        1.0,
        0.0,
        ty,
        0.0,
        0.0,
        1.0,
        tz,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _uniform_scale(s: Float32) -> Mat44f32:
    return Mat44f32(
        s,
        0.0,
        0.0,
        0.0,
        0.0,
        s,
        0.0,
        0.0,
        0.0,
        0.0,
        s,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _rotation_z_90() -> Mat44f32:
    # Maps (x, y, z) -> (-y, x, z).
    return Mat44f32(
        0.0,
        -1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _assert_hit(ray: Ray, inst: UInt32, prim: UInt32, t: Float32) raises:
    assert_true(ray.hit.t < 1.0e20, "expected ray hit")
    assert_true(ray.hit.inst == inst)
    assert_true(ray.hit.prim == prim)
    assert_almost_equal(ray.hit.t, t, atol=1.0e-4)


def _assert_miss(ray: Ray) raises:
    assert_true(ray.hit.t > 1.0e20, "expected ray miss")


def _append_tri(mut verts: List[Vec3f32], cx: Float32, z: Float32):
    verts.append(Vec3f32(cx - 1.0, -1.0, z))
    verts.append(Vec3f32(cx + 1.0, -1.0, z))
    verts.append(Vec3f32(cx, 1.0, z))


def _make_single_triangle_verts(z: Float32 = 2.0) -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=3)
    _append_tri(verts, 0.0, z)
    return verts^


def _make_strip(count: Int, z: Float32 = 2.0) -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=count * 3)
    for i in range(count):
        _append_tri(verts, Float32(i) * 4.0, z)
    return verts^


def _make_instance(
    transform: Mat44f32,
    inv_transform: Mat44f32,
    blas_idx: UInt32,
    bmin: Vec3f32,
    bmax: Vec3f32,
) -> BvhInstance:
    return BvhInstance(transform, inv_transform, blas_idx, bmin, bmax)


def _bounds_contains(
    outer_min: Vec3f32,
    outer_max: Vec3f32,
    inner_min: Vec3f32,
    inner_max: Vec3f32,
) -> Bool:
    return (
        outer_min.x() <= inner_min.x()
        and outer_min.y() <= inner_min.y()
        and outer_min.z() <= inner_min.z()
        and outer_max.x() >= inner_max.x()
        and outer_max.y() >= inner_max.y()
        and outer_max.z() >= inner_max.z()
    )


def _bruteforce_instances(
    mut ray: Ray,
    instances: List[BvhInstance],
    blases: UnsafePointer[BinaryBvh, MutAnyOrigin],
):
    for i in range(len(instances)):
        ref inst = instances[i]
        var local_origin = transform_point(inst.inv_transform, ray.O)
        var local_dir = transform_vector(inst.inv_transform, ray.D)
        var local_ray = Ray(local_origin, local_dir, ray.hit.t)

        blases[Int(inst.blas_idx)].traverse(local_ray)

        if local_ray.hit.t < ray.hit.t:
            ray.hit.t = local_ray.hit.t
            ray.hit.u = local_ray.hit.u
            ray.hit.v = local_ray.hit.v
            ray.hit.prim = local_ray.hit.prim
            ray.hit.inst = UInt32(i)


def test_tlas_instance_bounds_translation() raises:
    var inst = _make_instance(
        _translation(10.0, 2.0, -3.0),
        _translation(-10.0, -2.0, 3.0),
        0,
        Vec3f32(-1.0, -1.0, -1.0),
        Vec3f32(1.0, 1.0, 1.0),
    )

    assert_vec_equal(inst.bounds_min, Vec3f32(9.0, 1.0, -4.0))
    assert_vec_equal(inst.bounds_max, Vec3f32(11.0, 3.0, -2.0))


def test_tlas_instance_bounds_uniform_scale() raises:
    var inst = _make_instance(
        _uniform_scale(2.0),
        _uniform_scale(0.5),
        0,
        Vec3f32(-1.0, -1.0, -1.0),
        Vec3f32(1.0, 1.0, 1.0),
    )

    assert_vec_equal(inst.bounds_min, Vec3f32(-2.0, -2.0, -2.0))
    assert_vec_equal(inst.bounds_max, Vec3f32(2.0, 2.0, 2.0))


def test_tlas_instance_bounds_rotation_z_90() raises:
    var inst = _make_instance(
        _rotation_z_90(),
        _rotation_z_90().transpose(),
        0,
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(2.0, 1.0, 1.0),
    )

    assert_vec_equal(inst.bounds_min, Vec3f32(-1.0, 0.0, 0.0))
    assert_vec_equal(inst.bounds_max, Vec3f32(0.0, 2.0, 1.0))


def test_tlas_build_invariants() raises:
    var verts = _make_single_triangle_verts()
    var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    blas.build["median", False]()

    var instances = List[BvhInstance]()
    for i in range(12):
        var x = Float32(i - 6) * 3.0
        instances.append(
            BvhInstance.from_blas(
                _translation(x, 0.0, 0.0),
                _translation(-x, 0.0, 0.0),
                0,
                blas,
            )
        )

    var tlas = Tlas(instances)
    tlas.build()

    assert_true(tlas.inst_count == 12)
    assert_true(tlas.nodes_used > 1)
    assert_true(Int(tlas.nodes_used) <= len(tlas.tlas_nodes))

    var leaf_sum = UInt32(0)
    for i in range(Int(tlas.nodes_used)):
        ref node = tlas.tlas_nodes[i]
        if node.is_leaf():
            assert_true(node.tri_count > 0)
            assert_true(
                Int(node.left_first + node.tri_count) <= len(tlas.inst_indices)
            )
            leaf_sum += node.tri_count
        else:
            assert_true(node.tri_count == 0)
            assert_true(node.left_first + 1 < tlas.nodes_used)

    assert_true(leaf_sum == tlas.inst_count)

    ref root = tlas.tlas_nodes[0]
    for i in range(len(tlas.instances)):
        ref inst = tlas.instances[i]
        assert_true(
            _bounds_contains(
                root.aabb._min,
                root.aabb._max,
                inst.bounds_min,
                inst.bounds_max,
            )
        )


def test_tlas_single_instance_matches_blas() raises:
    var verts = _make_single_triangle_verts()
    var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    blas.build["median", False]()

    var instances = List[BvhInstance]()
    instances.append(
        BvhInstance.from_blas(Mat44f32.identity(), Mat44f32.identity(), 0, blas)
    )

    var tlas = Tlas(instances)
    tlas.build()

    var blases = List[BinaryBvh]()
    blases.append(blas.copy())

    var blas_ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    blas.traverse(blas_ray)

    var tlas_ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    tlas.traverse(tlas_ray, blases.unsafe_ptr())

    assert_almost_equal(tlas_ray.hit.t, blas_ray.hit.t, atol=1.0e-4)
    assert_true(tlas_ray.hit.prim == blas_ray.hit.prim)
    assert_true(tlas_ray.hit.inst == 0)


def test_tlas_two_translated_instances_report_instance_id() raises:
    var verts = _make_single_triangle_verts()
    var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    blas.build["median", False]()

    var instances = List[BvhInstance]()
    instances.append(
        BvhInstance.from_blas(
            _translation(-10.0, 0.0, 0.0),
            _translation(10.0, 0.0, 0.0),
            0,
            blas,
        )
    )
    instances.append(
        BvhInstance.from_blas(
            _translation(10.0, 0.0, 0.0),
            _translation(-10.0, 0.0, 0.0),
            0,
            blas,
        )
    )

    var tlas = Tlas(instances)
    tlas.build()

    var blases = List[BinaryBvh]()
    blases.append(blas^)

    var ray_right = Ray(Vec3f32(10.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    tlas.traverse(ray_right, blases.unsafe_ptr())
    _assert_hit(ray_right, 1, 0, 2.0)

    var ray_left = Ray(Vec3f32(-10.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    tlas.traverse(ray_left, blases.unsafe_ptr())
    _assert_hit(ray_left, 0, 0, 2.0)


def test_tlas_nearest_instance_wins() raises:
    var verts = _make_single_triangle_verts()
    var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    blas.build["median", False]()

    var instances = List[BvhInstance]()
    instances.append(
        BvhInstance.from_blas(
            _translation(0.0, 0.0, 3.0),
            _translation(0.0, 0.0, -3.0),
            0,
            blas,
        )
    )
    instances.append(
        BvhInstance.from_blas(Mat44f32.identity(), Mat44f32.identity(), 0, blas)
    )

    var tlas = Tlas(instances)
    tlas.build()

    var blases = List[BinaryBvh]()
    blases.append(blas^)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    tlas.traverse(ray, blases.unsafe_ptr())

    _assert_hit(ray, 1, 0, 2.0)


def test_tlas_miss() raises:
    var verts = _make_single_triangle_verts()
    var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    blas.build["median", False]()

    var instances = List[BvhInstance]()
    instances.append(
        BvhInstance.from_blas(Mat44f32.identity(), Mat44f32.identity(), 0, blas)
    )

    var tlas = Tlas(instances)
    tlas.build()

    var blases = List[BinaryBvh]()
    blases.append(blas^)

    var ray = Ray(Vec3f32(10.0, 10.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    tlas.traverse(ray, blases.unsafe_ptr())

    _assert_miss(ray)


def test_tlas_matches_bruteforce_instances() raises:
    var verts = _make_single_triangle_verts()
    var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    blas.build["median", False]()

    var instances = List[BvhInstance]()
    for i in range(8):
        var x = Float32((i % 4) - 2) * 6.0
        var y = Float32(i // 4) * 4.0
        instances.append(
            BvhInstance.from_blas(
                _translation(x, y, 0.0),
                _translation(-x, -y, 0.0),
                0,
                blas,
            )
        )

    var tlas = Tlas(instances)
    tlas.build()

    var blases = List[BinaryBvh]()
    blases.append(blas^)

    for i in range(8):
        var x = Float32((i % 4) - 2) * 6.0
        var y = Float32(i // 4) * 4.0

        var tlas_ray = Ray(Vec3f32(x, y, 0.0), Vec3f32(0.0, 0.0, 1.0))
        tlas.traverse(tlas_ray, blases.unsafe_ptr())

        var brute_ray = Ray(Vec3f32(x, y, 0.0), Vec3f32(0.0, 0.0, 1.0))
        _bruteforce_instances(brute_ray, instances, blases.unsafe_ptr())

        assert_true(tlas_ray.hit.inst == brute_ray.hit.inst)
        assert_true(tlas_ray.hit.prim == brute_ray.hit.prim)
        assert_almost_equal(tlas_ray.hit.t, brute_ray.hit.t, atol=1.0e-4)


# Keep this last so `run_tests.sh` can discover and run the file directly.
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
