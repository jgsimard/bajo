from std.testing import (
    TestSuite,
    assert_true,
)

from bajo.core.vec import Vec3f32
from bajo.core.intersect import intersect_ray_aabb
from bajo.core.bvh.tinybvh import BVHNode, _sah, Ray, BVH
from bajo.core.random import Rng
from bajo.core.utils import print_size_of


def test_tiny_bvh_minimal_reproduction() raises:
    comptime TRIANGLE_COUNT = 8192

    var rng = Rng(123, 123)

    var triangles = List[Vec3f32](capacity=TRIANGLE_COUNT * 3)

    for _ in range(TRIANGLE_COUNT):
        var x = rng.f32()
        var y = rng.f32()
        var z = rng.f32()

        for _ in range(3):
            var vx = x + 0.1 * rng.f32()
            var vy = y + 0.1 * rng.f32()
            var vz = z + 0.1 * rng.f32()
            triangles.append(Vec3f32(vx, vy, vz))

    var origin = Vec3f32(0.5, 0.5, -1.0)
    var direction = Vec3f32(0.1, 0.0, 2.0)
    var ray = Ray(origin, direction)

    var bvh = BVH(triangles.unsafe_ptr(), TRIANGLE_COUNT)
    bvh.traverse(ray)

    assert_true(ray.hit.t > 0.0 and ray.hit.t < 2.0)
    print_size_of[BVHNode]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
