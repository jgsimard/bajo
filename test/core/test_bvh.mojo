from std.testing import (
    TestSuite,
    assert_true,
)

from bajo.core.vec import Vec3f32
from bajo.core.intersect import intersect_ray_aabb


def test_bvh_ray_query_inside_outside() raises:
    # https://github.com/NVIDIA/warp/issues/288
    # AABB spanning x=[0.5, 1.0], extending across y and z axes
    lower = Vec3f32(0.5, -1.0, -1.0)
    upper = Vec3f32(1.0, 1.0, 1.0)

    query_dir = Vec3f32(1.0, 0.0, 0.0)
    rcp_dir = 1.0 / query_dir
    t = Float32(0.0)

    # Test Case 1: Outside (x=0.0)
    query_start_outside = Vec3f32(0.0, 0.0, 0.0)
    hit_outside = intersect_ray_aabb(
        query_start_outside, rcp_dir, lower, upper, t
    )
    assert_true(hit_outside, "Ray starting outside failed to hit")

    # Test Case 2: Inside (x=0.75)
    query_start_inside = Vec3f32(0.75, 0.0, 0.0)
    hit_inside = intersect_ray_aabb(
        query_start_inside, rcp_dir, lower, upper, t
    )
    assert_true(hit_inside, "Ray starting inside failed to hit")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
