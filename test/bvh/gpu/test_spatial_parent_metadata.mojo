"""Device parent publication against the CPU forest reference."""
from std.testing import TestSuite, assert_equal
from std.math import ceildiv
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32, Vec3f32
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.bvh.spatial.parallel_refit import (
    forest_parents,
    forest_parents_kernel,
)
from bajo.bvh.gpu.utils import upload_list, _device_span


def check_parent_publication[block_size: Int, grouped: Bool]() raises:
    with DeviceContext() as ctx:
        var counts: List[Int] = [
            1,
            2,
            3,
            31,
            32,
            33,
            63,
            64,
            65,
            255,
            256,
            257,
            1025,
            1026,
            4097,
        ]
        for count in counts:
            var boxes = List[AABB[.WORLD]]()
            var groups = List[UInt32]()
            for i in range(count):
                var point = Point3f32[.WORLD](
                    Float32(i % 17), Float32((i // 17) % 13), Float32(i // 221)
                )
                boxes.append(AABB(point, point + Vec3f32[.WORLD](0.125)))
                comptime if grouped:
                    # Uneven populations and sparse labels; count=1 is singleton.
                    groups.append(UInt32((i * 7) % 19) * 31 + 5)
            var tree = SpatialBoundsBvh(boxes, groups)
            var expected = forest_parents(tree.nodes)
            var nodes = upload_list(ctx, tree.nodes)
            var parents = ctx.enqueue_create_buffer[.uint32](len(expected))
            with parents.map_to_host() as values:
                for i in range(len(expected)):
                    # A stale root or unwritten child cannot accidentally pass.
                    values[i] = 0x12345678
            ctx.enqueue_function[forest_parents_kernel](
                _device_span[mut=False](nodes),
                _device_span[mut=True](parents),
                grid_dim=ceildiv(len(expected), block_size),
                block_dim=block_size,
            )
            ctx.synchronize()
            with parents.map_to_host() as values:
                for i in range(len(expected)):
                    assert_equal(values[i], expected[i])


def test_parent_links_global_forests_and_block_tails() raises:
    check_parent_publication[32, False]()
    check_parent_publication[256, False]()


def test_parent_links_sparse_groups_singletons_and_many_roots() raises:
    check_parent_publication[32, True]()
    check_parent_publication[256, True]()


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
