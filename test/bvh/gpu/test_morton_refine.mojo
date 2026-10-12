"""Morton refinement preserves segments, stable ties and partial tiles."""
from std.testing import TestSuite, assert_equal, assert_true
from std.math import ceildiv, min
from max.gpu.host import DeviceContext
from bajo.gpu.buffers import upload_list, _device_span
from bajo.bvh.gpu.builder.morton_refine import refine_morton_tiles_kernel
from bajo.core import AABB, Point3f32, SegmentOffsets
from bajo.bvh.gpu.builder import GpuBvhBuildMethod, build_binary_bvh
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBuildWorkspace,
    GpuBinaryBoundsBvh,
)
from bajo.bvh.gpu.diagnostics import validate_binary_bvh
from bajo.core.morton import morton3


def check[
    tile: Int, segmented: Bool
](count: Int, same_code: Bool, same_center: Bool) raises:
    var boxes = List[Float32]()
    var codes = List[UInt32]()
    var ids = List[UInt32]()
    var offsets = List[UInt32]()
    offsets.extend(
        [UInt32(0), UInt32(count // 3), UInt32(count // 3), UInt32(count)]
    )
    for i in range(count):
        var x = Float32((i * 73) % 101) if not same_center else Float32(1)
        var y = Float32((i * 127) % 103) if not same_center else Float32(2)
        var z = Float32((i * 59) % 107) if not same_center else Float32(3)
        boxes.extend([x, y, z, x, y, z])
        codes.append(UInt32(0 if same_code else i // 19))
        ids.append(UInt32((i * 17) % count))
    var expected = ids.copy()
    for first in range(0, count, tile):
        var end = min(first + tile, count)
        var head = first
        while head < end:
            var last = head + 1
            var segment_end = count
            comptime if segmented:
                segment_end = count // 3 if head < count // 3 else count
            while (
                last < end and last < segment_end and codes[last] == codes[head]
            ):
                last += 1
            if last - head >= 8:
                var bounds = AABB[.WORLD].invalid()
                for i in range(head, last):
                    var b = Int(ids[i]) * 6
                    bounds.grow(
                        Point3f32[.WORLD](boxes[b], boxes[b + 1], boxes[b + 2])
                    )
                var inverse = bounds.extent().safe_inv()
                var keys = List[UInt64]()
                for i in range(head, last):
                    var b = Int(ids[i]) * 6
                    var p = Point3f32[.WORLD](
                        boxes[b], boxes[b + 1], boxes[b + 2]
                    )
                    var c = (p - bounds._min) * inverse
                    keys.append(
                        (UInt64(morton3(c.x, c.y, c.z)) << 8)
                        | UInt64(i - first)
                    )
                sort(keys)
                for i in range(len(keys)):
                    expected[head + i] = ids[first + Int(keys[i] & UInt64(255))]
            head = last
    with DeviceContext() as ctx:
        var db = upload_list(ctx, boxes)
        var dc = upload_list(ctx, codes)
        var ds = upload_list(ctx, offsets)
        var di = upload_list(ctx, ids)
        ctx.enqueue_function[refine_morton_tiles_kernel[tile, segmented]](
            _device_span[mut=False](db),
            _device_span[mut=False](dc),
            _device_span[mut=False](ds),
            _device_span[mut=True](di),
            grid_dim=ceildiv(count, tile),
            block_dim=tile,
        )
        ctx.synchronize()
        with di.map_to_host() as actual:
            for i in range(count):
                assert_equal(actual[i], expected[i])
        with dc.map_to_host() as actual:
            for i in range(count):
                assert_equal(actual[i], codes[i])


def test_runs_and_tail128() raises:
    check[128, False](300, False, False)


def test_runs_and_tail256() raises:
    check[256, False](300, False, False)


def test_cluster_crossing_tiles() raises:
    check[256, False](777, True, False)


def test_identical_centroid_stable_ties() raises:
    check[128, False](513, True, True)


def test_segment_boundaries_and_empty_segment() raises:
    check[256, True](777, True, False)


def test_short_tail_and_no_eligible_runs() raises:
    check[128, True](17, False, False)


def check_binary[method: GpuBvhBuildMethod]() raises:
    var boxes = List[Float32]()
    var ids = List[UInt32]()
    for i in range(257):
        var x = Float32(255 - i) * 0.001
        var y = Float32(0)
        var z = Float32(0)
        if i == 256:
            x = 100000
            y = 100000
            z = 100000
        boxes.extend([x, y, z, x + 0.001, y + 0.001, z + 0.001])
        ids.append(UInt32(i))
    with DeviceContext() as ctx:
        var workspace = GpuBinaryBuildWorkspace(ctx, SegmentOffsets.single(257))
        var binary = GpuBinaryBoundsBvh(
            ctx, upload_list(ctx, boxes), upload_list(ctx, ids), workspace
        )
        _ = build_binary_bvh[method](ctx, binary, workspace)
        ctx.synchronize()
        var validation = validate_binary_bvh(
            binary, workspace, binary.root_bounds()
        )
        assert_true(validation.sorted_ok)
        assert_true(validation.values_ok)
        assert_true(validation.topology_ok)
        assert_true(validation.bounds_ok)
        with binary.leaf_ids.map_to_host() as mapped:
            var seen = List[Bool](length=257, fill=False)
            for i in range(257):
                assert_true(mapped[i] < UInt32(257))
                assert_true(not seen[Int(mapped[i])])
                seen[Int(mapped[i])] = True
                assert_equal(mapped[i], UInt32(255 - i if i < 256 else 256))


def test_refined_lbvh_topology_and_bounds() raises:
    check_binary[.LBVH]()


def test_refined_hploc_topology_and_bounds() raises:
    check_binary[.HPLOC]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
