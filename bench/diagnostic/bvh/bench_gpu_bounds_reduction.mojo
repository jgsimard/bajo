"""GPU root and centroid bounds reduction latency.

Report 31-sample medians after three warmup runs; exclude allocation and input
upload. Check all output coordinates against a host reference outside timing.
"""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.testing import assert_equal
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.bvh.constants import GPU_BOUNDS_BVH_BLOCK_SIZE
from bajo.core import AABB, Point3f32, SegmentOffsets
from bajo.bvh.gpu.utils import _device_span, upload_list
from bajo.bvh.gpu.builder.binary_layout import (
    _reduced_segment_offsets,
    compute_segment_bounds_partials_kernel,
    reduce_segment_bounds_partials_kernel,
)


def main() raises:
    var sizes: List[Int] = [
        1,
        255,
        256,
        257,
        4097,
        100003,
        249882,
        1000003,
        4000003,
    ]
    with DeviceContext() as ctx:
        for size in sizes:
            var input_bounds = ctx.enqueue_create_buffer[.float32](size * 6)
            var expected = AABB[.WORLD].invalid()
            var expected_centroids = AABB[.WORLD].invalid()
            with input_bounds.map_to_host() as host:
                for i in range(size):
                    var x = Float32(i % 1024)
                    var y = Float32(i // 1024)
                    var z = Float32(i % 17)
                    var bounds = AABB[.WORLD](
                        Point3f32[.WORLD](x, y, z),
                        Point3f32[.WORLD](x + 0.5, y + 0.5, z + 0.5),
                    )
                    expected.grow(bounds)
                    expected_centroids.grow(bounds.centroid())
                    bounds.store6(
                        Span(unsafe_ptr=host.unsafe_ptr(), length=len(host)),
                        i * 6,
                    )
            var segments = SegmentOffsets.single(size)
            var leaf_offsets = upload_list(ctx, segments.offsets)
            var partials = _reduced_segment_offsets(segments)
            var counts = List[Int]()
            var offsets = List[DeviceBuffer[.uint32]]()
            counts.append(partials.item_count())
            offsets.append(upload_list(ctx, partials.offsets))
            while partials.item_count() > 1:
                partials = _reduced_segment_offsets(partials)
                counts.append(partials.item_count())
                offsets.append(upload_list(ctx, partials.offsets))
            var scratch_a = ctx.enqueue_create_buffer[.float32](counts[0] * 12)
            var scratch_b = ctx.enqueue_create_buffer[.float32](counts[0] * 12)
            var times = List[Int]()
            var final_result = scratch_a.copy()
            for repeat in range(34):
                ctx.synchronize()
                var start = perf_counter_ns()
                ctx.enqueue_function[
                    compute_segment_bounds_partials_kernel[
                        GPU_BOUNDS_BVH_BLOCK_SIZE
                    ]
                ](
                    _device_span[mut=False](input_bounds),
                    _device_span[mut=False](leaf_offsets),
                    _device_span[mut=False](offsets[0]),
                    _device_span[mut=True](scratch_a),
                    grid_dim=ceildiv(
                        counts[0], GPU_BOUNDS_BVH_BLOCK_SIZE // WARP_SIZE
                    ),
                    block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
                )
                var input = scratch_a.copy()
                var output = scratch_b.copy()
                for level in range(1, len(counts)):
                    ctx.enqueue_function[
                        reduce_segment_bounds_partials_kernel[
                            GPU_BOUNDS_BVH_BLOCK_SIZE
                        ]
                    ](
                        _device_span[mut=False](input),
                        _device_span[mut=False](offsets[level - 1]),
                        _device_span[mut=False](offsets[level]),
                        _device_span[mut=True](output),
                        grid_dim=ceildiv(
                            counts[level],
                            GPU_BOUNDS_BVH_BLOCK_SIZE // WARP_SIZE,
                        ),
                        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
                    )
                    swap(input, output)
                ctx.synchronize()
                if repeat >= 3:
                    times.append(Int(perf_counter_ns() - start))
                final_result = input.copy()
            with final_result.map_to_host() as host:
                var span = Span(unsafe_ptr=host.unsafe_ptr(), length=len(host))
                var actual = AABB[.WORLD].load6(span, 0)
                var centroids = AABB[.WORLD].load6(span, 6)
                assert_equal(actual._min.x, expected._min.x)
                assert_equal(actual._min.y, expected._min.y)
                assert_equal(actual._min.z, expected._min.z)
                assert_equal(actual._max.x, expected._max.x)
                assert_equal(actual._max.y, expected._max.y)
                assert_equal(actual._max.z, expected._max.z)
                assert_equal(centroids._min.x, expected_centroids._min.x)
                assert_equal(centroids._min.y, expected_centroids._min.y)
                assert_equal(centroids._min.z, expected_centroids._min.z)
                assert_equal(centroids._max.x, expected_centroids._max.x)
                assert_equal(centroids._max.y, expected_centroids._max.y)
                assert_equal(centroids._max.z, expected_centroids._max.z)
            sort(times)
            print("BOUNDS", size, times[len(times) // 2], sep="\t")
