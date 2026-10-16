"""Resident bounds refits, including submission and synchronization overhead.

Build with `mojo build -O3 -I . --target-accelerator sm_120
bench/bvh/bench_gpu_spatial_bounds_updates.mojo`. Each result is the median of
seven batches of ten updates, after warmup. CPU reference hashes cover every
bound and topology word; validation is outside the timed region.
"""

from std.memory import bitcast
from std.testing import assert_equal
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.gpu.buffers import upload_list


def _hash(bounds: ImmSpan[Float32, _], nodes: ImmSpan[UInt32, _]) -> UInt64:
    var digest = UInt64(14695981039346656037)
    for value in bounds:
        digest = (digest ^ UInt64(bitcast[.uint32](value))) * UInt64(
            1099511628211
        )
    for value in nodes:
        digest = (digest ^ UInt64(value)) * UInt64(1099511628211)
    return digest


def main() raises:
    print("items,groups,graph,median_us,hash")
    with DeviceContext() as ctx:
        for count in [
            64,
            256,
            512,
            1023,
            1024,
            1025,
            4097,
            16384,
            65536,
            262144,
        ]:
            for group_count in [0, 17]:
                var boxes = List[AABB[.WORLD]](capacity=count)
                var moved = List[AABB[.WORLD]](capacity=count)
                var flat = List[Float32](capacity=count * 6)
                var groups = List[UInt32](capacity=count if group_count else 0)
                for i in range(count):
                    var x = Float32(i % 512)
                    var y = Float32(i // 512)
                    boxes.append(
                        AABB(
                            Point3f32[.WORLD](x, y, 0.0),
                            Point3f32[.WORLD](x + 0.5, y + 0.5, 0.5),
                        )
                    )
                    moved.append(
                        AABB(
                            Point3f32[.WORLD](x + 1.125, y + 2.25, -0.5),
                            Point3f32[.WORLD](x + 1.625, y + 2.75, 0.0),
                        )
                    )
                    comptime for axis in range(3):
                        flat.append(moved[i]._min[axis])
                    comptime for axis in range(3):
                        flat.append(moved[i]._max[axis])
                    if group_count:
                        groups.append(UInt32(i % group_count) * 65537)
                var cpu = SpatialBoundsBvh(boxes, groups)
                var gpu = GpuSpatialBoundsBvh(ctx, cpu.copy())
                var input = upload_list(ctx, flat)
                var graph = gpu.create_update_graph[False](ctx, input)
                cpu.refit(moved)
                var expected = _hash(cpu.bounds, cpu.nodes)
                for use_graph in [False, True]:
                    var times = List[Float64](capacity=7)
                    for repeat in range(8):
                        var start = perf_counter_ns()
                        for _ in range(10):
                            if use_graph:
                                graph.replay()
                            else:
                                gpu.enqueue_update[False](ctx, input)
                        ctx.synchronize()
                        if repeat > 0:
                            times.append(
                                Float64(perf_counter_ns() - start) * 0.0001
                            )
                    gpu.check_update(ctx)
                    with gpu.bounds.map_to_host() as bounds:
                        with gpu.nodes.map_to_host() as nodes:
                            assert_equal(
                                _hash(
                                    Span(
                                        unsafe_ptr=bounds.unsafe_ptr(),
                                        length=len(bounds),
                                    ),
                                    Span(
                                        unsafe_ptr=nodes.unsafe_ptr(),
                                        length=len(nodes),
                                    ),
                                ),
                                expected,
                            )
                    sort(times)
                    print(
                        count,
                        group_count,
                        use_graph,
                        times[3],
                        expected,
                        sep=",",
                    )
