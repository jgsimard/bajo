"""Exact device winding caches follow deformation, failures and graph replay."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32
from bajo.bvh.spatial import SpatialMesh, MeshQueryView, BoundsQueryView
from bajo.bvh.spatial.winding import DeviceWindingView
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import _device_span
from test.bvh.cpu.test_spatial_winding import append_tetra


def _winding_kernel[
    count: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    velocities: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    wb: ImmSpan[Float32, ImmutAnyOrigin],
    wn: ImmSpan[UInt32, ImmutAnyOrigin],
    offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    faces: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float64, MutAnyOrigin],
):
    var lane = Int(global_idx.x)
    if lane >= count:
        return
    var view = MeshQueryView[ImmutAnyOrigin, .WORLD](
        points,
        indices,
        velocities,
        BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes),
        DeviceWindingView[ImmutAnyOrigin, .WORLD](
            BoundsQueryView[ImmutAnyOrigin, .WORLD](wb, wn), offsets, faces
        ),
    )
    var p = Point3f32[.WORLD](
        Float32(lane % 11) * 0.17,
        Float32(lane % 7) * 0.13,
        Float32(lane % 5) * 0.11,
    )
    output[lane] = view.winding_number(p)


def test_device_winding_updates_graphs_open_faces_and_orientation() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    append_tetra(points, indices, Point3f32[.WORLD](0.0), 1.0)
    append_tetra(points, indices, Point3f32[.WORLD](0.25), 0.5, reverse=True)
    append_tetra(
        points, indices, Point3f32[.WORLD](2.0), 1.0, bad_orientation=True
    )
    var cpu = SpatialMesh(points, indices)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy(), enable_winding=True)
        var input = ctx.enqueue_create_buffer[.float32](len(points) * 3)
        var output = ctx.enqueue_create_buffer[.float64](77)
        var graph = gpu.create_update_graph[True](ctx, input)
        var before = gpu.device_winding_view()
        var address = Int(before.bvh.bounds.unsafe_ptr())
        for phase in range(5):
            if phase > 0:
                for vertex in range(len(points)):
                    points[vertex] = points[vertex] + Vec3f32[.WORLD](
                        0.03, Float32(vertex % 3) * 0.005, -0.02
                    )
                cpu.refit(points)
                with input.map_to_host() as data:
                    for vertex in range(len(points)):
                        comptime for axis in range(3):
                            data[vertex * 3 + axis] = points[vertex][axis]
                if phase == 1:
                    gpu.refit_device(ctx, input)
                elif phase == 2:
                    graph.replay()
                    gpu.check_update(ctx)
                elif phase == 3:
                    gpu.rebuild(ctx, points)
                else:
                    gpu.refit(ctx, points)
            var cache = gpu.device_winding_view()
            assert_equal(Int(cache.bvh.bounds.unsafe_ptr()), address)
            ctx.enqueue_function[_winding_kernel[77]](
                _device_span[mut=False](gpu.points),
                _device_span[mut=False](gpu.indices),
                _device_span[mut=False](gpu.velocities),
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                cache.bvh.bounds,
                cache.bvh.nodes,
                cache.offsets,
                cache.faces,
                _device_span[mut=True](output),
                grid_dim=1,
                block_dim=128,
            )
            ctx.synchronize()
            with output.map_to_host() as data:
                for lane in range(77):
                    var p = Point3f32[.WORLD](
                        Float32(lane % 11) * 0.17,
                        Float32(lane % 7) * 0.13,
                        Float32(lane % 5) * 0.11,
                    )
                    assert_almost_equal(
                        data[lane], cpu.view().winding_number(p), atol=1e-10
                    )
        var old = List[Float32]()
        with gpu.winding.value().tree.bounds.map_to_host() as data:
            for i in range(len(data)):
                old.append(data[i])
        with input.map_to_host() as data:
            data[1] = bitcast[.float32](UInt32(0x7FC00000))
        graph.replay()
        var rejected = False
        try:
            gpu.check_update(ctx)
        except:
            rejected = True
        assert_true(rejected)
        with gpu.winding.value().tree.bounds.map_to_host() as data:
            for i in range(len(old)):
                assert_equal(data[i], old[i])


def check_parallel_winding_refit[
    components: Int, block_size: Int, rebuild: Bool = False
]() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    append_tetra(points, indices, Point3f32[.WORLD](3.0), 1.0)
    for component in range(1, components):
        append_tetra(
            points,
            indices,
            Point3f32[.WORLD](4.0 + Float32(component) * 3.0, 3.0, 3.0),
            1.0,
            reverse=component % 2 == 0,
        )
    # Validation must include vertices that no face references.
    points.append(Point3f32[.WORLD](100.0))
    var original = points.copy()
    var cpu = SpatialMesh(points, indices)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy(), enable_winding=True)
        var input = ctx.enqueue_create_buffer[.float32](len(points) * 3)
        var output = ctx.enqueue_create_buffer[.float64](77)
        var graph = gpu.create_update_graph[rebuild, block_size](ctx, input)
        for phase in range(4):
            var shift = Vec3f32[.WORLD](
                Float32(-3.0 if phase % 2 == 0 else 0.0)
            )
            for vertex in range(len(points)):
                points[vertex] = original[vertex] + shift
            cpu.refit(points)
            with input.map_to_host() as data:
                for vertex in range(len(points)):
                    comptime for axis in range(3):
                        data[vertex * 3 + axis] = points[vertex][axis]
            if phase == 2:
                gpu.enqueue_update[rebuild, block_size](ctx, input)
                gpu.check_update(ctx)
            else:
                graph.replay()
                gpu.check_update(ctx)
            comptime if rebuild:
                with gpu.nodes.map_to_host() as data:
                    for i in range(len(cpu.bvh.nodes)):
                        cpu.bvh.nodes[i] = data[i]
                cpu.refit(points)
            with gpu.winding.value().tree.bounds.map_to_host() as data:
                for i in range(len(data)):
                    assert_equal(data[i], cpu.winding.bvh.bounds[i])
            var cache = gpu.device_winding_view()
            ctx.enqueue_function[_winding_kernel[77]](
                _device_span[mut=False](gpu.points),
                _device_span[mut=False](gpu.indices),
                _device_span[mut=False](gpu.velocities),
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                cache.bvh.bounds,
                cache.bvh.nodes,
                cache.offsets,
                cache.faces,
                _device_span[mut=True](output),
                grid_dim=1,
                block_dim=128,
            )
            ctx.synchronize()
            with output.map_to_host() as data:
                for lane in range(77):
                    var p = Point3f32[.WORLD](
                        Float32(lane % 11) * 0.17,
                        Float32(lane % 7) * 0.13,
                        Float32(lane % 5) * 0.11,
                    )
                    assert_almost_equal(
                        data[lane], cpu.view().winding_number(p), atol=1e-10
                    )
        with input.map_to_host() as data:
            data[len(data) - 1] = bitcast[.float32](UInt32(0x7FC00000))
        graph.replay()
        var rejected = False
        try:
            gpu.check_update(ctx)
        except:
            rejected = True
        assert_true(rejected)
        with gpu.winding.value().tree.bounds.map_to_host() as data:
            for i in range(len(data)):
                assert_equal(data[i], cpu.winding.bvh.bounds[i])
        with gpu.bounds.map_to_host() as data:
            for i in range(len(data)):
                assert_equal(data[i], cpu.bvh.bounds[i])
        with gpu.points.map_to_host() as data:
            for vertex in range(len(points)):
                comptime for axis in range(3):
                    assert_equal(data[vertex * 3 + axis], points[vertex][axis])


def test_parallel_winding_refit_graph_and_rejected_unused_vertex() raises:
    check_parallel_winding_refit[257, 256]()
    check_parallel_winding_refit[1025, 32]()


def test_gpu_winding_enables_cache_from_uncached_host() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    append_tetra(points, indices, Point3f32[.WORLD](0.0), 1.0)
    append_tetra(points, indices, Point3f32[.WORLD](0.25), 0.5, reverse=True)
    var cpu = SpatialMesh(points, indices, cache_winding=False)
    assert_equal(len(cpu.winding.face_components), 0)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy(), enable_winding=True)
        assert_equal(len(gpu.host.winding.face_components), len(indices) // 3)
        var output = ctx.enqueue_create_buffer[.float64](77)
        ref cache = gpu.winding.value()
        ctx.enqueue_function[_winding_kernel[77]](
            _device_span[mut=False](gpu.points),
            _device_span[mut=False](gpu.indices),
            _device_span[mut=False](gpu.velocities),
            _device_span[mut=False](gpu.bounds),
            _device_span[mut=False](gpu.nodes),
            _device_span[mut=False](cache.tree.bounds),
            _device_span[mut=False](cache.tree.nodes),
            _device_span[mut=False](cache.offsets),
            _device_span[mut=False](cache.faces),
            _device_span[mut=True](output),
            grid_dim=3,
            block_dim=32,
        )
        ctx.synchronize()
        with output.map_to_host() as values:
            for q in range(77):
                var point = Point3f32[.WORLD](
                    Float32(q % 11) * 0.17,
                    Float32(q % 7) * 0.13,
                    Float32(q % 5) * 0.11,
                )
                assert_almost_equal(
                    values[q],
                    cpu.view().winding_number(point),
                    atol=1e-11,
                    rtol=1e-11,
                )


def test_parallel_winding_rebuild_graph_and_rejected_unused_vertex() raises:
    check_parallel_winding_refit[1024, 256, True]()
    check_parallel_winding_refit[1025, 32, True]()


def test_winding_storage_empty_open_mixed_and_disjoint_views() raises:
    # Include zero component/node buffers, all-open faces and mixed topology.
    for mode in range(3):
        var points = List[Point3f32[.WORLD]]()
        var indices = List[UInt32]()
        if mode == 2:
            append_tetra(points, indices, Point3f32[.WORLD](0.0), 1.0)
        if mode > 0:
            var first = UInt32(len(points))
            points.append(Point3f32[.WORLD](3.0, 0.0, 0.0))
            points.append(Point3f32[.WORLD](4.0, 0.0, 0.0))
            points.append(Point3f32[.WORLD](3.0, 1.0, 0.0))
            indices.extend([first, first + 1, first + 2])
        var host = SpatialMesh(points, indices)
        with DeviceContext() as ctx:
            var gpu = GpuSpatialMesh(ctx, host.copy(), enable_winding=True)
            ref cache = gpu.winding.value()
            var component_count = len(host.winding.bvh.item_bounds)
            assert_equal(component_count, 1 if mode == 2 else 0)
            assert_equal(len(cache.offsets), component_count + 2)
            assert_equal(len(cache.faces), len(indices) // 3)
            assert_equal(len(cache.tree.bounds), len(host.winding.bvh.bounds))
            assert_equal(len(cache.tree.nodes), len(host.winding.bvh.nodes))
            assert_equal(len(cache.parents), len(cache.tree.nodes) // 2)
            assert_equal(len(cache.tree.update_items), component_count * 6)
            assert_equal(
                len(cache.tree.update_order), len(cache.tree.nodes) // 2
            )
            assert_equal(len(cache.tree.update_status), 1)
            var starts = List[Int]()
            var lengths = List[Int]()
            starts.extend(
                [
                    Int(cache.offsets.unsafe_ptr()),
                    Int(cache.faces.unsafe_ptr()),
                    Int(cache.tree.bounds.unsafe_ptr()),
                    Int(cache.tree.nodes.unsafe_ptr()),
                    Int(cache.parents.unsafe_ptr()),
                    Int(cache.tree.update_status.unsafe_ptr()),
                    Int(cache.tree.update_items.unsafe_ptr()),
                    Int(cache.tree.update_order.unsafe_ptr()),
                ]
            )
            lengths.extend(
                [
                    len(cache.offsets),
                    len(cache.faces),
                    len(cache.tree.bounds),
                    len(cache.tree.nodes),
                    len(cache.parents),
                    len(cache.tree.update_status),
                    len(cache.tree.update_items),
                    len(cache.tree.update_order),
                ]
            )
            for left in range(len(starts)):
                if lengths[left] == 0:
                    continue
                assert_equal(starts[left] % 64, 0)
                for right in range(left + 1, len(starts)):
                    if lengths[right] > 0:
                        assert_true(
                            starts[left] + lengths[left] * 4 <= starts[right]
                            or starts[right] + lengths[right] * 4
                            <= starts[left]
                        )
            with cache.offsets.map_to_host() as data:
                assert_equal(data[0], 0)
                assert_equal(data[len(data) - 1], UInt32(len(indices) // 3))
                if mode == 2:
                    assert_equal(data[1], 4)
            if len(cache.faces) > 0:
                with cache.faces.map_to_host() as data:
                    for face in range(len(data)):
                        assert_equal(data[face], UInt32(face))
            if len(cache.tree.bounds) > 0:
                with cache.tree.bounds.map_to_host() as data:
                    for i in range(len(data)):
                        assert_equal(data[i], host.winding.bvh.bounds[i])
            if len(cache.tree.nodes) > 0:
                with cache.tree.nodes.map_to_host() as data:
                    for i in range(len(data)):
                        assert_equal(data[i], host.winding.bvh.nodes[i])
            with cache.tree.update_status.map_to_host() as data:
                assert_equal(data[0], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
