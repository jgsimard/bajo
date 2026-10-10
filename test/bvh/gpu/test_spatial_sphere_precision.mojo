"""Exact GPU sphere sets against exhaustive Float64 triangle projection."""
from std.testing import TestSuite, assert_equal
from std.math import sqrt, min
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32
from bajo.core.intersect import project_triangle
from bajo.bvh.constants import f32_max
from bajo.bvh.spatial import SpatialMesh, MeshQueryView, BoundsQueryView
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import upload_list, _device_span


def sphere_kernel[
    count: Int, faces: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    velocities: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    params: ImmSpan[Float32, ImmutAnyOrigin],
    roots: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
):
    var lane = Int(global_idx.x)
    if lane >= count:
        return
    for face in range(faces + 1):
        output[lane * (faces + 1) + face] = 0
    var view = MeshQueryView[ImmutAnyOrigin, .WORLD](
        points,
        indices,
        velocities,
        BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes),
    )
    var center = Point3f32[.WORLD](
        params[lane * 4], params[lane * 4 + 1], params[lane * 4 + 2]
    )
    var query = view.query_sphere(center, params[lane * 4 + 3], roots[lane])
    var face = UInt32(12345)
    while query.next(face):
        output[lane * (faces + 1) + Int(face)] += 1
    var preserved = face
    if query.next(face) or face != preserved:
        output[lane * (faces + 1) + faces] = 1


def test_sphere_exact_sets_groups_tangencies_and_extreme_scales() raises:
    comptime count = 256
    comptime faces = 258
    with DeviceContext() as ctx:
        var scales = List[Float32](
            [
                Float32(1e-22),
                Float32(1e-15),
                Float32(1e-12),
                Float32(1.0),
                Float32(1e12),
                Float32(1e18),
                Float32(1e20),
            ]
        )
        for fixture in range(len(scales) * 2):
            var scale = scales[fixture // 2]
            var shift = Vec3f32[.WORLD](
                scale * (Float32(1048576.0) if fixture % 2 else Float32(0.0))
            )
            var points = List[Point3f32[.WORLD]]()
            var indices = List[UInt32]()
            var groups = List[UInt32]()
            for face in range(faces):
                var seed = face % 129
                var p = (
                    Point3f32[.WORLD](
                        Float32(seed % 17 - 8) * scale * 0.125,
                        Float32((seed * 7) % 19 - 9) * scale * 0.125,
                        Float32(seed % 5 - 2) * scale * 0.125,
                    )
                    + shift
                )
                var base = UInt32(len(points))
                points.extend(
                    [
                        p,
                        p
                        + Vec3f32[.WORLD](
                            scale * 0.2 if seed % 17 else Float32(0.0), 0.0, 0.0
                        ),
                        p
                        + Vec3f32[.WORLD](
                            0.0, scale * 0.2 if seed % 13 else Float32(0.0), 0.0
                        ),
                    ]
                )
                indices.extend([base, base + 1, base + 2])
                groups.append(UInt32(face % 3))
            var cpu = SpatialMesh(points, indices, groups)
            var gpu = GpuSpatialMesh(ctx, cpu.copy())
            var params = List[Float32]()
            var roots = List[UInt32]()
            var expected = List[UInt32]()
            for lane in range(count):
                var p = (
                    Point3f32[.WORLD](
                        Float32(lane % 11 - 5) * scale * 0.173,
                        Float32((lane * 17) % 13 - 6) * scale * 0.157,
                        Float32(lane % 7 - 3) * scale * 0.191,
                    )
                    + shift
                )
                if lane % 7 == 0:
                    p = cpu.view().face_vertex(UInt32(lane % faces), 0)
                var group = lane % 4 - 1
                var nearest = Float64(f32_max) * Float64(f32_max)
                var distances = List[Float64]()
                for face in range(faces):
                    var d2 = project_triangle(
                        p,
                        cpu.view().face_vertex(UInt32(face), 0),
                        cpu.view().face_vertex(UInt32(face), 1),
                        cpu.view().face_vertex(UInt32(face), 2),
                    ).distance2
                    distances.append(d2)
                    if group < 0 or groups[face] == UInt32(group):
                        nearest = min(nearest, d2)
                var radius = Float32(sqrt(nearest))
                var bits = bitcast[.uint32](radius)
                if lane % 6 == 0:
                    radius = Float32(f32_max)
                elif lane % 6 == 2:
                    radius = bitcast[.float32](
                        bits - 1
                    ) if bits > 0 else Float32(0.0)
                elif lane % 6 == 3:
                    radius = bitcast[.float32](bits + 1)
                elif lane % 6 == 4:
                    radius = -1.0
                elif lane % 6 == 5:
                    radius = scale * 0.37
                var r2 = Float64(radius) * Float64(radius)
                for face in range(faces):
                    expected.append(
                        UInt32(
                            radius >= 0
                            and (group < 0 or groups[face] == UInt32(group))
                            and distances[face] <= r2
                        )
                    )
                expected.append(0)
                params.extend([p.x, p.y, p.z, radius])
                roots.append(
                    UInt32(0) if group < 0 else gpu.group_root(UInt32(group))
                )
            var dp = upload_list(ctx, params)
            var dr = upload_list(ctx, roots)
            var output = ctx.enqueue_create_buffer[.uint32](count * (faces + 1))
            ctx.enqueue_function[sphere_kernel[count, faces]](
                _device_span[mut=False](gpu.points),
                _device_span[mut=False](gpu.indices),
                _device_span[mut=False](gpu.velocities),
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                _device_span[mut=False](dp),
                _device_span[mut=False](dr),
                _device_span[mut=True](output),
                grid_dim=count // 128,
                block_dim=128,
            )
            ctx.synchronize()
            with output.map_to_host() as actual:
                for slot in range(len(expected)):
                    assert_equal(actual[slot], expected[slot])


def test_sphere_canonical_vertex_cancellation_and_cutoff_neighbors() raises:
    var points: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](1e10, 0.0, 0.0),
        Point3f32[.WORLD](4e-7, 0.0, 0.0),
        Point3f32[.WORLD](-1e-6, 1e10, 0.0),
    ]
    var indices: List[UInt32] = [0, 1, 2, 0, 1, 2]
    var bits = bitcast[.uint32](Float32(5e6))
    var params: List[Float32] = [
        -3e6,
        -4e6,
        0.0,
        5e6,
        -3e6,
        -4e6,
        0.0,
        bitcast[.float32](bits - 1),
        -3e6,
        -4e6,
        0.0,
        bitcast[.float32](bits + 1),
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    var roots: List[UInt32] = [0, 0, 0, 0]
    var expected: List[UInt32] = [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    with DeviceContext() as ctx:
        var cpu = SpatialMesh(points, indices)
        for lane in range(4):
            var p = Point3f32[.WORLD](
                params[lane * 4], params[lane * 4 + 1], params[lane * 4 + 2]
            )
            var radius = Float64(params[lane * 4 + 3])
            for face in range(2):
                var d2 = project_triangle(
                    p, points[0], points[1], points[2]
                ).distance2
                assert_equal(
                    UInt32(d2 <= radius * radius), expected[lane * 3 + face]
                )
        var gpu = GpuSpatialMesh(ctx, cpu^)
        var dp = upload_list(ctx, params)
        var dr = upload_list(ctx, roots)
        var output = ctx.enqueue_create_buffer[.uint32](12)
        ctx.enqueue_function[sphere_kernel[4, 2]](
            _device_span[mut=False](gpu.points),
            _device_span[mut=False](gpu.indices),
            _device_span[mut=False](gpu.velocities),
            _device_span[mut=False](gpu.bounds),
            _device_span[mut=False](gpu.nodes),
            _device_span[mut=False](dp),
            _device_span[mut=False](dr),
            _device_span[mut=True](output),
            grid_dim=1,
            block_dim=32,
        )
        ctx.synchronize()
        with output.map_to_host() as actual:
            for slot in range(len(expected)):
                assert_equal(actual[slot], expected[slot])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
