"""GPU furthest pruning against exhaustive Float64 vertex scans."""

from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.math import sqrt
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32, dot
from bajo.core.intersect import vector64
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial import SpatialMesh, MeshQueryView, BoundsQueryView
from bajo.bvh.spatial.mesh_query import MeshPointResult
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import upload_list, _device_span


def exhaustive(
    mesh: SpatialMesh[.WORLD],
    p: Point3f32[.WORLD],
    cutoff: Float32,
    group: Int,
) -> MeshPointResult[.WORLD]:
    var result = MeshPointResult[.WORLD].miss()
    if cutoff < 0.0:
        return result
    var best2 = Float64(cutoff) * Float64(cutoff)
    for face in range(len(mesh.indices) // 3):
        if group >= 0 and mesh.bvh.groups[face] != UInt32(group):
            continue
        for corner in range(3):
            var vertex = mesh.view().face_vertex(UInt32(face), corner)
            var delta = vector64(vertex) - vector64(p)
            var distance2 = dot(delta, delta)
            if distance2 >= best2 and (
                not result.result
                or distance2 > best2
                or UInt32(face) < result.face
            ):
                best2 = distance2
                result = MeshPointResult(
                    True,
                    UInt32(face),
                    Float32(1.0 if corner == 0 else 0.0),
                    Float32(1.0 if corner == 1 else 0.0),
                    0.0,
                    vertex,
                    0.0,
                )
    if result.result:
        result.distance = sqrt(best2)
    return result


def furthest_kernel[
    count: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    velocities: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    params: ImmSpan[Float32, ImmutAnyOrigin],
    roots: ImmSpan[UInt32, ImmutAnyOrigin],
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
    )
    var p = Point3f32[.WORLD](
        params[lane * 4], params[lane * 4 + 1], params[lane * 4 + 2]
    )
    var hit = view.query_furthest_point_no_sign(
        p, params[lane * 4 + 3], roots[lane]
    )
    output[lane * 8] = Float64(hit.result)
    output[lane * 8 + 1] = Float64(hit.face)
    output[lane * 8 + 2] = Float64(hit.u)
    output[lane * 8 + 3] = Float64(hit.v)
    output[lane * 8 + 4] = hit.distance
    comptime for axis in range(3):
        output[lane * 8 + 5 + axis] = Float64(hit.point[axis])


def test_furthest_groups_ties_cutoffs_and_extreme_scales() raises:
    comptime count = 256
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
            for face in range(258):
                # Repeated geometry exercises exact face and corner ties.
                var seed = face % 129
                var p = Point3f32[.WORLD](
                    Float32(seed % 17 - 8) * scale * 0.125,
                    Float32((seed * 7) % 19 - 9) * scale * 0.125,
                    Float32(seed % 5 - 2) * scale * 0.125,
                )
                var base = UInt32(len(points))
                points.extend(
                    [
                        p + shift,
                        p + shift + Vec3f32[.WORLD](scale * 0.2, 0.0, 0.0),
                        p + shift + Vec3f32[.WORLD](0.0, scale * 0.2, 0.0),
                    ]
                )
                indices.extend([base, base + 1, base + 2])
                groups.append(UInt32(face % 3))
            var cpu = SpatialMesh(points, indices, groups)
            var gpu = GpuSpatialMesh(ctx, cpu.copy())
            var params = List[Float32]()
            var roots = List[UInt32]()
            var expected = List[MeshPointResult[.WORLD]]()
            for lane in range(count):
                var p = (
                    Point3f32[.WORLD](
                        Float32(lane % 11 - 5) * scale * 0.173,
                        Float32((lane * 17) % 13 - 6) * scale * 0.157,
                        Float32(lane % 7 - 3) * scale * 0.191,
                    )
                    + shift
                )
                var group = lane % 4 - 1
                var full = exhaustive(cpu, p, 0.0, group)
                var cutoff = Float32(full.distance)
                var bits = bitcast[.uint32](cutoff)
                if lane % 5 == 0:
                    cutoff = 0.0
                elif lane % 5 == 2:
                    cutoff = bitcast[.float32](bits - 1)
                elif lane % 5 == 3:
                    cutoff = bitcast[.float32](bits + 1)
                elif lane % 5 == 4:
                    cutoff = -1.0
                expected.append(exhaustive(cpu, p, cutoff, group))
                params.extend([p.x, p.y, p.z, cutoff])
                roots.append(
                    UInt32(0) if group < 0 else gpu.group_root(UInt32(group))
                )
            var dp = upload_list(ctx, params)
            var dr = upload_list(ctx, roots)
            var output = ctx.enqueue_create_buffer[.float64](count * 8)
            ctx.enqueue_function[furthest_kernel[count]](
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
                for lane in range(count):
                    var hit = expected[lane]
                    assert_equal(actual[lane * 8], Float64(hit.result))
                    assert_equal(actual[lane * 8 + 1], Float64(hit.face))
                    assert_equal(actual[lane * 8 + 2], Float64(hit.u))
                    assert_equal(actual[lane * 8 + 3], Float64(hit.v))
                    assert_almost_equal(
                        actual[lane * 8 + 4], hit.distance, atol=0.0, rtol=1e-14
                    )
                    comptime for axis in range(3):
                        assert_equal(
                            actual[lane * 8 + 5 + axis],
                            Float64(hit.point[axis]),
                        )


def test_furthest_inclusive_unit_cutoff_and_equal_vertex_ties() raises:
    var points: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](-1.0, 0.0, 0.0),
        Point3f32[.WORLD](1.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 1.0, 0.0),
    ]
    var indices: List[UInt32] = [0, 1, 2, 0, 1, 2]
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, SpatialMesh(points, indices))
        var bits = bitcast[.uint32](Float32(1.0))
        var params: List[Float32] = [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            bitcast[.float32](bits - 1),
            0.0,
            0.0,
            0.0,
            bitcast[.float32](bits + 1),
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        var dp = upload_list(ctx, params)
        var dr = upload_list(ctx, List[UInt32](length=4, fill=0))
        var output = ctx.enqueue_create_buffer[.float64](32)
        ctx.enqueue_function[furthest_kernel[4]](
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
            for lane in range(4):
                assert_equal(actual[lane * 8], Float64(lane != 2))
                if lane == 2:
                    assert_equal(actual[lane * 8 + 1], Float64(EMPTY_LANE))
                    continue
                assert_equal(actual[lane * 8 + 1], Float64(0.0))
                assert_equal(actual[lane * 8 + 2], Float64(1.0))
                assert_equal(actual[lane * 8 + 3], Float64(0.0))
                assert_equal(actual[lane * 8 + 4], Float64(1.0))
                assert_equal(actual[lane * 8 + 5], Float64(-1.0))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
