"""GPU ray queries against exhaustive Float64 triangle intersections."""
from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32, Vec3, Rayf32, dot
from bajo.core.intersect import intersect_triangle, triangle_normal
from bajo.bvh.constants import f32_max
from bajo.bvh.spatial import SpatialMesh, MeshQueryView, BoundsQueryView
from bajo.bvh.spatial.mesh_query import MeshRayResult
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import upload_list, _device_span


def exhaustive(
    mesh: SpatialMesh[.WORLD], ray: Rayf32[.WORLD], group: Int
) -> MeshRayResult[.WORLD]:
    var result = MeshRayResult[.WORLD].miss()
    for face in range(len(mesh.indices) // 3):
        if group >= 0 and mesh.bvh.groups[face] != UInt32(group):
            continue
        var a = mesh.view().face_vertex(UInt32(face), 0)
        var b = mesh.view().face_vertex(UInt32(face), 1)
        var c = mesh.view().face_vertex(UInt32(face), 2)
        var hit = intersect_triangle(ray, a, b, c)
        if hit.result and (
            not result.result
            or hit.t < result.t
            or (hit.t == result.t and UInt32(face) < result.face)
        ):
            var n = triangle_normal(a, b, c)
            var direction = Vec3[.float64, .WORLD](
                Float64(ray.d.x), Float64(ray.d.y), Float64(ray.d.z)
            )
            result = MeshRayResult(
                True,
                UInt32(face),
                Float32(hit.u),
                Float32(hit.v),
                hit.t,
                Vec3f32[.WORLD](Float32(n.x), Float32(n.y), Float32(n.z)),
                dot(n, direction) < 0.0,
            )
    return result


def exhaustive_count(
    mesh: SpatialMesh[.WORLD], ray: Rayf32[.WORLD], group: Int
) -> UInt32:
    var hits = UInt32(0)
    for face in range(len(mesh.indices) // 3):
        if group >= 0 and mesh.bvh.groups[face] != UInt32(group):
            continue
        var hit = intersect_triangle(
            ray,
            mesh.view().face_vertex(UInt32(face), 0),
            mesh.view().face_vertex(UInt32(face), 1),
            mesh.view().face_vertex(UInt32(face), 2),
        )
        if hit.result:
            hits += 1
    return hits


def exhaustive_parity(
    mesh: SpatialMesh[.WORLD], ray: Rayf32[.WORLD], group: Int
) -> Tuple[UInt32, Bool]:
    var hits = UInt32(0)
    for face in range(len(mesh.indices) // 3):
        if group >= 0 and mesh.bvh.groups[face] != UInt32(group):
            continue
        var hit = intersect_triangle(
            ray,
            mesh.view().face_vertex(UInt32(face), 0),
            mesh.view().face_vertex(UInt32(face), 1),
            mesh.view().face_vertex(UInt32(face), 2),
        )
        if hit.result:
            if hit.u <= 1e-12 or hit.v <= 1e-12 or 1.0 - hit.u - hit.v <= 1e-12:
                return (0, True)
            hits += 1
    return (hits, False)


def ray_kernel[
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
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](
            params[lane * 8], params[lane * 8 + 1], params[lane * 8 + 2]
        ),
        Vec3f32[.WORLD](
            params[lane * 8 + 3], params[lane * 8 + 4], params[lane * 8 + 5]
        ),
        params[lane * 8 + 6],
        params[lane * 8 + 7],
    )
    var hit = view.query_ray(ray, roots[lane])
    output[lane * 13] = Float64(hit.result)
    output[lane * 13 + 1] = Float64(hit.face)
    output[lane * 13 + 2] = hit.t
    output[lane * 13 + 3] = Float64(hit.u)
    output[lane * 13 + 4] = Float64(hit.v)
    comptime for axis in range(3):
        output[lane * 13 + 5 + axis] = Float64(hit.normal[axis])
    output[lane * 13 + 8] = Float64(hit.front_face)
    output[lane * 13 + 9] = Float64(
        view.query_ray_count_intersections(ray, roots[lane])
    )
    output[lane * 13 + 10] = Float64(view.query_ray_anyhit(ray, roots[lane]))

    var votes, ambiguous = view._query_ray_hits[1, False, True](
        ray, roots[lane]
    )
    output[lane * 13 + 11] = Float64(votes)
    output[lane * 13 + 12] = Float64(ambiguous)


def test_ray_groups_ties_slabs_cutoffs_and_extreme_scales() raises:
    comptime count = 384
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
                var seed = face % 129
                var axis = seed % 3
                var p = (
                    Point3f32[.WORLD](
                        Float32(seed % 17 - 8) * scale * 0.125,
                        Float32((seed * 7) % 19 - 9) * scale * 0.125,
                        Float32(seed % 5 - 2) * scale * 0.125,
                    )
                    + shift
                )
                var b = p
                var c = p
                b[(axis + 1) % 3] += scale * 0.2 if seed % 17 else Float32(0.0)
                c[(axis + 2) % 3] += scale * 0.2 if seed % 13 else Float32(0.0)
                var base = UInt32(len(points))
                points.extend([p, b, c])
                indices.extend([base, base + 1, base + 2])
                groups.append(UInt32(face % 3))
            var cpu = SpatialMesh(points, indices, groups)
            var gpu = GpuSpatialMesh(ctx, cpu.copy())
            var params = List[Float32]()
            var roots = List[UInt32]()
            var expected = List[MeshRayResult[.WORLD]]()
            var expected_counts = List[UInt32]()
            var expected_votes = List[UInt32]()
            var expected_ambiguous = List[Bool]()
            for lane in range(count):
                var face = lane % 129
                var axis = face % 3
                var p = cpu.view().face_vertex(UInt32(face), 0)
                var b = cpu.view().face_vertex(UInt32(face), 1)
                var c = cpu.view().face_vertex(UInt32(face), 2)
                var u = Float32(0.25)
                var v = Float32(0.25)
                if lane % 4 == 1:
                    u = 0.0
                    v = 0.0
                elif lane % 4 == 2:
                    u = 0.5
                    v = 0.5
                elif lane % 4 == 3:
                    u = 1.0
                    v = 0.0
                var target = p + (b - p) * u + (c - p) * v
                var direction = Vec3f32[.WORLD](0.0)
                direction[axis] = scale
                if lane % 9 == 1:
                    direction[axis] = 1.0
                elif lane % 9 == 2:
                    direction[axis] = 1e-30
                elif lane % 9 == 3:
                    direction[axis] = 1e30
                elif lane % 9 == 4:
                    direction[axis] = bitcast[.float32](UInt32(1))
                elif lane % 9 == 5:
                    direction[(axis + 1) % 3] = -scale * 0.13
                    direction[(axis + 2) % 3] = scale * 0.27
                elif lane % 9 == 6:
                    direction[(axis + 1) % 3] = bitcast[.float32](
                        UInt32(0x80000000)
                    )
                elif lane % 9 == 7:
                    direction[axis] = -scale
                elif lane % 9 == 8:
                    direction = Vec3f32[.WORLD](0.0)
                var negative = lane % 7 == 0
                var origin = target + direction * (
                    Float32(2.0) if negative else Float32(-2.0)
                )
                var ray = Rayf32[.WORLD](
                    origin,
                    direction,
                    Float32(-4.0) if negative else Float32(0.0),
                    Float32(f32_max),
                )
                var group = lane % 4 - 1
                var full = exhaustive(cpu, ray, group)
                if full.result and lane % 5 != 0:
                    var cutoff = Float32(full.t)
                    var bits = bitcast[.uint32](cutoff)
                    if lane % 5 == 2 and bits > 0:
                        cutoff = bitcast[.float32](bits - 1)
                    elif lane % 5 == 3:
                        cutoff = bitcast[.float32](bits + 1)
                    if cutoff >= ray.t_min:
                        ray.t_max = cutoff
                expected.append(exhaustive(cpu, ray, group))
                expected_counts.append(exhaustive_count(cpu, ray, group))
                var votes, ambiguous = exhaustive_parity(cpu, ray, group)
                expected_votes.append(votes)
                expected_ambiguous.append(ambiguous)
                params.extend(
                    [
                        ray.o.x,
                        ray.o.y,
                        ray.o.z,
                        ray.d.x,
                        ray.d.y,
                        ray.d.z,
                        ray.t_min,
                        ray.t_max,
                    ]
                )
                roots.append(
                    UInt32(0) if group < 0 else gpu.group_root(UInt32(group))
                )
            var dp = upload_list(ctx, params)
            var dr = upload_list(ctx, roots)
            var output = ctx.enqueue_create_buffer[.float64](count * 13)
            ctx.enqueue_function[ray_kernel[count]](
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
                    assert_equal(actual[lane * 13], Float64(hit.result))
                    assert_equal(actual[lane * 13 + 1], Float64(hit.face))
                    assert_almost_equal(
                        actual[lane * 13 + 2], hit.t, atol=0.0, rtol=1e-14
                    )
                    assert_equal(actual[lane * 13 + 3], Float64(hit.u))
                    assert_equal(actual[lane * 13 + 4], Float64(hit.v))
                    comptime for axis in range(3):
                        assert_equal(
                            actual[lane * 13 + 5 + axis],
                            Float64(hit.normal[axis]),
                        )
                    assert_equal(actual[lane * 13 + 8], Float64(hit.front_face))
                    assert_equal(
                        actual[lane * 13 + 9], Float64(expected_counts[lane])
                    )
                    assert_equal(
                        actual[lane * 13 + 10],
                        Float64(expected_counts[lane] != 0),
                    )
                    assert_equal(
                        actual[lane * 13 + 11], Float64(expected_votes[lane])
                    )
                    assert_equal(
                        actual[lane * 13 + 12],
                        Float64(expected_ambiguous[lane]),
                    )


def test_ray_subnormal_slabs_with_normal_direction() raises:
    # A flushed subnormal box coordinate can become a normal entry distance
    # after multiplication by a large reciprocal. Keep that uncertainty.
    with DeviceContext() as ctx:
        for fixture in range(6):
            var axis = fixture % 3
            var sign = Float32(1.0 if fixture < 3 else -1.0)
            var a = Point3f32[.WORLD](0.0)
            a[axis] = sign * Float32(1e-38)
            var b = a
            var c = a
            b[(axis + 1) % 3] = 1.0
            c[(axis + 2) % 3] = 1.0
            var points: List[Point3f32[.WORLD]] = [a, b, c]
            var indices: List[UInt32] = [0, 1, 2, 0, 1, 2]
            var cpu = SpatialMesh(points, indices)
            var gpu = GpuSpatialMesh(ctx, cpu.copy())
            var origin = Point3f32[.WORLD](0.25)
            origin[axis] = 0.0
            var direction = Vec3f32[.WORLD](0.0)
            direction[axis] = sign * Float32(1e-10)
            var ray = Rayf32[.WORLD](
                origin, direction, Float32(5e-29), Float32(2e-28)
            )
            var hit = exhaustive(cpu, ray, -1)
            assert_equal(hit.result, True)
            var params: List[Float32] = [
                origin.x,
                origin.y,
                origin.z,
                direction.x,
                direction.y,
                direction.z,
                ray.t_min,
                ray.t_max,
            ]
            var roots: List[UInt32] = [0]
            var dp = upload_list(ctx, params)
            var dr = upload_list(ctx, roots)
            var output = ctx.enqueue_create_buffer[.float64](13)
            ctx.enqueue_function[ray_kernel[1]](
                _device_span[mut=False](gpu.points),
                _device_span[mut=False](gpu.indices),
                _device_span[mut=False](gpu.velocities),
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                _device_span[mut=False](dp),
                _device_span[mut=False](dr),
                _device_span[mut=True](output),
                grid_dim=1,
                block_dim=128,
            )
            ctx.synchronize()
            with output.map_to_host() as actual:
                assert_equal(actual[0], Float64(hit.result))
                assert_equal(actual[1], Float64(hit.face))
                assert_almost_equal(actual[2], hit.t, atol=0.0, rtol=1e-14)
                assert_equal(actual[3], Float64(hit.u))
                assert_equal(actual[4], Float64(hit.v))
                comptime for component in range(3):
                    assert_equal(
                        actual[5 + component], Float64(hit.normal[component])
                    )
                assert_equal(actual[8], Float64(hit.front_face))
                assert_equal(actual[9], Float64(exhaustive_count(cpu, ray, -1)))
                assert_equal(actual[10], Float64(1.0))
                var votes, ambiguous = exhaustive_parity(cpu, ray, -1)
                assert_equal(actual[11], Float64(votes))
                assert_equal(actual[12], Float64(ambiguous))


def test_ray_provisional_winners_near_coincident_planes_and_cutoffs() raises:
    comptime count = 96
    with DeviceContext() as ctx:
        for scale in List[Float32]([Float32(1e-5), Float32(1.0), Float32(1e5)]):
            var points = List[Point3f32[.WORLD]]()
            var indices = List[UInt32]()
            var groups = List[UInt32]()
            # Reverse depth order and repeat near-coincident planes. Different
            # face sizes force canonical barycentrics to identify the winner.
            for face in range(64):
                var z = (
                    bitcast[.float32](UInt32(0x3F800000 + (63 - face) % 16))
                    * scale
                )
                var extent = Float32(2.0 + Float32(face % 7) * 0.03125) * scale
                var base = UInt32(len(points))
                points.extend(
                    [
                        Point3f32[.WORLD](-extent, -extent, z),
                        Point3f32[.WORLD](extent, -extent, z),
                        Point3f32[.WORLD](-extent, extent, z),
                    ]
                )
                indices.extend([base, base + 1, base + 2])
                groups.append(UInt32(face % 2))
            var cpu = SpatialMesh(points, indices, groups)
            var gpu = GpuSpatialMesh(ctx, cpu.copy())
            var params = List[Float32]()
            var roots = List[UInt32]()
            var expected = List[MeshRayResult[.WORLD]]()
            for lane in range(count):
                var group = lane % 3 - 1
                var ray = Rayf32[.WORLD](
                    Point3f32[.WORLD](-0.25 * scale, -0.125 * scale, -scale),
                    Vec3f32[.WORLD](
                        0.013 * scale, 0.017 * scale, 1.011 * scale
                    ),
                    0.0,
                    4.0,
                )
                var first = exhaustive(cpu, ray, group)
                assert_equal(first.result, True)
                if lane % 4 != 0:
                    var cutoff = Float32(first.t)
                    var bits = bitcast[.uint32](cutoff)
                    if lane % 4 == 2:
                        cutoff = bitcast[.float32](bits - 1)
                    elif lane % 4 == 3:
                        cutoff = bitcast[.float32](bits + 1)
                    ray.t_max = cutoff
                expected.append(exhaustive(cpu, ray, group))
                params.extend(
                    [
                        ray.o.x,
                        ray.o.y,
                        ray.o.z,
                        ray.d.x,
                        ray.d.y,
                        ray.d.z,
                        ray.t_min,
                        ray.t_max,
                    ]
                )
                roots.append(
                    UInt32(0) if group < 0 else gpu.group_root(UInt32(group))
                )
            var dp = upload_list(ctx, params)
            var dr = upload_list(ctx, roots)
            var output = ctx.enqueue_create_buffer[.float64](count * 13)
            ctx.enqueue_function[ray_kernel[count]](
                _device_span[mut=False](gpu.points),
                _device_span[mut=False](gpu.indices),
                _device_span[mut=False](gpu.velocities),
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                _device_span[mut=False](dp),
                _device_span[mut=False](dr),
                _device_span[mut=True](output),
                grid_dim=3,
                block_dim=32,
            )
            ctx.synchronize()
            with output.map_to_host() as actual:
                for lane in range(count):
                    var hit = expected[lane]
                    assert_equal(actual[lane * 13], Float64(hit.result))
                    if hit.result:
                        assert_equal(actual[lane * 13 + 1], Float64(hit.face))
                        assert_almost_equal(
                            actual[lane * 13 + 2], hit.t, atol=0.0, rtol=1e-14
                        )
                        assert_equal(actual[lane * 13 + 3], Float64(hit.u))
                        assert_equal(actual[lane * 13 + 4], Float64(hit.v))
                        comptime for axis in range(3):
                            assert_equal(
                                actual[lane * 13 + 5 + axis],
                                Float64(hit.normal[axis]),
                            )
                        assert_equal(
                            actual[lane * 13 + 8], Float64(hit.front_face)
                        )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
