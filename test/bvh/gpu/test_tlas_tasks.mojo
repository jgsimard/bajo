"""Pending TLAS tasks respect closer hits and clipped ray intervals."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal

from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.gpu.ray_launch import _load_packed_ray, _store_packed_hit
from bajo.bvh.gpu.trace import _trace_bounds_bvh_state_bvh2_closest
from bajo.bvh.gpu.utils import upload_list, upload_rays
from bajo.bvh.types import Hit
from bajo.bvh.wide_meta import _pack_wide_meta
from bajo.core import Point3f32, Rayf32, Vec3f32


@fieldwise_init
struct CountingPlanes(TrivialRegisterPassable):
    var calls: Pointer[UInt32, MutUntrackedOrigin]


def _intersect_plane(
    state: CountingPlanes,
    data: UInt32,
    count: UInt32,
    ray: Rayf32[.WORLD],
    mut hit: Hit[.WORLD],
) -> Bool:
    state.calls[unsafe_offset=0] += 1
    var t = (Float32(data + 1) * 2 - ray.o.z) / ray.d.z
    if t >= ray.t_min and t <= hit.t:
        hit.t = t
        hit.prim = data
        return True
    return False


def _compare_pending_tasks[
    case_count: Int
](
    nodes: ImmPointer[Float32, ImmutAnyOrigin],
    rays: ImmPointer[Float32, ImmutAnyOrigin],
    hits: MutPointer[Float32, MutAnyOrigin],
    calls: MutPointer[UInt32, MutAnyOrigin],
):
    var index = global_idx.x
    if index >= case_count:
        return
    var ray = _load_packed_ray[.WORLD](rays, case_count, index)
    comptime for pruned in [False, True]:
        comptime offset = 1 if pruned else 0
        var state = CountingPlanes(
            calls.unsafe_offset(index * 2 + offset).unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
        )
        var hit = _trace_bounds_bvh_state_bvh2_closest[
            .WORLD, CountingPlanes, _intersect_plane, pruned
        ](nodes, state, UInt32(0), ray)
        _store_packed_hit(hit, hits, case_count * 2, index * 2 + offset)


def test_pending_tasks_skip_farther_callbacks() raises:
    # Root's near internal child contains planes at z=2 and z=4; its far
    # child is the plane at z=6. Near-first traversal finds z=2 while the
    # other two leaves remain pending. Distance pruning must skip both.
    var nodes = List[Float32](length=28, fill=0)
    for node in range(2):
        var base = node * 14
        for lane in range(2):
            nodes[base + lane] = -1
            nodes[base + 2 + lane] = -1
            nodes[base + 6 + lane] = 1
            nodes[base + 8 + lane] = 1
    nodes[4] = 2
    nodes[10] = 4
    nodes[5] = 6
    nodes[11] = 6
    nodes[12] = bitcast[.float32](_pack_wide_meta(UInt32(1), UInt32(0)))
    nodes[13] = bitcast[.float32](_pack_wide_meta(UInt32(2), UInt32(1)))
    nodes[18] = 2
    nodes[24] = 2
    nodes[19] = 4
    nodes[25] = 4
    nodes[26] = bitcast[.float32](_pack_wide_meta(UInt32(0), UInt32(1)))
    nodes[27] = bitcast[.float32](_pack_wide_meta(UInt32(1), UInt32(1)))
    var rays = List[Rayf32[.WORLD]](
        [
            Rayf32[.WORLD](
                Point3f32[.WORLD](0), Vec3f32[.WORLD](0, 0, 1), 0, 10
            ),
            Rayf32[.WORLD](
                Point3f32[.WORLD](0), Vec3f32[.WORLD](0, 0, 1), 3, 10
            ),
            Rayf32[.WORLD](
                Point3f32[.WORLD](0), Vec3f32[.WORLD](0, 0, 1), 0, 1
            ),
            Rayf32[.WORLD](
                Point3f32[.WORLD](0, 0, 3), Vec3f32[.WORLD](0, 0, 1), 0, 10
            ),
            Rayf32[.WORLD](
                Point3f32[.WORLD](0), Vec3f32[.WORLD](0, 0, 1), 0, 2
            ),
        ]
    )
    var expected_calls = [UInt32(3), 1, 3, 2, 0, 0, 2, 1, 1, 1]
    var expected_prim = [UInt32(0), 1, EMPTY_LANE, 1, 0]
    var expected_t = [Float32(2), 4, 1, 1, 2]
    with DeviceContext() as ctx:
        var d_nodes = upload_list(ctx, nodes)
        var d_rays = upload_rays(ctx, rays)
        var d_calls = upload_list(ctx, List[UInt32](length=10, fill=0))
        var d_hits = ctx.enqueue_create_buffer[.float32](10 * Hit.STRIDE)
        ctx.enqueue_function[_compare_pending_tasks[5]](
            d_nodes,
            d_rays,
            d_hits,
            d_calls,
            grid_dim=1,
            block_dim=32,
        )
        ctx.synchronize()
        with d_calls.map_to_host() as calls:
            for index in range(10):
                assert_equal(calls[index], expected_calls[index])
        with d_hits.map_to_host() as values:
            var span = Span(unsafe_ptr=values.unsafe_ptr(), length=len(values))
            for index in range(len(rays)):
                var pruned = Hit[.WORLD].load(span, index * 2 + 1)
                assert_equal(pruned.prim, expected_prim[index])
                assert_equal(pruned.t, expected_t[index])
                for field in range(Hit.STRIDE):
                    assert_equal(
                        bitcast[.uint32](
                            values[(index * 2) * Hit.STRIDE + field]
                        ),
                        bitcast[.uint32](
                            values[(index * 2 + 1) * Hit.STRIDE + field]
                        ),
                    )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
