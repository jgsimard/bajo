"""Bounded, world-space debug snapshots independent of the graphics API."""

from bajo.core import AABB, Frame, Point3f32, Point3W, Affine3f32
from bajo.bvh.constants import EMPTY_LANE, PrimitiveKind
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.bounds_bvh import WideBvhNode
from bajo.bvh.tagged_ref import is_leaf_ref, decode_ref_index
from bajo.bvh.types import BlasDesc
from bajo.bvh.wide_meta import _wide_meta_count, _wide_meta_data
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.scene_data import SceneData


@fieldwise_init
struct DebugBox(ImplicitlyCopyable):
    var minimum: Point3W
    var maximum: Point3W
    var depth: Int


def object_boxes(data: SceneData, limit: Int = 10000) -> List[DebugBox]:
    var boxes = List[DebugBox]()
    for sphere in data.spheres():
        if len(boxes) >= limit:
            break
        var bounds = sphere.bounds()
        boxes.append(
            DebugBox(
                bounds._min.unsafe_convert[new_frame=.WORLD](),
                bounds._max.unsafe_convert[new_frame=.WORLD](),
                0,
            )
        )
    for instance in data.triangle_instances():
        if len(boxes) >= limit:
            break
        boxes.append(
            DebugBox(
                instance.bounds._min.unsafe_convert[new_frame=.WORLD](),
                instance.bounds._max.unsafe_convert[new_frame=.WORLD](),
                0,
            )
        )
    for triangle in range(len(data.triangle_vertices()) // 3):
        if len(boxes) >= limit:
            break
        var bounds = AABB[.WORLD].invalid()
        comptime for corner in range(3):
            bounds.grow(data.triangle_vertices()[3 * triangle + corner])
        boxes.append(
            DebugBox(
                bounds._min.unsafe_convert[new_frame=.WORLD](),
                bounds._max.unsafe_convert[new_frame=.WORLD](),
                0,
            )
        )
    return boxes^


def _node_boxes[
    frame: Frame, width: SIMDLength, packed: Bool = False
](
    nodes: ImmSpan[WideBvhNode[frame, width], _],
    index: Int,
    depth: Int,
    transform: Affine3f32[.LOCAL, .WORLD],
    mut boxes: List[DebugBox],
    limit: Int,
):
    ref node = nodes[index]
    comptime for lane in range(width):
        if node.data[lane] != EMPTY_LANE and len(boxes) < limit:
            var bounds = AABB[frame](
                Point3f32[frame](
                    node.aabb._min.x[lane],
                    node.aabb._min.y[lane],
                    node.aabb._min.z[lane],
                ),
                Point3f32[frame](
                    node.aabb._max.x[lane],
                    node.aabb._max.y[lane],
                    node.aabb._max.z[lane],
                ),
            )
            comptime if frame == .LOCAL:
                var world = bounds.unsafe_convert_frame[
                    .LOCAL
                ]().apply_transform(transform)
                boxes.append(
                    DebugBox(
                        world._min.unsafe_convert[new_frame=.WORLD](),
                        world._max.unsafe_convert[new_frame=.WORLD](),
                        depth,
                    )
                )
            else:
                boxes.append(
                    DebugBox(
                        bounds._min.unsafe_convert[new_frame=.WORLD](),
                        bounds._max.unsafe_convert[new_frame=.WORLD](),
                        depth,
                    )
                )
            var internal = not is_leaf_ref(node.data[lane])
            var child = decode_ref_index(node.data[lane])
            comptime if packed:
                internal = _wide_meta_count(node.data[lane]) == 0
                child = _wide_meta_data(node.data[lane])
            if internal:
                _node_boxes[frame, width, packed](
                    nodes, Int(child), depth + 1, transform, boxes, limit
                )


def _blas_boxes[
    kind: PrimitiveKind, width: SIMDLength, frame: Frame
](
    blases: CpuBlasSet[kind, width],
    index: Int,
    transform: Affine3f32[.LOCAL, .WORLD],
    mut boxes: List[DebugBox],
    limit: Int,
):
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), UInt32(index))
    if desc.node_count:
        var view = blases._view[frame](desc)
        _node_boxes[frame, width, True](
            view.nodes, Int(desc.root_idx), 0, transform, boxes, limit
        )


def cpu_bvh_boxes(
    world: CpuScene[16, 16], limit: Int = 20000
) -> List[DebugBox]:
    var boxes = List[DebugBox]()
    var identity = Affine3f32[.LOCAL, .WORLD].identity()
    if world.sphere_bvh:
        _blas_boxes[.SPHERE, 16, .WORLD](
            world.sphere_bvh.value(), 0, identity, boxes, limit
        )
    if world.triangle_bvh:
        _blas_boxes[.TRIANGLE, 16, .WORLD](
            world.triangle_bvh.value(), 0, identity, boxes, limit
        )
    if world.triangle_tlas:
        ref nodes = world.triangle_tlas.value().tree.nodes
        if len(nodes):
            _node_boxes(nodes, 0, 0, identity, boxes, limit)
    if world.triangle_mesh_blases:
        for instance in world.scene_data().triangle_instances():
            if len(boxes) >= limit:
                break
            _blas_boxes[.TRIANGLE, 16, .LOCAL](
                world.triangle_mesh_blases.value(),
                Int(instance.blas_idx),
                instance.transform,
                boxes,
                limit,
            )
    return boxes^
