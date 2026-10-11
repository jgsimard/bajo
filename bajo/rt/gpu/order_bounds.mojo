"""Scene bounds for geometry-independent spatial ray grouping."""

from std.math import min, max
from std.memory import bitcast
from bajo.bvh.constants import EMPTY_LANE, f32_max
from bajo.bvh.gpu.compressed_bounds_bvh import _scale_from_exponent
from bajo.core import Point3f32
from bajo.rt.gpu.config import GpuRtSceneKind, GpuRtBvhFormat
from bajo.rt.gpu.views import GpuRtSceneView


@always_inline
def _wide_order_bounds[
    width: Int
](
    nodes: Pointer[Float32, ImmUntrackedOrigin],
    root: UInt32,
) -> Tuple[
    Point3f32[.WORLD], Point3f32[.WORLD]
]:
    var low = Point3f32[.WORLD](f32_max, f32_max, f32_max)
    var high = Point3f32[.WORLD](-f32_max, -f32_max, -f32_max)
    var base = Int(root) * 7 * width
    comptime for lane in range(width):
        if (
            bitcast[.uint32](nodes[unsafe_offset=base + 6 * width + lane])
            != EMPTY_LANE
        ):
            low.x = min(low.x, nodes[unsafe_offset=base + lane])
            low.y = min(low.y, nodes[unsafe_offset=base + width + lane])
            low.z = min(low.z, nodes[unsafe_offset=base + 2 * width + lane])
            high.x = max(high.x, nodes[unsafe_offset=base + 3 * width + lane])
            high.y = max(high.y, nodes[unsafe_offset=base + 4 * width + lane])
            high.z = max(high.z, nodes[unsafe_offset=base + 5 * width + lane])
    return low, high


@always_inline
def _compressed_order_bounds(
    nodes: Pointer[Float32, ImmUntrackedOrigin],
    root: UInt32,
) -> Tuple[Point3f32[.WORLD], Point3f32[.WORLD]]:
    var base = Int(root) * 20
    var low = Point3f32[.WORLD](
        nodes[unsafe_offset=base],
        nodes[unsafe_offset=base + 1],
        nodes[unsafe_offset=base + 2],
    )
    var exp = bitcast[.uint32](nodes[unsafe_offset=base + 3])
    var high = Point3f32[.WORLD](
        low.x + 255.0 * _scale_from_exponent(exp & 255),
        low.y + 255.0 * _scale_from_exponent((exp >> 8) & 255),
        low.z + 255.0 * _scale_from_exponent((exp >> 16) & 255),
    )
    return low, high


@always_inline
def gpu_rt_order_bounds[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
](scene: GpuRtSceneView) -> Tuple[Point3f32[.WORLD], Point3f32[.WORLD]]:
    var low = Point3f32[.WORLD](f32_max, f32_max, f32_max)
    var high = Point3f32[.WORLD](-f32_max, -f32_max, -f32_max)

    @always_inline
    def include(a: Point3f32[.WORLD], b: Point3f32[.WORLD]) {mut}:
        low.x = min(low.x, a.x)
        low.y = min(low.y, a.y)
        low.z = min(low.z, a.z)
        high.x = max(high.x, b.x)
        high.y = max(high.y, b.y)
        high.z = max(high.z, b.z)

    comptime if kind.has_spheres():
        var s = scene.spheres.unsafe_value()
        var a, b = _wide_order_bounds[sphere_format.node_width](s.nodes, s.root)
        include(a, b)
    comptime if kind.has_triangles():
        var t = scene.triangles.unsafe_value()
        var a: Point3f32[.WORLD]
        var b: Point3f32[.WORLD]
        comptime if triangle_format.layout.compressed:
            a, b = _compressed_order_bounds(t.nodes, t.root)
        else:
            a, b = _wide_order_bounds[triangle_format.node_width](
                t.nodes, t.root
            )
        include(a, b)
    comptime if kind.has_instances():
        var i = scene.instances.unsafe_value()
        var a, b = _wide_order_bounds[tlas_format.node_width](
            i.tlas_nodes, i.tlas_root
        )
        include(a, b)
    return low, high
