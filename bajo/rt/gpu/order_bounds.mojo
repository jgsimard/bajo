"""Scene bounds for geometry-independent spatial ray grouping."""

from layout import TileTensor, row_major
from std.utils.numerics import max_finite

from std.math import min, max
from std.memory import bitcast
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.gpu.compressed_bounds_bvh import _scale_from_exponent
from bajo.core import Point3f32
from bajo.rt.gpu.config import (
    _GpuRtKernelFormats,
    GpuRtSceneKind,
)
from bajo.rt.gpu.views import GpuRtSceneView


def _wide_order_bounds[
    width: Int
](
    nodes: Pointer[Float32, ImmUntrackedOrigin],
    root: UInt32,
) -> Tuple[
    Point3f32[.WORLD], Point3f32[.WORLD]
]:
    var low = Point3f32[.WORLD](
        max_finite[.float32](), max_finite[.float32](), max_finite[.float32]()
    )
    var high = Point3f32[.WORLD](
        -max_finite[.float32](),
        -max_finite[.float32](),
        -max_finite[.float32](),
    )
    var fields = TileTensor(
        ptr=nodes, layout=row_major(Int(root) + 1, 7, width)
    )
    comptime for lane in range(width):
        if bitcast[.uint32](fields[Int(root), 6, lane]) != EMPTY_LANE:
            low.x = min(low.x, fields[Int(root), 0, lane])
            low.y = min(low.y, fields[Int(root), 1, lane])
            low.z = min(low.z, fields[Int(root), 2, lane])
            high.x = max(high.x, fields[Int(root), 3, lane])
            high.y = max(high.y, fields[Int(root), 4, lane])
            high.z = max(high.z, fields[Int(root), 5, lane])
    return low, high


def _compressed_order_bounds(
    nodes: Pointer[Float32, ImmUntrackedOrigin],
    root: UInt32,
) -> Tuple[Point3f32[.WORLD], Point3f32[.WORLD]]:
    var fields = TileTensor(ptr=nodes, layout=row_major(Int(root) + 1, 20))
    var low = Point3f32[.WORLD](fields.ptr_at_offset((Int(root), 0)))
    var exp = bitcast[.uint32](fields[Int(root), 3])
    var high = Point3f32[.WORLD](
        low.x + 255.0 * _scale_from_exponent(exp & 255),
        low.y + 255.0 * _scale_from_exponent((exp >> 8) & 255),
        low.z + 255.0 * _scale_from_exponent((exp >> 16) & 255),
    )
    return low, high


def gpu_rt_order_bounds[
    kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
](scene: GpuRtSceneView) -> Tuple[Point3f32[.WORLD], Point3f32[.WORLD]]:
    var low = Point3f32[.WORLD](
        max_finite[.float32](), max_finite[.float32](), max_finite[.float32]()
    )
    var high = Point3f32[.WORLD](
        -max_finite[.float32](),
        -max_finite[.float32](),
        -max_finite[.float32](),
    )

    def include(a: Point3f32[.WORLD], b: Point3f32[.WORLD]) {mut}:
        low.x = min(low.x, a.x)
        low.y = min(low.y, a.y)
        low.z = min(low.z, a.z)
        high.x = max(high.x, b.x)
        high.y = max(high.y, b.y)
        high.z = max(high.z, b.z)

    comptime if kind.has_spheres():
        var s = scene.spheres.unsafe_value()
        var a, b = _wide_order_bounds[Formats.sphere.node_width](
            s.nodes, s.root
        )
        include(a, b)
    comptime if kind.has_triangles():
        var t = scene.triangles.unsafe_value()
        var a: Point3f32[.WORLD]
        var b: Point3f32[.WORLD]
        comptime if Formats.triangle.layout == .CWBVH8:
            a, b = _compressed_order_bounds(t.nodes, t.root)
        else:
            a, b = _wide_order_bounds[Formats.triangle.node_width](
                t.nodes, t.root
            )
        include(a, b)
    comptime if kind.has_instances():
        var i = scene.instances.unsafe_value()
        var a, b = _wide_order_bounds[Formats.tlas.node_width](
            i.tlas_nodes, i.tlas_root
        )
        include(a, b)
    return low, high
