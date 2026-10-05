"""Allocation-free device-input updates for the preorder spatial forest.

One lane validates and updates the forest. This deliberately simple baseline
retains every published address, needs no host geometry transfer and supports
stream capture. Rebuild sorts leaf IDs by X centroid within each independent
root, then refits the retained balanced skeleton. Group membership stays fixed.
"""

from max.gpu import global_idx
from std.math import min, max
from std.utils.numerics import isfinite
from bajo.bvh.constants import EMPTY_LANE


@no_inline
def refit_forest(
    items: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
):
    for node in range(len(nodes) // 2 - 1, -1, -1):
        var item = nodes[node * 2 + 1]
        if item != EMPTY_LANE:
            for axis in range(6):
                bounds[node * 6 + axis] = items[Int(item) * 6 + axis]
        else:
            var left = node + 1
            var right = Int(nodes[left * 2])
            for axis in range(3):
                bounds[node * 6 + axis] = min(
                    bounds[left * 6 + axis], bounds[right * 6 + axis]
                )
                bounds[node * 6 + axis + 3] = max(
                    bounds[left * 6 + axis + 3], bounds[right * 6 + axis + 3]
                )


@no_inline
def reorder_forest(
    items: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    order: MutSpan[UInt32, MutAnyOrigin],
):
    var root = 0
    while root < len(nodes) // 2:
        var end = Int(nodes[root * 2])
        var count = 0
        for node in range(root, end):
            var item = nodes[node * 2 + 1]
            if item != EMPTY_LANE:
                order[root + count] = item
                count += 1
        # Stable insertion sort is a correctness baseline, not a throughput
        # claim. Float64 centroids avoid overflow for finite Float32 boxes.
        for i in range(1, count):
            var item = order[root + i]
            var center = Float64(items[Int(item) * 6]) + Float64(
                items[Int(item) * 6 + 3]
            )
            var j = i
            while j > 0:
                var previous = order[root + j - 1]
                var previous_center = Float64(
                    items[Int(previous) * 6]
                ) + Float64(items[Int(previous) * 6 + 3])
                if previous_center < center or (
                    previous_center == center and previous < item
                ):
                    break
                order[root + j] = previous
                j -= 1
            order[root + j] = item
        count = 0
        for node in range(root, end):
            if nodes[node * 2 + 1] != EMPTY_LANE:
                nodes[node * 2 + 1] = order[root + count]
                count += 1
        root = end


def bounds_update_kernel[
    rebuild: Bool
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    items: MutSpan[Float32, MutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    order: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    if global_idx.x != 0:
        return
    status[0] = 1
    for item in range(len(input) // 6):
        for axis in range(3):
            var lo = input[item * 6 + axis]
            var hi = input[item * 6 + axis + 3]
            if not isfinite(lo) or not isfinite(hi) or lo > hi:
                return
    # Validation completes before any geometry or topology buffer is mutated.
    for i in range(len(input)):
        items[i] = input[i]
    comptime if rebuild:
        reorder_forest(items, nodes, order)
    refit_forest(items, bounds, nodes)
    status[0] = 0


def mesh_update_kernel[
    rebuild: Bool
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    points: MutSpan[Float32, MutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    items: MutSpan[Float32, MutAnyOrigin],
    bounds: MutSpan[Float32, MutAnyOrigin],
    nodes: MutSpan[UInt32, MutAnyOrigin],
    order: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    if global_idx.x != 0:
        return
    status[0] = 1
    for value in input:
        if not isfinite(value):
            return
    for face in range(len(indices) // 3):
        for axis in range(3):
            var lo = input[Int(indices[face * 3]) * 3 + axis]
            var hi = lo
            for corner in range(1, 3):
                var value = input[Int(indices[face * 3 + corner]) * 3 + axis]
                lo = min(lo, value)
                hi = max(hi, value)
            items[face * 6 + axis] = lo
            items[face * 6 + axis + 3] = hi
    for i in range(len(input)):
        points[i] = input[i]
    comptime if rebuild:
        reorder_forest(items, nodes, order)
    refit_forest(items, bounds, nodes)
    status[0] = 0


def velocity_update_kernel[
    checked: Bool
](
    input: ImmSpan[Float32, ImmutAnyOrigin],
    velocities: MutSpan[Float32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    if global_idx.x != 0:
        return
    status[0] = 1
    for value in input:
        if not isfinite(value):
            return
    for i in range(len(velocities)):
        velocities[i] = input[i] if len(input) > 0 else 0.0
    status[0] = 0
