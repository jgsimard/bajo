from bajo.core.bvh.types import Ray
from bajo.core.intersect import intersect_ray_tri
from bajo.core.vec import Vec3f32


@always_inline
def push_near_far[
    order_by_dist: Bool = True,
    stack_size: Int = 64,
](
    mut stack: InlineArray[UInt32, stack_size],
    mut stack_ptr: Int,
    left_idx: UInt32,
    right_idx: UInt32,
    dist_left: Float32,
    dist_right: Float32,
) -> UInt32:
    """Push the farther child and return the nearer child.

    Caller should only call this when both children were hit.
    """

    var near = left_idx
    var far = right_idx

    comptime if order_by_dist:
        if dist_left > dist_right:
            near = right_idx
            far = left_idx

    debug_assert["safe"](stack_ptr < stack_size, "BVH traversal stack overflow")

    stack[stack_ptr] = far
    stack_ptr += 1

    return near


@always_inline
def pop_stack_or_done[
    stack_size: Int = 64,
](
    mut stack: InlineArray[UInt32, stack_size],
    mut stack_ptr: Int,
    mut node_idx: UInt32,
) -> Bool:
    """Pop the next node from the traversal stack.

    Returns False when traversal is finished.
    """

    if stack_ptr == 0:
        return False

    stack_ptr -= 1
    node_idx = stack[stack_ptr]
    return True


@always_inline
def advance_to_child_hits[
    order_by_dist: Bool = True,
    stack_size: Int = 64,
](
    mut stack: InlineArray[UInt32, stack_size],
    mut stack_ptr: Int,
    mut node_idx: UInt32,
    left_idx: UInt32,
    right_idx: UInt32,
    hit_left: Bool,
    hit_right: Bool,
    dist_left: Float32,
    dist_right: Float32,
) -> Bool:
    """Advance traversal after testing both children.

    Returns False when traversal is finished.
    """

    if not hit_left and not hit_right:
        return pop_stack_or_done(stack, stack_ptr, node_idx)

    if hit_left and not hit_right:
        node_idx = left_idx
        return True

    if not hit_left and hit_right:
        node_idx = right_idx
        return True

    node_idx = push_near_far[order_by_dist](
        stack,
        stack_ptr,
        left_idx,
        right_idx,
        dist_left,
        dist_right,
    )
    return True


@always_inline
def intersect_prim[
    is_shadow: Bool,
](
    vertices: UnsafePointer[Vec3f32, MutAnyOrigin],
    mut ray: Ray,
    prim_idx: UInt32,
) -> Bool:
    """Intersect one triangle primitive and optionally update the ray hit."""

    var base = Int(prim_idx) * 3

    ref v0 = vertices[base + 0]
    ref v1 = vertices[base + 1]
    ref v2 = vertices[base + 2]

    var h = intersect_ray_tri(
        ray.O,
        ray.D,
        v0,
        v1,
        v2,
        ray.hit.t,
    )

    if not h.mask[0]:
        return False

    comptime if is_shadow:
        return True

    ray.hit.t = h.t[0]
    ray.hit.u = h.u[0]
    ray.hit.v = h.v[0]
    ray.hit.prim = prim_idx

    return True
