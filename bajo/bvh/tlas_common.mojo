"""Hit policies shared by CPU and GPU TLAS traversal."""

from bajo.bvh.types import Hit
from bajo.core import Affine3f32, Frame


@always_inline
def promote_tlas_local_hit[
    local_frame: Frame
](
    local_hit: Hit[local_frame],
    instance_idx: UInt32,
    mut world_hit: Hit[.WORLD],
) -> Bool:
    """Promote a closer BLAS-local candidate without transforming its normal."""
    comptime assert local_frame == .LOCAL
    if not local_hit.is_hit() or local_hit.t >= world_hit.t:
        return False

    world_hit.t = local_hit.t
    world_hit.u = local_hit.u
    world_hit.v = local_hit.v
    world_hit.prim = local_hit.prim
    world_hit.inst = instance_idx
    # The TLAS transforms only the final winning normal after traversal.
    world_hit.normal = local_hit.normal.unsafe_convert[new_frame=.WORLD]()
    return True


@always_inline
def finalize_tlas_hit_normal(
    mut hit: Hit[.WORLD],
    inverse: Affine3f32[.WORLD, .LOCAL],
):
    """Transform the final provisional BLAS-local normal into world space."""
    hit.normal = Affine3f32[.LOCAL, .WORLD].normal_from_inverse(
        hit.normal.unsafe_convert[new_frame=.LOCAL](), inverse
    )
