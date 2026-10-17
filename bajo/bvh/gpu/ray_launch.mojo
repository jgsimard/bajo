"""Shared packed-ray launch ABI for GPU BVH front ends."""

from max.gpu.host import DeviceBuffer
from std.memory import bitcast

from bajo.bvh.types import Hit
from bajo.core import Frame, Point3f32, Rayf32, Vec3f32
from bajo.core.tensor_layout import field_tile


def validate_ray_launch(
    d_rays: DeviceBuffer[.float32],
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
) raises:
    if ray_count <= 0 or ray_count > Int(Int32.MAX):
        raise Error("ray count must fit a positive Int32")
    if ray_count > len(d_rays) // Rayf32.STRIDE:
        raise Error("packed ray input buffer is too short")
    if ray_count > len(d_hits) // Hit.STRIDE:
        raise Error("hit output buffer is too short")


@always_inline
def _load_packed_ray[
    frame: Frame,
](rays: ImmPointer[Float32, _], ray_count: Int, ray_idx: Int) -> Rayf32[frame]:
    # Field-major storage keeps each warp's scalar loads coalesced.
    var fields = field_tile[.float32, Rayf32.STRIDE](rays, ray_count)
    return Rayf32[frame](
        Point3f32[frame](
            fields[Rayf32.ORIGIN, ray_idx],
            fields[Rayf32.ORIGIN + 1, ray_idx],
            fields[Rayf32.ORIGIN + 2, ray_idx],
        ),
        Vec3f32[frame](
            fields[Rayf32.DIRECTION, ray_idx],
            fields[Rayf32.DIRECTION + 1, ray_idx],
            fields[Rayf32.DIRECTION + 2, ray_idx],
        ),
        fields[Rayf32.T_MIN, ray_idx],
        fields[Rayf32.T_MAX, ray_idx],
    )


@always_inline
def _store_packed_hit[
    frame: Frame,
](hit: Hit[frame], hits: MutPointer[Float32, _], ray_count: Int, ray_idx: Int,):
    hit._store_unchecked(
        Span(unsafe_ptr=hits, length=ray_count * Hit.STRIDE), ray_idx
    )


@always_inline
def _store_packed_hit_vectorized[
    frame: Frame,
](hit: Hit[frame], hits: MutPointer[Float32, _], ray_count: Int, ray_idx: Int,):
    """Store a hit in two vectors when the output pointer is aligned.

    Instanced traversal benefits from wider stores, including scattered
    outputs after ray ordering. Standalone traversal keeps its scalar stores.
    """
    # A hit occupies 32 bytes, so each record preserves pointer alignment.
    # Pointer views can start at any float offset and need a scalar fallback.
    if (Int(hits) & 15) == 0:
        var base = ray_idx * Hit.STRIDE
        hits.unsafe_store[width=4, alignment=16](
            base,
            SIMD[.float32, 4](
                hit.u[0],
                hit.v[0],
                bitcast[.float32](hit.prim[0]),
                bitcast[.float32](hit.inst[0]),
            ),
        )
        hits.unsafe_store[width=4, alignment=16](
            base + 4,
            SIMD[.float32, 4](
                hit.normal.x[0],
                hit.normal.y[0],
                hit.normal.z[0],
                hit.t[0],
            ),
        )
    else:
        _store_packed_hit(hit, hits, ray_count, ray_idx)
