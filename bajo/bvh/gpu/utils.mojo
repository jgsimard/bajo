from max.gpu.host import DeviceContext, DeviceBuffer

# Keep the existing import path available to callers.
from bajo.gpu.buffers import _device_span, _packed_point3_span, upload_list

from bajo.core.camera import Camera
from bajo.core import Point3f32, Frame, Rayf32
from bajo.bvh.types import Hit
from bajo.bvh.constants import f32_max


@fieldwise_init
struct GpuBuildTimings(TrivialRegisterPassable, Writable):
    var morton_ns: Int
    var sort_ns: Int
    var topology_ns: Int
    var refit_ns: Int
    var collapse_ns: Int
    var bounds_pack_ns: Int
    var leaf_pack_ns: Int

    def total(self) -> Int:
        return (
            self.morton_ns
            + self.sort_ns
            + self.topology_ns
            + self.refit_ns
            + self.collapse_ns
            + self.bounds_pack_ns
            + self.leaf_pack_ns
        )


@fieldwise_init
struct SortedKeysValidation(TrivialRegisterPassable):
    var sorted_ok: Bool
    var values_ok: Bool
    var first_bad_key: Int
    var first_bad_value: Int
    var first_code: UInt32
    var last_code: UInt32
    var guard: UInt64


@fieldwise_init
struct TopologyValidation(TrivialRegisterPassable):
    var ok: Bool
    var root_count: UInt32
    var root_idx: UInt32
    var guard: UInt64


@fieldwise_init
struct RefitBoundsValidation(TrivialRegisterPassable):
    var ok: Bool
    var diff: Float64
    var root_idx: UInt32
    var guard: UInt64


@fieldwise_init
struct GpuBVHValidation(TrivialRegisterPassable):
    var sorted_ok: Bool
    var values_ok: Bool
    var topology_ok: Bool
    var topology_root_count: UInt32
    var topology_root_idx: UInt32
    var bounds_ok: Bool
    var bounds_diff: Float64
    var root_idx: UInt32
    var guard: UInt64


def _download_full_hit_checksum(
    ctx: DeviceContext,
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32]:
    var checksum = 0.0
    var hit_count = UInt32(0)
    with d_hits.map_to_host() as h:
        for i in range(ray_count):
            var t = h[i * Hit.STRIDE + Hit.T]
            if t < f32_max:
                checksum += Float64(t)
                hit_count += 1
    ctx.synchronize()
    return (checksum, hit_count)


def upload_camera(
    mut ctx: DeviceContext,
    camera: Camera,
) raises -> DeviceBuffer[.float32]:
    var params = camera.flatten()
    return upload_list(ctx, params)


def upload_vertices[
    frame: Frame
](
    mut ctx: DeviceContext,
    verts: ImmSpan[Point3f32[frame], _],
) raises -> DeviceBuffer[.float32]:
    return upload_list(ctx, _packed_point3_span(verts))


def upload_rays[
    frame: Frame
](
    mut ctx: DeviceContext,
    rays: ImmSpan[Rayf32[frame], _],
) raises -> DeviceBuffer[.float32]:
    """Upload rays in the field-major, warp-coalesced tracing ABI."""
    var flat = List[Float32](capacity=len(rays) * Rayf32.STRIDE)
    for ray in rays:
        flat.append(ray.o.x)
    for ray in rays:
        flat.append(ray.o.y)
    for ray in rays:
        flat.append(ray.o.z)
    for ray in rays:
        flat.append(ray.t_min)
    for ray in rays:
        flat.append(ray.d.x)
    for ray in rays:
        flat.append(ray.d.y)
    for ray in rays:
        flat.append(ray.d.z)
    for ray in rays:
        flat.append(ray.t_max)
    return upload_list(ctx, flat)
