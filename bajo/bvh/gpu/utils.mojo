from std.gpu import DeviceContext, DeviceBuffer

from bajo.bvh.camera import Camera
from bajo.core import Vec3f32


@fieldwise_init
struct GpuBuildTimings(TrivialRegisterPassable, Writable):
    var morton_ns: Int
    var sort_ns: Int
    var topology_ns: Int
    var refit_ns: Int
    var collapse_ns: Int
    var bounds_pack_ns: Int
    var leaf_pack_ns: Int

    @staticmethod
    def empty() -> Self:
        return Self(0, 0, 0, 0, 0, 0, 0)

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


@fieldwise_init
struct GpuBuildResult(TrivialRegisterPassable):
    var static_setup_ns: Int
    var timings: GpuBuildTimings
    var validation: GpuBVHValidation


def _download_full_hit_checksum(
    ctx: DeviceContext,
    d_hits_f32: DeviceBuffer[DType.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32]:
    var checksum = 0.0
    var hit_count = UInt32(0)
    with d_hits_f32.map_to_host() as h:
        for i in range(ray_count):
            var t = h[i * 3]
            if t < 1.0e20:
                checksum += Float64(t)
                hit_count += 1
    ctx.synchronize()
    return (checksum, hit_count)


def upload_camera(
    mut ctx: DeviceContext,
    camera: Camera,
) raises -> DeviceBuffer[DType.float32]:
    var params = camera.flatten()
    return upload_list(ctx, params)


def upload_vertices(
    mut ctx: DeviceContext,
    verts: List[Vec3f32],
) raises -> DeviceBuffer[DType.float32]:
    var flat = List[Float32](capacity=len(verts) * 3)
    for v in verts:
        flat.append(v.x)
        flat.append(v.y)
        flat.append(v.z)
    return upload_list(ctx, flat)


def upload_list[
    dtype: DType
](
    mut ctx: DeviceContext,
    a: List[Scalar[dtype]],
) raises -> DeviceBuffer[
    dtype
]:
    var h_a = ctx.enqueue_create_host_buffer[dtype](len(a))
    var d_a = ctx.enqueue_create_buffer[dtype](len(a))
    h_a.enqueue_copy_from(Span(a))
    h_a.enqueue_copy_to(d_a)
    return d_a^
