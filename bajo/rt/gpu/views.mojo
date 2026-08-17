"""Compact device-passable views for GPU RT scene and queue state."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host import DeviceBuffer

from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena


@always_inline
def _immut[
    dtype: DType
](buffer: DeviceBuffer[dtype]) -> Pointer[Scalar[dtype], ImmUntrackedOrigin]:
    return buffer.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()


@always_inline
def _mut[
    dtype: DType
](buffer: DeviceBuffer[dtype]) -> Pointer[Scalar[dtype], MutUntrackedOrigin]:
    return (
        buffer.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


@fieldwise_init
struct GpuRtSphereView:
    """Host-side inputs for the sphere portion of a flat scene view."""

    var nodes: Pointer[Float32, ImmUntrackedOrigin]
    var leaves: Pointer[Float32, ImmUntrackedOrigin]
    var root: UInt32
    var surfaces: Pointer[UInt32, ImmUntrackedOrigin]
    var signed_radii: Pointer[Float32, ImmUntrackedOrigin]


@fieldwise_init
struct GpuRtTriangleView:
    """Host-side inputs for the static-triangle portion of a scene view."""

    var nodes: Pointer[Float32, ImmUntrackedOrigin]
    var leaves: Pointer[Float32, ImmUntrackedOrigin]
    var root: UInt32
    var surfaces: Pointer[UInt32, ImmUntrackedOrigin]


@fieldwise_init
struct GpuRtInstanceView:
    """Host-side inputs for the triangle-instance portion of a scene view."""

    var tlas_nodes: Pointer[Float32, ImmUntrackedOrigin]
    var tlas_leaf_instances: Pointer[UInt32, ImmUntrackedOrigin]
    var inv_transforms: Pointer[Float32, ImmUntrackedOrigin]
    var blas_indices: Pointer[UInt32, ImmUntrackedOrigin]
    var blas_descs: Pointer[UInt32, ImmUntrackedOrigin]
    var blas_nodes: Pointer[Float32, ImmUntrackedOrigin]
    var blas_leaves: Pointer[Float32, ImmUntrackedOrigin]
    var tlas_root: UInt32
    var count: Int32
    var surfaces: Pointer[UInt32, ImmUntrackedOrigin]


@fieldwise_init
struct GpuRtShadingView:
    """Host-side inputs for material and direct-light tables."""

    var emissives: Pointer[Float32, ImmUntrackedOrigin]
    var lambertians: Pointer[Float32, ImmUntrackedOrigin]
    var metals: Pointer[Float32, ImmUntrackedOrigin]
    var dielectrics: Pointer[Float32, ImmUntrackedOrigin]
    var light_kinds: Pointer[UInt32, ImmUntrackedOrigin]
    var light_fields: Pointer[Float32, ImmUntrackedOrigin]
    var light_count: Int32
    var total_light_weight: Float32


@fieldwise_init
struct GpuRtSceneView(DevicePassable, TrivialRegisterPassable):
    """Non-owning pointers to every optional scene component.

    Geometry-presence parameters on traversal functions erase accesses to
    absent components, allowing one source implementation to specialize to
    spheres, triangles, instances, or any combination.
    """

    var sphere_nodes: Pointer[Float32, ImmUntrackedOrigin]
    var sphere_leaves: Pointer[Float32, ImmUntrackedOrigin]
    var sphere_root: UInt32
    var sphere_surfaces: Pointer[UInt32, ImmUntrackedOrigin]
    var signed_radii: Pointer[Float32, ImmUntrackedOrigin]
    var triangle_nodes: Pointer[Float32, ImmUntrackedOrigin]
    var triangle_leaves: Pointer[Float32, ImmUntrackedOrigin]
    var triangle_root: UInt32
    var triangle_surfaces: Pointer[UInt32, ImmUntrackedOrigin]
    var tlas_nodes: Pointer[Float32, ImmUntrackedOrigin]
    var tlas_leaf_instances: Pointer[UInt32, ImmUntrackedOrigin]
    var inst_inv_transform: Pointer[Float32, ImmUntrackedOrigin]
    var inst_blas_indices: Pointer[UInt32, ImmUntrackedOrigin]
    var blas_descs: Pointer[UInt32, ImmUntrackedOrigin]
    var blas_nodes: Pointer[Float32, ImmUntrackedOrigin]
    var blas_leaves: Pointer[Float32, ImmUntrackedOrigin]
    var tlas_root: UInt32
    var instance_count: Int32
    var instance_surfaces: Pointer[UInt32, ImmUntrackedOrigin]
    var emissives: Pointer[Float32, ImmUntrackedOrigin]
    var lambertians: Pointer[Float32, ImmUntrackedOrigin]
    var metals: Pointer[Float32, ImmUntrackedOrigin]
    var dielectrics: Pointer[Float32, ImmUntrackedOrigin]
    var light_kinds: Pointer[UInt32, ImmUntrackedOrigin]
    var light_fields: Pointer[Float32, ImmUntrackedOrigin]
    var light_count: Int32
    var total_light_weight: Float32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GpuRtSceneView"


@always_inline
def gpu_rt_scene_view(
    spheres: GpuRtSphereView,
    triangles: GpuRtTriangleView,
    instances: GpuRtInstanceView,
    shading: GpuRtShadingView,
) -> GpuRtSceneView:
    """Flatten host-side component views into the stable device ABI."""
    return GpuRtSceneView(
        spheres.nodes,
        spheres.leaves,
        spheres.root,
        spheres.surfaces,
        spheres.signed_radii,
        triangles.nodes,
        triangles.leaves,
        triangles.root,
        triangles.surfaces,
        instances.tlas_nodes,
        instances.tlas_leaf_instances,
        instances.inv_transforms,
        instances.blas_indices,
        instances.blas_descs,
        instances.blas_nodes,
        instances.blas_leaves,
        instances.tlas_root,
        instances.count,
        instances.surfaces,
        shading.emissives,
        shading.lambertians,
        shading.metals,
        shading.dielectrics,
        shading.light_kinds,
        shading.light_fields,
        shading.light_count,
        shading.total_light_weight,
    )


@always_inline
def empty_gpu_rt_sphere_view(
    dummy_f32: Pointer[Float32, ImmUntrackedOrigin],
    dummy_u32: Pointer[UInt32, ImmUntrackedOrigin],
) -> GpuRtSphereView:
    return GpuRtSphereView(
        dummy_f32, dummy_f32, UInt32(0), dummy_u32, dummy_f32
    )


@always_inline
def empty_gpu_rt_triangle_view(
    dummy_f32: Pointer[Float32, ImmUntrackedOrigin],
    dummy_u32: Pointer[UInt32, ImmUntrackedOrigin],
) -> GpuRtTriangleView:
    return GpuRtTriangleView(dummy_f32, dummy_f32, UInt32(0), dummy_u32)


@always_inline
def empty_gpu_rt_instance_view(
    dummy_f32: Pointer[Float32, ImmUntrackedOrigin],
    dummy_u32: Pointer[UInt32, ImmUntrackedOrigin],
) -> GpuRtInstanceView:
    return GpuRtInstanceView(
        dummy_f32,
        dummy_u32,
        dummy_f32,
        dummy_u32,
        dummy_u32,
        dummy_f32,
        dummy_f32,
        UInt32(0),
        Int32(0),
        dummy_u32,
    )


@fieldwise_init
struct GpuRtTraceQueueView(DevicePassable, TrivialRegisterPassable):
    """Non-owning path input, shade outputs, counters, and radiance view."""

    var src_path_ids: Pointer[UInt32, ImmUntrackedOrigin]
    var src_path_fields: Pointer[Float32, ImmUntrackedOrigin]
    var dst_path_ids: Pointer[UInt32, MutUntrackedOrigin]
    var dst_path_fields: Pointer[Float32, MutUntrackedOrigin]
    var shade_path_refs: Pointer[UInt32, MutUntrackedOrigin]
    var shade_surfaces: Pointer[UInt32, MutUntrackedOrigin]
    var shade_fields: Pointer[Float32, MutUntrackedOrigin]
    var shadow_path_ids: Pointer[UInt32, MutUntrackedOrigin]
    var shadow_fields: Pointer[Float32, MutUntrackedOrigin]
    var counters: Pointer[UInt32, MutUntrackedOrigin]
    var sample_radiance: Pointer[Float32, MutUntrackedOrigin]
    var capacity: Int32
    var sample_base: UInt32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GpuRtTraceQueueView"


@always_inline
def gpu_rt_trace_queue_view(
    arena: GpuWavefrontArena,
    src_path_ids: DeviceBuffer[DType.uint32],
    src_path_fields: DeviceBuffer[DType.float32],
    dst_path_ids: DeviceBuffer[DType.uint32],
    dst_path_fields: DeviceBuffer[DType.float32],
) -> GpuRtTraceQueueView:
    """Assemble the shared wavefront queues for one bounce submission."""
    return GpuRtTraceQueueView(
        _immut(src_path_ids),
        _immut(src_path_fields),
        _mut(dst_path_ids),
        _mut(dst_path_fields),
        _mut(arena.shade.path_refs),
        _mut(arena.shade.surface_values),
        _mut(arena.shade.fields),
        _mut(arena.shadow.path_ids),
        _mut(arena.shadow.fields),
        _mut(arena.counters),
        _mut(arena.sample_radiance),
        Int32(arena.capacity),
        arena.sample_base,
    )
