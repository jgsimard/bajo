"""Construction policies for immutable prepared GPU RT snapshots.

Preparation uploads a finalized `SceneData` snapshot. Author with
`SceneBuilder`, call its consuming `finish()`, and prepare each backend from the
result; there is no implicit device update, dirty tracking, or refit.
"""

from max.gpu.host import DeviceContext

from bajo.bvh.gpu import GpuBvhBuildMethod, GpuBvhLayout
from bajo.rt.types import SceneData
from bajo.rt.gpu.sphere_path import GpuRtSphereScene
from bajo.rt.gpu.triangle_path import GpuRtTriangleScene
from bajo.rt.gpu.mixed_path import GpuRtMixedScene
from bajo.rt.gpu.instance_path import GpuRtTriangleInstanceScene
from bajo.rt.gpu.combined_instance_path import GpuRtCombinedInstanceScene


@fieldwise_init
struct GpuRtBvhPolicy:
    """Compile-time BVH layout and builder selection."""

    var node_width: Int
    var leaf_width: Int
    var build_method: GpuBvhBuildMethod
    var layout: GpuBvhLayout


comptime GPU_RT_BVH_WIDE4_LBVH = GpuRtBvhPolicy(4, 4, .LBVH, .WIDE)
comptime GPU_RT_BVH_CWBVH8_HPLOC = GpuRtBvhPolicy(
    8, 4, .HPLOC, .CWBVH8
)
comptime GPU_RT_BVH_TLAS2_LBVH = GpuRtBvhPolicy(
    2, 1, .LBVH, .WIDE
)


def prepare_gpu_sphere_scene[
    policy: GpuRtBvhPolicy = GPU_RT_BVH_WIDE4_LBVH,
](
    mut ctx: DeviceContext,
    data: SceneData,
) raises -> GpuRtSphereScene[
    policy.node_width, policy.leaf_width
]:
    """Upload a sphere scene using one compile-time BVH policy value."""
    return GpuRtSphereScene[policy.node_width, policy.leaf_width](ctx, data)


def prepare_gpu_triangle_scene[
    policy: GpuRtBvhPolicy = GPU_RT_BVH_CWBVH8_HPLOC,
](
    mut ctx: DeviceContext,
    data: SceneData,
) raises -> GpuRtTriangleScene[
    policy.node_width,
    policy.leaf_width,
    policy.build_method,
    policy.layout,
]:
    """Upload a static-triangle scene using one BVH policy value."""
    return GpuRtTriangleScene[
        policy.node_width,
        policy.leaf_width,
        policy.build_method,
        policy.layout,
    ](ctx, data)


def prepare_gpu_mixed_scene[
    sphere_policy: GpuRtBvhPolicy = GPU_RT_BVH_WIDE4_LBVH,
    triangle_policy: GpuRtBvhPolicy = GPU_RT_BVH_CWBVH8_HPLOC,
](
    mut ctx: DeviceContext,
    data: SceneData,
) raises -> GpuRtMixedScene[
    sphere_policy.node_width,
    sphere_policy.leaf_width,
    triangle_policy.node_width,
    triangle_policy.leaf_width,
    triangle_policy.build_method,
    triangle_policy.layout,
]:
    """Upload a mixed static scene using sphere and triangle policies."""
    return GpuRtMixedScene[
        sphere_policy.node_width,
        sphere_policy.leaf_width,
        triangle_policy.node_width,
        triangle_policy.leaf_width,
        triangle_policy.build_method,
        triangle_policy.layout,
    ](ctx, data)


def prepare_gpu_triangle_instance_scene[
    tlas_policy: GpuRtBvhPolicy = GPU_RT_BVH_TLAS2_LBVH,
    blas_policy: GpuRtBvhPolicy = GPU_RT_BVH_CWBVH8_HPLOC,
](
    mut ctx: DeviceContext,
    data: SceneData,
) raises -> GpuRtTriangleInstanceScene[
    tlas_policy.node_width,
    blas_policy.node_width,
    tlas_policy.leaf_width,
    blas_policy.leaf_width,
    blas_policy.build_method,
    blas_policy.layout,
    tlas_policy.build_method,
]:
    """Upload an instanced scene using one TLAS and one BLAS policy."""
    return GpuRtTriangleInstanceScene[
        tlas_policy.node_width,
        blas_policy.node_width,
        tlas_policy.leaf_width,
        blas_policy.leaf_width,
        blas_policy.build_method,
        blas_policy.layout,
        tlas_policy.build_method,
    ](ctx, data)


def prepare_gpu_combined_instance_scene[
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    sphere_policy: GpuRtBvhPolicy = GPU_RT_BVH_WIDE4_LBVH,
    triangle_policy: GpuRtBvhPolicy = GPU_RT_BVH_CWBVH8_HPLOC,
    tlas_policy: GpuRtBvhPolicy = GPU_RT_BVH_TLAS2_LBVH,
    blas_policy: GpuRtBvhPolicy = GPU_RT_BVH_CWBVH8_HPLOC,
](
    mut ctx: DeviceContext,
    data: SceneData,
) raises -> GpuRtCombinedInstanceScene[
    HAS_SPHERES,
    HAS_TRIANGLES,
    sphere_policy.node_width,
    sphere_policy.leaf_width,
    tlas_policy.node_width,
    tlas_policy.leaf_width,
    blas_policy.node_width,
    blas_policy.leaf_width,
    blas_policy.build_method,
    blas_policy.layout,
    triangle_policy.node_width,
    triangle_policy.leaf_width,
    triangle_policy.build_method,
    triangle_policy.layout,
    tlas_policy.build_method,
]:
    """Upload the full scene with four named compile-time policies."""
    return GpuRtCombinedInstanceScene[
        HAS_SPHERES,
        HAS_TRIANGLES,
        sphere_policy.node_width,
        sphere_policy.leaf_width,
        tlas_policy.node_width,
        tlas_policy.leaf_width,
        blas_policy.node_width,
        blas_policy.leaf_width,
        blas_policy.build_method,
        blas_policy.layout,
        triangle_policy.node_width,
        triangle_policy.leaf_width,
        triangle_policy.build_method,
        triangle_policy.layout,
        tlas_policy.build_method,
    ](ctx, data)
