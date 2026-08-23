"""Generic GPU RT entry point selecting a geometry-specialized pipeline."""

from bajo.bvh import Camera
from bajo.bvh.gpu import GpuBvhBuildMethod, GpuBvhLayout
from max.gpu.host import DeviceContext
from bajo.rt.types import Integrator, RenderResult, RenderSettings, SceneData
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.resources import GpuRtRenderTarget
from bajo.rt.gpu.instance_path import (
    GpuRtTriangleInstanceScene,
    enqueue_render_gpu_triangle_instances,
    render_gpu_triangle_instances,
)
from bajo.rt.gpu.combined_instance_path import (
    GpuRtCombinedInstanceScene,
    enqueue_render_gpu_combined_instances,
    render_gpu_combined_instances,
)
from bajo.rt.gpu.mixed_path import (
    GpuRtMixedScene,
    enqueue_render_gpu_mixed,
    render_gpu_mixed,
)
from bajo.rt.gpu.sphere_path import (
    GpuRtSphereScene,
    enqueue_render_gpu_spheres,
    render_gpu_spheres,
)
from bajo.rt.gpu.triangle_path import (
    GpuRtTriangleScene,
    enqueue_render_gpu_triangles,
    render_gpu_triangles,
)


# A CWBVH node can encode up to 24 leaf triangles. Below this scale there is
# little hierarchy to compress, and the task-mask setup costs more than wide4
# traversal. Select once on the host so device kernels remain format-specialized.
comptime GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD = 32


def _prefer_cwbvh8_triangles(world: SceneData) -> Bool:
    return (
        len(world.triangle_vertices()) / 3
        >= GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD
    )


def _prefer_cwbvh8_blases(world: SceneData) -> Bool:
    var weighted_triangles = 0
    for instance in world.triangle_instances():
        weighted_triangles += (
            len(world.triangle_meshes()[Int(instance.blas_idx)]) / 3
        )
    return weighted_triangles >= (
        len(world.triangle_instances()) * GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD
    )


@always_inline
def enqueue_render_gpu[
    ALGORITHM: Integrator = .PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    scene: GpuRtSphereScene[node_width, leaf_width],
    settings: RenderSettings,
) raises:
    """Submit a prepared sphere scene using the common GPU entry point."""
    enqueue_render_gpu_spheres[ALGORITHM, node_width, leaf_width](
        ctx, target, scene, settings
    )


@always_inline
def enqueue_render_gpu[
    ALGORITHM: Integrator = .PATH,
    node_width: SIMDLength = 8,
    leaf_width: SIMDLength = 4,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
    build_method: GpuBvhBuildMethod = .HPLOC,
    layout: GpuBvhLayout = GpuBvhLayout(
        node_width == 8 and leaf_width == 4
    ),
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    scene: GpuRtTriangleScene[node_width, leaf_width, build_method, layout],
    settings: RenderSettings,
) raises:
    """Submit a prepared static-triangle scene."""
    enqueue_render_gpu_triangles[
        ALGORITHM,
        node_width,
        leaf_width,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
        build_method,
        layout,
    ](ctx, target, scene, settings)


@always_inline
def enqueue_render_gpu[
    ALGORITHM: Integrator = .PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = .HPLOC,
    triangle_layout: GpuBvhLayout = GpuBvhLayout(
        triangle_node_width == 8 and triangle_leaf_width == 4
    ),
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    scene: GpuRtMixedScene[
        node_width,
        leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_layout,
    ],
    settings: RenderSettings,
) raises:
    """Submit a prepared mixed static scene."""
    enqueue_render_gpu_mixed[
        ALGORITHM,
        node_width,
        leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_layout,
    ](ctx, target, scene, settings)


@always_inline
def enqueue_render_gpu[
    ALGORITHM: Integrator = .PATH,
    tlas_node_width: SIMDLength = 2,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_node_width: SIMDLength = 8,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = .HPLOC,
    blas_layout: GpuBvhLayout = GpuBvhLayout(
        blas_node_width == 8 and blas_leaf_width == 4
    ),
    tlas_build_method: GpuBvhBuildMethod = .LBVH,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    scene: GpuRtTriangleInstanceScene[
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        blas_build_method,
        blas_layout,
        tlas_build_method,
    ],
    settings: RenderSettings,
) raises:
    """Submit a prepared triangle-instance scene."""
    enqueue_render_gpu_triangle_instances[
        ALGORITHM,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_layout,
        tlas_build_method,
    ](ctx, target, scene, settings)


@always_inline
def enqueue_render_gpu[
    ALGORITHM: Integrator = .PATH,
    HAS_SPHERES: Bool = False,
    HAS_TRIANGLES: Bool = False,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
    tlas_node_width: SIMDLength = 2,
    tlas_leaf_width: SIMDLength = 2,
    blas_node_width: SIMDLength = 8,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = .HPLOC,
    blas_layout: GpuBvhLayout = GpuBvhLayout(
        blas_node_width == 8 and blas_leaf_width == 4
    ),
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = .HPLOC,
    triangle_layout: GpuBvhLayout = GpuBvhLayout(
        triangle_node_width == 8 and triangle_leaf_width == 4
    ),
    tlas_build_method: GpuBvhBuildMethod = .LBVH,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    scene: GpuRtCombinedInstanceScene[
        HAS_SPHERES,
        HAS_TRIANGLES,
        node_width,
        leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_layout,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_layout,
        tlas_build_method,
    ],
    settings: RenderSettings,
) raises:
    """Submit a prepared static-plus-instanced scene."""
    enqueue_render_gpu_combined_instances[
        ALGORITHM,
        HAS_SPHERES,
        HAS_TRIANGLES,
        node_width,
        leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_layout,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_layout,
        tlas_build_method,
    ](ctx, target, scene, settings)


def _render_gpu_combined_default[
    ALGORITHM: Integrator,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    if _prefer_cwbvh8_blases(world):
        if HAS_TRIANGLES and _prefer_cwbvh8_triangles(world):
            return render_gpu_combined_instances[
                ALGORITHM,
                HAS_SPHERES,
                HAS_TRIANGLES,
                node_width,
                leaf_width,
                2,
                1,
                8,
                4,
                .HPLOC,
                .CWBVH8,
                8,
                4,
                .HPLOC,
                .CWBVH8,
            ](settings, camera, world)
        return render_gpu_combined_instances[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            node_width,
            leaf_width,
            2,
            1,
            8,
            4,
            .HPLOC,
            .CWBVH8,
            4,
            4,
            .LBVH,
            .WIDE,
        ](settings, camera, world)
    if HAS_TRIANGLES and _prefer_cwbvh8_triangles(world):
        return render_gpu_combined_instances[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            node_width,
            leaf_width,
            2,
            1,
            4,
            4,
            .LBVH,
            .WIDE,
            8,
            4,
            .HPLOC,
            .CWBVH8,
        ](settings, camera, world)
    return render_gpu_combined_instances[
        ALGORITHM,
        HAS_SPHERES,
        HAS_TRIANGLES,
        node_width,
        leaf_width,
        2,
        1,
        4,
        4,
        .LBVH,
        .WIDE,
        4,
        4,
        .LBVH,
        .WIDE,
    ](settings, camera, world)


def render_gpu[
    ALGORITHM: Integrator = .PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render supported geometry with compile-time algorithm/BVH specialization.
    """
    comptime assert ALGORITHM.is_valid()
    
    if len(world.triangle_instances()) > 0:
        if len(world.spheres()) > 0:
            if len(world.triangle_vertices()) > 0:
                return _render_gpu_combined_default[
                    ALGORITHM,
                    True,
                    True,
                    node_width,
                    leaf_width,
                ](settings, camera, world)
            return _render_gpu_combined_default[
                ALGORITHM,
                True,
                False,
                node_width,
                leaf_width,
            ](settings, camera, world)
        if len(world.triangle_vertices()) > 0:
            return _render_gpu_combined_default[
                ALGORITHM,
                False,
                True,
                node_width,
                leaf_width,
            ](settings, camera, world)
        if _prefer_cwbvh8_blases(world):
            return render_gpu_triangle_instances[
                ALGORITHM,
                2,
                1,
                8,
                4,
                .HPLOC,
                .CWBVH8,
            ](settings, camera, world)
        return render_gpu_triangle_instances[
            ALGORITHM,
            2,
            1,
            4,
            4,
            .LBVH,
            .WIDE,
        ](settings, camera, world)
    if len(world.spheres()) > 0:
        if len(world.triangle_vertices()) > 0:
            if not _prefer_cwbvh8_triangles(world):
                return render_gpu_mixed[
                    ALGORITHM,
                    node_width,
                    leaf_width,
                    4,
                    4,
                    .LBVH,
                    .WIDE,
                ](settings, camera, world)
            return render_gpu_mixed[
                ALGORITHM,
                node_width,
                leaf_width,
            ](settings, camera, world)
        return render_gpu_spheres[
            ALGORITHM,
            node_width,
            leaf_width,
        ](settings, camera, world)
    if _prefer_cwbvh8_triangles(world):
        return render_gpu_triangles[
            ALGORITHM,
            8,
            4,
            .HPLOC,
            .CWBVH8,
        ](settings, camera, world)
    return render_gpu_triangles[
        ALGORITHM,
        4,
        4,
        .LBVH,
        .WIDE,
    ](settings, camera, world)
