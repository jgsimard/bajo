"""Generic GPU RT entry point selecting a geometry-specialized pipeline."""

from bajo.bvh.camera import Camera
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.rt.types import RENDER, RenderResult, RenderSettings, World
from bajo.rt.gpu.instance_path import render_gpu_triangle_instances
from bajo.rt.gpu.combined_instance_path import render_gpu_combined_instances
from bajo.rt.gpu.mixed_path import render_gpu_mixed
from bajo.rt.gpu.sphere_path import render_gpu_spheres
from bajo.rt.gpu.triangle_path import render_gpu_triangles


# A CWBVH node can encode up to 24 leaf triangles. Below this scale there is
# little hierarchy to compress, and the task-mask setup costs more than wide4
# traversal. Select once on the host so device kernels remain format-specialized.
comptime GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD = 32


def _prefer_cwbvh8_triangles[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](world: World[world_bvh_width, instance_bvh_width]) -> Bool:
    return (
        len(world.triangle_vertices) / 3
        >= GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD
    )


def _prefer_cwbvh8_blases[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](world: World[world_bvh_width, instance_bvh_width]) -> Bool:
    var weighted_triangles = 0
    for instance in world.triangle_instances:
        weighted_triangles += (
            len(world.triangle_meshes[Int(instance.blas_idx)]) / 3
        )
    return weighted_triangles >= (
        len(world.triangle_instances) * GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD
    )


def _render_gpu_combined_default[
    ALGORITHM: RENDER,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    settings: RenderSettings,
    camera: Camera,
    world: World[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    if _prefer_cwbvh8_blases(world):
        if HAS_TRIANGLES and _prefer_cwbvh8_triangles(world):
            return render_gpu_combined_instances[
                ALGORITHM,
                HAS_SPHERES,
                HAS_TRIANGLES,
                node_width,
                leaf_width,
                world_bvh_width,
                instance_bvh_width,
                2,
                1,
                8,
                4,
                GpuBvhBuildMethod.HPLOC,
                True,
                8,
                4,
                GpuBvhBuildMethod.HPLOC,
                True,
            ](settings, camera, world)
        return render_gpu_combined_instances[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            node_width,
            leaf_width,
            world_bvh_width,
            instance_bvh_width,
            2,
            1,
            8,
            4,
            GpuBvhBuildMethod.HPLOC,
            True,
            4,
            4,
            GpuBvhBuildMethod.LBVH,
            False,
        ](settings, camera, world)
    if HAS_TRIANGLES and _prefer_cwbvh8_triangles(world):
        return render_gpu_combined_instances[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            node_width,
            leaf_width,
            world_bvh_width,
            instance_bvh_width,
            2,
            1,
            4,
            4,
            GpuBvhBuildMethod.LBVH,
            False,
            8,
            4,
            GpuBvhBuildMethod.HPLOC,
            True,
        ](settings, camera, world)
    return render_gpu_combined_instances[
        ALGORITHM,
        HAS_SPHERES,
        HAS_TRIANGLES,
        node_width,
        leaf_width,
        world_bvh_width,
        instance_bvh_width,
        2,
        1,
        4,
        4,
        GpuBvhBuildMethod.LBVH,
        False,
        4,
        4,
        GpuBvhBuildMethod.LBVH,
        False,
    ](settings, camera, world)


def render_gpu[
    ALGORITHM: RENDER = RENDER.PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
](
    settings: RenderSettings,
    camera: Camera,
    world: World[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    """Render supported geometry with compile-time algorithm/BVH specialization.
    """
    comptime assert ALGORITHM in (
        RENDER.PATH,
        RENDER.NORMALS,
        RENDER.AO,
        RENDER.NEE,
        RENDER.MIS,
    )
    if len(world.triangle_instances) > 0:
        if len(world.spheres) > 0:
            if len(world.triangle_vertices) > 0:
                return _render_gpu_combined_default[
                    ALGORITHM,
                    True,
                    True,
                    node_width,
                    leaf_width,
                    world_bvh_width,
                    instance_bvh_width,
                ](settings, camera, world)
            return _render_gpu_combined_default[
                ALGORITHM,
                True,
                False,
                node_width,
                leaf_width,
                world_bvh_width,
                instance_bvh_width,
            ](settings, camera, world)
        if len(world.triangle_vertices) > 0:
            return _render_gpu_combined_default[
                ALGORITHM,
                False,
                True,
                node_width,
                leaf_width,
                world_bvh_width,
                instance_bvh_width,
            ](settings, camera, world)
        if _prefer_cwbvh8_blases(world):
            return render_gpu_triangle_instances[
                ALGORITHM,
                2,
                1,
                8,
                4,
                world_bvh_width,
                instance_bvh_width,
                GpuBvhBuildMethod.HPLOC,
                True,
            ](settings, camera, world)
        return render_gpu_triangle_instances[
            ALGORITHM,
            2,
            1,
            4,
            4,
            world_bvh_width,
            instance_bvh_width,
            GpuBvhBuildMethod.LBVH,
            False,
        ](settings, camera, world)
    if len(world.spheres) > 0:
        if len(world.triangle_vertices) > 0:
            if not _prefer_cwbvh8_triangles(world):
                return render_gpu_mixed[
                    ALGORITHM,
                    node_width,
                    leaf_width,
                    world_bvh_width,
                    instance_bvh_width,
                    4,
                    4,
                    GpuBvhBuildMethod.LBVH,
                    False,
                ](settings, camera, world)
            return render_gpu_mixed[
                ALGORITHM,
                node_width,
                leaf_width,
                world_bvh_width,
                instance_bvh_width,
            ](settings, camera, world)
        return render_gpu_spheres[
            ALGORITHM,
            node_width,
            leaf_width,
            world_bvh_width,
            instance_bvh_width,
        ](settings, camera, world)
    if _prefer_cwbvh8_triangles(world):
        return render_gpu_triangles[
            ALGORITHM,
            8,
            4,
            world_bvh_width,
            instance_bvh_width,
            GpuBvhBuildMethod.HPLOC,
            True,
        ](settings, camera, world)
    return render_gpu_triangles[
        ALGORITHM,
        4,
        4,
        world_bvh_width,
        instance_bvh_width,
        GpuBvhBuildMethod.LBVH,
        False,
    ](settings, camera, world)
