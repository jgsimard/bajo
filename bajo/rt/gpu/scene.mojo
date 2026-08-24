"""Unified compile-time-specialized GPU RT scene ownership."""

from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh import Sphere
from bajo.bvh.gpu import (
    GpuBlasSet,
    GpuBvhBuildMethod,
    GpuSphereBvh,
    GpuTlas,
    build_gpu_sphere_bvh,
    build_gpu_triangle_blas_set,
    build_gpu_tlas,
)
from bajo.bvh.gpu.utils import upload_list
from bajo.core import Point3f32
from bajo.rt.geometry import sphere_for_acceleration
from bajo.rt.gpu.path_shading import GpuRtLights, GpuRtMaterials
from bajo.rt.gpu.config import (
    GpuRtBvhFormat,
    GpuRtSceneKind,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_TLAS2,
    GPU_RT_BVH_WIDE4,
)
from bajo.rt.gpu.resources import upload_surface_ids
from bajo.rt.gpu.views import (
    GpuRtInstanceView,
    GpuRtSceneView,
    GpuRtSphereView,
    GpuRtTriangleView,
    _immut,
)
from bajo.rt.types import SceneData


@fieldwise_init
struct GpuRtScene[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS2,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
]:
    """Own exactly the device resources selected by `kind`."""

    var _sphere_bvh: Optional[
        GpuSphereBvh[
            .WORLD,
            Self.sphere_format.node_width,
            Self.sphere_format.leaf_width,
        ]
    ]
    var _sphere_surfaces: Optional[DeviceBuffer[.uint32]]
    var _signed_radii: Optional[DeviceBuffer[.float32]]
    var _triangle_blas: Optional[
        GpuBlasSet[
            .TRIANGLE,
            Self.triangle_format.layout,
            Self.triangle_format.node_width,
            Self.triangle_format.leaf_width,
        ]
    ]
    var _triangle_surfaces: Optional[DeviceBuffer[.uint32]]
    var _instance_blases: Optional[
        GpuBlasSet[
            .TRIANGLE,
            Self.blas_format.layout,
            Self.blas_format.node_width,
            Self.blas_format.leaf_width,
        ]
    ]
    var _tlas: Optional[
        GpuTlas[
            .TRIANGLE,
            Self.tlas_format.node_width,
            Self.blas_format.node_width,
            Self.tlas_format.leaf_width,
            Self.blas_format.leaf_width,
            Self.blas_format.layout,
        ]
    ]
    var _instance_surfaces: Optional[DeviceBuffer[.uint32]]
    var materials: GpuRtMaterials
    var lights: GpuRtLights

    def view(self) -> GpuRtSceneView:
        """Borrow the selected owner fields through the common device ABI."""
        var spheres = Optional[GpuRtSphereView]()
        comptime if Self.kind.has_spheres():
            ref bvh = self._sphere_bvh.value()
            ref surfaces = self._sphere_surfaces.value()
            ref signed_radii = self._signed_radii.value()
            spheres = Optional(
                GpuRtSphereView(
                    _immut(bvh.tree.wide_nodes),
                    _immut(bvh.leaf_spheres),
                    bvh.tree.root_idx,
                    _immut(surfaces),
                    _immut(signed_radii),
                )
            )

        var triangles = Optional[GpuRtTriangleView]()
        comptime if Self.kind.has_triangles():
            ref blas = self._triangle_blas.value()
            ref surfaces = self._triangle_surfaces.value()
            triangles = Optional(
                GpuRtTriangleView(
                    _immut(blas.nodes),
                    _immut(blas.leaves),
                    UInt32(0),
                    _immut(surfaces),
                )
            )

        var instances = Optional[GpuRtInstanceView]()
        comptime if Self.kind.has_instances():
            ref blases = self._instance_blases.value()
            ref tlas = self._tlas.value()
            ref surfaces = self._instance_surfaces.value()
            instances = Optional(
                GpuRtInstanceView(
                    _immut(tlas._tree.wide_nodes),
                    _immut(tlas._tree.leaf_block_indices),
                    _immut(tlas._inst_inv_transform),
                    _immut(tlas._inst_blas_indices),
                    _immut(blases.descs),
                    _immut(blases.nodes),
                    _immut(blases.leaves),
                    tlas._tree.root_idx,
                    Int32(tlas._inst_count),
                    Int32(blases.blas_count),
                    _immut(surfaces),
                )
            )

        return GpuRtSceneView(
            spheres^,
            triangles^,
            instances^,
            _immut(self.materials.emissives),
            _immut(self.materials.lambertians),
            _immut(self.materials.metals),
            _immut(self.materials.dielectrics),
            _immut(self.lights.kinds),
            _immut(self.lights.fields),
            Int32(self.lights.count),
            self.lights.total_weight,
        )


def prepare_gpu_scene[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS2,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    triangle_build_method: GpuBvhBuildMethod = .HPLOC,
    tlas_build_method: GpuBvhBuildMethod = .LBVH,
    blas_build_method: GpuBvhBuildMethod = .HPLOC,
](
    mut ctx: DeviceContext,
    data: SceneData,
) raises -> GpuRtScene[
    kind, sphere_format, triangle_format, tlas_format, blas_format
]:
    """Build the typed owner selected by scene shape and ready BVH formats."""
    comptime assert kind.is_valid()
    comptime assert sphere_format.layout == .WIDE
    comptime assert tlas_format.layout == .WIDE
    debug_assert["safe", _use_compiler_assume=True](
        (len(data.spheres()) > 0) == kind.has_spheres()
        and (len(data.triangle_vertices()) > 0) == kind.has_triangles()
        and (len(data.triangle_instances()) > 0) == kind.has_instances(),
        "GPU RT scene kind does not match the scene geometry",
    )

    var sphere_bvh = Optional[
        GpuSphereBvh[.WORLD, sphere_format.node_width, sphere_format.leaf_width]
    ]()
    var sphere_surfaces = Optional[DeviceBuffer[.uint32]]()
    var signed_radii = Optional[DeviceBuffer[.float32]]()
    comptime if kind.has_spheres():
        var build_spheres = List[Sphere[.WORLD]](capacity=len(data.spheres()))
        var host_signed_radii = List[Float32](capacity=len(data.spheres()))
        for sphere in data.spheres():
            build_spheres.append(sphere_for_acceleration(sphere))
            host_signed_radii.append(sphere.radius)
        sphere_bvh = Optional(
            build_gpu_sphere_bvh[
                .WORLD, sphere_format.node_width, sphere_format.leaf_width
            ](ctx, build_spheres)
        )
        sphere_surfaces = Optional(
            upload_surface_ids(ctx, data.sphere_surfaces())
        )
        signed_radii = Optional(upload_list(ctx, host_signed_radii))

    var triangle_blas = Optional[
        GpuBlasSet[
            .TRIANGLE,
            triangle_format.layout,
            triangle_format.node_width,
            triangle_format.leaf_width,
        ]
    ]()
    var triangle_surfaces = Optional[DeviceBuffer[.uint32]]()
    comptime if kind.has_triangles():
        var vertices = List[Point3f32[.WORLD]](
            capacity=len(data.triangle_vertices())
        )
        for vertex in data.triangle_vertices():
            vertices.append(vertex)
        triangle_blas = Optional(
            build_gpu_triangle_blas_set[
                triangle_format.node_width,
                triangle_format.leaf_width,
                triangle_build_method,
                triangle_format.layout,
                .WORLD,
            ](ctx, [vertices^])
        )
        triangle_surfaces = Optional(
            upload_surface_ids(ctx, data.triangle_surfaces())
        )

    var instance_blases = Optional[
        GpuBlasSet[
            .TRIANGLE,
            blas_format.layout,
            blas_format.node_width,
            blas_format.leaf_width,
        ]
    ]()
    var tlas = Optional[
        GpuTlas[
            .TRIANGLE,
            tlas_format.node_width,
            blas_format.node_width,
            tlas_format.leaf_width,
            blas_format.leaf_width,
            blas_format.layout,
        ]
    ]()
    var instance_surfaces = Optional[DeviceBuffer[.uint32]]()
    comptime if kind.has_instances():
        instance_blases = Optional(
            build_gpu_triangle_blas_set[
                blas_format.node_width,
                blas_format.leaf_width,
                blas_build_method,
                blas_format.layout,
            ](ctx, data.triangle_meshes())
        )
        tlas = Optional(
            build_gpu_tlas[
                .TRIANGLE,
                tlas_format.node_width,
                blas_format.node_width,
                tlas_format.leaf_width,
                blas_format.leaf_width,
                tlas_build_method,
                blas_format.layout,
            ](ctx, data.triangle_instances())
        )
        instance_surfaces = Optional(
            upload_surface_ids(ctx, data.triangle_instance_surfaces())
        )

    var materials = GpuRtMaterials(ctx, data)
    var lights = GpuRtLights(ctx, data)
    return GpuRtScene[
        kind, sphere_format, triangle_format, tlas_format, blas_format
    ](
        sphere_bvh^,
        sphere_surfaces^,
        signed_radii^,
        triangle_blas^,
        triangle_surfaces^,
        instance_blases^,
        tlas^,
        instance_surfaces^,
        materials^,
        lights^,
    )
