"""Reusable GPU RT trace representation for signed sphere geometry."""

from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu import GpuSphereBvh, build_sphere_bvh
from bajo.bvh.gpu.utils import upload_list
from bajo.bvh import Sphere
from bajo.core import Frame, Point3f32
from bajo.rt.geometry import sphere_for_acceleration
from bajo.rt.gpu.resources import upload_surface_ids
from bajo.rt.types import SurfaceId


struct GpuRtSphereGeometry[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Sphere BVH plus surface and signed-radius trace sidecars."""

    var bvh: GpuSphereBvh[Self.frame, Self.node_width, Self.leaf_width]
    var surfaces: DeviceBuffer[DType.uint32]
    var signed_radii: DeviceBuffer[DType.float32]

    def __init__[
        enabled: Bool = True,
        surface_width: SIMDLength = 1,
    ](
        out self,
        mut ctx: DeviceContext,
        spheres: ImmSpan[Sphere[Self.frame], _],
        surface_ids: ImmSpan[SurfaceId[surface_width], _],
    ) raises:
        comptime assert surface_width == 1
        var build_spheres: List[Sphere[Self.frame]]
        var dummy_surfaces: List[UInt32]
        var signed_radii: List[Float32]
        comptime if enabled:
            debug_assert["safe", _use_compiler_assume=True](
                len(spheres) > 0 and len(surface_ids) == len(spheres),
                "GPU sphere geometry requires one surface per sphere",
            )
            build_spheres = List[Sphere[Self.frame]](capacity=len(spheres))
            dummy_surfaces = []
            signed_radii = List[Float32](capacity=len(spheres))
            for sphere in spheres:
                build_spheres.append(sphere_for_acceleration(sphere))
                signed_radii.append(sphere.radius)
        else:
            build_spheres = [Sphere(Point3f32[Self.frame](0.0), 1.0)]
            dummy_surfaces = [UInt32(0)]
            signed_radii = [1.0]

        self.bvh = build_sphere_bvh[
            Self.frame, Self.node_width, Self.leaf_width
        ](ctx, build_spheres)
        comptime if enabled:
            self.surfaces = upload_surface_ids(ctx, surface_ids)
        else:
            self.surfaces = upload_list(ctx, dummy_surfaces)
        self.signed_radii = upload_list(ctx, signed_radii)
