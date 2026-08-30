"""CPU-prepared scene ownership and traversal."""

from bajo.bvh.constants import EMPTY_LANE, PrimitiveKind, f32_max
from bajo.bvh import Sphere
from bajo.bvh.cpu import (
    CpuBlasSet,
    CpuBvhBuildMethod,
    CpuTraversalMode,
    CpuTlas,
    build_cpu_sphere_blas_set,
    build_cpu_triangle_blas_set,
    trace_blas_set,
    trace_blas_set_packet,
    trace_blas_set_packet_any_hit,
    trace_blas_set_packet_adaptive,
    trace_blas_set_packet_selected,
)
from bajo.bvh.types import Hit
from bajo.core import (
    Frame,
    GeoKind,
    Point3,
    Point3f32,
    Ray,
    Rayf32,
    Vec3,
    Vec3f32,
)
from bajo.rt.geometry import orient_surface_normal, sphere_for_acceleration
from bajo.rt.types import (
    Color,
    HitRecord,
    PrimitiveId,
    SceneData,
    SurfaceHit,
    SurfaceId,
)


@fieldwise_init
struct _WorldHit(Copyable, Writable):
    """Canonical closest-hit record shared by public and renderer queries."""

    var primitive: PrimitiveId
    var normal: Vec3f32[.WORLD]
    var surface: SurfaceId[1]
    var t: Float32
    var front_face: Bool
    var hit: Bool

    @staticmethod
    def miss(t: Float32 = f32_max) -> Self:
        return Self(
            PrimitiveId(PrimitiveKind.SPHERE, UInt32(0)),
            Vec3f32[.WORLD](0.0),
            SurfaceId(.LAMBERTIAN, UInt32(0)),
            t,
            True,
            False,
        )


struct CpuScene[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
]:
    """Immutable CPU-prepared snapshot with backend-specific acceleration.

    The input `SceneData` is consumed into this owner. `scene_data()` exposes a
    read-only view; geometry, surfaces, and lights cannot be mutated through the
    public prepared-scene API.

    `CpuScene[]` keeps the general-purpose BVH16/BVH16 policy for world geometry
    and instance BLASes. CPU instance traversal uses one instance per TLAS leaf
    independently of those SIMD widths. A packet-oriented scene can instead
    select, for example, `CpuScene[8, 16]` without changing the packet length
    passed to `render_wavefront`.
    """

    var sphere_bvh: Optional[CpuBlasSet[.SPHERE, Self.world_bvh_width]]
    var triangle_bvh: Optional[CpuBlasSet[.TRIANGLE, Self.world_bvh_width]]
    var triangle_tlas: Optional[CpuTlas[Self.instance_bvh_width, 1]]
    var triangle_mesh_blases: Optional[
        CpuBlasSet[.TRIANGLE, Self.instance_bvh_width]
    ]
    var _scene: SceneData

    def __init__[
        build_method: CpuBvhBuildMethod = .SAH
    ](out self, var scene: SceneData):
        self._scene = scene^
        self.sphere_bvh = Optional[CpuBlasSet[.SPHERE, Self.world_bvh_width]]()
        self.triangle_bvh = Optional[
            CpuBlasSet[.TRIANGLE, Self.world_bvh_width]
        ]()
        self.triangle_tlas = Optional[CpuTlas[Self.instance_bvh_width, 1]]()
        self.triangle_mesh_blases = Optional[
            CpuBlasSet[.TRIANGLE, Self.instance_bvh_width]
        ]()
        self._build_acceleration[build_method]()

    def scene_data(self) -> ref[self._scene] SceneData:
        """Return the immutable authoring snapshot used for preparation."""
        return self._scene

    def take_data(deinit self) -> SceneData:
        """Consume the CPU owner and recover its authoring snapshot."""
        return self._scene^

    def _build_acceleration[build_method: CpuBvhBuildMethod](mut self):
        if len(self._scene.spheres()) > 0:
            var bvh_spheres = List[Sphere[.WORLD]](
                capacity=len(self._scene.spheres())
            )
            for s in self._scene.spheres():
                bvh_spheres.append(sphere_for_acceleration(s))

            self.sphere_bvh = Optional[
                CpuBlasSet[.SPHERE, Self.world_bvh_width]
            ](
                build_cpu_sphere_blas_set[
                    Self.world_bvh_width, build_method, .WORLD
                ]([bvh_spheres^])
            )

        if len(self._scene.triangle_vertices()) > 0:
            self.triangle_bvh = Optional[
                CpuBlasSet[.TRIANGLE, Self.world_bvh_width]
            ](
                build_cpu_triangle_blas_set[
                    Self.world_bvh_width,
                    Self.world_bvh_width,
                    build_method,
                    .WORLD,
                ]([self._scene.triangle_vertices().copy()])
            )

        if len(self._scene.triangle_instances()) > 0:
            self.triangle_mesh_blases = Optional[
                CpuBlasSet[.TRIANGLE, Self.instance_bvh_width]
            ](
                build_cpu_triangle_blas_set[
                    Self.instance_bvh_width,
                    Self.instance_bvh_width,
                    build_method,
                    .LOCAL,
                ](self._scene.triangle_meshes())
            )

            self.triangle_tlas = Optional[CpuTlas[Self.instance_bvh_width, 1]](
                CpuTlas[Self.instance_bvh_width, 1](
                    self._scene.triangle_instances()
                )
            )

    @always_inline
    def occluded[
        length: SIMDLength
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> SIMD[.bool, length]:
        """Trace bounded visibility rays together where packet BVHs exist."""
        comptime if length == 1:
            var result = SIMD[.bool, length](fill=False)
            if not valid[0]:
                return result
            var ray = Rayf32[.WORLD](
                Point3f32[.WORLD](rays.o.x[0], rays.o.y[0], rays.o.z[0]),
                Vec3f32[.WORLD](rays.d.x[0], rays.d.y[0], rays.d.z[0]),
                rays.t_min[0],
                rays.t_max[0],
            )
            if self.sphere_bvh:
                var hit = trace_blas_set[
                    Self.world_bvh_width,
                    Self.world_bvh_width,
                    .ANY_HIT,
                    .WORLD,
                ](self.sphere_bvh.value(), UInt32(0), ray)
                if hit.is_occluded():
                    result[0] = True
                    return result
            if self.triangle_bvh:
                var hit = trace_blas_set[
                    Self.world_bvh_width,
                    Self.world_bvh_width,
                    .ANY_HIT,
                    .WORLD,
                ](self.triangle_bvh.value(), UInt32(0), ray)
                if hit.is_occluded():
                    result[0] = True
                    return result
            if self.triangle_tlas and self.triangle_mesh_blases:
                var hit = self.triangle_tlas.value().trace_blases[
                    Self.instance_bvh_width,
                    Self.instance_bvh_width,
                    .ANY_HIT,
                ](ray, self.triangle_mesh_blases.value())
                if hit.is_occluded():
                    result[0] = True
            return result
        else:
            var result = SIMD[.bool, length](fill=False)
            if self.sphere_bvh:
                var sphere_occluded = trace_blas_set_packet_any_hit[
                    Self.world_bvh_width,
                    Self.world_bvh_width,
                    length,
                    Frame.WORLD,
                ](self.sphere_bvh.value(), UInt32(0), rays, valid)
                result |= sphere_occluded

            if self.triangle_bvh:
                var active = valid & ~result
                if active.reduce_or():
                    var triangle_occluded = trace_blas_set_packet_any_hit[
                        Self.world_bvh_width,
                        Self.world_bvh_width,
                        length,
                        False,
                        .WORLD,
                    ](self.triangle_bvh.value(), UInt32(0), rays, active)
                    result |= triangle_occluded

            if self.triangle_tlas and self.triangle_mesh_blases:
                var active = valid & ~result
                if active.reduce_or():
                    result |= (
                        self.triangle_tlas.value().trace_blases_packet_any_hit[
                            Self.instance_bvh_width,
                            Self.instance_bvh_width,
                            length,
                        ](rays, self.triangle_mesh_blases.value(), active)
                    )
            return result

    def _trace_closest(self, ray: Rayf32[.WORLD]) -> _WorldHit:
        var closest = self._trace_spheres(ray)
        var triangle_hit = self._trace_triangles(ray)
        if triangle_hit.hit and (not closest.hit or triangle_hit.t < closest.t):
            closest = triangle_hit^

        var instance_hit = self._trace_triangle_instances(ray)
        if instance_hit.hit and (not closest.hit or instance_hit.t < closest.t):
            closest = instance_hit^
        return closest^

    def trace(self, ray: Rayf32[.WORLD]) -> Optional[HitRecord]:
        var hit = self._trace_closest(ray)
        if not hit.hit:
            return None
        return HitRecord(
            hit.primitive.copy(),
            ray.at(hit.t),
            hit.normal,
            hit.surface.copy(),
            hit.t,
            hit.front_face,
        )

    @always_inline
    def trace_surface[
        length: SIMDLength
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> SurfaceHit[length]:
        comptime if length == 1:
            if valid[0]:
                var ray = Rayf32[.WORLD](
                    Point3f32[.WORLD](rays.o.x[0], rays.o.y[0], rays.o.z[0]),
                    Vec3f32[.WORLD](rays.d.x[0], rays.d.y[0], rays.d.z[0]),
                    rays.t_min[0],
                    rays.t_max[0],
                )
                var scalar_hit = self._trace_closest(ray)
                var result = SurfaceHit[length](rays.t_max)
                result.normal.x[0] = scalar_hit.normal.x
                result.normal.y[0] = scalar_hit.normal.y
                result.normal.z[0] = scalar_hit.normal.z
                result.surface.value[0] = scalar_hit.surface.value
                result.t[0] = scalar_hit.t
                result.front_face[0] = scalar_hit.front_face
                result.hit[0] = scalar_hit.hit
                return result^
            return SurfaceHit[length](rays.t_max)
        else:
            return self._trace_surface_shared_stack[
                length, CpuTraversalMode.AUTO_COHERENT, 16, 8, 4
            ](rays, valid)

    @always_inline
    def trace_surface_configured[
        length: SIMDLength,
        traversal_mode: CpuTraversalMode,
        *adaptive_packet_sizes: SIMDLength,
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> SurfaceHit[length]:
        """Trace using an explicit compile-time traversal configuration."""
        comptime if length == 1:
            return self.trace_surface(rays, valid)
        else:
            return self._trace_surface_shared_stack[
                length, traversal_mode, *adaptive_packet_sizes
            ](rays, valid)

    @always_inline
    def _trace_world_triangle_packet[
        length: SIMDLength,
        traversal_mode: CpuTraversalMode,
        *adaptive_packet_sizes: SIMDLength,
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        valid: SIMD[.bool, length],
    ) -> Hit[.WORLD, length]:
        """Instantiate the viewer-selected triangle packet traversal."""
        comptime if traversal_mode == .ADAPTIVE:
            comptime assert len(adaptive_packet_sizes) > 0
            return trace_blas_set_packet_adaptive[
                Self.world_bvh_width,
                Self.world_bvh_width,
                length,
                *adaptive_packet_sizes,
                frame=.WORLD,
            ](self.triangle_bvh.value(), UInt32(0), rays, valid)
        else:
            return trace_blas_set_packet_selected[
                Self.world_bvh_width,
                Self.world_bvh_width,
                length,
                traversal_mode,
                .WORLD,
            ](self.triangle_bvh.value(), UInt32(0), rays, valid)

    def _trace_surface_shared_stack[
        length: SIMDLength,
        traversal_mode: CpuTraversalMode,
        *adaptive_packet_sizes: SIMDLength,
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        valid: SIMD[.bool, length],
    ) -> SurfaceHit[length]:
        """Trace SIMD packets through world and instance acceleration."""
        comptime assert length > 1
        var result = SurfaceHit[length](rays.t_max)
        if self.sphere_bvh:
            var sphere_hits = trace_blas_set_packet[
                Self.world_bvh_width,
                Self.world_bvh_width,
                length,
                Frame.WORLD,
            ](self.sphere_bvh.value(), UInt32(0), rays, valid)
            var sphere_mask = sphere_hits.is_hit()
            var center_x = SIMD[.float32, length](0.0)
            var center_y = SIMD[.float32, length](0.0)
            var center_z = SIMD[.float32, length](0.0)
            var radius = SIMD[.float32, length](1.0)
            var surface_values = SIMD[.uint32, length](0)
            for lane in range(length):
                if sphere_mask[lane]:
                    var sphere_idx = Int(sphere_hits.prim[lane])
                    ref sphere = self._scene.spheres()[sphere_idx]
                    center_x[lane] = sphere.center.x
                    center_y[lane] = sphere.center.y
                    center_z[lane] = sphere.center.z
                    radius[lane] = sphere.radius
                    surface_values[lane] = self._scene.sphere_surfaces()[
                        sphere_idx
                    ].value
            var inverse_radius = Float32(1.0) / radius
            var outward_normal = Vec3[.float32, .WORLD, length](
                (rays.o.x + sphere_hits.t * rays.d.x - center_x)
                * inverse_radius,
                (rays.o.y + sphere_hits.t * rays.d.y - center_y)
                * inverse_radius,
                (rays.o.z + sphere_hits.t * rays.d.z - center_z)
                * inverse_radius,
            )
            var oriented = orient_surface_normal(rays.d, outward_normal)
            result.normal = Vec3.select(
                sphere_mask, oriented.normal, result.normal
            )
            result.surface.value = sphere_mask.select(
                surface_values, result.surface.value
            )
            result.t = sphere_mask.select(sphere_hits.t, result.t)
            result.front_face = sphere_mask.select(
                oriented.front_face, result.front_face
            )
            result.hit |= sphere_mask

        if self.triangle_bvh:
            var triangle_hits = self._trace_world_triangle_packet[
                length, traversal_mode, *adaptive_packet_sizes
            ](rays, valid)
            var triangle_mask = triangle_hits.is_hit() & triangle_hits.t.lt(
                result.t
            )
            var surface_values = SIMD[.uint32, length](0)
            for lane in range(length):
                if triangle_mask[lane]:
                    var triangle_idx = Int(triangle_hits.prim[lane])
                    surface_values[lane] = self._scene.triangle_surfaces()[
                        triangle_idx
                    ].value
            var triangle_normal = triangle_hits.normal.unsafe_convert[
                new_kind=GeoKind.VECTOR
            ]()
            var oriented = orient_surface_normal(rays.d, triangle_normal)
            result.normal = Vec3.select(
                triangle_mask, oriented.normal, result.normal
            )
            result.surface.value = triangle_mask.select(
                surface_values, result.surface.value
            )
            result.t = triangle_mask.select(triangle_hits.t, result.t)
            result.front_face = triangle_mask.select(
                oriented.front_face, result.front_face
            )
            result.hit |= triangle_mask

        if self.triangle_tlas and self.triangle_mesh_blases:
            var bounded_rays = Ray[.float32, .WORLD, length](
                rays.o, rays.d, rays.t_min, result.t
            )
            var instance_hits = self.triangle_tlas.value().trace_blases_packet[
                Self.instance_bvh_width,
                Self.instance_bvh_width,
                length,
            ](bounded_rays, self.triangle_mesh_blases.value(), valid)
            var instance_mask = instance_hits.is_hit()
            instance_mask &= instance_hits.t.lt(result.t)
            var surface_values = SIMD[.uint32, length](0)
            for lane in range(length):
                if instance_mask[lane]:
                    var instance_idx = Int(instance_hits.inst[lane])
                    surface_values[
                        lane
                    ] = self._scene.triangle_instance_surfaces()[
                        instance_idx
                    ].value
            var instance_normal = instance_hits.normal.unsafe_convert[
                new_kind=GeoKind.VECTOR
            ]()
            var oriented = orient_surface_normal(rays.d, instance_normal)
            result.normal = Vec3.select(
                instance_mask, oriented.normal, result.normal
            )
            result.surface.value = instance_mask.select(
                surface_values, result.surface.value
            )
            result.t = instance_mask.select(instance_hits.t, result.t)
            result.front_face = instance_mask.select(
                oriented.front_face, result.front_face
            )
            result.hit |= instance_mask

        return result^

    def _trace_spheres(self, ray: Rayf32[.WORLD]) -> _WorldHit:
        if not self.sphere_bvh:
            return _WorldHit.miss(ray.t_max)

        var bvh_hit = trace_blas_set[
            Self.world_bvh_width,
            Self.world_bvh_width,
            .CLOSEST_HIT,
            .WORLD,
        ](self.sphere_bvh.value(), UInt32(0), ray)
        if not bvh_hit.is_hit():
            return _WorldHit.miss(ray.t_max)

        var sphere_idx = Int(bvh_hit.prim)
        debug_assert["safe", _use_compiler_assume=True](
            sphere_idx >= 0 and sphere_idx < len(self._scene.spheres()),
            "BVH returned an out-of-range sphere index",
        )
        ref sphere = self._scene.spheres()[sphere_idx]
        var p = ray.at(bvh_hit.t)
        var outward_normal = (p - sphere.center) / sphere.radius
        var oriented = orient_surface_normal(ray.d, outward_normal)
        return _WorldHit(
            PrimitiveId(PrimitiveKind.SPHERE, bvh_hit.prim),
            oriented.normal,
            self._scene.sphere_surfaces()[sphere_idx].copy(),
            bvh_hit.t,
            oriented.front_face,
            True,
        )

    def _trace_triangles(self, ray: Rayf32[.WORLD]) -> _WorldHit:
        if not self.triangle_bvh:
            return _WorldHit.miss(ray.t_max)

        var bvh_hit = trace_blas_set[
            Self.world_bvh_width,
            Self.world_bvh_width,
            .CLOSEST_HIT,
            .WORLD,
        ](self.triangle_bvh.value(), UInt32(0), ray)
        if not bvh_hit.is_hit():
            return _WorldHit.miss(ray.t_max)

        var tri_idx = Int(bvh_hit.prim)
        debug_assert["safe", _use_compiler_assume=True](
            tri_idx >= 0 and tri_idx < len(self._scene.triangle_surfaces()),
            "BVH returned an out-of-range triangle index",
        )
        var outward_normal = bvh_hit.normal.unsafe_convert[
            new_kind=GeoKind.VECTOR
        ]()
        var oriented = orient_surface_normal(ray.d, outward_normal)
        return _WorldHit(
            PrimitiveId(PrimitiveKind.TRIANGLE, bvh_hit.prim),
            oriented.normal,
            self._scene.triangle_surfaces()[tri_idx].copy(),
            bvh_hit.t,
            oriented.front_face,
            True,
        )

    def _trace_triangle_instances(self, ray: Rayf32[.WORLD]) -> _WorldHit:
        if not self.triangle_tlas or not self.triangle_mesh_blases:
            return _WorldHit.miss(ray.t_max)

        var bvh_hit = self.triangle_tlas.value().trace_blases[
            Self.instance_bvh_width,
            Self.instance_bvh_width,
            .CLOSEST_HIT,
        ](ray, self.triangle_mesh_blases.value())
        if not bvh_hit.is_hit() or bvh_hit.inst == EMPTY_LANE:
            return _WorldHit.miss(ray.t_max)

        var instance_idx = Int(bvh_hit.inst)
        debug_assert["safe", _use_compiler_assume=True](
            instance_idx >= 0
            and instance_idx < len(self._scene.triangle_instances()),
            "TLAS returned an out-of-range triangle instance index",
        )
        var outward_normal = bvh_hit.normal.unsafe_convert[
            new_kind=GeoKind.VECTOR
        ]()
        var oriented = orient_surface_normal(ray.d, outward_normal)
        return _WorldHit(
            PrimitiveId(PrimitiveKind.TRIANGLE_INSTANCE, bvh_hit.inst),
            oriented.normal,
            self._scene.triangle_instance_surfaces()[instance_idx].copy(),
            bvh_hit.t,
            oriented.front_face,
            True,
        )
