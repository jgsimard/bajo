"""CPU-prepared scene ownership and traversal."""

from std.utils.numerics import max_finite

from std.bit import count_trailing_zeros
from std.memory import pack_bits

from bajo.bvh.constants import EMPTY_LANE, PrimitiveKind
from bajo.core.geometry import (
    Sphere,
    orient_surface_normal,
    interpolate_triangle_attribute,
)
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
    Affine3,
    Frame,
    GeoKind,
    Point3,
    Point3f32,
    Ray,
    Rayf32,
    Vec3,
    Vec3f32,
    dot,
)

from bajo.core.tensor_layout import record_tile
from bajo.rt.material_types import PrimitiveId, SurfaceId
from bajo.rt.shading_types import HitRecord, SurfaceHit
from bajo.rt.scene_data import SceneData
from bajo.rt.media import SceneMedia
from bajo.rt.material_storage import pack_coated_materials


@always_inline
def _triangle_instance_shading[
    length: SIMDLength = 1
](
    data: SceneData,
    instance_idx: SIMD[.uint32, length],
    triangle_idx: SIMD[.uint32, length],
    hit_u: SIMD[.float32, length],
    hit_v: SIMD[.float32, length],
    geometric_normal: Vec3[.float32, .WORLD, length],
    active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Tuple[
    Vec3[.float32, .WORLD, length],
    SIMD[.float32, length],
    SIMD[.float32, length],
]:
    """Interpolate attributes across rays, sharing the scalar width-one math."""
    var outward_normal = geometric_normal
    var texture_u = SIMD[.float32, length](0.0)
    var texture_v = SIMD[.float32, length](0.0)
    if not active.reduce_or():
        return (outward_normal, texture_u, texture_v)
    var first = 0
    comptime if length > 1:
        var bits = UInt64(pack_bits(active))
        first = Int(count_trailing_zeros(bits))
        # Coherent packets share attribute buffers and one inverse transform.
        # Mixed instances stay linear and reuse this same kernel at width one.
        if (
            bits.is_power_of_two()
            or not (~active | instance_idx.eq(instance_idx[first])).reduce_and()
        ):
            for lane in range(Int(length)):
                if active[lane]:
                    var shading = _triangle_instance_shading(
                        data,
                        instance_idx[lane],
                        triangle_idx[lane],
                        hit_u[lane],
                        hit_v[lane],
                        Vec3f32[.WORLD](
                            geometric_normal.x[lane],
                            geometric_normal.y[lane],
                            geometric_normal.z[lane],
                        ),
                    )
                    outward_normal.x[lane] = shading[0].x
                    outward_normal.y[lane] = shading[0].y
                    outward_normal.z[lane] = shading[0].z
                    texture_u[lane] = shading[1]
                    texture_v[lane] = shading[2]
            return (outward_normal, texture_u, texture_v)
    ref instance = data.triangle_instances()[Int(instance_idx[first])]
    var mesh_idx = Int(instance.blas_idx)
    # Inactive lanes may contain miss sentinels, never valid gather indices.
    var triangles = active.select(triangle_idx, UInt32(0)).cast[.int64]()
    ref normals = data.triangle_mesh_normals()[mesh_idx]
    if len(normals) != 0:
        var records = record_tile[.float32, 9](
            normals.unsafe_ptr(), len(normals) // 9
        )
        var local_normal = Vec3[.float32, .LOCAL, length](
            interpolate_triangle_attribute[3, 0](
                records, triangles, hit_u, hit_v
            ),
            interpolate_triangle_attribute[3, 1](
                records, triangles, hit_u, hit_v
            ),
            interpolate_triangle_attribute[3, 2](
                records, triangles, hit_u, hit_v
            ),
        ).unsafe_convert[new_kind=GeoKind.NORMAL]()
        ref inverse = instance.inv_transform
        var wide_inverse = Affine3[.float32, .WORLD, .LOCAL, length](
            inverse.m00,
            inverse.m01,
            inverse.m02,
            0.0,
            inverse.m10,
            inverse.m11,
            inverse.m12,
            0.0,
            inverse.m20,
            inverse.m21,
            inverse.m22,
            0.0,
        )
        var transformed = (
            Affine3[.float32, .LOCAL, .WORLD, length]
            .normal_from_inverse(local_normal, wide_inverse)
            .unsafe_convert[new_kind=GeoKind.VECTOR]()
        )
        transformed = Vec3.select(
            dot(transformed, geometric_normal).lt(0.0),
            -transformed,
            transformed,
        )
        outward_normal = Vec3.select(active, transformed, outward_normal)
    ref texcoords = data.triangle_mesh_texcoords()[mesh_idx]
    if len(texcoords) != 0:
        var records = record_tile[.float32, 6](
            texcoords.unsafe_ptr(), len(texcoords) // 6
        )
        texture_u = active.select(
            interpolate_triangle_attribute[2, 0](
                records, triangles, hit_u, hit_v
            ),
            0.0,
        )
        texture_v = active.select(
            interpolate_triangle_attribute[2, 1](
                records, triangles, hit_u, hit_v
            ),
            0.0,
        )
    return (outward_normal, texture_u, texture_v)


@always_inline
def _blas_occluded[
    kind: PrimitiveKind, width: SIMDLength, length: SIMDLength
](
    blases: CpuBlasSet[kind, width, width],
    rays: Ray[.float32, .WORLD, length],
    valid: SIMD[.bool, length],
) -> SIMD[.bool, length]:
    """Dispatch BLAS visibility without changing its traversal policy."""
    # Mojo's traversal overloads require a concrete primitive kind here.
    comptime if length == 1:
        var hit: Hit[.WORLD]
        comptime if kind == .SPHERE:
            hit = trace_blas_set[width, width, .ANY_HIT, .WORLD](
                rebind[CpuBlasSet[.SPHERE, width, width]](blases),
                UInt32(0),
                rebind[Rayf32[.WORLD]](rays),
            )
        else:
            comptime assert kind == .TRIANGLE
            hit = trace_blas_set[width, width, .ANY_HIT, .WORLD](
                rebind[CpuBlasSet[.TRIANGLE, width, width]](blases),
                UInt32(0),
                rebind[Rayf32[.WORLD]](rays),
            )
        return SIMD[.bool, length](fill=hit.is_occluded()[0])
    else:
        comptime if kind == .SPHERE:
            return trace_blas_set_packet_any_hit[
                width, width, length, frame=.WORLD
            ](
                rebind[CpuBlasSet[.SPHERE, width, width]](blases),
                UInt32(0),
                rays,
                valid,
            )
        else:
            comptime assert kind == .TRIANGLE
            return trace_blas_set_packet_any_hit[
                width, width, length, frame=.WORLD
            ](
                rebind[CpuBlasSet[.TRIANGLE, width, width]](blases),
                UInt32(0),
                rays,
                valid,
            )


@fieldwise_init
struct _WorldHit(Copyable):
    var primitive: PrimitiveId
    var surface: SurfaceHit[1]

    @staticmethod
    def miss(t: Float32 = max_finite[.float32]()) -> Self:
        return Self(
            PrimitiveId(PrimitiveKind.SPHERE, UInt32(0)), SurfaceHit[1](t)
        )


@always_inline
def _merge_surface_hit[
    length: SIMDLength
](mut result: SurfaceHit[length], candidate: SurfaceHit[length]):
    """Keep the first hit on equal distances, independently in each lane."""
    var take = candidate.hit & (~result.hit | candidate.t.lt(result.t))
    result.normal = Vec3.select(take, candidate.normal, result.normal)
    result.geometric_normal = Vec3.select(
        take, candidate.geometric_normal, result.geometric_normal
    )
    result.surface.value = take.select(
        candidate.surface.value, result.surface.value
    )
    result.uv_u = take.select(candidate.uv_u, result.uv_u)
    result.uv_v = take.select(candidate.uv_v, result.uv_v)
    result.t = take.select(candidate.t, result.t)
    result.front_face = take.select(candidate.front_face, result.front_face)
    result.medium_boundary = take.select(
        candidate.medium_boundary, result.medium_boundary
    )
    result.hit |= candidate.hit


struct CpuScene[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
]:
    """Immutable CPU-prepared snapshot with backend-specific acceleration.

    The input `SceneData` is consumed into this owner. `scene_data()` exposes a
    read-only view; geometry, surfaces, and lights cannot be mutated through the
    public prepared-scene API.
    """

    var sphere_bvh: Optional[CpuBlasSet[.SPHERE, Self.world_bvh_width]]
    var triangle_bvh: Optional[CpuBlasSet[.TRIANGLE, Self.world_bvh_width]]
    var triangle_tlas: Optional[CpuTlas[Self.instance_bvh_width, 1]]
    var triangle_mesh_blases: Optional[
        CpuBlasSet[.TRIANGLE, Self.instance_bvh_width]
    ]
    var _scene: SceneData
    var _coated_materials: List[Float32]
    var _has_instance_attributes: Bool

    def __init__[
        build_method: CpuBvhBuildMethod = .SAH
    ](out self, var scene: SceneData):
        self._scene = scene^
        self._has_instance_attributes = False
        for mesh in range(len(self._scene.triangle_meshes())):
            if (
                len(self._scene.triangle_mesh_normals()[mesh]) != 0
                or len(self._scene.triangle_mesh_texcoords()[mesh]) != 0
            ):
                self._has_instance_attributes = True
                break
        self._coated_materials = pack_coated_materials[True](
            self._scene.surfaces()
        )
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

    def coated_materials(self) -> ref[self._coated_materials] List[Float32]:
        """Immutable column-major constants prepared once for SIMD shading."""
        return self._coated_materials

    def take_data(deinit self) -> SceneData:
        """Consume the CPU owner and recover its authoring snapshot."""
        return self._scene^

    def _build_acceleration[build_method: CpuBvhBuildMethod](mut self):
        if len(self._scene.spheres()) > 0:
            var bvh_spheres = List[Sphere[.WORLD]](
                capacity=len(self._scene.spheres())
            )
            for s in self._scene.spheres():
                bvh_spheres.append(s.for_acceleration())

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
                ](self._scene.triangle_vertices())
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
        """Trace visibility in geometry order, skipping already blocked lanes.
        """
        var result = SIMD[.bool, length](fill=False)
        var active = valid
        if not active.reduce_or():
            return result
        if self.sphere_bvh:
            result |= _blas_occluded(self.sphere_bvh.value(), rays, active)
            active &= ~result
        if not active.reduce_or():
            return result
        if self.triangle_bvh:
            result |= _blas_occluded(self.triangle_bvh.value(), rays, active)
            active &= ~result
        if (
            active.reduce_or()
            and self.triangle_tlas
            and self.triangle_mesh_blases
        ):
            result |= self._instances_occluded(rays, active)
        return result

    @always_inline
    def _instances_occluded[
        length: SIMDLength
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        valid: SIMD[.bool, length],
    ) -> SIMD[.bool, length]:
        comptime if length == 1:
            var hit = self.triangle_tlas.value().trace_blases[
                Self.instance_bvh_width,
                Self.instance_bvh_width,
                .ANY_HIT,
            ](rebind[Rayf32[.WORLD]](rays), self.triangle_mesh_blases.value())
            return SIMD[.bool, length](fill=hit.is_occluded()[0])
        else:
            return self.triangle_tlas.value().trace_blases_packet_any_hit[
                Self.instance_bvh_width,
                Self.instance_bvh_width,
                length,
            ](rays, self.triangle_mesh_blases.value(), valid)

    def _trace_closest(self, ray: Rayf32[.WORLD]) -> _WorldHit:
        var closest = self._trace_geometry[.SPHERE](ray)
        var triangle_hit = self._trace_geometry[.TRIANGLE](ray)
        if triangle_hit.surface.hit and (
            not closest.surface.hit
            or triangle_hit.surface.t < closest.surface.t
        ):
            closest = triangle_hit^

        var instance_hit = self._trace_geometry[.TRIANGLE_INSTANCE](ray)
        if instance_hit.surface.hit and (
            not closest.surface.hit
            or instance_hit.surface.t < closest.surface.t
        ):
            closest = instance_hit^
        return closest^

    def trace(self, ray: Rayf32[.WORLD]) -> Optional[HitRecord]:
        var hit = self._trace_closest(ray)
        if not hit.surface.hit:
            return None
        return HitRecord(
            hit.primitive.copy(),
            ray.at(hit.surface.t),
            hit.surface.normal,
            hit.surface.surface.copy(),
            hit.surface.t,
            hit.surface.front_face,
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
            if not valid[0]:
                return SurfaceHit[length](rays.t_max)
            var hit = self._trace_closest(rebind[Rayf32[.WORLD]](rays))
            return rebind[SurfaceHit[length]](hit.surface).copy()
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
        comptime match traversal_mode:
            case .ADAPTIVE:
                comptime assert len(adaptive_packet_sizes) > 0
                return trace_blas_set_packet_adaptive[
                    Self.world_bvh_width,
                    Self.world_bvh_width,
                    length,
                    *adaptive_packet_sizes,
                    frame=.WORLD,
                ](self.triangle_bvh.value(), UInt32(0), rays, valid)
            case .FIXED_PACKET | .AUTO_COHERENT:
                return trace_blas_set_packet_selected[
                    Self.world_bvh_width,
                    Self.world_bvh_width,
                    length,
                    traversal_mode,
                    .WORLD,
                ](self.triangle_bvh.value(), UInt32(0), rays, valid)

    @always_inline
    def _surface_from_hit[
        kind: PrimitiveKind, length: SIMDLength
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        hits: Hit[.WORLD, length],
        active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> SurfaceHit[length]:
        """Assemble shading attributes at every width, including one."""
        var mask = active & hits.is_hit()
        comptime if kind == .TRIANGLE_INSTANCE:
            mask &= hits.inst.ne(EMPTY_LANE)
        if not mask.reduce_or():
            return SurfaceHit[length](rays.t_max)

        var surfaces = SIMD[.uint32, length](0)
        var boundaries = SIMD[.uint32, length](0)
        var center = Point3[.float32, .WORLD, length](0.0)
        var radius = SIMD[.float32, length](1.0)
        for lane in range(length):
            if mask[lane]:
                comptime if kind == .SPHERE:
                    var index = Int(hits.prim[lane])
                    ref sphere = self._scene.spheres()[index]
                    center.x[lane] = sphere.center.x
                    center.y[lane] = sphere.center.y
                    center.z[lane] = sphere.center.z
                    radius[lane] = sphere.radius
                    surfaces[lane] = self._scene.sphere_surfaces()[index].value
                    boundaries[lane] = SceneMedia.boundary(
                        self._scene.media().spheres, index
                    )
                elif kind == .TRIANGLE:
                    var index = Int(hits.prim[lane])
                    surfaces[lane] = self._scene.triangle_surfaces()[
                        index
                    ].value
                    boundaries[lane] = SceneMedia.boundary(
                        self._scene.media().triangles, index
                    )
                else:
                    comptime assert kind == .TRIANGLE_INSTANCE
                    var index = Int(hits.inst[lane])
                    surfaces[lane] = self._scene.triangle_instance_surfaces()[
                        index
                    ].value
                    boundaries[lane] = SceneMedia.boundary(
                        self._scene.media().instances, index
                    )

        var geometric_normal = hits.normal.unsafe_convert[
            new_kind=GeoKind.VECTOR
        ]()
        comptime if kind == .SPHERE:
            geometric_normal = (rays.at(hits.t) - center) * (
                Float32(1.0) / radius
            )
        var normal = geometric_normal
        var u = SIMD[.float32, length](0.0)
        var v = SIMD[.float32, length](0.0)
        comptime if kind == .TRIANGLE_INSTANCE:
            if self._has_instance_attributes:
                var shading = _triangle_instance_shading(
                    self._scene,
                    hits.inst,
                    hits.prim,
                    hits.u,
                    hits.v,
                    geometric_normal,
                    mask,
                )
                normal = shading[0]
                u = shading[1]
                v = shading[2]
        var oriented = orient_surface_normal(rays.d, normal)
        var geometric = orient_surface_normal(rays.d, geometric_normal)
        var zero = Vec3[.float32, .WORLD, length](0.0)
        return SurfaceHit[length](
            Vec3.select(mask, oriented.normal, zero),
            SurfaceId[length](surfaces),
            u,
            v,
            mask.select(hits.t, rays.t_max),
            mask.select(geometric.front_face, SIMD[.bool, length](fill=True)),
            mask,
            boundaries,
            Vec3.select(mask, geometric.normal, zero),
        )

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
            var hits = trace_blas_set_packet[
                Self.world_bvh_width,
                Self.world_bvh_width,
                length,
                Frame.WORLD,
            ](self.sphere_bvh.value(), UInt32(0), rays, valid)
            result = self._surface_from_hit[.SPHERE](rays, hits, valid)

        if self.triangle_bvh:
            var hits = self._trace_world_triangle_packet[
                length, traversal_mode, *adaptive_packet_sizes
            ](rays, valid)
            _merge_surface_hit(
                result,
                self._surface_from_hit[.TRIANGLE](
                    rays, hits, valid & hits.t.lt(result.t)
                ),
            )

        if self.triangle_tlas and self.triangle_mesh_blases:
            var bounded_rays = Ray[.float32, .WORLD, length](
                rays.o, rays.d, rays.t_min, result.t
            )
            var hits = self.triangle_tlas.value().trace_blases_packet[
                Self.instance_bvh_width,
                Self.instance_bvh_width,
                length,
            ](bounded_rays, self.triangle_mesh_blases.value(), valid)
            _merge_surface_hit(
                result,
                self._surface_from_hit[.TRIANGLE_INSTANCE](
                    rays, hits, valid & hits.t.lt(result.t)
                ),
            )
        return result^

    def _trace_geometry[
        kind: PrimitiveKind
    ](self, ray: Rayf32[.WORLD]) -> _WorldHit:
        """Use scalar traversal and the same SIMD hit assembly at width one."""
        var hits: Hit[.WORLD]
        comptime if kind == .SPHERE:
            if not self.sphere_bvh:
                return _WorldHit.miss(ray.t_max)
            hits = trace_blas_set[
                Self.world_bvh_width,
                Self.world_bvh_width,
                .CLOSEST_HIT,
                .WORLD,
            ](self.sphere_bvh.value(), UInt32(0), ray)
        elif kind == .TRIANGLE:
            if not self.triangle_bvh:
                return _WorldHit.miss(ray.t_max)
            hits = trace_blas_set[
                Self.world_bvh_width,
                Self.world_bvh_width,
                .CLOSEST_HIT,
                .WORLD,
            ](self.triangle_bvh.value(), UInt32(0), ray)
        else:
            comptime assert kind == .TRIANGLE_INSTANCE
            if not self.triangle_tlas or not self.triangle_mesh_blases:
                return _WorldHit.miss(ray.t_max)
            hits = self.triangle_tlas.value().trace_blases[
                Self.instance_bvh_width,
                Self.instance_bvh_width,
                .CLOSEST_HIT,
            ](ray, self.triangle_mesh_blases.value())
        var surface = self._surface_from_hit[kind](ray, hits)
        if not surface.hit:
            return _WorldHit.miss(ray.t_max)
        var index = hits.inst if kind == .TRIANGLE_INSTANCE else hits.prim
        return _WorldHit(PrimitiveId(kind, index), surface^)
