"""Validated backend-neutral scene construction and ownership."""

from max.algorithm import parallelize

from std.algorithm import vectorize
from std.math import pi
from std.sys import simd_width_of
from std.utils.numerics import isfinite

from bajo.core import AABB, Affine3f32, Point3f32
from bajo.bvh import Instance
from bajo.core.geometry import Sphere, triangle_area, triangle_is_valid
from bajo.bvh.constants import PrimitiveKind

from bajo.rt.lighting_types import LightRecord, LightStore, _light_importance
from bajo.core.sampling import build_alias_distribution
from bajo.rt.material_types import (
    NO_TEXTURE,
    Environment,
    PrimitiveId,
    SurfaceId,
    SurfaceStore,
    PRIMITIVE_INDEX_BITS,
    PRIMITIVE_INDEX_MASK,
)
from bajo.core.color import Color
from bajo.rt.media import SceneMedia
from bajo.rt.scene_preparation import (
    _validate_mesh_bounds,
    _transformed_triangles,
    _append_instance_lights,
    _instance_light_record,
)
from bajo.rt.opacity import SceneOpacity
from bajo.rt.emission import AreaEmitter, SceneEmission


struct SceneBuilder(
    Deinitable where (
        False,
        "call finish() or abort() to consume the scene builder",
    )
):
    var spheres: List[Sphere[.WORLD]]
    var sphere_surfaces: List[SurfaceId[1]]
    var triangle_vertices: List[Point3f32[.WORLD]]
    var triangle_surfaces: List[SurfaceId[1]]
    var triangle_meshes: List[List[Point3f32[.LOCAL]]]
    var triangle_mesh_normals: List[List[Float32]]
    var triangle_mesh_texcoords: List[List[Float32]]
    var triangle_instances: List[Instance]
    var triangle_instance_surfaces: List[SurfaceId[1]]
    var surfaces: SurfaceStore
    var environment: Environment
    var media: SceneMedia
    var opacity: SceneOpacity
    var emission: SceneEmission

    def __init__(
        out self,
        var spheres: List[Sphere[.WORLD]] = List[Sphere[.WORLD]](),
        var sphere_surfaces: List[SurfaceId[1]] = List[SurfaceId[1]](),
        var triangle_vertices: List[Point3f32[.WORLD]] = List[
            Point3f32[.WORLD]
        ](),
        var triangle_surfaces: List[SurfaceId[1]] = List[SurfaceId[1]](),
        var triangle_meshes: List[List[Point3f32[.LOCAL]]] = List[
            List[Point3f32[.LOCAL]]
        ](),
        var triangle_mesh_normals: List[List[Float32]] = List[List[Float32]](),
        var triangle_mesh_texcoords: List[List[Float32]] = List[
            List[Float32]
        ](),
        var triangle_instances: List[Instance] = List[Instance](),
        var triangle_instance_surfaces: List[SurfaceId[1]] = List[
            SurfaceId[1]
        ](),
        var surfaces: SurfaceStore = SurfaceStore(),
        environment: Environment = Environment(),
    ):
        self.spheres = spheres^
        self.sphere_surfaces = sphere_surfaces^
        self.triangle_vertices = triangle_vertices^
        self.triangle_surfaces = triangle_surfaces^
        self.triangle_meshes = triangle_meshes^
        self.triangle_mesh_normals = triangle_mesh_normals^
        self.triangle_mesh_texcoords = triangle_mesh_texcoords^
        self.triangle_instances = triangle_instances^
        self.triangle_instance_surfaces = triangle_instance_surfaces^
        self.surfaces = surfaces^
        self.environment = environment.copy()
        self.media = SceneMedia()
        self.opacity = SceneOpacity()
        self.emission = SceneEmission()

    def set_environment(mut self, environment: Environment):
        self.environment = environment.copy()

    def add_lambertian(mut self, albedo: Color) -> SurfaceId[1]:
        return self.surfaces.add_lambertian(albedo)

    def add_metal(mut self, albedo: Color, fuzz: Float32) -> SurfaceId[1]:
        return self.surfaces.add_metal(albedo, fuzz)

    def add_dielectric(mut self, refraction_index: Float32) -> SurfaceId[1]:
        return self.surfaces.add_dielectric(refraction_index)

    def add_coated_diffuse(
        mut self, albedo: Color, roughness: Float32, eta: Float32
    ) -> SurfaceId[1]:
        return self.surfaces.add_coated_diffuse(albedo, roughness, eta)

    def add_emissive(mut self, radiance: Color) -> SurfaceId[1]:
        return self.surfaces.add_emissive(radiance)

    def add_sphere(
        mut self,
        center: Point3f32[.WORLD],
        radius: Float32,
        surface: SurfaceId[1],
    ):
        self.spheres.append(Sphere[.WORLD](center, radius))
        self.sphere_surfaces.append(surface.copy())

    def add_triangle(
        mut self,
        v0: Point3f32[.WORLD],
        v1: Point3f32[.WORLD],
        v2: Point3f32[.WORLD],
        surface: SurfaceId[1],
    ):
        self.triangle_vertices.extend([v0, v1, v2])
        self.triangle_surfaces.append(surface.copy())

    def add_quad(
        mut self,
        a: Point3f32[.WORLD],
        b: Point3f32[.WORLD],
        c: Point3f32[.WORLD],
        d: Point3f32[.WORLD],
        surface: SurfaceId[1],
    ):
        """Append two consistently wound triangles: `(a, b, c)` and `(a, c, d)`.
        """
        self.add_triangle(a, b, c, surface)
        self.add_triangle(a, c, d, surface)

    def add_box(
        mut self,
        minimum: Point3f32[.WORLD],
        maximum: Point3f32[.WORLD],
        surface: SurfaceId[1],
    ):
        """Append an outward-wound box as six quads, in face order."""
        var x0 = minimum.x
        var y0 = minimum.y
        var z0 = minimum.z
        var x1 = maximum.x
        var y1 = maximum.y
        var z1 = maximum.z

        self.add_quad(
            Point3f32[.WORLD](x0, y0, z0),
            Point3f32[.WORLD](x1, y0, z0),
            Point3f32[.WORLD](x1, y0, z1),
            Point3f32[.WORLD](x0, y0, z1),
            surface,
        )
        self.add_quad(
            Point3f32[.WORLD](x0, y1, z1),
            Point3f32[.WORLD](x1, y1, z1),
            Point3f32[.WORLD](x1, y1, z0),
            Point3f32[.WORLD](x0, y1, z0),
            surface,
        )
        self.add_quad(
            Point3f32[.WORLD](x0, y0, z1),
            Point3f32[.WORLD](x1, y0, z1),
            Point3f32[.WORLD](x1, y1, z1),
            Point3f32[.WORLD](x0, y1, z1),
            surface,
        )
        self.add_quad(
            Point3f32[.WORLD](x1, y0, z0),
            Point3f32[.WORLD](x0, y0, z0),
            Point3f32[.WORLD](x0, y1, z0),
            Point3f32[.WORLD](x1, y1, z0),
            surface,
        )
        self.add_quad(
            Point3f32[.WORLD](x0, y0, z0),
            Point3f32[.WORLD](x0, y0, z1),
            Point3f32[.WORLD](x0, y1, z1),
            Point3f32[.WORLD](x0, y1, z0),
            surface,
        )
        self.add_quad(
            Point3f32[.WORLD](x1, y0, z1),
            Point3f32[.WORLD](x1, y0, z0),
            Point3f32[.WORLD](x1, y1, z0),
            Point3f32[.WORLD](x1, y1, z1),
            surface,
        )

    def add_triangle_mesh(
        mut self,
        vertices: ImmSpan[Point3f32[.WORLD], _],
        surface: SurfaceId[1],
    ):
        self.triangle_vertices.extend(vertices)
        self.triangle_surfaces.resize(
            len(self.triangle_surfaces) + len(vertices) / 3, fill=surface
        )

    def add_triangle_mesh_instance(
        mut self,
        vertices: ImmSpan[Point3f32[.LOCAL], _],
        transform: Affine3f32[.LOCAL, .WORLD],
        bounds: AABB[.LOCAL],
        surface: SurfaceId[1],
    ) -> UInt32:
        var owned_vertices = List(vertices)
        return self.add_owned_triangle_mesh_instance(
            owned_vertices^,
            List[Float32](),
            List[Float32](),
            transform,
            bounds,
            surface,
        )

    def add_owned_triangle_mesh(
        mut self,
        var vertices: List[Point3f32[.LOCAL]],
        var normals: List[Float32],
        var texcoords: List[Float32],
    ) -> UInt32:
        """Register a shared mesh, keeping its attribute arrays aligned."""
        var mesh_idx = UInt32(len(self.triangle_meshes))
        self.triangle_meshes.append(vertices^)
        self.triangle_mesh_normals.append(normals^)
        self.triangle_mesh_texcoords.append(texcoords^)
        return mesh_idx

    def add_owned_triangle_mesh_instance(
        mut self,
        var vertices: List[Point3f32[.LOCAL]],
        var normals: List[Float32],
        var texcoords: List[Float32],
        transform: Affine3f32[.LOCAL, .WORLD],
        bounds: AABB[.LOCAL],
        surface: SurfaceId[1],
    ) -> UInt32:
        """Transfer mesh attributes into the builder without cloning them."""
        var mesh_idx = self.add_owned_triangle_mesh(
            vertices^, normals^, texcoords^
        )
        self._add_triangle_instance_unchecked(mesh_idx, transform, bounds)
        self.triangle_instance_surfaces.append(surface.copy())
        return mesh_idx

    def add_triangle_instance(
        mut self,
        mesh_idx: UInt32,
        transform: Affine3f32[.LOCAL, .WORLD],
        mesh_bounds: AABB[.LOCAL],
        surface: SurfaceId[1],
    ):
        self._add_triangle_instance_unchecked(mesh_idx, transform, mesh_bounds)
        self.triangle_instance_surfaces.append(surface.copy())

    def _add_triangle_instance_unchecked(
        mut self,
        mesh_idx: UInt32,
        transform: Affine3f32[.LOCAL, .WORLD],
        mesh_bounds: AABB[.LOCAL],
    ):
        var instance = Instance()
        instance.transform = transform.copy()
        instance.bounds = mesh_bounds.apply_transform(transform)
        instance.blas_idx = mesh_idx
        instance.kind = .TRIANGLE
        self.triangle_instances.append(instance^)

    def finish(deinit self) raises -> SceneData:
        """Consume the builder and produce one validated immutable snapshot."""
        return SceneData(
            self.spheres^,
            self.sphere_surfaces^,
            self.triangle_vertices^,
            self.triangle_surfaces^,
            self.triangle_meshes^,
            self.triangle_mesh_normals^,
            self.triangle_mesh_texcoords^,
            self.triangle_instances^,
            self.triangle_instance_surfaces^,
            self.surfaces^,
            self.environment,
            self.media^,
            self.opacity^,
            self.emission^,
        )

    def abort(deinit self):
        """Release authoring buffers when ingestion fails before finalization.
        """
        pass


struct SceneData:
    """Validated backend-neutral scene snapshot."""

    var _spheres: List[Sphere[.WORLD]]
    var _sphere_surfaces: List[SurfaceId[1]]
    var _triangle_vertices: List[Point3f32[.WORLD]]
    var _triangle_surfaces: List[SurfaceId[1]]
    var _triangle_meshes: List[List[Point3f32[.LOCAL]]]
    var _triangle_mesh_normals: List[List[Float32]]
    var _triangle_mesh_texcoords: List[List[Float32]]
    var _triangle_instances: List[Instance]
    var _triangle_instance_surfaces: List[SurfaceId[1]]
    var _surfaces: SurfaceStore
    var _lights: LightStore
    var _media: SceneMedia
    var _opacity: SceneOpacity
    var _emission: SceneEmission
    var _environment: Environment
    var _environment_weight: Float32
    var _environment_alias_probabilities: List[Float32]
    var _environment_alias_indices: List[UInt32]

    def __init__(
        out self,
        var spheres: List[Sphere[.WORLD]],
        var sphere_surfaces: List[SurfaceId[1]],
        var triangle_vertices: List[Point3f32[.WORLD]],
        var triangle_surfaces: List[SurfaceId[1]],
        var triangle_meshes: List[List[Point3f32[.LOCAL]]],
        var triangle_mesh_normals: List[List[Float32]],
        var triangle_mesh_texcoords: List[List[Float32]],
        var triangle_instances: List[Instance],
        var triangle_instance_surfaces: List[SurfaceId[1]],
        var surfaces: SurfaceStore,
        environment: Environment,
        var media: SceneMedia = SceneMedia(),
        var opacity: SceneOpacity = SceneOpacity(),
        var emission: SceneEmission = SceneEmission(),
    ) raises:
        self._spheres = spheres^
        self._sphere_surfaces = sphere_surfaces^
        self._triangle_vertices = triangle_vertices^
        self._triangle_surfaces = triangle_surfaces^
        self._triangle_meshes = triangle_meshes^
        self._triangle_mesh_normals = triangle_mesh_normals^
        self._triangle_mesh_texcoords = triangle_mesh_texcoords^
        self._triangle_instances = triangle_instances^
        self._triangle_instance_surfaces = triangle_instance_surfaces^
        self._surfaces = surfaces^
        self._lights = LightStore()
        self._media = media^
        self._opacity = opacity^
        self._emission = emission^
        self._environment = environment.copy()
        self._environment_weight = 0.0
        self._environment_alias_probabilities = List[Float32]()
        self._environment_alias_indices = List[UInt32]()
        self._validate()
        self._build_environment_distribution()
        self._build_light_store()
        if not isfinite(self.total_light_weight()):
            raise Error("combined light weight must be finite")
        self._lights.build_alias_table()

    def opacity(self) -> ref[self._opacity] SceneOpacity:
        return self._opacity

    def emission(self) -> ref[self._emission] SceneEmission:
        return self._emission

    def emitter(self, primitive: UInt32, surface: SurfaceId[1]) -> AreaEmitter:
        if case .Some(value) = self._emission.records.get(primitive):
            return value.copy()
        if surface.kind() == .EMISSIVE:
            return AreaEmitter(
                self._surfaces.emissives[Int(surface.index())].radiance
            )
        return AreaEmitter()

    def spheres(self) -> ref[self._spheres] List[Sphere[.WORLD]]:
        return self._spheres

    def sphere_surfaces(
        self,
    ) -> ref[self._sphere_surfaces] List[SurfaceId[1]]:
        return self._sphere_surfaces

    def triangle_vertices(
        self,
    ) -> ref[self._triangle_vertices] List[Point3f32[.WORLD]]:
        return self._triangle_vertices

    def triangle_surfaces(
        self,
    ) -> ref[self._triangle_surfaces] List[SurfaceId[1]]:
        return self._triangle_surfaces

    def triangle_meshes(
        self,
    ) -> ref[self._triangle_meshes] List[List[Point3f32[.LOCAL]]]:
        return self._triangle_meshes

    def triangle_mesh_normals(
        self,
    ) -> ref[self._triangle_mesh_normals] List[List[Float32]]:
        return self._triangle_mesh_normals

    def triangle_mesh_texcoords(
        self,
    ) -> ref[self._triangle_mesh_texcoords] List[List[Float32]]:
        return self._triangle_mesh_texcoords

    def triangle_instances(
        self,
    ) -> ref[self._triangle_instances] List[Instance]:
        return self._triangle_instances

    def triangle_instance_surfaces(
        self,
    ) -> ref[self._triangle_instance_surfaces] List[SurfaceId[1]]:
        return self._triangle_instance_surfaces

    def media(self) -> ref[self._media] SceneMedia:
        return self._media

    def surfaces(self) -> ref[self._surfaces] SurfaceStore:
        return self._surfaces

    def lights(self) -> ref[self._lights] LightStore:
        return self._lights

    def environment(self) -> ref[self._environment] Environment:
        return self._environment

    def environment_weight(self) -> Float32:
        return self._environment_weight

    def environment_alias_probabilities(
        self,
    ) -> ref[self._environment_alias_probabilities] List[Float32]:
        return self._environment_alias_probabilities

    def environment_alias_indices(
        self,
    ) -> ref[self._environment_alias_indices] List[UInt32]:
        return self._environment_alias_indices

    def total_light_weight(self) -> Float32:
        return self._lights.total_weight + self._environment_weight

    def _validate(mut self) raises:
        if (
            len(self._spheres) == 0
            and len(self._triangle_vertices) == 0
            and len(self._triangle_instances) == 0
        ):
            raise Error("scene requires at least one primitive")
        if len(self._spheres) != len(self._sphere_surfaces):
            raise Error("sphere and surface sidecar lengths must match")
        if len(self._triangle_vertices) % 3 != 0:
            raise Error("triangle vertex count must be a multiple of three")
        if len(self._triangle_vertices) / 3 != len(self._triangle_surfaces):
            raise Error("triangle and surface sidecar lengths must match")
        if len(self._triangle_instances) != len(
            self._triangle_instance_surfaces
        ):
            raise Error(
                "triangle instance and surface sidecar lengths must match"
            )
        if len(self._triangle_meshes) != len(self._triangle_mesh_normals):
            raise Error("triangle mesh normal sidecar lengths must match")
        if len(self._triangle_meshes) != len(self._triangle_mesh_texcoords):
            raise Error("triangle mesh texcoord sidecar lengths must match")

        self._media.validate(
            len(self._spheres),
            len(self._triangle_surfaces),
            len(self._triangle_instances),
        )
        self._validate_materials()
        self._emission.validate(
            len(self._spheres),
            len(self._triangle_surfaces),
            len(self._triangle_instances),
        )
        for entry in self._opacity.records.items():
            var primitive = entry.key >> PRIMITIVE_INDEX_BITS
            var index = entry.key & PRIMITIVE_INDEX_MASK
            var count = (
                len(self._triangle_surfaces) if primitive
                == PrimitiveKind.TRIANGLE.value else len(
                    self._spheres
                ) if primitive
                == PrimitiveKind.SPHERE.value else len(
                    self._triangle_instances
                ) if primitive
                == PrimitiveKind.TRIANGLE_INSTANCE.value else 0
            )
            if index >= UInt32(count):
                raise Error("opacity primitive index is out of range")
            entry.value.validate(len(self._surfaces.image_textures))

        if not self._environment.world_to_light.is_finite()[0]:
            raise Error("environment transform must be finite")
        if not self._environment.light_to_world.is_finite()[0]:
            raise Error("inverse environment transform must be finite")
        if not self._environment.scale.is_finite()[0]:
            raise Error("environment scale must be finite")
        if (
            self._environment.scale.x[0] < 0.0
            or self._environment.scale.y[0] < 0.0
            or self._environment.scale.z[0] < 0.0
        ):
            raise Error("environment scale must be non-negative")
        match self._environment.kind:
            case .IMAGE:
                if self._environment.texture_index >= UInt32(
                    len(self._surfaces.image_textures)
                ):
                    raise Error("environment texture index is out of range")
            case .BLACK | .PROCEDURAL | .UNIFORM:
                pass

        for i, sphere in enumerate(self._spheres):
            if not sphere.center.is_finite()[0]:
                raise Error("sphere center must be finite")
            if not isfinite(sphere.radius):
                raise Error("sphere radius must be finite")
            if sphere.radius == 0.0:
                raise Error("sphere radius must be non-zero")
            var radius = sphere.physical_radius()
            if not (
                isfinite(sphere.center.x[0] - radius)
                and isfinite(sphere.center.y[0] - radius)
                and isfinite(sphere.center.z[0] - radius)
                and isfinite(sphere.center.x[0] + radius)
                and isfinite(sphere.center.y[0] + radius)
                and isfinite(sphere.center.z[0] + radius)
            ):
                raise Error("sphere bounds must be finite")
            if not self._surfaces.contains(self._sphere_surfaces[i]):
                raise Error("sphere surface id is out of range")

        for triangle_idx, surface in enumerate(self._triangle_surfaces):
            if not self._surfaces.contains(surface):
                raise Error("triangle surface id is out of range")
            var base = 3 * triangle_idx
            if not triangle_is_valid(
                self._triangle_vertices[base],
                self._triangle_vertices[base + 1],
                self._triangle_vertices[base + 2],
            ):
                raise Error(
                    "triangle vertices must be finite and non-degenerate"
                )

        var triangle_mesh_bounds = List[AABB[.LOCAL]](
            capacity=len(self._triangle_meshes)
        )
        for mesh_idx, vertices in enumerate(self._triangle_meshes):
            if len(vertices) == 0 or len(vertices) % 3 != 0:
                raise Error(
                    "triangle mesh vertex count must be a positive multiple of"
                    " three"
                )
            ref normals = self._triangle_mesh_normals[mesh_idx]
            ref texcoords = self._triangle_mesh_texcoords[mesh_idx]
            if len(normals) != 0 and len(normals) != 3 * len(vertices):
                raise Error("triangle mesh normals must contain xyz per vertex")
            if len(texcoords) != 0 and len(texcoords) != 2 * len(vertices):
                raise Error(
                    "triangle mesh texcoords must contain uv per vertex"
                )
            var local_bounds = _validate_mesh_bounds(vertices)
            triangle_mesh_bounds.append(local_bounds)

        for i, inst in enumerate(self._triangle_instances):
            if inst.kind != .TRIANGLE:
                raise Error(
                    "triangle instance must have triangle primitive kind"
                )
            if inst.blas_idx >= UInt32(len(self._triangle_meshes)):
                raise Error("triangle instance blas_idx is out of range")
            ref surface = self._triangle_instance_surfaces[i]
            if not self._surfaces.contains(surface):
                raise Error("triangle instance surface id is out of range")

            if not inst.transform.is_finite()[0]:
                raise Error("triangle instance transform must be finite")
            var inverse = inst.transform.inverse()
            if not inverse.mask[0] or not inverse.inv.is_finite()[0]:
                raise Error("triangle instance transform must be invertible")

            var world_bounds = triangle_mesh_bounds[
                Int(inst.blas_idx)
            ].apply_transform(inst.transform)
            if not world_bounds.is_valid()[0]:
                raise Error(
                    "triangle instance transformed bounds must be finite"
                )

            ref finalized_instance = self._triangle_instances[i]
            finalized_instance.inv_transform = inverse.inv.copy()
            finalized_instance.bounds = world_bounds

    def _validate_materials(mut self) raises:
        if len(self._surfaces.mix_texture_indices) != len(
            self._surfaces.mix_amounts
        ) or len(self._surfaces.mix_uv_scales) != 2 * len(
            self._surfaces.mix_amounts
        ):
            raise Error("mix texture sidecars have invalid lengths")
        for scale in self._surfaces.mix_uv_scales:
            if not isfinite(scale):
                raise Error("mix UV scale must be finite")
        for texture_index in self._surfaces.mix_texture_indices:
            if texture_index != NO_TEXTURE and texture_index >= UInt32(
                len(self._surfaces.image_textures)
            ):
                raise Error("mix texture index is out of range")
        if len(self._surfaces.mix_surfaces) != 2 * len(
            self._surfaces.mix_amounts
        ):
            raise Error("mix material reference storage has an invalid length")
        for index, amount in enumerate(self._surfaces.mix_amounts):
            if not isfinite(amount) or amount < 0.0 or amount > 1.0:
                raise Error("mix amount must be within [0, 1]")
            for lane in range(2):
                var surface = SurfaceId.from_raw(
                    self._surfaces.mix_surfaces[2 * index + lane]
                )
                if (
                    not self._surfaces.contains(surface)
                    or surface.kind() == .EMISSIVE
                ):
                    raise Error("mix material must reference an existing BSDF")
                if surface.kind() == .MIX and Int(surface.index()) >= index:
                    raise Error("mix material references must be acyclic")
        for entry in self._surfaces.normal_maps.items():
            if not self._surfaces.contains(SurfaceId.from_raw(entry.key)):
                raise Error("normal map material is out of range")
            if entry.value >= UInt32(len(self._surfaces.image_textures)):
                raise Error("normal map image index is out of range")
        for material in self._surfaces.lambertians:
            material.validate()
            if (
                material.texture_index != NO_TEXTURE
                and material.texture_index
                >= UInt32(len(self._surfaces.image_textures))
            ):
                raise Error("lambertian texture index is out of range")

        for material in self._surfaces.metals:
            material.validate()
            if (
                material.texture_index != NO_TEXTURE
                and material.texture_index
                >= UInt32(len(self._surfaces.image_textures))
            ):
                raise Error("metal texture index is out of range")

        for material in self._surfaces.dielectrics:
            material.validate()
            if (
                material.eta_spectrum != NO_TEXTURE
                and material.eta_spectrum
                >= UInt32(len(self._surfaces.optical_spectra))
            ):
                raise Error("dielectric optical spectrum index is out of range")
            if (
                material.displacement_texture_index != NO_TEXTURE
                and material.displacement_texture_index
                >= UInt32(len(self._surfaces.image_textures))
            ):
                raise Error(
                    "dielectric displacement texture index is out of range"
                )

        for material in self._surfaces.coated_diffuses:
            material.validate()
            for spectrum in [
                material.conductor_eta_spectrum,
                material.conductor_k_spectrum,
            ]:
                if spectrum != NO_TEXTURE and spectrum >= UInt32(
                    len(self._surfaces.optical_spectra)
                ):
                    raise Error(
                        "conductor optical spectrum index is out of range"
                    )
            if (
                material.roughness_texture_index != NO_TEXTURE
                and material.roughness_texture_index
                >= UInt32(len(self._surfaces.image_textures))
            ):
                raise Error("conductor roughness texture index is out of range")
            if (
                material.texture_index != NO_TEXTURE
                and material.texture_index
                >= UInt32(len(self._surfaces.image_textures))
            ):
                raise Error("coated diffuse texture index is out of range")
            if (
                material.displacement_texture_index != NO_TEXTURE
                and material.displacement_texture_index
                >= UInt32(len(self._surfaces.image_textures))
            ):
                raise Error(
                    "coated diffuse displacement texture index is out of range"
                )

        for material in self._surfaces.emissives:
            material.validate()

        for texture_idx in range(len(self._surfaces.image_textures)):
            ref texture = self._surfaces.image_textures[texture_idx]
            if texture.width <= 0 or texture.height <= 0:
                raise Error("image texture dimensions must be positive")
            if len(texture.pixels) != 3 * texture.width * texture.height:
                raise Error("image texture RGB storage has an invalid length")
            if (
                len(texture.alpha) != 0
                and len(texture.alpha) != texture.width * texture.height
            ):
                raise Error("image texture alpha storage has an invalid length")
            for value in texture.alpha:
                if not isfinite(value) or value < 0.0 or value > 1.0:
                    raise Error(
                        "image texture alpha must be finite and within [0, 1]"
                    )
            texture.has_scalar_alpha = False
            for value in texture.alpha:
                if value != 1.0:
                    texture.has_scalar_alpha = True
                    break
            comptime width = simd_width_of[Float32]()
            var end = len(texture.pixels) // width * width
            for first in range(0, end, width):
                var channels = (
                    texture.pixels.unsafe_ptr()
                    .unsafe_offset(first)
                    .unsafe_load[width=width, alignment=1]()
                )
                if not (isfinite(channels) & channels.ge(0.0)).reduce_and():
                    raise Error(
                        "image texture channels must be finite and non-negative"
                    )
            for first in range(end, len(texture.pixels)):
                var channel = texture.pixels[first]
                if not isfinite(channel) or channel < 0.0:
                    raise Error(
                        "image texture channels must be finite and non-negative"
                    )

    def _build_environment_distribution(mut self) raises:
        var average_radiance = self._environment.scale
        match self._environment.kind:
            case .BLACK:
                return
            case .IMAGE:
                ref texture = self._surfaces.image_textures[
                    Int(self._environment.texture_index)
                ]
                var count = texture.width * texture.height
                if count > Int(UInt32.MAX):
                    raise Error(
                        "environment importance table exceeds UInt32 indexing"
                    )
                var weights = List[Float32](length=count, fill=0.0)
                var total = Float64(0.0)
                for pixel_idx in range(count):
                    var importance = _light_importance(
                        Color.load(texture.pixels, 3 * pixel_idx)
                        * self._environment.scale
                    )
                    weights[pixel_idx] = importance
                    total += Float64(importance)
                if not isfinite(total) or not isfinite(Float32(total)):
                    raise Error(
                        "environment importance integral must be finite"
                    )
                build_alias_distribution(
                    Span(weights),
                    total,
                    self._environment_alias_probabilities,
                    self._environment_alias_indices,
                )
                if total > 0.0:
                    self._environment_weight = Float32(
                        4.0 * Float64(pi) * total / Float64(count)
                    )
                if not isfinite(self._environment_weight):
                    raise Error("environment light weight must be finite")
                return
            case .PROCEDURAL:
                # The procedural sky is linear in direction.y, whose sphere-wide
                # average is zero.
                average_radiance = Color(0.75, 0.85, 1.0)
            case .UNIFORM:
                pass
        self._environment_weight = (
            4.0 * pi * _light_importance(average_radiance)
        )
        if not isfinite(self._environment_weight):
            raise Error("environment light weight must be finite")

    def _prepare_instance_light_records(self) -> List[LightRecord]:
        var count = len(self.triangle_instances())
        var work = List[Int]()
        var offsets = List[Int]()
        offsets.append(0)
        for i in range(count):
            if (
                self.emitter(
                    PrimitiveId(.TRIANGLE_INSTANCE, UInt32(i)).value,
                    self.triangle_instance_surfaces()[i],
                ).radiance.max_component()
                > 0
            ):
                work.append(i)
                offsets.append(
                    offsets[len(offsets) - 1]
                    + len(
                        self.triangle_meshes()[
                            Int(self.triangle_instances()[i].blas_idx)
                        ]
                    )
                    // 3
                )
        count = len(work)
        var records = List[LightRecord](unsafe_uninit_length=offsets[count])

        def prepare(task: Int) {imm, mut records}:
            var i = work[task]
            if offsets[task] == offsets[task + 1]:
                return
            ref surface = self.triangle_instance_surfaces()[i]
            var emitter = self.emitter(
                PrimitiveId(.TRIANGLE_INSTANCE, UInt32(i)).value, surface
            )
            var radiance = emitter.radiance
            ref instance = self.triangle_instances()[i]
            ref vertices = self.triangle_meshes()[Int(instance.blas_idx)]
            ref texcoords = self.triangle_mesh_texcoords()[
                Int(instance.blas_idx)
            ]
            var reverse = instance.transform.reverses_orientation()[0]

            def block[lanes: Int](first: Int) {imm, mut records}:
                var values = _transformed_triangles[lanes](
                    vertices, first, instance.transform, reverse, radiance
                )
                comptime for lane in range(lanes):
                    records[
                        offsets[task] + first + lane
                    ] = _instance_light_record(
                        values,
                        lane,
                        surface,
                        i,
                        radiance,
                        emitter.two_sided,
                        texcoords,
                        first + lane,
                        reverse,
                    )

            vectorize[simd_width_of[Float32]()](
                offsets[task + 1] - offsets[task], block
            )

        if count > 1:
            parallelize(prepare, count, min(8, count))
        elif count:
            prepare(0)
        return records^

    def _build_light_store(mut self) raises:
        for idx, surface in enumerate(self._triangle_surfaces):
            var emitter = self.emitter(
                PrimitiveId(.TRIANGLE, UInt32(idx)).value, surface
            )
            if emitter.radiance.max_component() > 0.0:
                var radiance = emitter.radiance
                ref p0 = self._triangle_vertices[3 * idx + 0]
                ref p1 = self._triangle_vertices[3 * idx + 1]
                ref p2 = self._triangle_vertices[3 * idx + 2]
                var weight = (
                    triangle_area(p0, p1, p2)
                    * _light_importance(radiance)
                    * Float32(2.0 if emitter.two_sided else 1.0)
                )
                if not isfinite(weight):
                    raise Error("triangle light weight must be finite")
                if weight > 0.0:
                    self._append_light(
                        LightRecord.triangle(
                            PrimitiveId(PrimitiveKind.TRIANGLE, UInt32(idx)),
                            surface,
                            weight,
                            p0,
                            p1,
                            p2,
                            radiance=radiance,
                            two_sided=emitter.two_sided,
                        )
                    )

        for idx, surface in enumerate(self._sphere_surfaces):
            var emitter = self.emitter(
                PrimitiveId(.SPHERE, UInt32(idx)).value, surface
            )
            if emitter.radiance.max_component() > 0.0:
                var radiance = emitter.radiance
                var radius = self._spheres[idx].physical_radius()
                var weight = (
                    4.0 * pi * radius * radius * _light_importance(radiance)
                ) * Float32(2.0 if emitter.two_sided else 1.0)
                if not isfinite(weight):
                    raise Error("sphere light weight must be finite")
                if weight > 0.0:
                    self._append_light(
                        LightRecord.sphere(
                            PrimitiveId(PrimitiveKind.SPHERE, UInt32(idx)),
                            surface,
                            weight,
                            self._spheres[idx].center,
                            radius,
                            radiance,
                            emitter.two_sided,
                        )
                    )

        if len(self._triangle_instances) > 1 and (
            len(self._surfaces.emissives) > 0 or len(self._emission.records) > 0
        ):
            var instances = 0
            var triangles = 0
            for i in range(len(self._triangle_instances)):
                if (
                    self.emitter(
                        PrimitiveId(.TRIANGLE_INSTANCE, UInt32(i)).value,
                        self._triangle_instance_surfaces[i],
                    ).radiance.max_component()
                    > 0
                ):
                    instances += 1
                    triangles += (
                        len(
                            self._triangle_meshes[
                                Int(self._triangle_instances[i].blas_idx)
                            ]
                        )
                        // 3
                    )
            if instances > 1 and triangles >= 4096:
                var records = self._prepare_instance_light_records()
                self._lights.records.reserve(
                    len(self._lights.records) + len(records)
                )
                for record in records:
                    if not isfinite(record.weight):
                        raise Error(
                            "triangle instance light weight must be finite"
                        )
                    if record.weight > 0.0:
                        self._append_light(record.copy())
                return

        for instance_idx, surface in enumerate(
            self._triangle_instance_surfaces
        ):
            var emitter = self.emitter(
                PrimitiveId(.TRIANGLE_INSTANCE, UInt32(instance_idx)).value,
                surface,
            )
            if emitter.radiance.max_component() <= 0.0:
                continue
            var radiance = emitter.radiance
            var transform = self._triangle_instances[
                instance_idx
            ].transform.copy()
            var mesh_idx = Int(self._triangle_instances[instance_idx].blas_idx)
            _append_instance_lights(
                self._triangle_meshes[mesh_idx],
                transform,
                surface,
                instance_idx,
                radiance,
                self._lights,
                self._triangle_mesh_texcoords[mesh_idx],
                emitter.two_sided,
            )

    def _append_light(mut self, var light: LightRecord) raises:
        self._lights.append(light^)
        if not isfinite(self._lights.total_weight):
            raise Error("total light weight must be finite")
