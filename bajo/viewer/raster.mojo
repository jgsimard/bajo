"""Cached raylib geometry and native presentation of Bajo scenes."""

from std.math import atan, pi
from std.os import abort
from bajo.core.camera import Camera
from bajo.core import (
    Affine3f32,
    Point3f32,
    Vec3W,
    Frame as CoordinateFrame,
    cross,
    normalize,
)
from bajo.rt.image_output import color_to_byte
from bajo.rt.material_types import SurfaceId, SurfaceStore
from bajo.rt.scene_data import SceneData
from raylib import (
    Library,
    Texture2D as DisplayTexture,
    Camera3D,
    Color,
    Frame,
    Scene3D,
    Material,
    Matrix,
    Mesh,
    Vector3,
    Window,
)


comptime VERTEX_SHADER = """#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec4 vertexColor;
uniform mat4 mvp;
uniform mat4 matNormal;
out vec3 normal;
out vec4 color;
void main() {
    normal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    color = vertexColor;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""
comptime FRAGMENT_SHADER = """#version 330
in vec3 normal;
in vec4 color;
uniform vec4 colDiffuse;
out vec4 finalColor;
void main() {
    float diffuse = abs(dot(normalize(normal), normalize(vec3(-0.4, 0.8, 0.5))));
    float lighting = mix(1.0, 0.30 + 0.70 * diffuse, color.a * colDiffuse.a);
    finalColor = vec4(color.rgb * colDiffuse.rgb * lighting, 1.0);
}
"""


def raster_camera(camera: Camera) -> Camera3D:
    return Camera3D(
        Vector3(camera.origin.x, camera.origin.y, camera.origin.z),
        Vector3(
            camera.origin.x + camera.forward.x,
            camera.origin.y + camera.forward.y,
            camera.origin.z + camera.forward.z,
        ),
        Vector3(camera.up.x, camera.up.y, camera.up.z),
        2 * atan(camera.fov_scale) * (180 / pi),
        0,
    )


def identity_matrix() -> Matrix:
    return Matrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


def instance_matrix[
    source: CoordinateFrame, target: CoordinateFrame
](transform: Affine3f32[source, target]) -> Matrix:
    return Matrix(
        transform.m00,
        transform.m01,
        transform.m02,
        transform.tx,
        transform.m10,
        transform.m11,
        transform.m12,
        transform.ty,
        transform.m20,
        transform.m21,
        transform.m22,
        transform.tz,
        0,
        0,
        0,
        1,
    )


def raster_view_matrix(camera: Camera) raises -> Matrix:
    """Keep the authored camera axes, including PBRT's opposite handedness."""
    var camera_to_world = Affine3f32[.LOCAL, .WORLD](
        camera.right.x,
        camera.up.x,
        -camera.forward.x,
        camera.origin.x,
        camera.right.y,
        camera.up.y,
        -camera.forward.y,
        camera.origin.y,
        camera.right.z,
        camera.up.z,
        -camera.forward.z,
        camera.origin.z,
    )
    var inverse = camera_to_world.inverse()
    if not inverse.mask[0] or not inverse.inv.is_finite()[0]:
        raise Error("viewer camera basis must be finite and invertible")
    return instance_matrix(inverse.inv)


def surface_color(surfaces: SurfaceStore, surface: SurfaceId[1]) -> Color:
    var selected = surfaces.resolve_mix(surface, 0.5)
    var albedo = Vec3W(0.8, 0.9, 1.0)
    var emissive = False
    var index = Int(selected.index())
    match selected.kind():
        case .LAMBERTIAN:
            albedo = surfaces.lambertians[index].albedo
        case .METAL:
            albedo = surfaces.metals[index].albedo
        case .COATED_DIFFUSE:
            albedo = surfaces.coated_diffuses[index].albedo
        case .EMISSIVE:
            albedo = surfaces.emissives[index].radiance
            emissive = True
        case .DIELECTRIC | .MIX:
            pass
    return Color(
        color_to_byte(albedo.x),
        color_to_byte(albedo.y),
        color_to_byte(albedo.z),
        UInt8(0 if emissive else 255),
    )


@fieldwise_init
struct RasterDraw(ImplicitlyCopyable):
    var mesh: Int
    var transform: Matrix
    var color: Color


def _upload_triangles[
    frame: CoordinateFrame
](
    window: Window,
    vertices: ImmSpan[Point3f32[frame], _],
    normals: ImmSpan[Float32, _],
    surfaces: SurfaceStore,
    ids: ImmSpan[SurfaceId[1], _],
) raises -> Mesh:
    var positions = List[Float32](capacity=len(vertices) * 3)
    var packed_normals = List[Float32](capacity=len(vertices) * 3)
    var colors = List[UInt8](capacity=len(vertices) * 4)
    for triangle in range(len(vertices) // 3):
        var base = 3 * triangle
        var normal = normalize(
            cross(
                vertices[base + 1] - vertices[base],
                vertices[base + 2] - vertices[base],
            )
        )
        var color = Color(255, 255, 255, 255)
        if len(ids):
            color = surface_color(surfaces, ids[triangle])
        for corner in range(3):
            var index = base + corner
            positions.append(vertices[index].x)
            positions.append(vertices[index].y)
            positions.append(vertices[index].z)
            comptime for axis in range(3):
                packed_normals.append(
                    normals[3 * index + axis] if len(normals) else normal[axis]
                )
            colors.append(color.r)
            colors.append(color.g)
            colors.append(color.b)
            colors.append(color.a)
    return window.create_mesh(positions, packed_normals, colors)


def _unload_meshes(var meshes: List[Mesh], library: Library):
    """Consume every mesh, including empty placeholders, before its context closes.
    """

    def unload(var mesh: Mesh) {imm library}:
        library.unload_mesh(mesh^)

    meshes^.deinit_with(unload)


def _unload_material(material: Material, library: Library):
    try:
        library.unload_material(material)
    except error:
        abort(String(t"raylib raster cleanup failed: {error}"))


struct RasterScene:
    """Own uploaded meshes once; draw shared meshes with per-instance matrices.

    Keep this owner inside the window session so GPU resources are destroyed
    before the OpenGL context. Its library handle retains all cleanup symbols.
    """

    var library: Library
    var meshes: List[Mesh]
    var draws: List[RasterDraw]
    var material: Material

    def __init__(
        out self, path: String, window: Window, data: SceneData
    ) raises:
        var library = Library(path)
        var material = library.default_material()
        var meshes = List[Mesh]()
        var draws = List[RasterDraw]()
        try:
            var vertex_source = String(VERTEX_SHADER)
            var fragment_source = String(FRAGMENT_SHADER)
            var shader = library.load_shader(vertex_source, fragment_source)
            material.shader = shader
            library.set_shader_location(
                shader, 10, library.shader_location(shader, String("matNormal"))
            )
            if len(data.triangle_vertices()):
                meshes.append(
                    _upload_triangles(
                        window,
                        data.triangle_vertices(),
                        List[Float32](),
                        data.surfaces(),
                        data.triangle_surfaces(),
                    )
                )
                draws.append(
                    RasterDraw(0, identity_matrix(), Color(255, 255, 255, 255))
                )
            var mesh_base = len(meshes)
            for index in range(len(data.triangle_meshes())):
                if len(data.triangle_meshes()[index]):
                    meshes.append(
                        _upload_triangles(
                            window,
                            data.triangle_meshes()[index],
                            data.triangle_mesh_normals()[index],
                            data.surfaces(),
                            List[SurfaceId[1]](),
                        )
                    )
                else:
                    meshes.append(Mesh())
            for index in range(len(data.triangle_instances())):
                ref instance = data.triangle_instances()[index]
                draws.append(
                    RasterDraw(
                        mesh_base + Int(instance.blas_idx),
                        instance_matrix(instance.transform),
                        surface_color(
                            data.surfaces(),
                            data.triangle_instance_surfaces()[index],
                        ),
                    )
                )
            if len(data.spheres()):
                var sphere_mesh = len(meshes)
                meshes.append(window.sphere_mesh(1, 20, 24))
                for index in range(len(data.spheres())):
                    ref sphere = data.spheres()[index]
                    var r = sphere.radius
                    draws.append(
                        RasterDraw(
                            sphere_mesh,
                            Matrix(
                                r,
                                0,
                                0,
                                sphere.center.x,
                                0,
                                r,
                                0,
                                sphere.center.y,
                                0,
                                0,
                                r,
                                sphere.center.z,
                                0,
                                0,
                                0,
                                1,
                            ),
                            surface_color(
                                data.surfaces(), data.sphere_surfaces()[index]
                            ),
                        )
                    )
        except error:
            _unload_meshes(meshes^, library)
            library.unload_material(material)
            raise error
        self.library = library^
        self.material = material
        self.meshes = meshes^
        self.draws = draws^

    def draw(self, frame: Scene3D[Frame]):
        for draw in self.draws:
            if self.meshes[draw.mesh].vao_id:
                self.library.set_material_color(self.material, 0, draw.color)
                frame.draw_mesh(
                    self.meshes[draw.mesh], self.material, draw.transform
                )

    def __deinit__(deinit self):
        _unload_meshes(self.meshes^, self.library)
        _unload_material(self.material, self.library)
