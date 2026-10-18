import std.os

from bajo.parser.obj.types import ObjMaterial, ObjMesh, ObjTexture
from bajo.parser.asset_loader import AssetLoader
from bajo.parser.obj.cursor import ObjLineCursor
from bajo.parser.obj.constants import MINUS


def _map_name_from_tail(mut cur: ObjLineCursor) -> StringSpan[cur.origin]:
    var candidate = StringSpan[cur.origin]()
    while cur.has_next():
        var word = cur.next_word()
        if word.byte_length() == 0:
            break
        # Skip flags like '-clamp' or '-s'
        var word_bytes = word.as_bytes()
        if word_bytes[0] == MINUS:
            continue
        candidate = word
    return candidate


def _upsert_material(
    mut mesh: ObjMesh, var material: ObjMaterial
) raises -> Int:
    var next_index = len(mesh.materials)
    var index = mesh.material_names.setdefault(material.name, next_index)
    if index != next_index and mesh.materials[index].fallback:
        mesh.materials[index] = material^
        return index
    mesh.materials.append(material^)
    return next_index


def _add_texture(
    mut mesh: ObjMesh, name: ImmStringSpan, base: String
) raises -> Int:
    var filename = String(name)
    var path = std.os.path.join(base, filename)
    if case .Some(index) = mesh.texture_names.get(path):
        return index

    var index = len(mesh.textures)
    mesh.textures.append(ObjTexture(filename^, path))
    mesh.texture_names[path] = index
    return index


def _read_mtl_text(mut mesh: ObjMesh, base: String, text: String) raises:
    var current = ObjMaterial()
    var have_current = False
    var found_d = False

    var text_span = StringSpan(text)
    var text_len = text_span.byte_length()
    var bytes = text_span.as_bytes()
    var line_start = 0

    while line_start < text_len:
        # Fast newline scan. CR and comments are handled lazily by ObjLineCursor.
        var line_end = text_span.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        var cur = ObjLineCursor(bytes[line_start:line_end])
        var tag = cur.next_word()

        if tag.byte_length() == 0:
            line_start = line_end + 1
            continue

        if tag == "newmtl":
            if have_current:
                _ = _upsert_material(mesh, current^)
            var name = cur.joined_rest_of_line()
            current = ObjMaterial(name, fallback=False)
            have_current = True
            found_d = False

        elif have_current:
            match tag:
                case "Ka":
                    current.Ka = (
                        cur.next_f32(),
                        cur.next_f32(),
                        cur.next_f32(),
                    )
                case "Kd":
                    current.Kd = (
                        cur.next_f32(),
                        cur.next_f32(),
                        cur.next_f32(),
                    )
                case "Ks":
                    current.Ks = (
                        cur.next_f32(),
                        cur.next_f32(),
                        cur.next_f32(),
                    )
                case "Ke":
                    current.Ke = (
                        cur.next_f32(),
                        cur.next_f32(),
                        cur.next_f32(),
                    )
                case "Kt":
                    current.Kt = (
                        cur.next_f32(),
                        cur.next_f32(),
                        cur.next_f32(),
                    )
                case "Tf":
                    current.Tf = (
                        cur.next_f32(),
                        cur.next_f32(),
                        cur.next_f32(),
                    )
                case "Ns":
                    current.Ns = cur.next_f32()
                case "Ni":
                    current.Ni = cur.next_f32()
                case "illum":
                    current.illum = Int(
                        cur.next_f32()
                    )  # Can just cast the f32 parser for illum ints
                case "d":
                    current.d = cur.next_f32()
                    found_d = True
                case "Tr":
                    if not found_d:
                        current.d = 1.0 - cur.next_f32()
                case _:
                    if tag[byte=:4] == "map_" or tag == "bump":
                        var name = _map_name_from_tail(cur)
                        if name.byte_length() > 0:
                            var val = _add_texture(mesh, name, base)
                            match tag:
                                case "map_Ka":
                                    current.map_Ka = val
                                case "map_Kd":
                                    current.map_Kd = val
                                case "map_Ks":
                                    current.map_Ks = val
                                case "map_Ke":
                                    current.map_Ke = val
                                case "map_Kt":
                                    current.map_Kt = val
                                case "map_Ns":
                                    current.map_Ns = val
                                case "map_Ni":
                                    current.map_Ni = val
                                case "map_d":
                                    current.map_d = val
                                case "map_bump" | "bump":
                                    current.map_bump = val
                                case _:
                                    pass

        line_start = line_end + 1

    if have_current:
        _ = _upsert_material(mesh, current^)


def _read_mtl_file[
    Loader: AssetLoader
](mut mesh: ObjMesh, obj_path: String, mtl_name: String, loader: Loader) raises:
    var base = std.os.path.dirname(obj_path)
    var mtl_path = std.os.path.join(base, mtl_name)
    var text = loader.read_text(mtl_path)
    _read_mtl_text(mesh, std.os.path.dirname(mtl_path), text)
