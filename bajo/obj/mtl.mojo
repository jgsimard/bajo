import std.os

from bajo.obj.types import ObjMaterial
from bajo.obj.obj import ObjLineCursor, MINUS


def _map_name_from_tail[o: Origin](mut cur: ObjLineCursor[o]) -> String:
    var candidate = ""
    while cur.has_next():
        var word = cur.next_word()
        if word.byte_length() == 0:
            break
        # Skip flags like '-clamp' or '-s'
        if word.unsafe_ptr().load(0) == MINUS:
            continue
        candidate = String(word)
    return candidate


def _read_mtl_text(mut mesh: ObjMesh, base: String, text: String) raises:
    var current = ObjMaterial()
    var have_current = False
    var found_d = False

    var text_slice = StringSlice(text)
    var text_len = text_slice.byte_length()
    var line_start = 0

    while line_start < text_len:
        # Fast newline scan. CR and comments are handled lazily by ObjLineCursor.
        var line_end = text_slice.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        var cur = ObjLineCursor(text_slice, line_start, line_end)
        var tag = cur.next_word()

        if tag.byte_length() == 0:
            line_start = line_end + 1
            continue

        if tag == "newmtl":
            if have_current:
                _ = mesh._upsert_material(current)
            var name = cur.joined_rest_of_line()
            current = ObjMaterial(name, fallback=False)
            have_current = True
            found_d = False

        elif have_current:
            if tag == "Ka":
                current.Ka = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Kd":
                current.Kd = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Ks":
                current.Ks = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Ke":
                current.Ke = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Kt":
                current.Kt = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Tf":
                current.Tf = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Ns":
                current.Ns = cur.next_f32()
            elif tag == "Ni":
                current.Ni = cur.next_f32()
            elif tag == "illum":
                current.illum = Int(
                    cur.next_f32()
                )  # Can just cast the f32 parser for illum ints
            elif tag == "d":
                current.d = cur.next_f32()
                found_d = True
            elif tag == "Tr":
                if not found_d:
                    current.d = 1.0 - cur.next_f32()
            elif tag[byte=:4] == "map_" or tag == "bump":
                var name = _map_name_from_tail(cur)
                if name.byte_length() > 0:
                    var val = mesh._add_texture(name, base)
                    if tag == "map_Ka":
                        current.map_Ka = val
                    elif tag == "map_Kd":
                        current.map_Kd = val
                    elif tag == "map_Ks":
                        current.map_Ks = val
                    elif tag == "map_Ke":
                        current.map_Ke = val
                    elif tag == "map_Kt":
                        current.map_Kt = val
                    elif tag == "map_Ns":
                        current.map_Ns = val
                    elif tag == "map_Ni":
                        current.map_Ni = val
                    elif tag == "map_d":
                        current.map_d = val
                    elif tag == "map_bump" or tag == "bump":
                        current.map_bump = val

        line_start = line_end + 1

    if have_current:
        _ = mesh._upsert_material(current)


def _read_mtl_file[
    Loader: ObjTextLoader
](mut mesh: ObjMesh, obj_path: String, mtl_name: String, loader: Loader) raises:
    var base = std.os.path.dirname(obj_path)
    var mtl_path = std.os.path.join(base, mtl_name)
    var text = loader.read_text(mtl_path)
    _read_mtl_text(mesh, std.os.path.dirname(mtl_path), text)
