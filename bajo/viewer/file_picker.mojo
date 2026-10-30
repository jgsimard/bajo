"""A raylib file browser for PBRT scenes, with no external GUI dependency."""

from std.math import min, max, clamp
from .raylib import Window, Frame, Font, Color, Vector2


def contains(point: Vector2, x: Int, y: Int, width: Int, height: Int) -> Bool:
    return (
        point.x >= Float32(x)
        and point.x < Float32(x + width)
        and point.y >= Float32(y)
        and point.y < Float32(y + height)
    )


def is_scene_file(window: Window, path: String) raises -> Bool:
    var extension = window.file_extension(path)
    return extension == ".pbrt" or extension == ".pbrt-v4"


@fieldwise_init
struct FileEntry(Copyable):
    var path: String
    var name: String
    var directory: Bool


struct FilePicker:
    var active: Bool
    var directory: String
    var entries: List[FileEntry]
    var selected: Int
    var offset: Int
    var last_click: Float64
    var error: String

    def __init__(out self):
        self.active = False
        self.directory = "."
        self.entries = List[FileEntry]()
        self.selected = -1
        self.offset = 0
        self.last_click = -1
        self.error = ""

    def navigate(mut self, window: Window, path: String) raises:
        var canonical = window.canonical_path(path)
        if not window.directory_exists(canonical):
            raise Error("directory does not exist: " + path)
        var folders = List[String]()
        var files = List[String]()
        for entry in window.directory_files(canonical):
            var name = window.file_name(entry)
            if name == "." or name == "..":
                continue
            if window.directory_exists(entry):
                folders.append(entry)
            elif is_scene_file(window, entry):
                files.append(entry)
        sort(folders)
        sort(files)
        var entries = List[FileEntry](capacity=len(folders) + len(files))
        for folder in folders:
            entries.append(FileEntry(folder, window.file_name(folder), True))
        for file in files:
            entries.append(FileEntry(file, window.file_name(file), False))
        self.entries = entries^
        self.directory = canonical
        self.selected = 0 if len(self.entries) else -1
        self.offset = 0
        self.last_click = -1
        self.error = ""

    def show(mut self, window: Window, current_path: String) raises:
        var directory = self.directory
        if current_path != "":
            directory = window.directory_path(current_path)
        elif self.directory == "." and window.directory_exists("assets/pbrt"):
            directory = "assets/pbrt"
        self.navigate(window, directory)
        self.active = True

    def open_path(
        mut self, window: Window, path: String
    ) raises -> Optional[String]:
        if window.directory_exists(path):
            self.navigate(window, path)
        elif not window.file_exists(path):
            self.error = "File does not exist: " + path
        elif not is_scene_file(window, path):
            self.error = "Select a .pbrt or .pbrt-v4 scene."
        else:
            return path
        return Optional[String]()

    def update(mut self, window: Window) raises -> Optional[String]:
        if window.is_key_pressed(256):
            self.active = False
            return Optional[String]()
        if (
            window.is_key_down(341) or window.is_key_down(345)
        ) and window.is_key_pressed(86):
            return self.open_path(window, window.clipboard_text())
        var width = window.width()
        var height = window.height()
        var visible = max(1, (height - 264) // 28)
        var mouse = window.mouse_position()
        var clicked = window.mouse_pressed()
        if window.is_key_pressed(259) or (
            clicked and contains(mouse, 32, 96, 80, 28)
        ):
            self.navigate(window, window.parent_directory(self.directory))
        if window.is_key_pressed(264):
            self.selected = min(len(self.entries) - 1, self.selected + 1)
        if window.is_key_pressed(265):
            self.selected = max(0, self.selected - 1) if len(
                self.entries
            ) else -1
        self.offset -= Int(window.mouse_wheel() * 3)
        if window.is_key_pressed(264) or window.is_key_pressed(265):
            self.offset = clamp(
                self.offset, self.selected - visible + 1, self.selected
            )
        self.offset = clamp(self.offset, 0, max(0, len(self.entries) - visible))
        var activate = window.is_key_pressed(257)
        if clicked and contains(mouse, 32, 136, width - 64, visible * 28):
            var selected = self.offset + (Int(mouse.y) - 136) // 28
            if selected < len(self.entries):
                var now = window.time()
                activate = (
                    selected == self.selected
                    and self.last_click >= 0
                    and now - self.last_click < 0.35
                )
                self.selected = selected
                self.last_click = now
        if clicked and contains(mouse, width - 252, height - 60, 100, 32):
            self.active = False
            return Optional[String]()
        if clicked and contains(mouse, width - 140, height - 60, 108, 32):
            activate = True
        if (
            activate
            and self.selected >= 0
            and self.selected < len(self.entries)
        ):
            var path = self.entries[self.selected].path
            return self.open_path(window, path)
        return Optional[String]()

    def draw(self, frame: Frame, font: Font, width: Int, height: Int) raises:
        frame.draw_rectangle(0, 0, width, height, Color(0, 0, 0, 180))
        frame.draw_rectangle(
            16, 16, width - 32, height - 32, Color(28, 33, 42, 255)
        )
        frame.draw_text_ex(
            font,
            String("Open PBRT scene"),
            32,
            28,
            24,
            Color(240, 245, 255, 255),
        )
        frame.begin_scissor(32, 64, width - 64, 24)
        frame.draw_text_ex(
            font, self.directory, 32, 64, 16, Color(190, 205, 220, 255)
        )
        frame.end_scissor()
        frame.draw_rectangle(32, 96, 80, 28, Color(55, 67, 85, 255))
        frame.draw_text_ex(
            font, String("Up"), 48, 100, 18, Color(240, 245, 255, 255)
        )
        frame.draw_text_ex(
            font,
            String("Folders and .pbrt / .pbrt-v4 files"),
            128,
            102,
            16,
            Color(190, 205, 220, 255),
        )
        var visible = max(1, (height - 264) // 28)
        frame.begin_scissor(32, 136, width - 64, visible * 28)
        for row in range(visible):
            var index = self.offset + row
            if index >= len(self.entries):
                break
            var y = 136 + row * 28
            if index == self.selected:
                frame.draw_rectangle(
                    32, y, width - 64, 28, Color(48, 78, 112, 255)
                )
            var label = self.entries[index].name
            if self.entries[index].directory:
                label += "/"
            frame.draw_text_ex(
                font, label, 40, y + 4, 18, Color(230, 235, 245, 255)
            )
        if not len(self.entries):
            frame.draw_text_ex(
                font,
                String("No PBRT files or subfolders here."),
                40,
                140,
                18,
                Color(180, 195, 210, 255),
            )
        frame.end_scissor()
        frame.draw_text_ex(
            font,
            String(
                "Double-click / Enter: open | Backspace: up | Ctrl+V: open"
                " copied path"
            ),
            32,
            height - 116,
            14,
            Color(180, 195, 210, 255),
        )
        frame.begin_scissor(32, height - 92, width - 64, 24)
        frame.draw_text_ex(
            font, self.error, 32, height - 92, 16, Color(255, 130, 110, 255)
        )
        frame.end_scissor()
        frame.draw_rectangle(
            width - 252, height - 60, 100, 32, Color(55, 67, 85, 255)
        )
        frame.draw_text_ex(
            font,
            String("Cancel"),
            width - 236,
            height - 55,
            18,
            Color(240, 245, 255, 255),
        )
        frame.draw_rectangle(
            width - 140, height - 60, 108, 32, Color(45, 105, 160, 255)
        )
        frame.draw_text_ex(
            font,
            String("Open"),
            width - 118,
            height - 55,
            18,
            Color(240, 245, 255, 255),
        )
