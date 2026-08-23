from std.pathlib import Path


trait TextLoader:
    def read_text(self, path: String) raises -> String:
        ...


@fieldwise_init
struct PathTextLoader(Copyable, TextLoader):
    def read_text(self, path: String) raises -> String:
        return Path(path).read_text()


struct MemoryTextLoader(TextLoader):
    var files: Dict[String, String]

    def __init__(out self):
        self.files = Dict[String, String]()

    def add_file(mut self, path: String, text: String):
        self.files[path] = text

    def read_text(self, path: String) raises -> String:
        if path in self.files:
            return self.files[path]
        raise Error("MemoryTextLoader: file not found: " + path)
