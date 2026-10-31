// Script UI input while retaining real raylib drawing and file handling.
#include <raylib.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int frames;
static int scenes;
static int browser_frames;
static int custom_scene;
static int load_error;
static int dropped_unloaded;
static int custom_trace;
static int completed_samples;
static int total_samples;
static int completed_batches;
static int sample_resets;
static int batch_control_frame = -1;
static int batch_setting_four;
static int orientation_trace;
static int live_meshes;
static int drawing;
static int scene_3d;
static int scissor;
static int ownership_errors;
static int partial_mesh_unloads;

static void *real_symbol(const char *name) {
    static void *library;
    if (!library) library = dlopen(getenv("BAJO_TEST_RAYLIB"), RTLD_NOW);
    void *symbol = library ? dlsym(library, name) : NULL;
    if (!symbol) abort();
    return symbol;
}

static bool mode(const char *name) {
    const char *value = getenv("BAJO_TEST_SCENE_INPUT");
    return strcmp(value ? value : "cycle", name) == 0;
}

static bool batch_mode(void) {
    return mode("batch") || mode("batch-key") || mode("batch-click");
}

void BeginDrawing(void) {
    if (drawing || scene_3d || scissor) ++ownership_errors;
    drawing = 1;
    void (*begin)(void) = real_symbol("BeginDrawing");
    begin();
}

void BeginMode3D(Camera3D camera) {
    if (!drawing || scene_3d) ++ownership_errors;
    scene_3d = 1;
    void (*begin)(Camera3D) = real_symbol("BeginMode3D");
    begin(camera);
}

void EndMode3D(void) {
    if (!drawing || !scene_3d) ++ownership_errors;
    scene_3d = 0;
    void (*end)(void) = real_symbol("EndMode3D");
    end();
}

void BeginScissorMode(int x, int y, int width, int height) {
    if (!drawing || scissor) ++ownership_errors;
    scissor = 1;
    void (*begin)(int, int, int, int) = real_symbol("BeginScissorMode");
    begin(x, y, width, height);
}

void EndScissorMode(void) {
    if (!drawing || !scissor) ++ownership_errors;
    scissor = 0;
    void (*end)(void) = real_symbol("EndScissorMode");
    end();
}

void UploadMesh(Mesh *mesh, bool dynamic) {
    if (mode("upload-failure")) return;
    void (*upload)(Mesh *, bool) = real_symbol("UploadMesh");
    upload(mesh, dynamic);
    if (mesh->vaoId) ++live_meshes;
}

Mesh GenMeshSphere(float radius, int rings, int slices) {
    Mesh (*generate)(float, int, int) = real_symbol("GenMeshSphere");
    int before = live_meshes;
    Mesh mesh = generate(radius, rings, slices);
    // Native generation can call the interposed UploadMesh itself.
    if (mesh.vaoId && live_meshes == before) ++live_meshes;
    return mesh;
}

void UnloadMesh(Mesh mesh) {
    if (mesh.vaoId) --live_meshes;
    else if (mesh.vertices) ++partial_mesh_unloads;
    if (live_meshes < 0 || drawing) ++ownership_errors;
    void (*unload)(Mesh) = real_symbol("UnloadMesh");
    unload(mesh);
}

bool IsKeyPressed(int key) {
    if (mode("orientation")) return frames == 3 && key == KEY_TAB;
    if (batch_mode()) {
        if (mode("batch-key") && key == KEY_K) {
            if (batch_control_frame < 0 && completed_samples >= 2) batch_control_frame = frames;
            return frames == batch_control_frame;
        }
        return false;
    }
    if (mode("cycle")) return frames == 2 && key == KEY_C;
    if (mode("cancel")) return (frames == 1 && key == KEY_O) || (frames == 3 && key == KEY_ESCAPE);
    if (mode("drop") || mode("bad-drop") || mode("drop-trace")) return false;
    return (frames == 1 && key == KEY_O && !mode("button")) ||
           (frames == 2 && key == KEY_V) ||
           (frames == 3 && key == KEY_ENTER && mode("open")) ||
           (frames == 4 && key == KEY_ESCAPE && mode("invalid"));
}

bool IsKeyDown(int key) {
    return frames == 2 && key == KEY_LEFT_CONTROL &&
           (mode("open") || mode("button") || mode("invalid") || mode("double"));
}

bool IsMouseButtonPressed(int button) {
    if (mode("batch-click") && button == MOUSE_BUTTON_LEFT) {
        if (batch_control_frame < 0 && completed_samples >= 2) batch_control_frame = frames;
        return frames == batch_control_frame;
    }
    return button == MOUSE_BUTTON_LEFT &&
        ((mode("button") && (frames == 1 || frames == 3)) ||
         (mode("double") && (frames == 3 || frames == 4)));
}

bool IsMouseButtonDown(int button) { (void)button; return false; }

Vector2 GetMousePosition(void) {
    int (*width)(void) = real_symbol("GetScreenWidth");
    int (*height)(void) = real_symbol("GetScreenHeight");
    if (mode("batch-click")) return (Vector2){(float)(width() - 24), 46.0f};
    if (mode("double")) return (Vector2){60.0f, 150.0f};
    Vector2 point = {(float)(width() - 100), frames == 1 ? 70.0f : (float)(height() - 45)};
    return point;
}

const char *GetClipboardText(void) { return getenv("BAJO_TEST_SCENE_PATH"); }

bool IsFileDropped(void) { return frames == 2 && (mode("drop") || mode("bad-drop") || mode("drop-trace")); }

FilePathList LoadDroppedFiles(void) {
    static char *paths[1];
    paths[0] = getenv("BAJO_TEST_SCENE_PATH");
    FilePathList files = {1, paths};
    return files;
}

void UnloadDroppedFiles(FilePathList files) { (void)files; ++dropped_unloaded; }

bool WindowShouldClose(void) {
    // Bound a broken scene-switch loop as well as a normal rendering loop.
    if (mode("orientation")) return orientation_trace || frames >= 600;
    if (batch_mode()) return (frames >= 8 && total_samples > 0 && completed_samples >= total_samples) || frames >= 600;
    if (mode("drop-trace")) return (frames >= 8 && custom_trace) || frames >= 600;
    return frames >= (mode("cycle") ? 6 : 8) || scenes >= 4;
}

Material LoadMaterialDefault(void) {
    ++scenes;
    Material (*load)(void) = real_symbol("LoadMaterialDefault");
    return load();
}

void DrawTextEx(Font font, const char *text, Vector2 position, float size, float spacing, Color color) {
    const char *backend = strstr(text, "| Bajo ");
    const char *stats = backend ? strchr(backend + 7, '|') : NULL;
    int samples, total, batches;
    if (stats && sscanf(stats + 1, " %d/%d spp | %d batch", &samples, &total, &batches) == 3) {
        if (samples < completed_samples && total == total_samples) ++sample_resets;
        completed_samples = samples;
        total_samples = total;
        completed_batches = batches;
    }
    if (strstr(text, "Batch: 4 spp")) batch_setting_four = 1;
    if (strcmp(text, "Open PBRT scene") == 0) ++browser_frames;
    if (strncmp(text, "PBRT: ", 6) == 0) {
        custom_scene = 1;
        if (strstr(text, "| TRACE |")) custom_trace = 1;
    }
    if (strstr(text, "missing-scene-asset") || strstr(text, "Scene load failed:")) load_error = 1;
    void (*draw)(Font, const char *, Vector2, float, float, Color) = real_symbol("DrawTextEx");
    draw(font, text, position, size, spacing, color);
}

void EndDrawing(void) {
    if (!drawing || scene_3d || scissor) ++ownership_errors;
    drawing = 0;
    void (*end)(void) = real_symbol("EndDrawing");
    end();
    if (mode("orientation")) {
        const char *image = NULL;
        if (frames == 2) image = "raster.png";
        if (completed_samples > 0 && completed_samples >= total_samples && !orientation_trace) {
            image = "trace.png";
            orientation_trace = 1;
        }
        if (image) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s", getenv("BAJO_TEST_SCENE_PATH"), image);
            void (*capture)(const char *) = real_symbol("TakeScreenshot");
            capture(path);
        }
    }
    ++frames;
}

void CloseWindow(void) {
    if (drawing || scene_3d || scissor || live_meshes) ++ownership_errors;
    void (*close)(void) = real_symbol("CloseWindow");
    close();
    printf("SCENE_INPUT_TEST frames=%d scenes=%d\n", frames, scenes);
    printf("BATCH_INPUT_TEST samples=%d total=%d batches=%d resets=%d control=%d setting_four=%d\n",
           completed_samples, total_samples, completed_batches, sample_resets, batch_control_frame >= 0, batch_setting_four);
    printf("FILE_INPUT_TEST browser=%d custom=%d error=%d drop_unloaded=%d trace=%d\n",
           browser_frames, custom_scene, load_error, dropped_unloaded, custom_trace);
    printf("OWNERSHIP_TEST meshes=%d drawing=%d scene=%d scissor=%d errors=%d partial_unloads=%d\n",
           live_meshes, drawing, scene_3d, scissor, ownership_errors, partial_mesh_unloads);
}
