// Real rendering with one C press, latched until EndDrawing polls input.
#include <raylib.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static int frames;
static int scenes;

static void *real_symbol(const char *name) {
    static void *library;
    if (!library) library = dlopen(getenv("BAJO_TEST_RAYLIB"), RTLD_NOW);
    void *symbol = library ? dlsym(library, name) : NULL;
    if (!symbol) abort();
    return symbol;
}

bool IsKeyPressed(int key) { return frames == 2 && key == KEY_C; }
bool IsKeyDown(int key) { (void)key; return false; }

bool WindowShouldClose(void) {
    // Bound a broken switch loop as well as a normal rendering loop.
    return frames >= 6 || scenes >= 4;
}

Material LoadMaterialDefault(void) {
    ++scenes;
    Material (*load)(void) = real_symbol("LoadMaterialDefault");
    return load();
}

void EndDrawing(void) {
    void (*end)(void) = real_symbol("EndDrawing");
    end();
    ++frames;
}

void CloseWindow(void) {
    void (*close)(void) = real_symbol("CloseWindow");
    close();
    printf("SCENE_INPUT_TEST frames=%d scenes=%d\n", frames, scenes);
}
