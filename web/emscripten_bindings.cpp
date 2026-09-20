// ---------------------------------------------------------------------------
// web/emscripten_bindings.cpp
//
// Role in BackendRace: the WASM <-> JS boundary.  It owns the singleton
// simulation state for the browser build and exports a tiny C ABI that
// web/main.js drives through Module.ccall / Module.cwrap:
//
//     brInit(w, h)              allocate the grid and the RGBA frame buffer
//     brStep()                  advance one frame with the active backend
//     brRender()                re-render without stepping (paused mode)
//     brAddCell(x, y, mat)      set a single cell
//     brBrush(x, y, r, mat)     paint a disc of cells
//     brSetBackend(id)          live backend switch (returns 0 if unavailable)
//     brBackendName(id)         UTF-8 name, or the active one when id < 0
//     brBackendDesc(id)         human-readable description
//     brBackendAvailable(id)    1/0
//     brPixels()                pointer into WASM linear memory (RGBA8, W*H*4)
//     brWidth() / brHeight()    grid geometry
//     brLastStepMs()            wall-clock of the most recent step
//     brClear() / brSeed(seed)  world management
//
// Everything is EMSCRIPTEN_KEEPALIVE + extern "C" so the names survive
// -O3 dead-code elimination and appear in EXPORTED_FUNCTIONS.  The file also
// compiles (as a no-op-free plain C++ TU) outside Emscripten so that IDEs and
// native CI can still syntax-check it.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "core/grid.hpp"

#if defined(__EMSCRIPTEN__)
#  include <emscripten/emscripten.h>
#else
#  define EMSCRIPTEN_KEEPALIVE
#endif

namespace {

struct App {
    std::unique_ptr<br::Grid>                   grid;
    std::unique_ptr<br::IBackend>               active;
    std::unique_ptr<br::IBackend>               probes[4];
    std::vector<std::uint8_t>                   pixels;
    std::string                                 nameScratch;
    std::string                                 descScratch;
    int                                         backendId = 0;
    double                                      lastStepMs = 0.0;
};

App& app() {
    static App a;
    return a;
}

void ensureProbes() {
    App& a = app();
    for (int i = 0; i < 4; ++i) {
        if (!a.probes[i]) {
            a.probes[i] = br::createBackend(static_cast<br::BackendId>(i));
        }
    }
}

void renderInto(App& a) {
    if (!a.grid) return;
    a.grid->renderRGBA(a.pixels.data());
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void brInit(int w, int h) {
    App& a = app();
    a.grid = std::make_unique<br::Grid>(w, h);
    a.pixels.assign(static_cast<std::size_t>(a.grid->width()) *
                    static_cast<std::size_t>(a.grid->height()) * 4u, 0u);
    ensureProbes();
    a.backendId = 0;
    a.active = br::createBackend(br::BackendId::Naive);
    renderInto(a);
}

EMSCRIPTEN_KEEPALIVE
void brStep() {
    App& a = app();
    if (!a.grid || !a.active) return;
    a.active->step(*a.grid);
    a.lastStepMs = a.active->stats().lastStepMs;
    renderInto(a);
}

EMSCRIPTEN_KEEPALIVE
void brRender() {
    renderInto(app());
}

EMSCRIPTEN_KEEPALIVE
void brAddCell(int x, int y, int material) {
    App& a = app();
    if (!a.grid) return;
    if (material < 0 || material >= br::kCellKinds) return;
    a.grid->set(x, y, static_cast<br::Cell>(material));
}

EMSCRIPTEN_KEEPALIVE
void brBrush(int x, int y, int radius, int material) {
    App& a = app();
    if (!a.grid) return;
    if (material < 0 || material >= br::kCellKinds) return;
    a.grid->paintDisc(x, y, radius, static_cast<br::Cell>(material));
}

EMSCRIPTEN_KEEPALIVE
int brSetBackend(int id) {
    App& a = app();
    if (id < 0 || id >= static_cast<int>(br::BackendId::Count)) return 0;
    ensureProbes();
    if (!a.probes[id] || !a.probes[id]->available()) return 0;
    a.active = br::createBackend(static_cast<br::BackendId>(id));
    a.backendId = id;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
const char* brBackendName(int id) {
    App& a = app();
    ensureProbes();
    const int q = (id < 0) ? a.backendId : id;
    if (q < 0 || q >= 4 || !a.probes[q]) return "?";
    a.nameScratch = a.probes[q]->name();
    return a.nameScratch.c_str();
}

EMSCRIPTEN_KEEPALIVE
const char* brBackendDesc(int id) {
    App& a = app();
    ensureProbes();
    const int q = (id < 0) ? a.backendId : id;
    if (q < 0 || q >= 4 || !a.probes[q]) return "";
    a.descScratch = a.probes[q]->description();
    return a.descScratch.c_str();
}

EMSCRIPTEN_KEEPALIVE
int brBackendAvailable(int id) {
    ensureProbes();
    App& a = app();
    if (id < 0 || id >= 4 || !a.probes[id]) return 0;
    return a.probes[id]->available() ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
std::uint8_t* brPixels() {
    return app().pixels.data();
}

EMSCRIPTEN_KEEPALIVE
int brWidth() {
    App& a = app();
    return a.grid ? a.grid->width() : 0;
}

EMSCRIPTEN_KEEPALIVE
int brHeight() {
    App& a = app();
    return a.grid ? a.grid->height() : 0;
}

EMSCRIPTEN_KEEPALIVE
double brLastStepMs() {
    return app().lastStepMs;
}

EMSCRIPTEN_KEEPALIVE
void brClear() {
    App& a = app();
    if (!a.grid) return;
    a.grid->clear();
    renderInto(a);
}

EMSCRIPTEN_KEEPALIVE
void brSeed(unsigned int seed) {
    App& a = app();
    if (!a.grid) return;
    a.grid->clear();
    a.grid->seedRandom(seed, 0.18f, 0.12f);
    renderInto(a);
}

}  // extern "C"

#if !defined(__EMSCRIPTEN__)
// Keeps the TU linkable in native builds/tests without providing a main().
namespace br_web_detail { inline void referenceExports() { (void)brWidth; } }
#endif