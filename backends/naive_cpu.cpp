// ---------------------------------------------------------------------------
// backends/naive_cpu.cpp
//
// Role in BackendRace: the BASELINE.  Straightforward nested loops, one cell at
// a time, every neighbour fetched through the generic `sampleCell()` helper
// (which recomputes the full tiled address -- two shifts, a multiply and an add
// -- for each of the up-to-six neighbour reads).  Nothing is cached, nothing is
// vectorised, nothing is blocked.  This is deliberately the "obvious" version:
// it is the reference for correctness AND the denominator of every speedup
// number in the benchmark table.
//
// This file also provides the shared scalar passes declared in core/passes.hpp,
// which the other backends fall back to when a pass is not offloaded.
// ---------------------------------------------------------------------------
#include <chrono>
#include <memory>

#include "core/backend.hpp"
#include "core/grid.hpp"
#include "core/passes.hpp"

namespace br::passes {

void verticalScalar(const std::uint8_t* s, std::uint8_t* d,
                    const Grid& g, int parity) {
    const int W = g.width(), H = g.height(), TX = g.tilesX();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::uint8_t c = sampleCell(s, x, y, W, H, TX);
            std::uint8_t out = c;

            if ((y & 1) == parity) {
                // This row is the UPPER half of its pair.
                const std::uint8_t b = sampleCell(s, x, y + 1, W, H, TX);
                if (swapVertical(c, b)) out = b;
            } else {
                // This row is the LOWER half of its pair.
                const std::uint8_t a = sampleCell(s, x, y - 1, W, H, TX);
                if (swapVertical(a, c)) out = a;
            }
            d[tiledIndex(x, y, TX)] = out;
        }
    }
}

void horizontalScalar(const std::uint8_t* s, std::uint8_t* d,
                      const Grid& g, int parity) {
    const int W = g.width(), H = g.height(), TX = g.tilesX();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::uint8_t c = sampleCell(s, x, y, W, H, TX);
            std::uint8_t out = c;

            if ((x & 1) == parity) {
                // LEFT element of the pair (x, x+1).
                const std::uint8_t b  = sampleCell(s, x + 1, y,     W, H, TX);
                const std::uint8_t ba = sampleCell(s, x,     y + 1, W, H, TX);
                const std::uint8_t bb = sampleCell(s, x + 1, y + 1, W, H, TX);
                const std::uint8_t aa = sampleCell(s, x,     y - 1, W, H, TX);
                const std::uint8_t ab = sampleCell(s, x + 1, y - 1, W, H, TX);
                if (swapHorizontal(c, b, ba, bb, aa, ab)) out = b;
            } else {
                // RIGHT element of the pair (x-1, x).
                const std::uint8_t a  = sampleCell(s, x - 1, y,     W, H, TX);
                const std::uint8_t ba = sampleCell(s, x - 1, y + 1, W, H, TX);
                const std::uint8_t bb = sampleCell(s, x,     y + 1, W, H, TX);
                const std::uint8_t aa = sampleCell(s, x - 1, y - 1, W, H, TX);
                const std::uint8_t ab = sampleCell(s, x,     y - 1, W, H, TX);
                if (swapHorizontal(a, c, ba, bb, aa, ab)) out = a;
            }
            d[tiledIndex(x, y, TX)] = out;
        }
    }
}

void decayScalar(const std::uint8_t* s, std::uint8_t* d,
                 const Grid& g, std::uint32_t frame) {
    const int W = g.width(), H = g.height(), TX = g.tilesX();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::uint8_t c = sampleCell(s, x, y, W, H, TX);
            const std::uint8_t l = sampleCell(s, x - 1, y, W, H, TX);
            const std::uint8_t r = sampleCell(s, x + 1, y, W, H, TX);
            const std::uint8_t u = sampleCell(s, x, y - 1, W, H, TX);
            const std::uint8_t dn= sampleCell(s, x, y + 1, W, H, TX);

            d[tiledIndex(x, y, TX)] =
                decayFire(c, l, r, u, dn,
                          randBit(x, y, frame, kSaltA),
                          randBit(x, y, frame, kSaltB),
                          randBit(x, y, frame, kSaltC));
        }
    }
}

}  // namespace br::passes

namespace br {

namespace {

class NaiveBackend final : public IBackend {
public:
    const char* name() const noexcept override { return "Naive CPU"; }
    const char* description() const noexcept override {
        return "scalar nested loops, generic tiled addressing, no blocking";
    }
    bool available() const noexcept override { return true; }

    void step(Grid& g) override {
        const auto t0 = std::chrono::steady_clock::now();

        const int p1 = static_cast<int>( g.frame()       & 1ull);
        const int p2 = static_cast<int>((g.frame() >> 1) & 1ull);
        const auto f = static_cast<std::uint32_t>(g.frame());

        passes::verticalScalar  (g.src(), g.dst(), g, p1); g.swapBuffers();
        passes::horizontalScalar(g.src(), g.dst(), g, p2); g.swapBuffers();
        passes::decayScalar     (g.src(), g.dst(), g, f ); g.swapBuffers();
        g.advanceFrame();

        const auto t1 = std::chrono::steady_clock::now();
        stats_.lastStepMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++stats_.stepsRun;
    }
};

}  // namespace

std::unique_ptr<IBackend> createNaiveBackend() {
    return std::make_unique<NaiveBackend>();
}

}  // namespace br