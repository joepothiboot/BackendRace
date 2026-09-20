// ---------------------------------------------------------------------------
// tests/test_tiling.cpp
//
// Role in BackendRace: validates the 32x32 tiled memory layout itself and the
// halo/tile-boundary handling that every backend depends on.
//
// These are the tests that catch the classic tiling bug: a simulation that is
// perfectly correct *inside* a tile and silently wrong on the seams. They check
// the addressing algebra directly, then drive material across both a vertical
// seam (y = 31 -> 32) and a horizontal seam (x = 31 -> 32) through the scalar,
// AVX2 and toy-ISA paths and require identical results.
// ---------------------------------------------------------------------------
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <set>
#include <vector>

#include "backends/toy_isa/isa.hpp"
#include "core/backend.hpp"
#include "core/grid.hpp"
#include "core/passes.hpp"

namespace {

std::size_t firstDifference(const std::uint8_t* a, const std::uint8_t* b,
                            std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return i;
    }
    return n;
}

}  // namespace

// ---------------------------------------------------------------------------
// Addressing algebra
// ---------------------------------------------------------------------------

TEST_CASE("tiledIndex matches the documented formula", "[tiling][layout]") {
    const int tilesX = 4;                 // 128-wide grid

    REQUIRE(br::tiledIndex(0, 0, tilesX)   == 0u);
    REQUIRE(br::tiledIndex(31, 0, tilesX)  == 31u);
    REQUIRE(br::tiledIndex(0, 1, tilesX)   == 32u);
    REQUIRE(br::tiledIndex(31, 31, tilesX) == 1023u);

    // First cell of tile (1,0) is the start of the second 1024-byte block.
    REQUIRE(br::tiledIndex(32, 0, tilesX)  == 1024u);
    // First cell of tile (0,1) is tilesX tiles further on.
    REQUIRE(br::tiledIndex(0, 32, tilesX)  == static_cast<std::size_t>(tilesX) * 1024u);

    // Explicit recomputation of the formula for a sample of coordinates.
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 128; ++x) {
            const std::size_t want =
                (static_cast<std::size_t>((y >> 5) * tilesX + (x >> 5)) << 10) +
                (static_cast<std::size_t>(y & 31) << 5) +
                static_cast<std::size_t>(x & 31);
            REQUIRE(br::tiledIndex(x, y, tilesX) == want);
        }
    }
}

TEST_CASE("tiledIndex is a bijection onto the buffer", "[tiling][layout]") {
    br::Grid g(128, 64);
    std::set<std::size_t> seen;
    for (int y = 0; y < g.height(); ++y) {
        for (int x = 0; x < g.width(); ++x) {
            const std::size_t i = br::tiledIndex(x, y, g.tilesX());
            REQUIRE(i < g.bytes());
            REQUIRE(seen.insert(i).second);      // never collides
        }
    }
    REQUIRE(seen.size() == g.bytes());           // and covers everything
}

TEST_CASE("a tile row is 32 contiguous bytes", "[tiling][layout]") {
    const int tilesX = 4;
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 127; ++x) {
            if ((x & 31) == 31) continue;        // seam handled below
            REQUIRE(br::tiledIndex(x + 1, y, tilesX) ==
                    br::tiledIndex(x, y, tilesX) + 1u);
        }
    }
    // Every tile row base is 32-byte aligned -> aligned AVX2 loads are legal.
    for (int ty = 0; ty < 2; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            for (int ly = 0; ly < 32; ++ly) {
                const std::size_t base =
                    br::tiledIndex(tx * 32, ty * 32 + ly, tilesX);
                REQUIRE((base % 32u) == 0u);
            }
        }
    }
}

TEST_CASE("seam strides are exactly what the backends assume",
          "[tiling][layout]") {
    const int tilesX = 4;
    const std::size_t tileRowStride = static_cast<std::size_t>(tilesX) * 1024u;

    // Vertical seam: bottom row of a tile -> top row of the tile below.
    for (int x = 0; x < 32; ++x) {
        const std::size_t a = br::tiledIndex(x, 31, tilesX);
        const std::size_t b = br::tiledIndex(x, 32, tilesX);
        REQUIRE(b == a + tileRowStride - 31u * 32u);
        // Which is exactly "tile base + tilesX*1024" as used in verticalAvx2.
        REQUIRE(b == br::tiledIndex(x, 0, tilesX) + tileRowStride);
    }

    // Horizontal seam: last column of a tile -> first column of the next.
    for (int y = 0; y < 32; ++y) {
        const std::size_t a = br::tiledIndex(31, y, tilesX);
        const std::size_t b = br::tiledIndex(32, y, tilesX);
        REQUIRE(b == a + 1024u - 31u);
    }
}

TEST_CASE("sampleCell returns Wall outside the grid", "[tiling][halo]") {
    br::Grid g(64, 64);
    g.clear();
    g.set(1, 1, br::Cell::Sand);

    const std::uint8_t* s = g.src();
    const int W = g.width(), H = g.height(), TX = g.tilesX();

    REQUIRE(br::sampleCell(s, 1, 1, W, H, TX)   == br::cid(br::Cell::Sand));
    REQUIRE(br::sampleCell(s, -1, 1, W, H, TX)  == br::cid(br::Cell::Wall));
    REQUIRE(br::sampleCell(s, 1, -1, W, H, TX)  == br::cid(br::Cell::Wall));
    REQUIRE(br::sampleCell(s, W, 1, W, H, TX)   == br::cid(br::Cell::Wall));
    REQUIRE(br::sampleCell(s, 1, H, W, H, TX)   == br::cid(br::Cell::Wall));
    REQUIRE(br::sampleCell(s, -99, -99, W, H, TX) == br::cid(br::Cell::Wall));
}

// ---------------------------------------------------------------------------
// Halo behaviour across seams
// ---------------------------------------------------------------------------

TEST_CASE("material crosses a vertical tile seam", "[tiling][halo][vertical]") {
    br::Grid g(64, 64);
    g.clear();
    // Row 31 is the last row of the upper tile; row 32 is the first row of the
    // tile below. Parity 1 makes row 31 the UPPER member of the pair (31,32).
    g.set(10, 31, br::Cell::Sand);
    REQUIRE(g.at(10, 32) == br::Cell::Empty);

    br::passes::verticalScalar(g.src(), g.dst(), g, /*parity=*/1);
    g.swapBuffers();

    REQUIRE(g.at(10, 31) == br::Cell::Empty);
    REQUIRE(g.at(10, 32) == br::Cell::Sand);     // it crossed the seam
}

TEST_CASE("material crosses a horizontal tile seam",
          "[tiling][halo][horizontal]") {
    br::Grid g(64, 64);
    g.clear();
    // Sand at x = 31 (last column of tile 0) resting on a wall, with an empty
    // cell at x = 32 (first column of tile 1) that also has a wall beneath it.
    const int y = 20;
    for (int x = 28; x < 36; ++x) g.set(x, y + 1, br::Cell::Wall);
    g.set(31, y, br::Cell::Sand);
    REQUIRE(g.at(32, y) == br::Cell::Empty);

    // Parity 1 makes x = 31 the LEFT member of the pair (31, 32).
    br::passes::horizontalScalar(g.src(), g.dst(), g, /*parity=*/1);
    g.swapBuffers();

    REQUIRE(g.at(31, y) == br::Cell::Empty);
    REQUIRE(g.at(32, y) == br::Cell::Sand);
}

TEST_CASE("AVX2 halo handling matches the scalar halo handling",
          "[tiling][halo][avx2]") {
    auto avx2 = br::createAvx2Backend();
    if (!avx2->available()) {
        WARN("AVX2 unavailable - seam comparison skipped");
        SUCCEED("skipped");
        return;
    }

    // A pattern engineered to be busy on EVERY seam: material on every row and
    // column that is a multiple of 31, 32 or 33.
    br::Grid a(128, 128);
    br::Grid b(128, 128);
    for (br::Grid* g : {&a, &b}) {
        g->clear();
        for (int y = 1; y < g->height() - 1; ++y) {
            for (int x = 1; x < g->width() - 1; ++x) {
                const bool nearSeamX = ((x % 32) <= 1) || ((x % 32) >= 30);
                const bool nearSeamY = ((y % 32) <= 1) || ((y % 32) >= 30);
                if (!nearSeamX && !nearSeamY) continue;
                const std::uint32_t h = br::hash32(
                    static_cast<std::uint32_t>(x),
                    static_cast<std::uint32_t>(y), 0x5A5Au);
                switch (h % 5u) {
                    case 0: g->set(x, y, br::Cell::Sand);  break;
                    case 1: g->set(x, y, br::Cell::Water); break;
                    case 2: g->set(x, y, br::Cell::Fire);  break;
                    case 3: g->set(x, y, br::Cell::Wall);  break;
                    default: break;
                }
            }
        }
    }
    REQUIRE(std::memcmp(a.src(), b.src(), a.bytes()) == 0);

    auto naive = br::createNaiveBackend();
    for (int frame = 0; frame < 40; ++frame) {
        naive->step(a);
        avx2->step(b);
        const std::size_t diff = firstDifference(a.src(), b.src(), a.bytes());
        INFO("seam divergence at frame " << frame << ", byte " << diff
             << " -> tile " << (diff >> 10)
             << " localY " << ((diff >> 5) & 31)
             << " localX " << (diff & 31));
        REQUIRE(diff == a.bytes());
    }
}

TEST_CASE("toy-ISA halo handling matches the scalar vertical pass",
          "[tiling][halo][toyisa]") {
    // 64x96 exercises all three compiled tile groups: top (no tile above),
    // interior (both neighbours), bottom (no tile below).
    br::Grid g(64, 96);
    g.clear();
    for (int y = 1; y < g.height() - 1; ++y) {
        for (int x = 1; x < g.width() - 1; ++x) {
            if ((y % 32) >= 30 || (y % 32) <= 1 || y % 7 == 0) {
                g.set(x, y, ((x + y) % 3 == 0) ? br::Cell::Sand
                                               : br::Cell::Water);
            }
        }
    }

    std::vector<std::uint8_t> reference(g.bytes(), 0u);
    for (int parity = 0; parity < 2; ++parity) {
        std::memset(g.dst(), 0xAB, g.bytes());
        br::passes::verticalScalar(g.src(), reference.data(), g, parity);

        const br::toy::Program p =
            br::toy::compileVerticalPass(g.width(), g.height(), parity);
        br::toy::MemoryImage mem{g.src(), g.dst(), g.bytes()};
        const br::toy::RunResult r = br::toy::execute(p, mem);

        INFO("parity " << parity << ": " << r.error);
        REQUIRE(r.ok);
        REQUIRE(firstDifference(reference.data(), g.dst(), g.bytes()) == g.bytes());
    }
}

TEST_CASE("the top and bottom tile rows are handled without leaking",
          "[tiling][halo][boundary]") {
    br::Grid g(64, 64);
    g.clear();
    // Fill the very first and very last interior rows.
    for (int x = 1; x < g.width() - 1; ++x) {
        g.set(x, 1, br::Cell::Water);
        g.set(x, g.height() - 2, br::Cell::Sand);
    }

    auto naive = br::createNaiveBackend();
    for (int i = 0; i < 20; ++i) naive->step(g);

    // The Wall ring is untouched: the top tile row never reads a tile "above"
    // and the bottom tile row never reads a tile "below".
    for (int x = 0; x < g.width(); ++x) {
        REQUIRE(g.at(x, 0) == br::Cell::Wall);
        REQUIRE(g.at(x, g.height() - 1) == br::Cell::Wall);
    }
    for (int y = 0; y < g.height(); ++y) {
        REQUIRE(g.at(0, y) == br::Cell::Wall);
        REQUIRE(g.at(g.width() - 1, y) == br::Cell::Wall);
    }
}

TEST_CASE("tiling does not change the physics versus a naive mental model",
          "[tiling][semantics]") {
    // Drop a column of sand that must fall through three tile rows.
    br::Grid g(64, 96);
    g.clear();
    for (int y = 2; y < 10; ++y) g.set(16, y, br::Cell::Sand);
    const std::size_t sand = 8u;

    auto naive = br::createNaiveBackend();
    naive->stepMany(g, 500);

    std::size_t found = 0;
    int lowest = -1;
    for (int y = 0; y < g.height(); ++y) {
        for (int x = 0; x < g.width(); ++x) {
            if (g.at(x, y) == br::Cell::Sand) { ++found; lowest = y; }
        }
    }
    REQUIRE(found == sand);
    REQUIRE(lowest == g.height() - 2);          // it reached the floor
}