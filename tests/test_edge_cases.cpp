// ---------------------------------------------------------------------------
// tests/test_edge_cases.cpp
//
// Role in BackendRace: the degenerate-input suite. Grid boundaries, a totally
// empty grid, a totally full grid, the smallest legal (single-tile) grid,
// out-of-range brush coordinates, and the qualitative behaviours a reviewer
// will eyeball in the demo (sand falls and rests, fire rises and burns out,
// water spreads). Every case is run against naive AND, when present, AVX2, so
// that boundary handling is validated in both code paths.
// ---------------------------------------------------------------------------
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <vector>

#include "core/backend.hpp"
#include "core/grid.hpp"
#include "core/passes.hpp"

namespace {

std::size_t countCells(const br::Grid& g, br::Cell c) {
    std::size_t n = 0;
    for (int y = 0; y < g.height(); ++y) {
        for (int x = 0; x < g.width(); ++x) {
            if (g.at(x, y) == c) ++n;
        }
    }
    return n;
}

bool bordersAreWall(const br::Grid& g) {
    for (int x = 0; x < g.width(); ++x) {
        if (g.at(x, 0) != br::Cell::Wall) return false;
        if (g.at(x, g.height() - 1) != br::Cell::Wall) return false;
    }
    for (int y = 0; y < g.height(); ++y) {
        if (g.at(0, y) != br::Cell::Wall) return false;
        if (g.at(g.width() - 1, y) != br::Cell::Wall) return false;
    }
    return true;
}

// Runs the body once per available backend.
template <typename F>
void forEachBackend(F&& f) {
    for (int id = 0; id < static_cast<int>(br::BackendId::Count); ++id) {
        auto be = br::createBackend(static_cast<br::BackendId>(id));
        if (!be->available()) continue;
        f(*be);
    }
}

}  // namespace

// ---------------------------------------------------------------------------

TEST_CASE("grid geometry is rounded up to whole tiles", "[edge][layout]") {
    br::Grid g(50, 33);
    REQUIRE(g.width()  == 64);
    REQUIRE(g.height() == 64);
    REQUIRE(g.tilesX() == 2);
    REQUIRE(g.tilesY() == 2);
    REQUIRE(g.bytes()  == 2u * 2u * 1024u);

    br::Grid tiny(1, 1);
    REQUIRE(tiny.width()  == 32);
    REQUIRE(tiny.height() == 32);
    REQUIRE(tiny.tilesX() == 1);
    REQUIRE(tiny.tilesY() == 1);
}

TEST_CASE("an empty grid is a fixed point", "[edge][empty]") {
    forEachBackend([](br::IBackend& be) {
        INFO("backend: " << be.name());
        br::Grid g(64, 64);
        g.clear();
        const std::uint64_t before = g.checksum();
        be.stepMany(g, 25);
        REQUIRE(g.checksum() == before);
        REQUIRE(countCells(g, br::Cell::Empty) ==
                g.cells() - countCells(g, br::Cell::Wall));
        REQUIRE(bordersAreWall(g));
    });
}

TEST_CASE("a completely packed grid cannot move", "[edge][full]") {
    forEachBackend([](br::IBackend& be) {
        INFO("backend: " << be.name());
        br::Grid g(64, 64);
        g.clear();
        for (int y = 1; y < g.height() - 1; ++y) {
            for (int x = 1; x < g.width() - 1; ++x) {
                g.set(x, y, br::Cell::Sand);
            }
        }
        const std::size_t sand = countCells(g, br::Cell::Sand);
        const std::uint64_t before = g.checksum();

        be.stepMany(g, 20);

        REQUIRE(countCells(g, br::Cell::Sand) == sand);
        REQUIRE(g.checksum() == before);     // nowhere to go: fully static
        REQUIRE(bordersAreWall(g));
    });
}

TEST_CASE("a grid of solid wall is inert", "[edge][full]") {
    br::Grid g(32, 32);
    for (int y = 0; y < g.height(); ++y) {
        for (int x = 0; x < g.width(); ++x) g.set(x, y, br::Cell::Wall);
    }
    const std::uint64_t before = g.checksum();
    auto be = br::createNaiveBackend();
    be->stepMany(g, 10);
    REQUIRE(g.checksum() == before);
}

TEST_CASE("nothing ever escapes through the boundary", "[edge][boundary]") {
    forEachBackend([](br::IBackend& be) {
        INFO("backend: " << be.name());
        br::Grid g(64, 64);
        g.clear();
        // Press material right up against all four inner edges.
        for (int x = 1; x < g.width() - 1; ++x) {
            g.set(x, 1, br::Cell::Water);
            g.set(x, g.height() - 2, br::Cell::Sand);
        }
        for (int y = 1; y < g.height() - 1; ++y) {
            g.set(1, y, br::Cell::Sand);
            g.set(g.width() - 2, y, br::Cell::Water);
        }
        const std::size_t total =
            countCells(g, br::Cell::Sand) + countCells(g, br::Cell::Water);

        be.stepMany(g, 60);

        REQUIRE(bordersAreWall(g));
        REQUIRE(countCells(g, br::Cell::Sand) +
                countCells(g, br::Cell::Water) == total);
    });
}

TEST_CASE("the smallest legal grid (one tile) simulates correctly",
          "[edge][single-tile]") {
    br::Grid naiveGrid(32, 32);
    br::Grid otherGrid(32, 32);
    naiveGrid.clear();
    otherGrid.clear();
    naiveGrid.paintDisc(16, 6, 4, br::Cell::Sand);
    otherGrid.paintDisc(16, 6, 4, br::Cell::Sand);

    auto naive = br::createNaiveBackend();
    naive->stepMany(naiveGrid, 40);

    for (int id = 1; id < static_cast<int>(br::BackendId::Count); ++id) {
        auto be = br::createBackend(static_cast<br::BackendId>(id));
        if (!be->available()) continue;
        INFO("backend: " << be->name());

        br::Grid g(32, 32);
        g.clear();
        g.paintDisc(16, 6, 4, br::Cell::Sand);
        be->stepMany(g, 40);
        REQUIRE(g.checksum() == naiveGrid.checksum());
    }
}

TEST_CASE("a single sand grain falls and comes to rest", "[edge][behaviour]") {
    br::Grid g(64, 64);
    g.clear();
    g.set(20, 2, br::Cell::Sand);

    auto be = br::createNaiveBackend();
    be->stepMany(g, 400);

    REQUIRE(countCells(g, br::Cell::Sand) == 1u);
    // It cannot spread sideways along the floor (the cell under the target is
    // Wall), so it must be parked directly on the bottom wall.
    REQUIRE(g.at(20, g.height() - 2) == br::Cell::Sand);
}

TEST_CASE("fire rises and eventually burns out", "[edge][behaviour]") {
    br::Grid g(64, 64);
    g.clear();
    g.paintDisc(32, 50, 4, br::Cell::Fire);
    REQUIRE(countCells(g, br::Cell::Fire) > 0u);

    auto be = br::createNaiveBackend();
    be->stepMany(g, 600);

    REQUIRE(countCells(g, br::Cell::Fire) == 0u);
    REQUIRE(bordersAreWall(g));
}

TEST_CASE("fire is extinguished on contact with water", "[edge][behaviour]") {
    br::Grid g(64, 64);
    g.clear();
    for (int x = 10; x < 54; ++x) {
        g.set(x, 40, br::Cell::Water);
        g.set(x, 41, br::Cell::Fire);
    }
    auto be = br::createNaiveBackend();
    be->stepMany(g, 3);
    REQUIRE(countCells(g, br::Cell::Fire) == 0u);
}

TEST_CASE("water spreads into a flat puddle", "[edge][behaviour]") {
    br::Grid g(64, 64);
    g.clear();
    for (int y = 40; y < 60; ++y) {
        for (int x = 30; x < 34; ++x) g.set(x, y, br::Cell::Water);
    }
    const std::size_t water = countCells(g, br::Cell::Water);

    auto be = br::createNaiveBackend();
    be->stepMany(g, 300);

    REQUIRE(countCells(g, br::Cell::Water) == water);

    int minX = g.width(), maxX = -1;
    for (int y = 1; y < g.height() - 1; ++y) {
        for (int x = 1; x < g.width() - 1; ++x) {
            if (g.at(x, y) == br::Cell::Water) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
            }
        }
    }
    INFO("water spans x=[" << minX << "," << maxX << "]");
    REQUIRE(maxX - minX > 8);      // it must have flowed outward
}

TEST_CASE("out-of-range edits are ignored, not crashes", "[edge][api]") {
    br::Grid g(64, 64);
    g.clear();
    const std::uint64_t before = g.checksum();

    g.set(-1, 10, br::Cell::Sand);
    g.set(10, -1, br::Cell::Sand);
    g.set(g.width(), 10, br::Cell::Sand);
    g.set(10, g.height(), br::Cell::Sand);
    g.set(1000000, 1000000, br::Cell::Sand);
    REQUIRE(g.checksum() == before);

    g.paintDisc(-50, -50, 10, br::Cell::Sand);   // entirely off-grid
    REQUIRE(g.checksum() == before);

    g.paintDisc(2, 2, 0, br::Cell::Sand);        // zero radius -> one cell
    REQUIRE(g.at(2, 2) == br::Cell::Sand);

    g.paintDisc(10, 10, -5, br::Cell::Water);    // negative radius is clamped
    REQUIRE(g.at(10, 10) == br::Cell::Water);

    REQUIRE(g.at(-1, -1) == br::Cell::Wall);     // reads outside -> Wall
    REQUIRE(g.at(g.width(), 0) == br::Cell::Wall);
}

TEST_CASE("rendering fills the whole RGBA buffer with opaque pixels",
          "[edge][render]") {
    br::Grid g(64, 32);
    g.clear();
    g.paintDisc(20, 16, 5, br::Cell::Water);

    std::vector<std::uint8_t> px(
        static_cast<std::size_t>(g.width()) *
        static_cast<std::size_t>(g.height()) * 4u, 0u);
    g.renderRGBA(px.data());

    for (std::size_t i = 3; i < px.size(); i += 4) {
        REQUIRE(px[i] == 255u);
    }
    // Water pixel must carry the water colour.
    const std::size_t o = (16u * static_cast<std::size_t>(g.width()) + 20u) * 4u;
    REQUIRE(px[o + 2] > px[o + 0]);      // blue dominant
}