// ---------------------------------------------------------------------------
// tests/test_correctness.cpp
//
// Role in BackendRace: proves that every backend computes the SAME universe.
// Because the rules are deterministic (pair exchanges + a pure hash), the
// comparison is bit-exact, not tolerance-based: any divergence is a bug, never
// floating-point noise.
//
//   * naive  vs  AVX2                 -> byte-identical after N full steps
//   * naive  vs  toy-ISA accelerator  -> byte-identical vertical pass
//   * naive  vs  toy-ISA backend      -> byte-identical after N full steps
//   * physical invariants (mass conservation, determinism) hold for all of them
// ---------------------------------------------------------------------------
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>
#include <vector>

#include "backends/toy_isa/isa.hpp"
#include "core/backend.hpp"
#include "core/grid.hpp"
#include "core/passes.hpp"

namespace {

void seedWorld(br::Grid& g, std::uint32_t seed) {
    g.clear();
    g.seedRandom(seed, 0.25f, 0.15f);
    g.paintDisc(g.width() / 3, g.height() - 10, 5, br::Cell::Fire);
    g.paintDisc((2 * g.width()) / 3, 20, 7, br::Cell::Water);
}

std::size_t firstDifference(const std::uint8_t* a, const std::uint8_t* b,
                            std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return i;
    }
    return n;
}

std::array<std::size_t, br::kCellKinds> histogram(const br::Grid& g) {
    std::array<std::size_t, br::kCellKinds> h{};
    for (int y = 0; y < g.height(); ++y) {
        for (int x = 0; x < g.width(); ++x) {
            const std::uint8_t c = br::cid(g.at(x, y));
            if (c < br::kCellKinds) ++h[c];
        }
    }
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST_CASE("naive and AVX2 backends are bit-exact", "[correctness][avx2]") {
    auto naive = br::createNaiveBackend();
    auto avx2  = br::createAvx2Backend();

    if (!avx2->available()) {
        WARN("AVX2 backend unavailable on this machine/build - test skipped");
        SUCCEED("skipped");
        return;
    }

    br::Grid a(128, 96);
    br::Grid b(128, 96);
    seedWorld(a, 0xA11CE5u);
    seedWorld(b, 0xA11CE5u);

    REQUIRE(a.bytes() == b.bytes());
    REQUIRE(std::memcmp(a.src(), b.src(), a.bytes()) == 0);

    for (int frame = 0; frame < 60; ++frame) {
        naive->step(a);
        avx2->step(b);

        const std::size_t diff = firstDifference(a.src(), b.src(), a.bytes());
        INFO("divergence at frame " << frame << ", byte offset " << diff
             << " (tile " << (diff >> 10) << ", localY " << ((diff >> 5) & 31)
             << ", localX " << (diff & 31) << ")");
        REQUIRE(diff == a.bytes());
    }

    REQUIRE(a.frame() == b.frame());
    REQUIRE(a.checksum() == b.checksum());
}

TEST_CASE("toy-ISA vertical pass matches the scalar reference",
          "[correctness][toyisa]") {
    br::Grid g(64, 96);
    seedWorld(g, 0x5EED01u);

    std::vector<std::uint8_t> reference(g.bytes(), 0u);

    for (int parity = 0; parity < 2; ++parity) {
        // Poison the destination so a missed store shows up immediately.
        std::memset(g.dst(), 0xCD, g.bytes());

        br::passes::verticalScalar(g.src(), reference.data(), g, parity);

        const br::toy::Program prog =
            br::toy::compileVerticalPass(g.width(), g.height(), parity);
        br::toy::MemoryImage mem{g.src(), g.dst(), g.bytes()};
        const br::toy::RunResult r = br::toy::execute(prog, mem);

        INFO("parity " << parity << " toy-ISA fault: " << r.error);
        REQUIRE(r.ok);
        REQUIRE(r.instructions > 0);
        REQUIRE(r.cycles >= r.instructions);     // single issue: >= 1 cycle each
        REQUIRE(r.vectorMemOps > 0);

        const std::size_t diff =
            firstDifference(reference.data(), g.dst(), g.bytes());
        INFO("first differing byte offset: " << diff);
        REQUIRE(diff == g.bytes());
    }
}

TEST_CASE("toy-ISA backend reproduces the naive simulation",
          "[correctness][toyisa]") {
    auto naive = br::createNaiveBackend();
    auto toy   = br::createToyIsaBackend();
    REQUIRE(toy->available());

    br::Grid a(64, 64);
    br::Grid b(64, 64);
    seedWorld(a, 0x7A11u);
    seedWorld(b, 0x7A11u);

    for (int frame = 0; frame < 24; ++frame) {
        naive->step(a);
        toy->step(b);
        INFO("frame " << frame);
        REQUIRE(firstDifference(a.src(), b.src(), a.bytes()) == a.bytes());
    }

    REQUIRE(toy->stats().modeledCycles > 0);
    REQUIRE(toy->stats().instructions > 0);
    REQUIRE(a.checksum() == b.checksum());
}

TEST_CASE("every available backend agrees with the naive baseline",
          "[correctness][matrix]") {
    br::Grid reference(96, 64);
    seedWorld(reference, 0xBEEF11u);
    auto naive = br::createNaiveBackend();
    for (int i = 0; i < 40; ++i) naive->step(reference);
    const std::uint64_t want = reference.checksum();

    for (int id = 0; id < static_cast<int>(br::BackendId::Count); ++id) {
        auto be = br::createBackend(static_cast<br::BackendId>(id));
        INFO("backend: " << be->name() << " (" << be->description() << ")");
        if (!be->available()) continue;

        br::Grid g(96, 64);
        seedWorld(g, 0xBEEF11u);
        be->stepMany(g, 40);

        REQUIRE(g.frame() == reference.frame());
        REQUIRE(g.checksum() == want);
    }
}

TEST_CASE("the update rule conserves material", "[correctness][invariant]") {
    br::Grid g(96, 96);
    g.clear();
    g.seedRandom(0x1234ABu, 0.30f, 0.20f);   // no fire -> nothing can be destroyed

    const auto before = histogram(g);
    auto be = br::createNaiveBackend();
    for (int i = 0; i < 50; ++i) be->step(g);
    const auto after = histogram(g);

    REQUIRE(after[br::cid(br::Cell::Sand)]  == before[br::cid(br::Cell::Sand)]);
    REQUIRE(after[br::cid(br::Cell::Water)] == before[br::cid(br::Cell::Water)]);
    REQUIRE(after[br::cid(br::Cell::Wall)]  == before[br::cid(br::Cell::Wall)]);
    REQUIRE(after[br::cid(br::Cell::Empty)] == before[br::cid(br::Cell::Empty)]);
}

TEST_CASE("simulation is deterministic across runs", "[correctness][determinism]") {
    br::Grid a(64, 64);
    br::Grid b(64, 64);
    seedWorld(a, 0xD0D0u);
    seedWorld(b, 0xD0D0u);

    auto be1 = br::createNaiveBackend();
    auto be2 = br::createNaiveBackend();
    be1->stepMany(a, 33);
    be2->stepMany(b, 33);

    REQUIRE(a.checksum() == b.checksum());
}

TEST_CASE("the density ordering encodes the physics we intend",
          "[correctness][rules]") {
    using br::Cell;
    using br::cid;

    // Fire < Empty < Water < Sand < Wall
    REQUIRE(br::denserThan(cid(Cell::Sand),  cid(Cell::Water)));
    REQUIRE(br::denserThan(cid(Cell::Water), cid(Cell::Empty)));
    REQUIRE(br::denserThan(cid(Cell::Empty), cid(Cell::Fire)));
    REQUIRE(br::denserThan(cid(Cell::Wall),  cid(Cell::Sand)));
    REQUIRE_FALSE(br::denserThan(cid(Cell::Fire),  cid(Cell::Empty)));
    REQUIRE_FALSE(br::denserThan(cid(Cell::Empty), cid(Cell::Empty)));

    // Sand sinks through water, water rises above sand.
    REQUIRE(br::swapVertical(cid(Cell::Sand), cid(Cell::Water)));
    // Fire rises: the Empty above a Fire is "denser" and falls into it.
    REQUIRE(br::swapVertical(cid(Cell::Empty), cid(Cell::Fire)));
    // Walls never move, and nothing displaces them.
    REQUIRE_FALSE(br::swapVertical(cid(Cell::Wall), cid(Cell::Empty)));
    REQUIRE_FALSE(br::swapVertical(cid(Cell::Sand), cid(Cell::Wall)));

    // Fire dies next to water regardless of the random bits.
    REQUIRE(br::decayFire(cid(Cell::Fire), cid(Cell::Water), 0, 0, 0,
                          false, false, false) == cid(Cell::Empty));
    // ...and survives when all three random bits are not set.
    REQUIRE(br::decayFire(cid(Cell::Fire), 0, 0, 0, 0,
                          true, true, false) == cid(Cell::Fire));
    // Non-fire cells are untouched by the decay pass.
    REQUIRE(br::decayFire(cid(Cell::Sand), cid(Cell::Water), 0, 0, 0,
                          true, true, true) == cid(Cell::Sand));
}