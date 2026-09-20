// ---------------------------------------------------------------------------
// core/grid.cpp
//
// Role in BackendRace: implementation of the tiled grid container -- aligned
// allocation, tiled addressing helpers, deterministic seeding, checksumming
// (used by the benchmark to prove every backend produced the same universe)
// and RGBA rendering for both the native terminal UI and the WASM canvas.
// ---------------------------------------------------------------------------
#include "core/grid.hpp"

#include <algorithm>
#include <new>

namespace br {

namespace {
constexpr std::size_t kAlign = 64;   // one cache line; 32B vector alignment implied

std::uint8_t* allocAligned(std::size_t n) {
    return static_cast<std::uint8_t*>(
        ::operator new[](n, std::align_val_t(kAlign)));
}
void freeAligned(std::uint8_t* p) noexcept {
    if (p) ::operator delete[](p, std::align_val_t(kAlign));
}
int roundUpTile(int v) {
    if (v < Grid::kTile) v = Grid::kTile;
    return ((v + Grid::kTile - 1) / Grid::kTile) * Grid::kTile;
}
}  // namespace

Grid::Grid(int w, int h)
    : w_(roundUpTile(w)),
      h_(roundUpTile(h)),
      tilesX_(0), tilesY_(0),
      bytes_(0),
      a_(nullptr), b_(nullptr),
      frame_(0) {
    tilesX_ = w_ / kTile;
    tilesY_ = h_ / kTile;
    bytes_  = static_cast<std::size_t>(tilesX_) *
              static_cast<std::size_t>(tilesY_) *
              static_cast<std::size_t>(kTileBytes);
    a_ = allocAligned(bytes_);
    b_ = allocAligned(bytes_);
    clear();
}

Grid::~Grid() {
    freeAligned(a_);
    freeAligned(b_);
}

Cell Grid::at(int x, int y) const noexcept {
    if (x < 0 || y < 0 || x >= w_ || y >= h_) return Cell::Wall;
    return static_cast<Cell>(a_[tiledIndex(x, y, tilesX_)]);
}

void Grid::set(int x, int y, Cell c) noexcept {
    if (x < 0 || y < 0 || x >= w_ || y >= h_) return;
    a_[tiledIndex(x, y, tilesX_)] = cid(c);
}

void Grid::clear() {
    std::memset(a_, cid(Cell::Empty), bytes_);
    std::memset(b_, cid(Cell::Empty), bytes_);
    fillBorders();
    frame_ = 0;
}

void Grid::fillBorders() {
    for (int x = 0; x < w_; ++x) {
        a_[tiledIndex(x, 0,      tilesX_)] = cid(Cell::Wall);
        a_[tiledIndex(x, h_ - 1, tilesX_)] = cid(Cell::Wall);
    }
    for (int y = 0; y < h_; ++y) {
        a_[tiledIndex(0,      y, tilesX_)] = cid(Cell::Wall);
        a_[tiledIndex(w_ - 1, y, tilesX_)] = cid(Cell::Wall);
    }
}

void Grid::copyToDst() {
    std::memcpy(b_, a_, bytes_);
}

void Grid::paintDisc(int cx, int cy, int r, Cell c) {
    if (r < 0) r = 0;
    const int x0 = std::max(1, cx - r), x1 = std::min(w_ - 2, cx + r);
    const int y0 = std::max(1, cy - r), y1 = std::min(h_ - 2, cy + r);
    const int rr = r * r;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const int dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy <= rr) {
                a_[tiledIndex(x, y, tilesX_)] = cid(c);
            }
        }
    }
}

void Grid::seedRandom(std::uint32_t seed, float sandFrac, float waterFrac) {
    const std::uint32_t sandT  =
        static_cast<std::uint32_t>(sandFrac  * 4294967295.0f);
    const std::uint32_t waterT =
        static_cast<std::uint32_t>((sandFrac + waterFrac) * 4294967295.0f);

    for (int y = 1; y < h_ - 1; ++y) {
        for (int x = 1; x < w_ - 1; ++x) {
            const std::uint32_t h = hash32(static_cast<std::uint32_t>(x),
                                           static_cast<std::uint32_t>(y),
                                           seed);
            std::uint8_t c = cid(Cell::Empty);
            if      (h < sandT)  c = cid(Cell::Sand);
            else if (h < waterT) c = cid(Cell::Water);
            a_[tiledIndex(x, y, tilesX_)] = c;
        }
    }
    // A couple of solid ledges so the fluids actually have something to do.
    const int ly1 = h_ / 3, ly2 = (2 * h_) / 3;
    for (int x = 1; x < w_ / 2; ++x)            a_[tiledIndex(x, ly1, tilesX_)] = cid(Cell::Wall);
    for (int x = w_ / 2; x < w_ - 1; ++x)       a_[tiledIndex(x, ly2, tilesX_)] = cid(Cell::Wall);
    frame_ = 0;
}

std::uint64_t Grid::checksum() const noexcept {
    std::uint64_t h = 1469598103934665603ull;           // FNV-1a 64
    for (std::size_t i = 0; i < bytes_; ++i) {
        h ^= static_cast<std::uint64_t>(a_[i]);
        h *= 1099511628211ull;
    }
    return h;
}

void Grid::renderRGBA(std::uint8_t* out) const {
    static const std::uint8_t pal[5][3] = {
        { 14,  14,  18},   // Empty
        {222, 184, 105},   // Sand
        { 58, 116, 220},   // Water
        {240, 120,  40},   // Fire
        { 88,  90, 102},   // Wall
    };
    for (int y = 0; y < h_; ++y) {
        for (int x = 0; x < w_; ++x) {
            std::uint8_t c = a_[tiledIndex(x, y, tilesX_)];
            if (c >= kCellKinds) c = 0;
            std::uint8_t r = pal[c][0], g = pal[c][1], b = pal[c][2];
            if (c == cid(Cell::Fire)) {                 // cosmetic flicker only
                const std::uint32_t h = hash32(static_cast<std::uint32_t>(x),
                                               static_cast<std::uint32_t>(y),
                                               static_cast<std::uint32_t>(frame_));
                r = static_cast<std::uint8_t>(220 + (h & 31));
                g = static_cast<std::uint8_t>( 80 + ((h >> 5) & 63));
                b = 30;
            }
            const std::size_t o =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(w_) +
                 static_cast<std::size_t>(x)) * 4u;
            out[o + 0] = r;
            out[o + 1] = g;
            out[o + 2] = b;
            out[o + 3] = 255;
        }
    }
}

}  // namespace br