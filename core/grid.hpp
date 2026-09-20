// ---------------------------------------------------------------------------
// core/grid.hpp
//
// Role in BackendRace:
//   * Declares the double-buffered, 32x32-TILED cell grid that every compute
//     backend (naive CPU, AVX2, CUDA, toy-ISA accelerator) reads and writes.
//   * Declares the SINGLE SOURCE OF TRUTH for the falling-sand update rules as
//     tiny inline (and __host__ __device__) functions.  Because every backend
//     calls/lowers exactly these rules, all backends are BIT-EXACT with each
//     other, which is what makes the "race" a fair comparison and what makes
//     the correctness tests meaningful.
//
// Design notes that matter for the optimisation story:
//   * Memory layout: tiles of 32x32 bytes stored contiguously (1024 B/tile).
//     One tile row == 32 bytes == exactly one AVX2 register == exactly one
//     toy-ISA vector register.  Tile bases are 1024-byte aligned, so every
//     tile row is 32-byte aligned -> aligned vector loads/stores everywhere.
//   * The update is expressed as three *gather*/pair-exchange passes so that
//     every output cell depends only on the read buffer.  That makes the rule
//     embarrassingly parallel (no update-order dependency), deterministic, and
//     exactly mass-conserving (all motion is a pair swap).
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#  define BR_HD __host__ __device__
#else
#  define BR_HD
#endif

namespace br {

// ---------------------------------------------------------------------------
// Cell kinds.  One byte per cell.  Values MUST stay inside [0,15] because the
// AVX2 backend and the toy ISA map cells to densities through a 16-entry
// pshufb-style lookup table.
// ---------------------------------------------------------------------------
enum class Cell : std::uint8_t {
    Empty = 0,
    Sand  = 1,
    Water = 2,
    Fire  = 3,
    Wall  = 4,
};

inline constexpr int kCellKinds = 5;

BR_HD inline constexpr std::uint8_t cid(Cell c) noexcept {
    return static_cast<std::uint8_t>(c);
}

// ---------------------------------------------------------------------------
// Density table, pre-XORed with 0x80.
//
// Real densities:  Fire=0 < Empty=1 < Water=2 < Sand=3 < Wall=255
// After XOR 0x80 the values are   0x80 0x81 0x82 0x83 0x7F
// which, interpreted as SIGNED bytes, are -128 -127 -126 -125 +127.
// The unsigned ordering is preserved, so a *signed* byte compare
// (vpcmpgtb / toy-ISA VCMPGT) implements the *unsigned* density comparison.
// This trick is why the AVX2 kernel needs zero unpacking.
// ---------------------------------------------------------------------------
BR_HD inline std::uint8_t densityBiased(std::uint8_t c) noexcept {
    const std::uint8_t kTable[16] = {
        0x81, 0x83, 0x82, 0x80, 0x7F, 0x7F, 0x7F, 0x7F,
        0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F
    };
    return kTable[c & 15u];
}

BR_HD inline bool denserThan(std::uint8_t a, std::uint8_t b) noexcept {
    return static_cast<std::int8_t>(densityBiased(a)) >
           static_cast<std::int8_t>(densityBiased(b));
}

// A cell can move iff it is neither vacuum nor immovable rock.
BR_HD inline bool isMobile(std::uint8_t c) noexcept {
    return c != cid(Cell::Empty) && c != cid(Cell::Wall);
}

// ---------------------------------------------------------------------------
// PASS 1 -- vertical pair exchange (gravity + buoyancy).
// Rows are paired (y, y+1) with the pairing alternating every frame.  A pair
// swaps when the upper cell is denser than the lower one.  Because Fire is the
// lightest material it "falls upwards" through Empty, i.e. it rises.
// ---------------------------------------------------------------------------
BR_HD inline bool swapVertical(std::uint8_t above, std::uint8_t below) noexcept {
    // `below == Wall` is rejected automatically (nothing is denser than Wall).
    return denserThan(above, below) && above != cid(Cell::Wall);
}

// ---------------------------------------------------------------------------
// PASS 2 -- horizontal pair exchange (lateral spreading).
// A material may only spread sideways once it cannot continue its vertical
// motion.  Sand additionally requires that the destination has empty space
// underneath it, which is what produces angle-of-repose piles instead of
// water-like puddles.
// ---------------------------------------------------------------------------
BR_HD inline bool blockedVertically(std::uint8_t c,
                                    std::uint8_t below,
                                    std::uint8_t above) noexcept {
    return (c == cid(Cell::Fire)) ? (above != cid(Cell::Empty))
                                  : (below != cid(Cell::Empty));
}

// `a` is at (x,y), `b` is at (x+1,y).  Returns true if the pair must swap.
// The two directions are mutually exclusive (one side must be Empty, and
// Empty is never mobile), so there is no ordering ambiguity.
BR_HD inline bool swapHorizontal(std::uint8_t a,      std::uint8_t b,
                                 std::uint8_t belowA, std::uint8_t belowB,
                                 std::uint8_t aboveA, std::uint8_t aboveB) noexcept {
    const std::uint8_t E = cid(Cell::Empty);
    const std::uint8_t S = cid(Cell::Sand);

    if (b == E && isMobile(a) && blockedVertically(a, belowA, aboveA)) {
        if (a != S || belowB == E) return true;
    }
    if (a == E && isMobile(b) && blockedVertically(b, belowB, aboveB)) {
        if (b != S || belowA == E) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Deterministic hash used by the fire-decay pass.
//
// IMPORTANT: the hash is evaluated *once per (tile-column, row, frame)* and
// yields 32 random bits -- exactly one bit per lane of a 32-wide tile row.
// The scalar, AVX2 and CUDA backends all consume the same 32-bit word, so the
// "randomness" is bit-identical everywhere and costs the vector backends one
// scalar hash per 32 cells instead of one per cell.
// ---------------------------------------------------------------------------
BR_HD inline std::uint32_t hash32(std::uint32_t a, std::uint32_t b,
                                  std::uint32_t c) noexcept {
    std::uint32_t h = a * 0x9E3779B1u + b * 0x85EBCA77u + c * 0xC2B2AE3Du;
    h ^= h >> 15; h *= 0x2545F491u;
    h ^= h >> 13; h *= 0x27220A95u;
    h ^= h >> 16;
    return h;
}

inline constexpr std::uint32_t kSaltA = 0x51ED2701u;
inline constexpr std::uint32_t kSaltB = 0x2F7C93A5u;
inline constexpr std::uint32_t kSaltC = 0x7B4DE19Fu;

BR_HD inline bool randBit(int x, int y, std::uint32_t frame,
                          std::uint32_t salt) noexcept {
    const std::uint32_t word =
        hash32(static_cast<std::uint32_t>(x >> 5),
               static_cast<std::uint32_t>(y),
               frame ^ salt);
    return ((word >> (x & 31)) & 1u) != 0u;
}

// ---------------------------------------------------------------------------
// PASS 3 -- per-cell fire decay.  Pure function of the 4-neighbourhood, so it
// vectorises trivially.  Fire dies instantly next to water and otherwise has a
// 1/8 chance per frame of burning out (three independent hash bit-planes).
// ---------------------------------------------------------------------------
BR_HD inline std::uint8_t decayFire(std::uint8_t c,
                                    std::uint8_t l, std::uint8_t r,
                                    std::uint8_t u, std::uint8_t d,
                                    bool rA, bool rB, bool rC) noexcept {
    if (c != cid(Cell::Fire)) return c;
    const std::uint8_t W = cid(Cell::Water);
    if (l == W || r == W || u == W || d == W) return cid(Cell::Empty);
    return (rA && rB && rC) ? cid(Cell::Empty) : c;
}

// ---------------------------------------------------------------------------
// Tiled addressing.  tile = 32x32 = 1024 bytes, tiles stored row-major.
//   index = (tileIndex << 10) | (localY << 5) | localX
// ---------------------------------------------------------------------------
BR_HD inline std::size_t tiledIndex(int x, int y, int tilesX) noexcept {
    const int tx = x >> 5, ty = y >> 5;
    const int lx = x & 31, ly = y & 31;
    return (static_cast<std::size_t>(ty * tilesX + tx) << 10) +
           (static_cast<std::size_t>(ly) << 5) +
           static_cast<std::size_t>(lx);
}

// Out-of-grid reads return Wall.  Every backend uses this convention, which is
// what lets the rules be written without a single boundary `if`.
BR_HD inline std::uint8_t sampleCell(const std::uint8_t* s, int x, int y,
                                     int W, int H, int tilesX) noexcept {
    if (x < 0 || y < 0 || x >= W || y >= H) return cid(Cell::Wall);
    return s[tiledIndex(x, y, tilesX)];
}

// ---------------------------------------------------------------------------
// The grid itself.  Two 64-byte-aligned buffers, ping-ponged by the passes.
// ---------------------------------------------------------------------------
class Grid {
public:
    static constexpr int kTile      = 32;
    static constexpr int kTileBytes = kTile * kTile;   // 1024

    Grid(int w, int h);
    ~Grid();
    Grid(const Grid&)            = delete;
    Grid& operator=(const Grid&) = delete;

    int  width()  const noexcept { return w_; }
    int  height() const noexcept { return h_; }
    int  tilesX() const noexcept { return tilesX_; }
    int  tilesY() const noexcept { return tilesY_; }
    std::size_t bytes() const noexcept { return bytes_; }
    std::size_t cells() const noexcept {
        return static_cast<std::size_t>(w_) * static_cast<std::size_t>(h_);
    }

    std::uint8_t*       src()       noexcept { return a_; }
    const std::uint8_t* src() const noexcept { return a_; }
    std::uint8_t*       dst()       noexcept { return b_; }
    const std::uint8_t* dst() const noexcept { return b_; }

    void swapBuffers() noexcept { std::uint8_t* t = a_; a_ = b_; b_ = t; }

    std::uint64_t frame() const noexcept { return frame_; }
    void advanceFrame()          noexcept { ++frame_; }
    void setFrame(std::uint64_t f) noexcept { frame_ = f; }

    Cell at(int x, int y) const noexcept;
    void set(int x, int y, Cell c) noexcept;

    void clear();                       // Empty everywhere + Wall border
    void fillBorders();                 // 1-cell Wall ring
    void copyToDst();                   // memcpy src -> dst
    void paintDisc(int cx, int cy, int r, Cell c);
    void seedRandom(std::uint32_t seed, float sandFrac, float waterFrac);

    std::uint64_t checksum() const noexcept;      // FNV-1a over src()
    void renderRGBA(std::uint8_t* out) const;     // width*height*4 bytes

private:
    int  w_, h_, tilesX_, tilesY_;
    std::size_t   bytes_;
    std::uint8_t* a_;
    std::uint8_t* b_;
    std::uint64_t frame_;
};

}  // namespace br