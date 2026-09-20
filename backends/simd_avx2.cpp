// ---------------------------------------------------------------------------
// backends/simd_avx2.cpp
//
// Role in BackendRace: the hand-vectorised CPU backend.
//
// ============================ VECTORISATION STRATEGY =======================
//
// 1) LAYOUT CHOICE -- why 32-wide horizontal lanes.
//    The grid is stored in 32x32 tiles (1024 B each, tile bases 1024-byte
//    aligned).  One tile ROW is therefore 32 contiguous bytes == exactly one
//    __m256i, and its address is always 32-byte aligned.  We process cells as
//    one byte per lane: no unpacking to 16/32-bit ever happens, so a single
//    vpcmpgtb does the work of 32 scalar density comparisons.
//
//    The alternative ("vertical"/SoA layout, i.e. 32 independent columns per
//    register) would make the *horizontal* pass free but would turn every
//    vertical neighbour access into a gather.  Since gravity (pass 1) is the
//    hottest pass and is purely vertical, the horizontal-lane layout wins: in
//    pass 1 the neighbour row is *another aligned 32-byte row*, so both
//    operands are plain aligned loads and the pass contains zero shuffles.
//
// 2) WHICH LOADS ARE VECTORISED.
//    Pass 1 (vertical)   : 2 aligned vmovdqa loads + 1 aligned store per row.
//                          The neighbour row for localY==31 is row 0 of the
//                          tile BELOW -- one `+ tilesX*1024` byte offset, still
//                          32-byte aligned.  No scalar fixup at all.
//    Pass 2 (horizontal) : needs x-1 and x+1, which cross the tile seam.  We
//                          gather a 34-byte halo row (1 + 32 + 1) into a small
//                          stack scratch buffer, then issue three *unaligned*
//                          loads at offsets 0/1/2.  Unaligned loads inside a
//                          64-byte-aligned 34-byte scratch never split a cache
//                          line, so they cost the same as aligned ones on any
//                          Haswell-or-later core.  The three rows (y-1,y,y+1)
//                          are rotated instead of re-gathered, so each row is
//                          copied exactly once per tile.
//    Pass 3 (decay)      : same halo gather; the per-cell "randomness" is a
//                          single scalar hash per (tile,row) whose 32 bits are
//                          expanded into a 32-lane 0x00/0xFF mask with
//                          broadcast + pshufb + pand + pcmpeqb (4 uops for 32
//                          random bits, versus 32 scalar hashes).
//
// 3) ALIGNMENT.
//    Both grid buffers come from aligned operator new (64 B).  tile base is a
//    multiple of 1024 and a row offset is a multiple of 32, therefore every
//    `_mm256_load_si256` / `_mm256_store_si256` below is provably aligned; we
//    use the aligned forms on purpose so that a mistake faults loudly instead
//    of silently costing performance.
//
// 4) TILING / L1 RESIDENCY.
//    The outer loops walk tile-by-tile.  A tile plus its two neighbour rows is
//    ~1.1 KB, and the three 34-byte scratch rows are ~102 B, so the working set
//    of the inner loop is far inside a 32-48 KB L1d.  Walking whole scanlines
//    instead (naive order) touches tilesX*1024 bytes before reusing a row.
//
// 5) PORTABILITY.
//    If the translation unit is not built with AVX2 (non-x86, Emscripten
//    without -mavx2, ...), everything below collapses to the shared scalar
//    passes and `available()` returns false so the UI can grey the button out.
// ---------------------------------------------------------------------------
#include <chrono>
#include <cstring>
#include <memory>

#include "core/backend.hpp"
#include "core/grid.hpp"
#include "core/passes.hpp"

#if defined(__AVX2__)
#  include <immintrin.h>
#  define BR_AVX2_ENABLED 1
#else
#  define BR_AVX2_ENABLED 0
#endif

namespace br {
namespace {

#if BR_AVX2_ENABLED

// --------------------------------------------------------------------------
// Small helpers
// --------------------------------------------------------------------------
inline __m256i vNot(__m256i x) {
    return _mm256_xor_si256(x, _mm256_set1_epi8(static_cast<char>(0xFF)));
}
inline __m256i vNeq(__m256i x, __m256i y) {
    return vNot(_mm256_cmpeq_epi8(x, y));
}

// The biased density table, duplicated into both 128-bit halves because
// vpshufb indexes within each lane independently.
inline __m256i densityLutVec() {
    return _mm256_setr_epi8(
        (char)0x81, (char)0x83, (char)0x82, (char)0x80,
        (char)0x7F, (char)0x7F, (char)0x7F, (char)0x7F,
        (char)0x7F, (char)0x7F, (char)0x7F, (char)0x7F,
        (char)0x7F, (char)0x7F, (char)0x7F, (char)0x7F,
        (char)0x81, (char)0x83, (char)0x82, (char)0x80,
        (char)0x7F, (char)0x7F, (char)0x7F, (char)0x7F,
        (char)0x7F, (char)0x7F, (char)0x7F, (char)0x7F,
        (char)0x7F, (char)0x7F, (char)0x7F, (char)0x7F);
}

// Expand the 32 bits of `bits` into 32 bytes of 0x00 / 0xFF, lane i <- bit i.
// Exactly mirrors the scalar `(word >> (x & 31)) & 1`.
inline __m256i expandBits(std::uint32_t bits) {
    const __m256i v  = _mm256_set1_epi32(static_cast<int>(bits));
    const __m256i sh = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                                        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bit = _mm256_setr_epi8(
        1,2,4,8,16,32,64,(char)128, 1,2,4,8,16,32,64,(char)128,
        1,2,4,8,16,32,64,(char)128, 1,2,4,8,16,32,64,(char)128);
    const __m256i spread = _mm256_shuffle_epi8(v, sh);
    return _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit), bit);
}

// Vector form of br::swapVertical(): ~(a == Wall) & (density(a) > density(b)).
inline __m256i vSwapVertical(__m256i a, __m256i b, __m256i lut, __m256i wall) {
    const __m256i da = _mm256_shuffle_epi8(lut, a);
    const __m256i db = _mm256_shuffle_epi8(lut, b);
    const __m256i gt = _mm256_cmpgt_epi8(da, db);          // signed == unsigned (biased)
    return _mm256_andnot_si256(_mm256_cmpeq_epi8(a, wall), gt);
}

// Vector form of br::swapHorizontal().
inline __m256i vSwapHorizontal(__m256i a,  __m256i b,
                               __m256i ba, __m256i bb,
                               __m256i aa, __m256i ab,
                               __m256i zero, __m256i wall,
                               __m256i fire, __m256i sand) {
    const __m256i mobA  = _mm256_and_si256(vNeq(a, zero), vNeq(a, wall));
    const __m256i mobB  = _mm256_and_si256(vNeq(b, zero), vNeq(b, wall));
    const __m256i fireA = _mm256_cmpeq_epi8(a, fire);
    const __m256i fireB = _mm256_cmpeq_epi8(b, fire);
    // blockedVertically(): fire looks up, everything else looks down.
    const __m256i blkA  = _mm256_blendv_epi8(vNeq(ba, zero), vNeq(aa, zero), fireA);
    const __m256i blkB  = _mm256_blendv_epi8(vNeq(bb, zero), vNeq(ab, zero), fireB);
    const __m256i okSA  = _mm256_or_si256(vNot(_mm256_cmpeq_epi8(a, sand)),
                                          _mm256_cmpeq_epi8(bb, zero));
    const __m256i okSB  = _mm256_or_si256(vNot(_mm256_cmpeq_epi8(b, sand)),
                                          _mm256_cmpeq_epi8(ba, zero));

    __m256i right = _mm256_and_si256(_mm256_cmpeq_epi8(b, zero), mobA);
    right = _mm256_and_si256(right, blkA);
    right = _mm256_and_si256(right, okSA);

    __m256i left = _mm256_and_si256(_mm256_cmpeq_epi8(a, zero), mobB);
    left = _mm256_and_si256(left, blkB);
    left = _mm256_and_si256(left, okSB);

    return _mm256_or_si256(right, left);   // mutually exclusive by construction
}

// Copy one 34-byte halo row (x0-1 .. x0+32) of global row `y` for tile col `tx`.
inline void gatherHaloRow(const std::uint8_t* s, int y, int tx,
                          int W, int H, int tilesX, std::uint8_t* out34) {
    if (y < 0 || y >= H) {
        std::memset(out34, cid(Cell::Wall), 34);
        return;
    }
    const int x0 = tx * 32;
    // A whole tile row is 32 contiguous bytes -> one memcpy, no gather.
    std::memcpy(out34 + 1, s + tiledIndex(x0, y, tilesX), 32);
    out34[0]  = (x0 - 1 >= 0)  ? s[tiledIndex(x0 - 1,  y, tilesX)] : cid(Cell::Wall);
    out34[33] = (x0 + 32 < W)  ? s[tiledIndex(x0 + 32, y, tilesX)] : cid(Cell::Wall);
}

// --------------------------------------------------------------------------
// PASS 1 -- vertical pair exchange.  Pure aligned loads, zero shuffles.
// --------------------------------------------------------------------------
void verticalAvx2(const std::uint8_t* s, std::uint8_t* d,
                  const Grid& g, int parity) {
    const int TX = g.tilesX(), TY = g.tilesY(), H = g.height();
    const std::size_t tileRowStride = static_cast<std::size_t>(TX) << 10;

    const __m256i lut  = densityLutVec();
    const __m256i wall = _mm256_set1_epi8(static_cast<char>(cid(Cell::Wall)));

    for (int ty = 0; ty < TY; ++ty) {
        for (int tx = 0; tx < TX; ++tx) {
            const std::size_t base =
                static_cast<std::size_t>(ty * TX + tx) << 10;
            const std::uint8_t* sp = s + base;
            std::uint8_t*       dp = d + base;

            for (int ly = 0; ly < 32; ++ly) {
                const int y = (ty << 5) + ly;
                const std::uint8_t* rowC = sp + (ly << 5);
                __m256i out;

                if ((y & 1) == parity) {
                    if (y + 1 >= H) {                     // bottom edge: copy
                        out = _mm256_load_si256(
                            reinterpret_cast<const __m256i*>(rowC));
                    } else {
                        const std::uint8_t* rowB =
                            (ly < 31) ? (rowC + 32) : (sp + tileRowStride);
                        const __m256i a = _mm256_load_si256(
                            reinterpret_cast<const __m256i*>(rowC));
                        const __m256i b = _mm256_load_si256(
                            reinterpret_cast<const __m256i*>(rowB));
                        out = _mm256_blendv_epi8(
                            a, b, vSwapVertical(a, b, lut, wall));
                    }
                } else {
                    if (y - 1 < 0) {                      // top edge: copy
                        out = _mm256_load_si256(
                            reinterpret_cast<const __m256i*>(rowC));
                    } else {
                        const std::uint8_t* rowA =
                            (ly > 0) ? (rowC - 32)
                                     : (sp - tileRowStride + (31 << 5));
                        const __m256i a = _mm256_load_si256(
                            reinterpret_cast<const __m256i*>(rowA));
                        const __m256i c = _mm256_load_si256(
                            reinterpret_cast<const __m256i*>(rowC));
                        out = _mm256_blendv_epi8(
                            c, a, vSwapVertical(a, c, lut, wall));
                    }
                }
                _mm256_store_si256(
                    reinterpret_cast<__m256i*>(dp + (ly << 5)), out);
            }
        }
    }
}

// --------------------------------------------------------------------------
// PASS 2 -- horizontal pair exchange.  Halo-padded scratch rows, rotated.
// --------------------------------------------------------------------------
void horizontalAvx2(const std::uint8_t* s, std::uint8_t* d,
                    const Grid& g, int parity) {
    const int TX = g.tilesX(), TY = g.tilesY(), W = g.width(), H = g.height();

    alignas(64) std::uint8_t buf0[34], buf1[34], buf2[34];
    std::uint8_t* up = buf0;
    std::uint8_t* cu = buf1;
    std::uint8_t* dn = buf2;

    const __m256i zero = _mm256_setzero_si256();
    const __m256i wall = _mm256_set1_epi8(static_cast<char>(cid(Cell::Wall)));
    const __m256i fire = _mm256_set1_epi8(static_cast<char>(cid(Cell::Fire)));
    const __m256i sand = _mm256_set1_epi8(static_cast<char>(cid(Cell::Sand)));

    // Lane i holds global column x = tx*32 + i; 32 is even, so x&1 == i&1 and
    // the "am I the left element of my pair" mask is a compile-time constant.
    const __m256i lanePar = _mm256_setr_epi8(
        0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1,
        0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1);
    const __m256i isLeft =
        _mm256_cmpeq_epi8(lanePar, _mm256_set1_epi8(static_cast<char>(parity)));

    for (int ty = 0; ty < TY; ++ty) {
        for (int tx = 0; tx < TX; ++tx) {
            const std::size_t base =
                static_cast<std::size_t>(ty * TX + tx) << 10;
            std::uint8_t* dp = d + base;

            gatherHaloRow(s, (ty << 5) - 1, tx, W, H, TX, up);
            gatherHaloRow(s, (ty << 5),     tx, W, H, TX, cu);

            for (int ly = 0; ly < 32; ++ly) {
                gatherHaloRow(s, (ty << 5) + ly + 1, tx, W, H, TX, dn);

                const __m256i cL = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cu + 0));
                const __m256i c0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cu + 1));
                const __m256i cR = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cu + 2));
                const __m256i dL = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(dn + 0));
                const __m256i d0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(dn + 1));
                const __m256i dR = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(dn + 2));
                const __m256i uL = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(up + 0));
                const __m256i u0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(up + 1));
                const __m256i uR = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(up + 2));

                // Decision for the pair (x, x+1) -- used by LEFT lanes.
                const __m256i decR = vSwapHorizontal(c0, cR, d0, dR, u0, uR,
                                                     zero, wall, fire, sand);
                // Decision for the pair (x-1, x) -- used by RIGHT lanes.
                const __m256i decL = vSwapHorizontal(cL, c0, dL, d0, uL, u0,
                                                     zero, wall, fire, sand);

                const __m256i asLeft  = _mm256_blendv_epi8(c0, cR, decR);
                const __m256i asRight = _mm256_blendv_epi8(c0, cL, decL);
                const __m256i out     = _mm256_blendv_epi8(asRight, asLeft, isLeft);

                _mm256_store_si256(
                    reinterpret_cast<__m256i*>(dp + (ly << 5)), out);

                std::uint8_t* t = up; up = cu; cu = dn; dn = t;  // rotate rows
            }
        }
    }
}

// --------------------------------------------------------------------------
// PASS 3 -- fire decay.  One scalar hash per (tile,row) -> 32 random lanes.
// --------------------------------------------------------------------------
void decayAvx2(const std::uint8_t* s, std::uint8_t* d,
               const Grid& g, std::uint32_t frame) {
    const int TX = g.tilesX(), TY = g.tilesY(), W = g.width(), H = g.height();

    alignas(64) std::uint8_t buf0[34], buf1[34], buf2[34];
    std::uint8_t* up = buf0;
    std::uint8_t* cu = buf1;
    std::uint8_t* dn = buf2;

    const __m256i fire  = _mm256_set1_epi8(static_cast<char>(cid(Cell::Fire)));
    const __m256i water = _mm256_set1_epi8(static_cast<char>(cid(Cell::Water)));

    for (int ty = 0; ty < TY; ++ty) {
        for (int tx = 0; tx < TX; ++tx) {
            const std::size_t base =
                static_cast<std::size_t>(ty * TX + tx) << 10;
            std::uint8_t* dp = d + base;

            gatherHaloRow(s, (ty << 5) - 1, tx, W, H, TX, up);
            gatherHaloRow(s, (ty << 5),     tx, W, H, TX, cu);

            for (int ly = 0; ly < 32; ++ly) {
                const int y = (ty << 5) + ly;
                gatherHaloRow(s, y + 1, tx, W, H, TX, dn);

                const __m256i cL = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cu + 0));
                const __m256i c0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cu + 1));
                const __m256i cR = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cu + 2));
                const __m256i u0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(up + 1));
                const __m256i d0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(dn + 1));

                const std::uint32_t hA = hash32(static_cast<std::uint32_t>(tx),
                                                static_cast<std::uint32_t>(y),
                                                frame ^ kSaltA);
                const std::uint32_t hB = hash32(static_cast<std::uint32_t>(tx),
                                                static_cast<std::uint32_t>(y),
                                                frame ^ kSaltB);
                const std::uint32_t hC = hash32(static_cast<std::uint32_t>(tx),
                                                static_cast<std::uint32_t>(y),
                                                frame ^ kSaltC);
                const __m256i rnd = _mm256_and_si256(
                    _mm256_and_si256(expandBits(hA), expandBits(hB)),
                    expandBits(hC));

                __m256i nearWater = _mm256_or_si256(
                    _mm256_cmpeq_epi8(cL, water), _mm256_cmpeq_epi8(cR, water));
                nearWater = _mm256_or_si256(nearWater,
                    _mm256_cmpeq_epi8(u0, water));
                nearWater = _mm256_or_si256(nearWater,
                    _mm256_cmpeq_epi8(d0, water));

                const __m256i isFire = _mm256_cmpeq_epi8(c0, fire);
                const __m256i kill   = _mm256_and_si256(
                    isFire, _mm256_or_si256(nearWater, rnd));
                const __m256i out    = _mm256_andnot_si256(kill, c0);

                _mm256_store_si256(
                    reinterpret_cast<__m256i*>(dp + (ly << 5)), out);

                std::uint8_t* t = up; up = cu; cu = dn; dn = t;
            }
        }
    }
}

#endif  // BR_AVX2_ENABLED

class Avx2Backend final : public IBackend {
public:
    const char* name() const noexcept override { return "AVX2 SIMD"; }
    const char* description() const noexcept override {
#if BR_AVX2_ENABLED
        return "32 cells/lane, aligned tile-row loads, halo scratch rows";
#else
        return "AVX2 not compiled in for this target (scalar fallback)";
#endif
    }
    bool available() const noexcept override {
        return BR_AVX2_ENABLED && cpuSupportsAvx2();
    }

    void step(Grid& g) override {
        const auto t0 = std::chrono::steady_clock::now();

        const int p1 = static_cast<int>( g.frame()       & 1ull);
        const int p2 = static_cast<int>((g.frame() >> 1) & 1ull);
        const auto f = static_cast<std::uint32_t>(g.frame());

#if BR_AVX2_ENABLED
        if (cpuSupportsAvx2()) {
            verticalAvx2  (g.src(), g.dst(), g, p1); g.swapBuffers();
            horizontalAvx2(g.src(), g.dst(), g, p2); g.swapBuffers();
            decayAvx2     (g.src(), g.dst(), g, f ); g.swapBuffers();
            g.advanceFrame();
        } else
#endif
        {
            passes::verticalScalar  (g.src(), g.dst(), g, p1); g.swapBuffers();
            passes::horizontalScalar(g.src(), g.dst(), g, p2); g.swapBuffers();
            passes::decayScalar     (g.src(), g.dst(), g, f ); g.swapBuffers();
            g.advanceFrame();
        }

        const auto t1 = std::chrono::steady_clock::now();
        stats_.lastStepMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++stats_.stepsRun;
    }
};

}  // namespace

std::unique_ptr<IBackend> createAvx2Backend() {
    return std::make_unique<Avx2Backend>();
}

}  // namespace br