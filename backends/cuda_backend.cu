// ---------------------------------------------------------------------------
// backends/cuda_backend.cu
//
// Role in BackendRace: the GPU backend.  *** REQUIRES AN NVIDIA GPU AND THE
// CUDA TOOLKIT TO BUILD AND RUN. ***  This file is only added to the build when
// `cmake -DBR_WITH_CUDA=ON` finds a CUDA compiler; otherwise
// backends/cuda_stub.cpp provides an `available() == false` placeholder and the
// rest of the project builds and runs untouched.
//
// It reuses the *same* inline rules from core/grid.hpp (they are marked
// __host__ __device__ via the BR_HD macro), so the GPU result is bit-exact with
// the naive CPU result -- the benchmark asserts this with a checksum.
//
// ===================== THREAD BLOCK / OCCUPANCY RATIONALE ==================
//
// Grid mapping : gridDim = (tilesX, tilesY), i.e. ONE BLOCK PER 32x32 TILE.
//                This matches the host memory layout exactly, so a block's
//                tile is 1024 contiguous bytes -> every global load in the
//                interior is perfectly coalesced (threadIdx.x = localX walks
//                32 consecutive bytes = one 32-byte sector).
//
// Block shape  : dim3(32, 8) = 256 threads.
//                * 32 in x so a warp covers one whole tile row.
//                * 8 in y (not 32) because a 32x32 = 1024-thread block would
//                  allow only 1-2 blocks per SM, and with 1024 threads we hit
//                  the per-SM thread cap long before we hit the register or
//                  shared-memory cap. With 256 threads and ~32 registers/thread
//                  an SM can hold 8 blocks = 2048 threads = 100% occupancy on
//                  sm_70+, which is what this fully memory-bound kernel needs
//                  to hide DRAM latency.
//                * Each thread loops over 4 rows (ly += blockDim.y), which also
//                  amortises the shared-memory halo fill over more work.
//
// Shared memory: vertical pass 34*32 = 1088 B/block, the 2-D passes
//                34*34 = 1156 B/block.  At 8 resident blocks that is < 10 KB of
//                the 64-100 KB budget, so shared memory is never the occupancy
//                limiter -- deliberate.
//
// Halo         : the ring is fetched through sampleCell(), which returns Wall
//                out of bounds; this is the same convention the CPU backends
//                use, so no special-case boundary code exists anywhere.
// ---------------------------------------------------------------------------
#include <chrono>
#include <cstdio>
#include <memory>

#include <cuda_runtime.h>

#include "core/backend.hpp"
#include "core/grid.hpp"

#define BR_CUDA_CHECK(expr)                                                    \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[BackendRace/CUDA] %s failed at %s:%d: %s\n", \
                         #expr, __FILE__, __LINE__, cudaGetErrorString(err__)); \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace br {
namespace {

__device__ __forceinline__ std::uint8_t sampleG(const std::uint8_t* s,
                                                int x, int y,
                                                int W, int H, int tilesX) {
    return sampleCell(s, x, y, W, H, tilesX);
}

// --------------------------------------------------------------------------
// PASS 1 -- vertical pair exchange.  Shared tile is [34][32]: one halo row
// above (row 31 of the tile above) and one below (row 0 of the tile below).
// --------------------------------------------------------------------------
__global__ void kVertical(const std::uint8_t* __restrict__ s,
                          std::uint8_t* __restrict__ d,
                          int tilesX, int tilesY, int W, int H, int parity) {
    __shared__ std::uint8_t sh[34][32];

    const int tx = blockIdx.x, ty = blockIdx.y;
    const int lx = threadIdx.x;
    const std::size_t base =
        (static_cast<std::size_t>(ty * tilesX + tx)) << 10;

    for (int ly = threadIdx.y; ly < 32; ly += blockDim.y) {
        sh[ly + 1][lx] = s[base + (static_cast<std::size_t>(ly) << 5) + lx];
    }
    if (threadIdx.y == 0) {
        sh[0][lx] = sampleG(s, (tx << 5) + lx, (ty << 5) - 1, W, H, tilesX);
    }
    if (threadIdx.y == 1) {
        sh[33][lx] = sampleG(s, (tx << 5) + lx, (ty << 5) + 32, W, H, tilesX);
    }
    __syncthreads();

    for (int ly = threadIdx.y; ly < 32; ly += blockDim.y) {
        const int y = (ty << 5) + ly;
        const std::uint8_t c = sh[ly + 1][lx];
        std::uint8_t out = c;
        if ((y & 1) == parity) {
            const std::uint8_t b = sh[ly + 2][lx];
            if (swapVertical(c, b)) out = b;
        } else {
            const std::uint8_t a = sh[ly][lx];
            if (swapVertical(a, c)) out = a;
        }
        d[base + (static_cast<std::size_t>(ly) << 5) + lx] = out;
    }
}

// --------------------------------------------------------------------------
// Cooperative 34x34 halo fill: coalesced interior + a 132-element ring.
// --------------------------------------------------------------------------
__device__ __forceinline__ void loadTileWithRing(const std::uint8_t* __restrict__ s,
                                                 std::uint8_t sh[34][34],
                                                 int tx, int ty,
                                                 int W, int H, int tilesX) {
    const std::size_t base =
        (static_cast<std::size_t>(ty * tilesX + tx)) << 10;
    const int lx    = threadIdx.x;
    const int tid   = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthr  = blockDim.x * blockDim.y;

    for (int ly = threadIdx.y; ly < 32; ly += blockDim.y) {
        sh[ly + 1][lx + 1] =
            s[base + (static_cast<std::size_t>(ly) << 5) + lx];
    }
    // ring: 34 top + 34 bottom + 32 left + 32 right = 132 entries
    for (int i = tid; i < 132; i += nthr) {
        int r, c;
        if      (i <  34) { r = 0;            c = i;        }
        else if (i <  68) { r = 33;           c = i - 34;   }
        else if (i < 100) { r = i - 68 + 1;   c = 0;        }
        else              { r = i - 100 + 1;  c = 33;       }
        sh[r][c] = sampleG(s, (tx << 5) + c - 1, (ty << 5) + r - 1,
                           W, H, tilesX);
    }
}

// --------------------------------------------------------------------------
// PASS 2 -- horizontal pair exchange.
// --------------------------------------------------------------------------
__global__ void kHorizontal(const std::uint8_t* __restrict__ s,
                            std::uint8_t* __restrict__ d,
                            int tilesX, int tilesY, int W, int H, int parity) {
    __shared__ std::uint8_t sh[34][34];

    const int tx = blockIdx.x, ty = blockIdx.y;
    const int lx = threadIdx.x;
    const std::size_t base =
        (static_cast<std::size_t>(ty * tilesX + tx)) << 10;

    loadTileWithRing(s, sh, tx, ty, W, H, tilesX);
    __syncthreads();

    for (int ly = threadIdx.y; ly < 32; ly += blockDim.y) {
        const int x = (tx << 5) + lx;
        const std::uint8_t c = sh[ly + 1][lx + 1];
        std::uint8_t out = c;

        if ((x & 1) == parity) {
            const std::uint8_t b = sh[ly + 1][lx + 2];
            if (swapHorizontal(c, b,
                               sh[ly + 2][lx + 1], sh[ly + 2][lx + 2],
                               sh[ly    ][lx + 1], sh[ly    ][lx + 2])) out = b;
        } else {
            const std::uint8_t a = sh[ly + 1][lx];
            if (swapHorizontal(a, c,
                               sh[ly + 2][lx], sh[ly + 2][lx + 1],
                               sh[ly    ][lx], sh[ly    ][lx + 1])) out = a;
        }
        d[base + (static_cast<std::size_t>(ly) << 5) + lx] = out;
    }
}

// --------------------------------------------------------------------------
// PASS 3 -- fire decay.
// --------------------------------------------------------------------------
__global__ void kDecay(const std::uint8_t* __restrict__ s,
                       std::uint8_t* __restrict__ d,
                       int tilesX, int tilesY, int W, int H,
                       unsigned int frame) {
    __shared__ std::uint8_t sh[34][34];

    const int tx = blockIdx.x, ty = blockIdx.y;
    const int lx = threadIdx.x;
    const std::size_t base =
        (static_cast<std::size_t>(ty * tilesX + tx)) << 10;

    loadTileWithRing(s, sh, tx, ty, W, H, tilesX);
    __syncthreads();

    for (int ly = threadIdx.y; ly < 32; ly += blockDim.y) {
        const int x = (tx << 5) + lx;
        const int y = (ty << 5) + ly;
        const std::uint8_t out =
            decayFire(sh[ly + 1][lx + 1],
                      sh[ly + 1][lx    ], sh[ly + 1][lx + 2],
                      sh[ly    ][lx + 1], sh[ly + 2][lx + 1],
                      randBit(x, y, frame, kSaltA),
                      randBit(x, y, frame, kSaltB),
                      randBit(x, y, frame, kSaltC));
        d[base + (static_cast<std::size_t>(ly) << 5) + lx] = out;
    }
}

// --------------------------------------------------------------------------
// Host-side backend object.
// --------------------------------------------------------------------------
class CudaBackend final : public IBackend {
public:
    CudaBackend() {
        int count = 0;
        if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                std::snprintf(desc_, sizeof(desc_),
                              "%s, sm_%d%d, %d SMs, blocks of 32x8",
                              prop.name, prop.major, prop.minor,
                              prop.multiProcessorCount);
                ok_ = true;
            }
        }
        if (!ok_) {
            std::snprintf(desc_, sizeof(desc_),
                          "no CUDA device found at runtime");
        }
    }

    ~CudaBackend() override { release(); }

    const char* name() const noexcept override { return "CUDA GPU"; }
    const char* description() const noexcept override { return desc_; }
    bool available() const noexcept override { return ok_; }

    void step(Grid& g) override { stepMany(g, 1); }

    // Device-resident loop: one H2D upload, N frames on the GPU, one D2H copy.
    void stepMany(Grid& g, int n) override {
        if (!ok_ || n <= 0) return;
        const auto t0 = std::chrono::steady_clock::now();
        if (!ensureBuffers(g.bytes())) { ok_ = false; return; }
        if (!runFrames(g, n))          { ok_ = false; return; }
        const auto t1 = std::chrono::steady_clock::now();
        stats_.lastStepMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / n;
        stats_.stepsRun += static_cast<std::uint64_t>(n);
    }

private:
    bool ensureBuffers(std::size_t bytes) {
        if (bytes_ == bytes && dA_ && dB_) return true;
        release();
        BR_CUDA_CHECK(cudaMalloc(&dA_, bytes));
        BR_CUDA_CHECK(cudaMalloc(&dB_, bytes));
        bytes_ = bytes;
        return true;
    }

    void release() {
        if (dA_) cudaFree(dA_);
        if (dB_) cudaFree(dB_);
        dA_ = dB_ = nullptr;
        bytes_ = 0;
    }

    bool runFrames(Grid& g, int n) {
        const int TX = g.tilesX(), TY = g.tilesY();
        const int W = g.width(), H = g.height();
        const dim3 block(32, 8);
        const dim3 grid(static_cast<unsigned>(TX), static_cast<unsigned>(TY));

        BR_CUDA_CHECK(cudaMemcpy(dA_, g.src(), g.bytes(),
                                 cudaMemcpyHostToDevice));

        std::uint8_t* a = dA_;
        std::uint8_t* b = dB_;
        for (int i = 0; i < n; ++i) {
            const int p1 = static_cast<int>( g.frame()       & 1ull);
            const int p2 = static_cast<int>((g.frame() >> 1) & 1ull);
            const auto f = static_cast<unsigned>(g.frame());

            kVertical  <<<grid, block>>>(a, b, TX, TY, W, H, p1); std::swap(a, b);
            kHorizontal<<<grid, block>>>(a, b, TX, TY, W, H, p2); std::swap(a, b);
            kDecay     <<<grid, block>>>(a, b, TX, TY, W, H, f ); std::swap(a, b);
            g.advanceFrame();
        }
        BR_CUDA_CHECK(cudaGetLastError());
        BR_CUDA_CHECK(cudaDeviceSynchronize());
        BR_CUDA_CHECK(cudaMemcpy(g.dst(), a, g.bytes(),
                                 cudaMemcpyDeviceToHost));
        g.swapBuffers();
        return true;
    }

    std::uint8_t* dA_ = nullptr;
    std::uint8_t* dB_ = nullptr;
    std::size_t   bytes_ = 0;
    bool          ok_ = false;
    char          desc_[192] = {0};
};

}  // namespace

std::unique_ptr<IBackend> createCudaBackend() {
    return std::make_unique<CudaBackend>();
}

}  // namespace br