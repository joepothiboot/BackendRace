// ---------------------------------------------------------------------------
// core/backend.hpp
//
// Role in BackendRace: the abstract compute-backend interface plus the factory
// used by the native app, the WASM bindings and the benchmark runner to swap
// implementations live at runtime.  Keys 1..4 in both UIs map directly onto
// BackendId 0..3.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>

#include "core/grid.hpp"

namespace br {

enum class BackendId : int {
    Naive  = 0,
    Avx2   = 1,
    Cuda   = 2,
    ToyIsa = 3,
    Count  = 4,
};

struct BackendStats {
    double        lastStepMs   = 0.0;   // wall-clock of the most recent step
    std::uint64_t stepsRun     = 0;
    std::uint64_t modeledCycles= 0;     // toy-ISA only: simulated accelerator cycles
    std::uint64_t instructions = 0;     // toy-ISA only
};

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual const char* name()        const noexcept = 0;
    virtual const char* description() const noexcept = 0;
    virtual bool        available()   const noexcept = 0;

    // Advance the simulation by one frame (three passes + frame counter).
    virtual void step(Grid& g) = 0;

    // Backends with an expensive transfer boundary (CUDA) override this to keep
    // the data device-resident across `n` steps.
    virtual void stepMany(Grid& g, int n) {
        for (int i = 0; i < n; ++i) step(g);
    }

    virtual BackendStats stats() const noexcept { return stats_; }

protected:
    BackendStats stats_{};
};

std::unique_ptr<IBackend> createNaiveBackend();
std::unique_ptr<IBackend> createAvx2Backend();
std::unique_ptr<IBackend> createToyIsaBackend();
std::unique_ptr<IBackend> createCudaBackend();   // stub when built without CUDA

std::unique_ptr<IBackend> createBackend(BackendId id);

const char* backendName(BackendId id);
bool        cpuSupportsAvx2();

}  // namespace br