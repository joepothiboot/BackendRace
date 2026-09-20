// ---------------------------------------------------------------------------
// backends/cuda_stub.cpp
//
// Role in BackendRace: graceful no-CUDA fallback.  When the project is built
// without BR_WITH_CUDA, this provides createCudaBackend() so that every UI,
// the benchmark runner and the tests keep compiling and linking; the backend
// simply reports available() == false and refuses to step.
// ---------------------------------------------------------------------------
#include "core/backend.hpp"

#if !defined(BR_WITH_CUDA)

#include <memory>

namespace br {
namespace {

class CudaUnavailableBackend final : public IBackend {
public:
    const char* name() const noexcept override { return "CUDA GPU"; }
    const char* description() const noexcept override {
        return "unavailable: rebuild with -DBR_WITH_CUDA=ON on a machine with "
               "an NVIDIA GPU + CUDA toolkit";
    }
    bool available() const noexcept override { return false; }
    void step(Grid&) override { /* intentionally does nothing */ }
};

}  // namespace

std::unique_ptr<IBackend> createCudaBackend() {
    return std::make_unique<CudaUnavailableBackend>();
}

}  // namespace br

#endif  // !BR_WITH_CUDA