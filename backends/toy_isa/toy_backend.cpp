// ---------------------------------------------------------------------------
// backends/toy_isa/toy_backend.cpp
//
// Role in BackendRace: adapts the TSA-32 toy accelerator to the IBackend
// interface so it can be selected live from the UI and measured by the
// benchmark runner.
//
// Offload model (deliberately realistic): the accelerator executes the gravity
// pass -- the only pass expressible in TSA-32 -- and the remaining two passes
// fall back to the host scalar implementation.  The compiled program is cached
// per (geometry, parity) so the compile cost is paid once, not per frame.
// ---------------------------------------------------------------------------
#include <chrono>
#include <memory>

#include "backends/toy_isa/isa.hpp"
#include "core/backend.hpp"
#include "core/grid.hpp"
#include "core/passes.hpp"

namespace br {
namespace {

class ToyIsaBackend final : public IBackend {
public:
    const char* name() const noexcept override { return "Toy ISA"; }
    const char* description() const noexcept override {
        return "TSA-32 accelerator: gravity pass lowered to 64-bit VLIW-ish "
               "vector ops, run on a cycle-model interpreter";
    }
    bool available() const noexcept override { return true; }

    void step(Grid& g) override {
        const auto t0 = std::chrono::steady_clock::now();

        const int p1 = static_cast<int>( g.frame()       & 1ull);
        const int p2 = static_cast<int>((g.frame() >> 1) & 1ull);
        const auto f = static_cast<std::uint32_t>(g.frame());

        // ---- pass 1 on the toy accelerator --------------------------------
        ensureProgram(g, p1);
        toy::MemoryImage mem{g.src(), g.dst(), g.bytes()};
        const toy::RunResult r = toy::execute(prog_[p1], mem);
        if (r.ok) {
            stats_.modeledCycles += r.cycles;
            stats_.instructions  += r.instructions;
            lastRun_ = r;
            g.swapBuffers();
        } else {
            // Accelerator fault -> host fallback so the demo keeps running.
            passes::verticalScalar(g.src(), g.dst(), g, p1);
            g.swapBuffers();
        }

        // ---- passes 2 and 3 on the host ------------------------------------
        passes::horizontalScalar(g.src(), g.dst(), g, p2); g.swapBuffers();
        passes::decayScalar     (g.src(), g.dst(), g, f ); g.swapBuffers();
        g.advanceFrame();

        const auto t1 = std::chrono::steady_clock::now();
        stats_.lastStepMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++stats_.stepsRun;
    }

    const toy::RunResult& lastRun() const noexcept { return lastRun_; }

private:
    void ensureProgram(const Grid& g, int parity) {
        if (compiledW_ != g.width() || compiledH_ != g.height()) {
            prog_[0] = toy::compileVerticalPass(g.width(), g.height(), 0);
            prog_[1] = toy::compileVerticalPass(g.width(), g.height(), 1);
            compiledW_ = g.width();
            compiledH_ = g.height();
        }
        (void)parity;
    }

    toy::Program   prog_[2];
    int            compiledW_ = -1;
    int            compiledH_ = -1;
    toy::RunResult lastRun_{};
};

}  // namespace

std::unique_ptr<IBackend> createToyIsaBackend() {
    return std::make_unique<ToyIsaBackend>();
}

}  // namespace br