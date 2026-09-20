// ---------------------------------------------------------------------------
// benchmarks/benchmark_runner.cpp
//
// Role in BackendRace: runs every available backend over an identical,
// deterministically seeded workload, verifies that they all produced the same
// universe (FNV-1a checksum of the final grid), measures cells/s, and prints a
// Markdown results table ready to paste into the README.  It also prints the
// exact perf / Nsight Compute command lines used to produce the profiling
// numbers quoted in the README.
//
// Usage:
//   br_bench [--width N] [--height N] [--steps N] [--warmup N] [--dump-isa]
// ---------------------------------------------------------------------------
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "backends/toy_isa/isa.hpp"
#include "core/backend.hpp"
#include "core/grid.hpp"

namespace {

struct Options {
    int  width   = 512;
    int  height  = 512;
    int  steps   = 200;
    int  warmup  = 20;
    bool dumpIsa = false;
};

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](int def) -> int {
            return (i + 1 < argc) ? std::atoi(argv[++i]) : def;
        };
        if      (a == "--width")    o.width   = next(o.width);
        else if (a == "--height")   o.height  = next(o.height);
        else if (a == "--steps")    o.steps   = next(o.steps);
        else if (a == "--warmup")   o.warmup  = next(o.warmup);
        else if (a == "--dump-isa") o.dumpIsa = true;
        else if (a == "--help") {
            std::printf(
                "usage: br_bench [--width N] [--height N] [--steps N] "
                "[--warmup N] [--dump-isa]\n");
            std::exit(0);
        }
    }
    return o;
}

struct Row {
    std::string   backend;
    std::string   notes;
    bool          ran         = false;
    double        seconds     = 0.0;
    double        cellsPerSec = 0.0;
    double        msPerStep   = 0.0;
    double        speedup     = 0.0;
    bool          matches     = false;
    std::uint64_t checksum    = 0;
    std::uint64_t modelCycles = 0;
};

void seed(br::Grid& g) {
    g.clear();
    g.seedRandom(0xC0FFEEu, 0.22f, 0.14f);
    // A block of fire at the bottom so pass 3 is actually exercised.
    g.paintDisc(g.width() / 2, g.height() - 8, 6, br::Cell::Fire);
}

Row runBackend(br::BackendId id, const Options& o) {
    Row row;
    auto be = br::createBackend(id);
    row.backend = be->name();
    row.notes   = be->description();
    if (!be->available()) {
        row.ran = false;
        return row;
    }

    br::Grid g(o.width, o.height);
    seed(g);

    if (o.warmup > 0) be->stepMany(g, o.warmup);

    seed(g);                                   // identical start state for all
    const auto t0 = std::chrono::steady_clock::now();
    be->stepMany(g, o.steps);
    const auto t1 = std::chrono::steady_clock::now();

    row.ran         = true;
    row.seconds     = std::chrono::duration<double>(t1 - t0).count();
    row.msPerStep   = (row.seconds * 1000.0) / o.steps;
    row.cellsPerSec = (static_cast<double>(g.cells()) *
                       static_cast<double>(o.steps)) / row.seconds;
    row.checksum    = g.checksum();
    row.modelCycles = be->stats().modeledCycles;
    return row;
}

void printMarkdown(const std::vector<Row>& rows, const Options& o) {
    std::printf("\n");
    std::printf("### Measured results (%dx%d grid, %d steps)\n\n",
                o.width, o.height, o.steps);
    std::printf("| Backend | Throughput (Mcell/s) | ms / frame | Speedup vs naive | Bit-exact vs naive | Notes |\n");
    std::printf("|---|---:|---:|---:|:--:|---|\n");
    for (const Row& r : rows) {
        if (!r.ran) {
            std::printf("| %s | _n/a_ | _n/a_ | _n/a_ | _n/a_ | %s |\n",
                        r.backend.c_str(), r.notes.c_str());
            continue;
        }
        std::printf("| %s | %.2f | %.3f | %.2fx | %s | %s |\n",
                    r.backend.c_str(),
                    r.cellsPerSec / 1.0e6,
                    r.msPerStep,
                    r.speedup,
                    r.matches ? "yes" : "**NO**",
                    r.notes.c_str());
    }
    std::printf("\n");
}

void printSampleTable() {
    std::printf(
"### Reference numbers  [SAMPLE - replace with your hardware's actual measurements]\n"
"\n"
"Captured on: Ryzen 9 5900X (Zen 3, 12C/24T, 32 KiB L1d/core) + RTX 3070,\n"
"Ubuntu 22.04, clang 17, CUDA 12.3, 512x512 grid, 200 steps, single thread\n"
"on the CPU backends.\n"
"\n"
"| Backend | Throughput (Mcell/s) | ms / frame | Speedup vs naive | Bit-exact | Notes |\n"
"|---|---:|---:|---:|:--:|---|\n"
"| Naive CPU  |    41.2 | 6.360 |  1.00x | ref  | scalar, generic tiled addressing |\n"
"| AVX2 SIMD  |   612.5 | 0.428 | 14.87x | yes  | 32 cells/lane, aligned tile rows |\n"
"| CUDA GPU   | 11840.0 | 0.022 | 287.4x | yes  | 256 thr/block, device-resident loop |\n"
"| Toy ISA    |     6.1 | 43.10 |  0.15x | yes  | interpreted; model says 1.9 Gcell/s @1 GHz |\n"
"\n"
"[SAMPLE - replace with your hardware's actual measurements]\n"
"\n"
"Toy-ISA model detail (from the cycle-accurate interpreter, vertical pass only):\n"
"  instructions   2,362,368\n"
"  cycles         2,681,092   (IPC 0.88, 318,724 stall cycles)\n"
"  cells/step     262,144  ->  ~1.96 Gcell/s at a nominal 1.0 GHz\n"
"[SAMPLE - replace with your hardware's actual measurements]\n"
"\n");
}

void printProfilingHelp(const Options& o) {
    std::printf(
"### Profiling recipes\n"
"\n"
"#### CPU (`perf`, Linux)\n"
"\n"
"```bash\n"
"# 1) Where does the time go? (expect verticalAvx2/horizontalAvx2 to dominate)\n"
"perf record -g --call-graph=dwarf -- ./build/br_bench --width %d --height %d --steps %d\n"
"perf report --stdio --sort=symbol | head -40\n"
"\n"
"# 2) Is the AVX2 path actually memory bound? Compare IPC and cache misses.\n"
"perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,\\\n"
"LLC-load-misses,branch-misses -- ./build/br_bench --steps %d\n"
"\n"
"# 3) Did we really emit 256-bit ops? (needs a Skylake+ PMU)\n"
"perf stat -e fp_arith_inst_retired.256b_packed_single,\\\n"
"cycle_activity.stalls_l1d_miss -- ./build/br_bench --steps %d\n"
"\n"
"# 4) Per-instruction hot spots inside the kernel\n"
"perf annotate --stdio -s _ZN2br12_GLOBAL__N_113verticalAvx2EPKhPhRKNS_4GridEi\n"
"```\n"
"\n"
"What to look for: L1-dcache-load-misses should stay under ~2%% of loads thanks\n"
"to the 32x32 tiling; if it climbs, the tile no longer fits alongside the halo\n"
"scratch rows. A branch-miss spike means the per-row boundary `if`s were not\n"
"hoisted.\n"
"\n"
"#### GPU (Nsight Compute)\n"
"\n"
"```bash\n"
"# Full section set for the three kernels\n"
"ncu --set full -k regex:'kVertical|kHorizontal|kDecay' \\\n"
"    -o backendrace_profile ./build/br_bench --steps 50\n"
"\n"
"# Focused: memory coalescing + occupancy only\n"
"ncu --section MemoryWorkloadAnalysis --section Occupancy \\\n"
"    -k kVertical ./build/br_bench --steps 20\n"
"\n"
"# Roofline\n"
"ncu --set roofline -k kVertical -o roofline ./build/br_bench --steps 20\n"
"\n"
"# Timeline of H2D/D2H vs kernels (is the PCIe copy dominating?)\n"
"nsys profile -t cuda,nvtx -o backendrace_timeline ./build/br_bench --steps 200\n"
"```\n"
"\n"
"What to look for: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` divided by\n"
"requests should be ~4 sectors/request (perfect 128B coalescing); Achieved\n"
"Occupancy should be >= 85%% at 256 threads/block; and in the nsys timeline the\n"
"two memcpys must be a small fraction of a 200-step run - if they are not, you\n"
"are measuring PCIe, not the kernels.\n"
"\n", o.width, o.height, o.steps, o.steps, o.steps);
}

}  // namespace

int main(int argc, char** argv) {
    const Options o = parseArgs(argc, argv);

    if (o.dumpIsa) {
        const br::toy::Program p =
            br::toy::compileVerticalPass(o.width, o.height, 0);
        std::printf("%s", p.disassemble().c_str());
        return 0;
    }

    std::printf("BackendRace benchmark\n");
    std::printf("  grid   : %d x %d (%d x %d tiles of 32x32)\n",
                o.width, o.height, o.width / 32, o.height / 32);
    std::printf("  steps  : %d (+%d warmup)\n", o.steps, o.warmup);
    std::printf("  x86 AVX2 at runtime: %s\n\n",
                br::cpuSupportsAvx2() ? "yes" : "no");

    std::vector<Row> rows;
    rows.push_back(runBackend(br::BackendId::Naive,  o));
    rows.push_back(runBackend(br::BackendId::Avx2,   o));
    rows.push_back(runBackend(br::BackendId::Cuda,   o));
    rows.push_back(runBackend(br::BackendId::ToyIsa, o));

    const Row& ref = rows[0];
    for (Row& r : rows) {
        if (!r.ran) continue;
        r.matches = (r.checksum == ref.checksum);
        r.speedup = (ref.cellsPerSec > 0.0) ? (r.cellsPerSec / ref.cellsPerSec)
                                            : 0.0;
    }

    printMarkdown(rows, o);

    // Toy-ISA cycle model detail.
    for (const Row& r : rows) {
        if (r.ran && r.modelCycles > 0) {
            const double cells = static_cast<double>(o.width) *
                                 static_cast<double>(o.height) *
                                 static_cast<double>(o.steps);
            const double secsAt1GHz = static_cast<double>(r.modelCycles) / 1.0e9;
            std::printf("Toy-ISA cycle model (vertical pass only): "
                        "%llu cycles -> %.2f Mcell/s at a nominal 1.0 GHz\n\n",
                        static_cast<unsigned long long>(r.modelCycles),
                        (cells / secsAt1GHz) / 1.0e6);
        }
    }

    bool allMatch = true;
    for (const Row& r : rows) if (r.ran && !r.matches) allMatch = false;
    std::printf("Cross-backend determinism: %s\n\n",
                allMatch ? "OK - every backend produced an identical grid"
                         : "FAILED - see the table above");

    printSampleTable();
    printProfilingHelp(o);
    return allMatch ? 0 : 1;
}