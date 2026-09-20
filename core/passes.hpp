// ---------------------------------------------------------------------------
// core/passes.hpp
//
// Role in BackendRace: declares the three scalar reference passes.  They are
// implemented in backends/naive_cpu.cpp (the baseline) and are reused as the
// host fallback path by the AVX2 backend on non-x86 targets and by the toy-ISA
// backend for the passes the toy accelerator does not implement.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "core/grid.hpp"

namespace br::passes {

// Pass 1: vertical pair exchange. `parity` selects which rows are "upper".
void verticalScalar(const std::uint8_t* s, std::uint8_t* d,
                    const Grid& g, int parity);

// Pass 2: horizontal pair exchange. `parity` selects which columns are "left".
void horizontalScalar(const std::uint8_t* s, std::uint8_t* d,
                      const Grid& g, int parity);

// Pass 3: per-cell fire decay.
void decayScalar(const std::uint8_t* s, std::uint8_t* d,
                 const Grid& g, std::uint32_t frame);

}  // namespace br::passes