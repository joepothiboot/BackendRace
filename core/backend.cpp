// ---------------------------------------------------------------------------
// core/backend.cpp
//
// Role in BackendRace: backend factory + x86 runtime feature detection.  The
// AVX2 translation unit is compiled with -mavx2, so we must never *call* into
// it on a machine without AVX2; this file (compiled with baseline flags) is the
// gatekeeper.
// ---------------------------------------------------------------------------
#include "core/backend.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#  include <intrin.h>
#endif

namespace br {

bool cpuSupportsAvx2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;     // EBX bit 5 = AVX2
#  elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#  else
    return false;
#  endif
#else
    return false;
#endif
}

const char* backendName(BackendId id) {
    switch (id) {
        case BackendId::Naive:  return "Naive CPU";
        case BackendId::Avx2:   return "AVX2 SIMD";
        case BackendId::Cuda:   return "CUDA GPU";
        case BackendId::ToyIsa: return "Toy ISA";
        default:                return "?";
    }
}

std::unique_ptr<IBackend> createBackend(BackendId id) {
    switch (id) {
        case BackendId::Naive:  return createNaiveBackend();
        case BackendId::Avx2:   return createAvx2Backend();
        case BackendId::Cuda:   return createCudaBackend();
        case BackendId::ToyIsa: return createToyIsaBackend();
        default:                return createNaiveBackend();
    }
}

}  // namespace br