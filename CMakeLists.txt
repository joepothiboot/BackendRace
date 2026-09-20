# ---------------------------------------------------------------------------
# CMakeLists.txt
#
# Role in BackendRace: one build description for three very different targets.
#
#   1. native desktop   : `backendrace`  (terminal UI, optional CUDA backend)
#   2. native benchmark : `br_bench`     (+ `br_tests` with Catch2)
#   3. WebAssembly      : `backendrace.js` / `.wasm` via Emscripten
#
# Native:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Native + CUDA (needs an NVIDIA GPU and the CUDA toolkit):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DBR_WITH_CUDA=ON
#
# Web:
#   emcmake cmake -B build-web -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-web -j
#   python3 -m http.server -d build-web/web 8080
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(BackendRace VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BR_WITH_CUDA   "Build the CUDA backend (requires an NVIDIA GPU + toolkit)" OFF)
option(BR_BUILD_TESTS "Build the Catch2 test suite" ON)
option(BR_BUILD_BENCH "Build the benchmark runner"  ON)
option(BR_BUILD_APP   "Build the native terminal app" ON)

include(CheckCXXCompilerFlag)
include(CheckLanguage)

# ---------------------------------------------------------------------------
# Target detection
# ---------------------------------------------------------------------------
set(BR_X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
  set(BR_X86 ON)
endif()
if(EMSCRIPTEN)
  set(BR_X86 OFF)
endif()

message(STATUS "BackendRace: system=${CMAKE_SYSTEM_NAME} proc=${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "BackendRace: x86 target        = ${BR_X86}")
message(STATUS "BackendRace: emscripten target = ${EMSCRIPTEN}")
message(STATUS "BackendRace: CUDA requested    = ${BR_WITH_CUDA}")

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(BR_CORE_SOURCES
  core/grid.cpp
  core/backend.cpp
  backends/naive_cpu.cpp
  backends/simd_avx2.cpp
  backends/toy_isa/compiler.cpp
  backends/toy_isa/interpreter.cpp
  backends/toy_isa/toy_backend.cpp
)

set(BR_CUDA_ENABLED OFF)
if(BR_WITH_CUDA AND NOT EMSCRIPTEN)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70 75 86 89)
    endif()
    list(APPEND BR_CORE_SOURCES backends/cuda_backend.cu)
    set(BR_CUDA_ENABLED ON)
    message(STATUS "BackendRace: CUDA backend ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING "BR_WITH_CUDA=ON but no CUDA compiler was found; "
                    "falling back to the stub backend.")
  endif()
endif()

# The stub is always compiled; its body is #if'd out when CUDA is enabled.
list(APPEND BR_CORE_SOURCES backends/cuda_stub.cpp)

add_library(br_core STATIC ${BR_CORE_SOURCES})
target_include_directories(br_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(BR_CUDA_ENABLED)
  target_compile_definitions(br_core PUBLIC BR_WITH_CUDA=1)
  set_target_properties(br_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# --- optimisation flags ----------------------------------------------------
if(MSVC)
  target_compile_options(br_core PRIVATE /O2 /W4 /permissive-)
else()
  target_compile_options(br_core PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# --- AVX2 flags applied to ONE translation unit only -----------------------
# The baseline stays portable; only backends/simd_avx2.cpp gets -mavx2, and a
# runtime CPUID check in core/backend.cpp gates every call into it.
if(BR_X86)
  if(MSVC)
    set(BR_AVX2_FLAGS /arch:AVX2)
  else()
    check_cxx_compiler_flag("-mavx2" BR_HAS_MAVX2)
    if(BR_HAS_MAVX2)
      set(BR_AVX2_FLAGS -mavx2 -mfma)
    endif()
  endif()
  if(BR_AVX2_FLAGS)
    set_source_files_properties(backends/simd_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "${BR_AVX2_FLAGS}")
    message(STATUS "BackendRace: AVX2 flags for simd_avx2.cpp = ${BR_AVX2_FLAGS}")
  endif()
else()
  message(STATUS "BackendRace: non-x86 target, AVX2 backend compiles to the "
                 "scalar fallback")
endif()

# ---------------------------------------------------------------------------
# TARGET 3: WebAssembly (Emscripten)
# ---------------------------------------------------------------------------
if(EMSCRIPTEN)
  add_executable(backendrace_wasm web/emscripten_bindings.cpp)
  target_link_libraries(backendrace_wasm PRIVATE br_core)
  set_target_properties(backendrace_wasm PROPERTIES
    OUTPUT_NAME "backendrace"
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/web")

  target_link_options(backendrace_wasm PRIVATE
    "-O3"
    "-sWASM=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sINITIAL_MEMORY=67108864"
    "-sEXPORTED_FUNCTIONS=['_brInit','_brStep','_brRender','_brAddCell','_brBrush','_brSetBackend','_brBackendName','_brBackendDesc','_brBackendAvailable','_brPixels','_brWidth','_brHeight','_brLastStepMs','_brClear','_brSeed','_malloc','_free']"
    "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','HEAPU8','HEAPU32']"
    "-sENVIRONMENT=web"
    "-sFILESYSTEM=0"
    "-sASSERTIONS=0"
    "--no-entry"
  )

  # Experimental: Emscripten >= 3.1.45 can emulate AVX2 on top of SIMD128.
  # Enable with -DBR_WASM_AVX2=ON and expect a large but sub-native speedup.
  option(BR_WASM_AVX2 "Try Emscripten's AVX2-over-SIMD128 emulation" OFF)
  if(BR_WASM_AVX2)
    set_source_files_properties(backends/simd_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "-msimd128;-mavx2")
    target_link_options(backendrace_wasm PRIVATE "-msimd128")
    message(STATUS "BackendRace: WASM AVX2 emulation ENABLED (experimental)")
  endif()

  foreach(asset index.html style.css main.js)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/web/${asset}
                   ${CMAKE_BINARY_DIR}/web/${asset} COPYONLY)
  endforeach()

  message(STATUS "BackendRace: WASM output -> ${CMAKE_BINARY_DIR}/web/")
endif()

# ---------------------------------------------------------------------------
# TARGET 1: native desktop app
# ---------------------------------------------------------------------------
if(BR_BUILD_APP AND NOT EMSCRIPTEN)
  add_executable(backendrace app/main_native.cpp)
  target_link_libraries(backendrace PRIVATE br_core)
endif()

# ---------------------------------------------------------------------------
# TARGET 2: benchmark runner
# ---------------------------------------------------------------------------
if(BR_BUILD_BENCH AND NOT EMSCRIPTEN)
  add_executable(br_bench benchmarks/benchmark_runner.cpp)
  target_link_libraries(br_bench PRIVATE br_core)
endif()

# ---------------------------------------------------------------------------
# Tests (Catch2 v3)
# ---------------------------------------------------------------------------
if(BR_BUILD_TESTS AND NOT EMSCRIPTEN)
  enable_testing()
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.5.2
      GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(Catch2)
  endif()

  add_executable(br_tests
    tests/test_correctness.cpp
    tests/test_edge_cases.cpp
    tests/test_toy_isa.cpp
    tests/test_tiling.cpp)
  target_link_libraries(br_tests PRIVATE br_core Catch2::Catch2WithMain)

  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
  include(CTest)
  if(EXISTS "${Catch2_SOURCE_DIR}/extras/Catch.cmake")
    include(Catch)
    catch_discover_tests(br_tests)
  else()
    add_test(NAME br_tests COMMAND br_tests)
  endif()
endif()

# ---------------------------------------------------------------------------
# Convenience target: IR / PTX dumps
# ---------------------------------------------------------------------------
if(UNIX AND NOT EMSCRIPTEN)
  add_custom_target(dumps
    COMMAND ${CMAKE_COMMAND} -E env
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/dump_ir.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Dumping LLVM IR (AVX2) and PTX (CUDA) into dumps/")
endif()