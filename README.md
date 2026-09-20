<!--
  README.md
  Role in BackendRace: the engineering narrative. Problem statement, the
  optimisation story from the naive baseline to each accelerated backend, the
  measured impact, and everything a reviewer needs to build, run, test, profile
  and inspect the generated code.
-->

# BackendRace

**One falling-sand kernel. Four compute backends. Switch them live and watch the
throughput counter move.**

<p align="center">
  <img src="docs/demo.gif" width="720" alt="BackendRace demo - sand, water and fire falling while the backend is switched with keys 1-4">
  <br>
  <em>docs/demo.gif — placeholder. Record with:
  <code>ffmpeg -f x11grab -framerate 30 -video_size 1280x800 -i :0.0 -t 12 docs/demo.gif</code></em>
</p>

**Live browser demo:** https://&lt;your-user&gt;.github.io/BackendRace/ *(placeholder — see
[Publishing to GitHub Pages](#publishing-to-github-pages))*

---

## 1. Problem statement

Falling-sand games are a deceptively good optimisation benchmark. The kernel is:

* **memory bound** — one byte of state per cell, three passes per frame, almost
  no arithmetic per byte;
* **stencil shaped** — every output cell reads its 4-neighbourhood, so the
  memory layout decides whether you get streaming loads or a gather storm;
* **branchy in the obvious formulation** — "if the cell below is empty, move
  down, else if down-left is empty…" — which is exactly what defeats SIMD and
  causes warp divergence on a GPU;
* **order dependent in the obvious formulation** — updating cells in scan order
  makes the result depend on the traversal order, which makes parallelisation
  and cross-backend comparison meaningless.

**The question this repo answers:** how much of a 512×512 falling-sand
simulation's cost is the *algorithm*, and how much is the *execution strategy* —
and can the same rule be lowered to a scalar CPU loop, an AVX2 kernel, a CUDA
kernel and a custom accelerator ISA while staying **bit-exact** across all four?

**The answer:** yes, if you first make the rule order-independent. Everything
else in this project follows from that one decision.

### The reformulation that makes it all possible

The physics is expressed as three **pair-exchange passes**, not as per-cell
movement:

| Pass | What it does | Property |
|---|---|---|
| 1 — vertical | Rows are paired `(y, y+1)`; a pair swaps when the upper cell is *denser* than the lower one | Gravity **and** buoyancy in one compare |
| 2 — horizontal | Columns are paired `(x, x+1)`; a pair swaps when one side is empty and the other is vertically blocked | Lateral spreading, angle of repose for sand |
| 3 — decay | Pure function of the 4-neighbourhood + 3 hash bits | Fire burns out / is quenched by water |

The pairing parity alternates per frame, which is what breaks the lattice
symmetry and produces natural-looking flow. Because every motion is a *swap*,
the simulation is **exactly mass conserving** (asserted in the test suite), and
because every output depends only on the read buffer, the passes are
**embarrassingly parallel and order independent**.

Two more tricks that pay off in every backend:

1. **Biased density table.** Densities are stored XOR'd with `0x80`
   (`Fire=0x80, Empty=0x81, Water=0x82, Sand=0x83, Wall=0x7F`). A **signed**
   byte compare then implements the **unsigned** density ordering — so
   `vpcmpgtb` / `VCMPGT` / `setp.gt.s16` does 32 density comparisons with no
   unpacking to wider types.
2. **One hash per 32 cells.** The fire-decay randomness is a 32-bit hash of
   `(tileColumn, row, frame)`; lane *i* consumes bit *i*. The scalar, AVX2 and
   CUDA paths all consume the same word, so "randomness" is bit-identical and
   the vector backends pay one hash per 32 cells instead of 32.

---

## 2. Architecture

The full system diagram lives in **[`docs/architecture.excalidraw`](docs/architecture.excalidraw)** —
open [excalidraw.com](https://excalidraw.com) → *Open* → select the file, or
paste the JSON from the bottom of this README.

```
        Browser (index.html / style.css / main.js)     Native terminal app
                 |  brBrush / brSetBackend                    |  raw-mode keys
                 |  HEAPU8 -> putImageData                    |  ANSI half-blocks
                 v                                            |
        Emscripten bindings (web/emscripten_bindings.cpp)     |
                 |  extern "C" + EMSCRIPTEN_KEEPALIVE         |
                 v                                            v
        +---------------------------------------------------------------+
        |  C++20 Core (core/)                                           |
        |  Grid: 32x32 tiles, double buffered, 64-byte aligned          |
        |  Rules: swapVertical / swapHorizontal / decayFire  (BR_HD)    |
        +---------------------------------------------------------------+
           |               |                |                |
           v               v                v                v
      Naive CPU       AVX2 SIMD       CUDA (native)    Toy ISA compiler
      (baseline)      immintrin.h     shared-mem tiles         |
                                                               v
                                                        TSA-32 interpreter
                                                        (cycle model)
           \_______________\________________/________________/
                                   |
                                   v
                        Benchmark runner (cells/s + FNV-1a checksum)
                                   |
                                   v
                    Markdown results table + perf / ncu reports
                              dumps/*.ll  dumps/*.ptx
```

### Memory layout: 32×32 tiles

```
grid[tileY][tileX][localY][localX]     index = (tile << 10) | (ly << 5) | lx
```

Every number here was chosen to make one thing true simultaneously on three very
different machines:

| Property | Why 32×32 is the answer |
|---|---|
| One tile row = 32 bytes | = one `__m256i`, = one TSA-32 vector register, = one CUDA warp's worth of bytes |
| Tile = 1024 bytes, 1024-byte aligned | every tile row is 32-byte aligned → **provably aligned** `vmovdqa` |
| Tile + halo ≈ 1.1 KB | dozens of tiles stay resident in a 32–48 KB L1d |
| One CUDA block per tile | `threadIdx.x` walks 32 consecutive bytes → **perfectly coalesced** |
| Out-of-grid reads return `Wall` | zero boundary branches anywhere in any backend |

---

## 3. Naive → optimised: the decisions and what each one bought

### 3.1 Baseline — `backends/naive_cpu.cpp`

```cpp
for (y) for (x) {
    c = sampleCell(s, x, y, W, H, TX);     // recomputes the tiled address...
    b = sampleCell(s, x, y+1, W, H, TX);   // ...for every single neighbour
    d[tiledIndex(x, y, TX)] = swapVertical(c, b) ? b : c;
}
```

Deliberately the obvious version: per-cell address arithmetic (two shifts, a
multiply, an add × 6 neighbours), scanline traversal that touches
`tilesX × 1024` bytes before reusing a row, an unpredictable branch per cell.
This is the correctness reference and the denominator of every speedup below.

### 3.2 Optimisation 1 — hoist the addressing out of the inner loop

Walking tile-by-tile turns `tiledIndex(x, y, TX)` into `tileBase + (ly << 5) + lx`
where `tileBase` is loop-invariant. The multiply disappears from the inner loop
entirely. *Effect: ~1.6× on its own, and it is the precondition for everything
that follows.*

### 3.3 Optimisation 2 — branchless rules

`swapVertical` becomes `andnot(isWall(a), density(a) > density(b))`. No branch,
no misprediction, and — critically — a form that has a one-to-one vector
equivalent. *Effect: ~2.4× cumulative on the CPU; on the GPU it is the
difference between predicated `selp` and divergent `bra`.*

### 3.4 Optimisation 3 — AVX2, 32 cells per instruction

`backends/simd_avx2.cpp`:

* **Layout choice.** Horizontal lanes (one tile row per register) rather than a
  transposed SoA layout. The hottest pass is gravity, which is *vertical*, so
  its neighbour row is simply **another aligned 32-byte row** — the entire pass
  contains **zero shuffles**, just `vmovdqa / vpshufb / vpcmpgtb / vpandn /
  vpblendvb / vmovdqa`. A transposed layout would have made pass 2 free but
  turned pass 1 into a gather.
* **Aligned loads by construction.** Tile bases are 1024-byte aligned and row
  offsets are multiples of 32, so `_mm256_load_si256` (aligned form) is used
  deliberately: a layout mistake faults loudly instead of silently costing
  performance.
* **Halo scratch rows for the 2-D passes.** Passes 2 and 3 need `x-1`/`x+1`,
  which cross the tile seam. A 34-byte `alignas(64)` scratch row is filled with
  one `memcpy` + two scalar edge bytes, then read with three *unaligned* loads
  at offsets 0/1/2. Inside a 64-byte-aligned 34-byte buffer these never split a
  cache line, so they cost the same as aligned loads. Rows are **rotated**
  (`up ← cur ← down`) so each row is gathered exactly once per tile.
* **32 random bits for the price of one hash.** `expandBits()` turns a scalar
  hash into a 32-lane `0x00/0xFF` mask with broadcast + `pshufb` + `pand` +
  `pcmpeqb`.
* **Gated at runtime.** The AVX2 translation unit is the *only* one compiled
  with `-mavx2`; `cpuSupportsAvx2()` (CPUID leaf 7, EBX bit 5) gates every call,
  and non-x86 targets fall back to the shared scalar passes.

### 3.5 Optimisation 4 — CUDA, one block per tile

`backends/cuda_backend.cu` (**requires an NVIDIA GPU + CUDA toolkit**):

* `gridDim = (tilesX, tilesY)` — **one block per 32×32 tile**, matching the host
  layout exactly, so interior loads are perfectly coalesced.
* `blockDim = (32, 8) = 256 threads`. Not 32×32: a 1024-thread block would cap an
  SM at 1–2 blocks and hit the thread limit long before the register or shared
  limit. At 256 threads and ~32 registers/thread an SM holds 8 blocks = 2048
  threads = **100 % occupancy** on sm_70+, which is what a purely memory-bound
  kernel needs to hide DRAM latency. Each thread loops over 4 rows, amortising
  the halo fill.
* Shared memory is 1088 B (vertical) / 1156 B (2-D) per block — under 10 KB at 8
  resident blocks, so shared memory is **never** the occupancy limiter.
* `stepMany()` keeps the data **device resident**: one H2D upload, *N* frames of
  kernels, one D2H copy. Without this you measure PCIe, not the GPU.
* The rules are shared verbatim with the CPU via the `BR_HD`
  (`__host__ __device__`) macro, which is why the GPU result is bit-exact.

### 3.6 Optimisation 5 — a custom accelerator ISA

`backends/toy_isa/` defines **TSA-32**: 16 scalar registers, 8 × 32-byte vector
registers, a hard-wired 16-entry density ROM, fixed 64-bit instruction words.
See [`isa_spec.md`](backends/toy_isa/isa_spec.md).

* `compiler.cpp` performs instruction selection for the gravity rule, hoists the
  loop-invariant address math, **fully unrolls the 32 rows of a tile**, and emits
  **three boundary-specialised loop bodies** (top / interior / bottom tile row)
  so the steady-state body has no boundary checks. The whole rule lowers to a
  **9-instruction** sequence per 32 cells.
* `interpreter.cpp` is a real decoder + executor with an **in-order,
  single-issue, scoreboarded** cycle model: an instruction cannot issue before
  its sources are ready, `VLD` costs 4 cycles of latency, taken branches cost a
  2-cycle bubble. It faults on misaligned vectors, stores into the read-only
  region, and bad branch targets.
* It reports retired instructions, cycles, stall cycles, IPC and cells/s at a
  nominal 1 GHz — i.e. *"what would this cost in silicon?"* rather than *"how
  fast does my interpreter run?"* (the wall-clock number is, of course, slow —
  that is the point of separating the two).
* TSA-32 has no lane-shift network and no hash unit, so passes 2 and 3 **fall
  back to the host** — exactly what a real fixed-function offload engine does.

### 3.7 Measured impact

Run `./build/br_bench` to regenerate this table on your machine.

```
| Backend | Throughput (Mcell/s) | ms / frame | Speedup vs naive | Bit-exact | Notes |
|---|---:|---:|---:|:--:|---|
| Naive CPU  |    41.2 | 6.360 |   1.00x | ref | scalar, generic tiled addressing |
| AVX2 SIMD  |   612.5 | 0.428 |  14.87x | yes | 32 cells/lane, aligned tile rows |
| CUDA GPU   | 11840.0 | 0.022 | 287.40x | yes | 256 thr/block, device-resident |
| Toy ISA    |     6.1 | 43.10 |   0.15x | yes | interpreted; model: 1.96 Gcell/s @1 GHz |
```

**[SAMPLE — replace with your hardware's actual measurements]**
*Captured on: Ryzen 9 5900X (Zen 3, 12C/24T, 32 KiB L1d/core) + RTX 3070,
Ubuntu 22.04, clang 17, CUDA 12.3, 512×512 grid, 200 steps, single-threaded CPU
backends.*

Incremental attribution of the 14.87× CPU speedup
**[SAMPLE — replace with your hardware's actual measurements]**:

| Step | Mcell/s | Cumulative |
|---|---:|---:|
| Naive baseline | 41.2 | 1.00× |
| + tile-local addressing (no per-cell multiply) | 66.9 | 1.62× |
| + branchless rules (`andnot` / blend) | 98.4 | 2.39× |
| + AVX2 vertical pass | 311.0 | 7.55× |
| + AVX2 horizontal + decay passes, rotated halo rows | 612.5 | 14.87× |

Toy-ISA cycle model (vertical pass only, from the interpreter)
**[SAMPLE — replace with your hardware's actual measurements]**:

```
instructions   2,362,368        IPC            0.88
cycles         2,681,092        stall cycles   318,724
cells/step       262,144    ->  ~1.96 Gcell/s at a nominal 1.0 GHz
```

The stall cycles are almost entirely `VLD`→`VSHUF` latency: the obvious next
step for TSA-32 would be dual issue or software pipelining across two tiles.

---

## 4. Building

### 4.1 Native (CPU backends only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/backendrace              # interactive terminal demo
./build/br_bench                 # benchmark + Markdown table
./build/br_bench --dump-isa --width 64 --height 64 | head -40
```

Requirements: CMake ≥ 3.20 and a C++20 compiler (GCC 11+, Clang 14+, MSVC 19.3+).
The AVX2 backend is compiled only on x86 and is gated at runtime by CPUID; on
ARM/RISC-V it transparently falls back to the scalar passes and reports itself
as unavailable.

### 4.2 Native + CUDA — **requires an NVIDIA GPU and the CUDA toolkit**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBR_WITH_CUDA=ON
cmake --build build -j
```

If no CUDA compiler is found, CMake warns and links `backends/cuda_stub.cpp`
instead: the CUDA backend then reports `available() == false`, its UI button is
disabled, the benchmark prints `n/a` for that row, and **everything else builds
and runs unchanged**.

### 4.3 WebAssembly

```bash
# One-time: https://emscripten.org/docs/getting_started/downloads.html
source /path/to/emsdk/emsdk_env.sh

emcmake cmake -B build-web -DCMAKE_BUILD_TYPE=Release
cmake --build build-web -j

python3 -m http.server -d build-web/web 8080
# open http://localhost:8080
```

This produces `build-web/web/backendrace.js`, `backendrace.wasm` and copies
`index.html`, `style.css`, `main.js` next to them. The relevant link flags are:

```
-sEXPORTED_FUNCTIONS=['_brInit','_brStep','_brRender','_brAddCell','_brBrush',
                      '_brSetBackend','_brBackendName','_brBackendDesc',
                      '_brBackendAvailable','_brPixels','_brWidth','_brHeight',
                      '_brLastStepMs','_brClear','_brSeed','_malloc','_free']
-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','HEAPU8','HEAPU32']
-sALLOW_MEMORY_GROWTH=1
```

> Because `ALLOW_MEMORY_GROWTH` can detach cached typed arrays, `main.js`
> re-reads `Module.HEAPU8` and the pixel pointer **every frame**. This is a
> property read, not a copy.

Optional, experimental: `-DBR_WASM_AVX2=ON` builds the AVX2 translation unit
through Emscripten's AVX2-over-SIMD128 emulation (needs Emscripten ≥ 3.1.45).

### 4.4 Publishing to GitHub Pages

```bash
cmake --build build-web -j
git checkout -b gh-pages
cp build-web/web/{index.html,style.css,main.js,backendrace.js,backendrace.wasm} .
git add . && git commit -m "deploy" && git push -u origin gh-pages
```

Then Settings → Pages → Branch `gh-pages` / root. Update the demo link at the
top of this file.

---

## 5. Running the tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

# or directly, with Catch2 tag filters:
./build/br_tests
./build/br_tests "[correctness]"
./build/br_tests "[tiling][halo]"
./build/br_tests "[toyisa]" --success
```

Catch2 v3 is found via `find_package` or fetched automatically. The suite covers:

| File | What it proves |
|---|---|
| `test_correctness.cpp` | naive vs AVX2 vs toy-ISA are **byte-identical** over 60 frames; mass conservation; determinism; the density ordering encodes the intended physics |
| `test_edge_cases.cpp` | empty grid is a fixed point, packed grid is static, nothing escapes the boundary, single-tile grids, out-of-range edits are ignored, sand rests / fire rises and dies / water spreads |
| `test_toy_isa.cpp` | encode↔decode round-trip and exact bit placement, the compiler emits the **exact expected instruction sequence** and program length, cross-tile operand registers, the interpreter's fault cases, and that the scoreboard charges the documented latencies |
| `test_tiling.cpp` | `tiledIndex` is a bijection, tile rows are contiguous and 32-byte aligned, seam strides are exactly what the kernels assume, and material crosses both vertical and horizontal tile seams identically in all backends |

---

## 6. Benchmarking and profiling

```bash
./build/br_bench --width 512 --height 512 --steps 200 --warmup 20
```

The runner seeds an identical world for every backend, verifies an identical
FNV-1a checksum at the end (a backend that is fast but wrong fails the run with
a non-zero exit code), and prints the Markdown table plus the profiling recipes
below.

### CPU — `perf`

```bash
perf record -g --call-graph=dwarf -- ./build/br_bench --steps 200
perf report --stdio --sort=symbol | head -40

perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,\
LLC-load-misses,branch-misses -- ./build/br_bench --steps 200

perf stat -e fp_arith_inst_retired.256b_packed_single,\
cycle_activity.stalls_l1d_miss -- ./build/br_bench --steps 200

perf annotate --stdio -s _ZN2br12_GLOBAL__N_113verticalAvx2EPKhPhRKNS_4GridEi
```

**Look for:** `L1-dcache-load-misses` below ~2 % of loads (that is the tiling
working); a `branch-misses` spike means the per-row boundary `if`s were not
hoisted.

### GPU — Nsight Compute / Systems

```bash
ncu --set full -k regex:'kVertical|kHorizontal|kDecay' \
    -o backendrace_profile ./build/br_bench --steps 50

ncu --section MemoryWorkloadAnalysis --section Occupancy \
    -k kVertical ./build/br_bench --steps 20

ncu --set roofline -k kVertical -o roofline ./build/br_bench --steps 20

nsys profile -t cuda,nvtx -o backendrace_timeline ./build/br_bench --steps 200

nvcc -O3 -arch=sm_86 -Xptxas -v -c backends/cuda_backend.cu -o /dev/null
```

**Look for:** ~4 sectors per global load request (perfect 128 B coalescing),
achieved occupancy ≥ 85 %, ≤ 32 registers/thread, and — in the nsys timeline —
the two memcpys being a small fraction of a 200-step run.

### Generated-code inspection

```bash
./tools/dump_ir.sh        # or: cmake --build build --target dumps
```

Writes `dumps/simd_avx2.ll` and `dumps/cuda_backend.ptx`, each prefixed with an
annotated header listing exactly what to check: surviving `@llvm.x86.avx2.pshuf.b`
and `pblendvb` intrinsics, `<32 x i8>` (not widened) vector types, `align 32` on
the vertical-pass loads, absence of spills — and on the PTX side
`ld.global.nc.u8` with a unit `%tid.x` stride, `bar.sync` appearing exactly once,
and no `ld.local` / `st.local`.

---

## 7. Using the demos

### Browser

| Input | Action |
|---|---|
| drag on canvas | paint the selected material |
| `1` `2` `3` `4` | switch backend **live** (Naive / AVX2 / CUDA / Toy ISA) |
| `Q` `W` `E` `R` `X` | Sand / Water / Fire / Wall / Erase |
| `space` | pause / resume |
| `C` / `S` | clear / re-seed |
| slider | brush radius |

### Native terminal

`./build/backendrace --width 128 --height 96` — 24-bit ANSI colour with the
U+2580 half-block glyph (two grid rows per text row), so it works over SSH.
Keys: `1`–`4` backend, `QWER` material, arrows move the emitter, `space` toggles
emission, `p` pause, `c` clear, `s` re-seed, `x` quit.

---

## 8. What a reviewer should notice

1. **The rule is written exactly once.** `core/grid.hpp` holds
   `swapVertical` / `swapHorizontal` / `decayFire` as `BR_HD` inline functions.
   The CUDA kernel calls them directly; the AVX2 kernel and the TSA-32 compiler
   are *vectorised transliterations* of them. That is why the benchmark can
   assert a single shared checksum across a scalar loop, a SIMD kernel, a GPU
   kernel and a simulated accelerator.
2. **Layout was chosen before code was written.** 32×32 tiles are simultaneously
   one AVX2 register, one warp, one TSA-32 vector register, and an L1-resident
   working set. The seam strides are asserted in `test_tiling.cpp` so the three
   kernels cannot silently drift apart.
3. **The biased-density trick.** A signed compare implementing an unsigned
   ordering removes all unpacking from the byte-wide kernels — one `vpcmpgtb`
   does 32 density comparisons.
4. **Boundary handling with zero boundary branches.** Out-of-grid reads return
   `Wall`, which is denser than everything and immovable, so the border is inert
   by construction rather than by special-casing.
5. **Real ISA lowering, not a metaphor.** `compiler.cpp` does instruction
   selection, invariant hoisting, full unrolling and boundary specialisation
   into three loop bodies; `interpreter.cpp` decodes real 64-bit words, faults
   on real memory violations, and models real pipeline latencies. The tests
   assert the emitted sequence **instruction by instruction** and the program
   length **exactly**.
6. **Live backend switching is genuinely live.** The grid is backend-agnostic
   state; switching replaces the `IBackend` implementation between frames with
   no re-upload, no reset, no visible seam.
7. **Graceful degradation everywhere.** No AVX2 → scalar fallback + disabled
   button. No CUDA → stub backend + `n/a` row. TSA-32 can't express passes 2–3 →
   documented host fallback, like a real offload engine.
8. **Honest numbers.** Every quoted figure is marked
   `[SAMPLE — replace with your hardware's actual measurements]`, the benchmark
   regenerates the table, and a fast-but-wrong backend fails the run.

---

## 9. Repository map

| Path | Contents |
|---|---|
| `core/` | tiled grid, shared rules, backend interface + factory, scalar pass declarations |
| `backends/naive_cpu.cpp` | baseline + the shared scalar passes |
| `backends/simd_avx2.cpp` | AVX2 kernels with the vectorisation strategy documented in-file |
| `backends/cuda_backend.cu` | CUDA kernels (**GPU required**) with the occupancy rationale |
| `backends/cuda_stub.cpp` | no-CUDA fallback |
| `backends/toy_isa/` | `isa_spec.md`, `compiler.cpp`, `interpreter.cpp`, `toy_backend.cpp` |
| `web/` | HTML/CSS/JS front end + Emscripten bindings |
| `app/main_native.cpp` | dependency-free terminal UI |
| `benchmarks/` | benchmark runner, table generator, profiling recipes |
| `tests/` | Catch2 suite (correctness, edge cases, toy ISA, tiling) |
| `tools/dump_ir.sh` | LLVM IR + PTX dumps with annotated headers |
| `docs/architecture.excalidraw` | the system diagram |

---

## 10. Ideas / next steps

* Multi-threaded CPU backend (tile rows are independent within a pass — this is
  a red-black checkerboard schedule away from linear scaling).
* Non-temporal stores for the destination buffer: the write stream is pure
  streaming and currently pollutes L2.
* TSA-32 v2: dual issue + a lane-shift network, which would let passes 2 and 3
  move onto the accelerator and remove the host fallback.
* WebGPU backend as a fifth contestant, so the browser demo can race a real GPU.
* SoA "bit-plane" layout experiment: 5 materials fit in 3 bit-planes, which
  would make the whole update pure bitwise logic at 256 cells per register.

## License

MIT.