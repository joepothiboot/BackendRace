<!--
  backends/toy_isa/isa_spec.md
  Role in BackendRace: normative specification of TSA-32, the toy
  accelerator ISA that the falling-sand gravity kernel is lowered onto by
  compiler.cpp and executed by the cycle-model interpreter in interpreter.cpp.
-->

# TSA-32 — Toy Sand Accelerator, 32 lanes

TSA-32 is a deliberately small, domain-specific SIMD accelerator ISA. It exists
to show the *whole* lowering path: rule → instruction selection → encoding →
cycle-accurate execution → measured throughput. It is not a general CPU; it has
exactly the operations the cell-update rule needs.

## 1. Machine model

| Resource | Count | Width | Notes |
|---|---|---|---|
| Scalar registers `r0..r15` | 16 | 32-bit | addresses, loop counters |
| Vector registers `v0..v7` | 8 | 32 bytes (32 lanes × 8-bit) | one lane = one cell |
| Density ROM | 1 | 16 bytes | hard-wired biased density table |
| Program counter | 1 | 32-bit | instruction index, not byte address |

A vector register holds **exactly one tile row** (32 cells), which is why the
host grid uses 32×32 tiles.

### Address space

The host maps two regions into a single flat byte address space:

```
[0,          size)      SRC region   — read-only
[size,     2*size)      DST region   — read/write
```

Any store outside the DST region raises `STORE_PROTECTION` and halts the
machine. All vector memory accesses must be **32-byte aligned**; a misaligned
access raises `ALIGNMENT` and halts. The compiler only ever emits aligned
addresses (tile base is a multiple of 1024, row offset a multiple of 32).

## 2. Instruction encoding

Fixed-width **64-bit** instruction words, little-endian in the program image:

```
 63    56 55  52 51  48 47  44 43  40 39                      8 7      0
+--------+------+------+------+------+-------------------------+--------+
| opcode |  rd  | rs1  | rs2  | rs3  |        imm32 (s32)      | rsvd=0 |
+--------+------+------+------+------+-------------------------+--------+
    8       4      4      4      4              32                 8
```

* `rd`, `rs1`, `rs2`, `rs3` index the scalar file for scalar opcodes and the
  vector file for vector opcodes (unused fields are 0).
* For `VST` the **`rd` field names the *source* vector register**.
* `imm32` is two's complement.

## 3. Opcode map

### Control / scalar

| Op | Enc | Semantics | Latency |
|---|---|---|---|
| `HALT`  | 0x00 | stop execution | 1 |
| `NOP`   | 0x01 | — | 1 |
| `LI`    | 0x10 | `rd = imm` | 1 |
| `ADD`   | 0x11 | `rd = rs1 + rs2` | 1 |
| `ADDI`  | 0x12 | `rd = rs1 + imm` | 1 |
| `SUB`   | 0x13 | `rd = rs1 - rs2` | 1 |
| `SHLI`  | 0x14 | `rd = rs1 << imm` | 1 |
| `ANDI`  | 0x15 | `rd = rs1 & imm` | 1 |
| `MUL`   | 0x16 | `rd = rs1 * rs2` | 3 |
| `CMPLT` | 0x20 | `rd = (int32)rs1 < (int32)rs2` | 1 |
| `CMPEQ` | 0x21 | `rd = rs1 == rs2` | 1 |
| `BNZ`   | 0x22 | `if (rs1 != 0) pc = imm` | 1 (+2 taken) |
| `BZ`    | 0x23 | `if (rs1 == 0) pc = imm` | 1 (+2 taken) |
| `JMP`   | 0x24 | `pc = imm` | 1 (+2) |

Branch targets are absolute **instruction indices**.

### Vector

| Op | Enc | Semantics (per lane `i`) | Latency |
|---|---|---|---|
| `VLD`     | 0x30 | `vd = mem[rs1 + imm .. +32]` | 4 |
| `VST`     | 0x31 | `mem[rs1 + imm .. +32] = vd` | 1 |
| `VSHUF`   | 0x32 | `vd[i] = vs1[ vs2[i] & 15 ]` (table = low 16 bytes of `vs1`) | 2 |
| `VCMPGT`  | 0x33 | `vd[i] = (int8)vs1[i] > (int8)vs2[i] ? 0xFF : 0x00` | 1 |
| `VCMPEQ`  | 0x34 | `vd[i] = vs1[i] == vs2[i] ? 0xFF : 0x00` | 1 |
| `VAND`    | 0x35 | `vd[i] = vs1[i] & vs2[i]` | 1 |
| `VANDN`   | 0x36 | `vd[i] = (~vs1[i]) & vs2[i]` | 1 |
| `VOR`     | 0x37 | `vd[i] = vs1[i] \| vs2[i]` | 1 |
| `VXOR`    | 0x38 | `vd[i] = vs1[i] ^ vs2[i]` | 1 |
| `VBLEND`  | 0x39 | `vd[i] = (vs3[i] & 0x80) ? vs2[i] : vs1[i]` | 2 |
| `VBCASTB` | 0x3A | `vd[i] = (uint8)imm` | 1 |
| `VMOV`    | 0x3B | `vd = vs1` | 1 |
| `VDENS`   | 0x3C | `vd = densityROM ++ densityROM` (16 B duplicated) | 1 |

`VSHUF` duplicates the x86 `vpshufb` contract when the table register holds two
identical 16-byte halves, which `VDENS` guarantees. `VBLEND` duplicates
`vpblendvb` (high bit of the mask selects).

### Density ROM contents

```
index :  0(Empty) 1(Sand) 2(Water) 3(Fire) 4(Wall)  5..15
value :   0x81     0x83    0x82     0x80    0x7F     0x7F
```

These are `density XOR 0x80`, so the *signed* `VCMPGT` implements an *unsigned*
density comparison — the same trick the AVX2 backend uses.

## 4. Pipeline / cycle model

The interpreter models a **single-issue, in-order, scoreboarded** pipeline:

* one instruction issues per cycle,
* an instruction cannot issue before all of its source registers are ready,
* a result register becomes ready `issueCycle + latency(op)` cycles later,
* a taken branch costs an extra 2 cycles of front-end bubble.

Reported metrics: retired instructions, total cycles, stall cycles, IPC, and
throughput in cells/s at a nominal 1.0 GHz accelerator clock
(`cells / (cycles / 1e9)`).

## 5. Calling convention emitted by `compiler.cpp`

| Reg | Meaning |
|---|---|
| `r0` | current tile index (loop counter) |
| `r1` | end tile index (exclusive) |
| `r2` | tile stride in bytes (1024) |
| `r3` | SRC→DST displacement (= grid size in bytes) |
| `r4` | SRC base of current tile |
| `r5` | SRC base of tile ABOVE |
| `r6` | SRC base of tile BELOW |
| `r7` | DST base of current tile |
| `r8` | loop condition temporary |
| `v6` | density ROM (loaded once by `VDENS`) |
| `v7` | broadcast `Cell::Wall` |

## 6. Lowered kernel: the vertical (gravity) pass

Scalar source rule:

```
out(x,y) = ((y&1)==parity) ? (swapVertical(c, below) ? below : c)
                           : (swapVertical(above, c) ? above : c)
swapVertical(a,b) = (density(a) > density(b)) && a != Wall
```

Lowered, fully unrolled over the 32 rows of a tile, 9 instructions per row:

```
VLD    v0, r4, ly*32          ; a = upper row
VLD    v1, rB, offB           ; b = lower row (tile-local or tile below)
VSHUF  v2, v6, v0             ; density(a)
VSHUF  v3, v6, v1             ; density(b)
VCMPGT v4, v2, v3             ; density(a) > density(b)
VCMPEQ v5, v0, v7             ; a == Wall
VANDN  v4, v5, v4             ; mask = ~isWall(a) & gt
VBLEND v2, v0, v1, v4         ; mask ? b : a
VST    v2, r7, ly*32
```

Rows whose partner lies outside the grid lower to a 2-instruction copy
(`VLD` / `VST`) — the Wall border guarantees those rows are inert.

The compiler specialises three loop bodies (top tile row / interior / bottom
tile row) so that the steady-state body contains no boundary checks at all.

## 7. Not implemented on the accelerator

`swapHorizontal` and `decayFire` require a lane-shifted neighbour network and a
hash unit that TSA-32 does not have. The toy backend therefore runs the gravity
pass on the accelerator and **falls back to the host** for passes 2 and 3 —
exactly what a real offload engine with a limited op set does. The bit-exactness
tests consequently compare the *vertical pass* against the scalar reference.