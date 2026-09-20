// ---------------------------------------------------------------------------
// tests/test_toy_isa.cpp
//
// Role in BackendRace: unit tests for the TSA-32 toolchain itself.
//
//   * instruction encode/decode round-trips through the 64-bit word format
//   * the compiler emits EXACTLY the expected instruction sequence for the
//     update rule (opcode-by-opcode, operand-by-operand), and the expected
//     program length for known geometries/parities
//   * boundary specialisation really removes the boundary work from the
//     steady-state loop body
//   * the interpreter executes real programs, faults on misaligned accesses,
//     read-only stores and bad branch targets
//   * the scoreboard cycle model charges the documented latencies
//   * VSHUF + VCMPGT + VANDN really implement br::swapVertical()
// ---------------------------------------------------------------------------
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>
#include <vector>

#include "backends/toy_isa/isa.hpp"
#include "core/grid.hpp"

using br::toy::Instr;
using br::toy::Op;
using br::toy::Program;

namespace {

Instr instrAt(const Program& p, std::size_t i) { return p.at(i); }

// Convenience: assert a single instruction's fields.
void expectInstr(const Program& p, std::size_t idx, Op op,
                 int rd, int rs1, int rs2, int rs3, std::int32_t imm) {
    const Instr in = instrAt(p, idx);
    INFO("at index " << idx << " expected " << br::toy::opName(op)
         << " but found " << br::toy::opName(in.op));
    REQUIRE(in.op == op);
    REQUIRE(static_cast<int>(in.rd)  == rd);
    REQUIRE(static_cast<int>(in.rs1) == rs1);
    REQUIRE(static_cast<int>(in.rs2) == rs2);
    REQUIRE(static_cast<int>(in.rs3) == rs3);
    REQUIRE(in.imm == imm);
}

std::size_t countOp(const Program& p, Op op) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (p.at(i).op == op) ++n;
    }
    return n;
}

}  // namespace

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

TEST_CASE("instruction encoding round-trips", "[toyisa][encoding]") {
    Instr in;
    in.op  = Op::Vblend;
    in.rd  = 5;
    in.rs1 = 1;
    in.rs2 = 2;
    in.rs3 = 3;
    in.imm = -123456;

    const std::uint64_t w = br::toy::encode(in);
    const Instr out = br::toy::decode(w);

    REQUIRE(out.op  == in.op);
    REQUIRE(out.rd  == in.rd);
    REQUIRE(out.rs1 == in.rs1);
    REQUIRE(out.rs2 == in.rs2);
    REQUIRE(out.rs3 == in.rs3);
    REQUIRE(out.imm == in.imm);

    // Field placement must match isa_spec.md exactly.
    REQUIRE(((w >> 56) & 0xFF) == static_cast<std::uint64_t>(Op::Vblend));
    REQUIRE(((w >> 52) & 0x0F) == 5u);
    REQUIRE(((w >> 48) & 0x0F) == 1u);
    REQUIRE(((w >> 44) & 0x0F) == 2u);
    REQUIRE(((w >> 40) & 0x0F) == 3u);
    REQUIRE((w & 0xFFull) == 0u);       // reserved byte stays zero
}

TEST_CASE("extreme immediates survive encoding", "[toyisa][encoding]") {
    for (std::int32_t imm : {std::int32_t(0), std::int32_t(1), std::int32_t(-1),
                             std::int32_t(2147483647),
                             std::int32_t(-2147483647 - 1),
                             std::int32_t(1024), std::int32_t(-2048)}) {
        Instr in;
        in.op  = Op::Li;
        in.rd  = 7;
        in.imm = imm;
        REQUIRE(br::toy::decode(br::toy::encode(in)).imm == imm);
    }
}

TEST_CASE("documented latencies and op classes", "[toyisa][encoding]") {
    REQUIRE(br::toy::latencyOf(Op::Vld)    == 4);
    REQUIRE(br::toy::latencyOf(Op::Vshuf)  == 2);
    REQUIRE(br::toy::latencyOf(Op::Vblend) == 2);
    REQUIRE(br::toy::latencyOf(Op::Mul)    == 3);
    REQUIRE(br::toy::latencyOf(Op::Add)    == 1);

    REQUIRE(br::toy::isVectorOp(Op::Vld));
    REQUIRE(br::toy::isVectorOp(Op::Vdens));
    REQUIRE_FALSE(br::toy::isVectorOp(Op::Add));
    REQUIRE_FALSE(br::toy::isVectorOp(Op::Bnz));

    REQUIRE(std::string(br::toy::opName(Op::Vcmpgt)) == "VCMPGT");
    REQUIRE(std::string(br::toy::opName(Op::Halt))   == "HALT");
}

// ---------------------------------------------------------------------------
// Compiler: update rule -> expected instruction sequence
// ---------------------------------------------------------------------------

TEST_CASE("compiler emits the documented prologue", "[toyisa][compiler]") {
    // 64x64 -> tilesX = 2, tilesY = 2, gridBytes = 4096, tileRowStride = 2048
    const Program p = br::toy::compileVerticalPass(64, 64, 0);

    expectInstr(p, 0, Op::Li,      /*rd=*/2, 0, 0, 0, 1024);   // r2 = tile stride
    expectInstr(p, 1, Op::Li,      /*rd=*/3, 0, 0, 0, 4096);   // r3 = src->dst disp
    expectInstr(p, 2, Op::Vdens,   /*rd=*/6, 0, 0, 0, 0);      // v6 = density ROM
    expectInstr(p, 3, Op::Vbcastb, /*rd=*/7, 0, 0, 0,
                static_cast<std::int32_t>(br::cid(br::Cell::Wall)));

    // First tile group: tiles [0, tilesX) -- the top row, no tile above.
    expectInstr(p, 4, Op::Li, /*rd=*/0, 0, 0, 0, 0);           // r0 = first tile
    expectInstr(p, 5, Op::Li, /*rd=*/1, 0, 0, 0, 2);           // r1 = end tile

    // Loop head: address computation, hoisted out of the 32 unrolled rows.
    expectInstr(p,  6, Op::Mul,  /*rd=*/4, 0, 2, 0, 0);        // r4 = r0 * 1024
    expectInstr(p,  7, Op::Add,  /*rd=*/7, 4, 3, 0, 0);        // r7 = r4 + r3
    expectInstr(p,  8, Op::Addi, /*rd=*/5, 4, 0, 0, -2048);    // r5 = tile above
    expectInstr(p,  9, Op::Addi, /*rd=*/6, 4, 0, 0,  2048);    // r6 = tile below
}

TEST_CASE("compiler lowers one pair-row to the documented 9 instructions",
          "[toyisa][compiler]") {
    const Program p = br::toy::compileVerticalPass(64, 64, 0);

    // parity 0, ly 0 -> this row is the UPPER half of the pair (0,1).
    expectInstr(p, 10, Op::Vld,    0, 4, 0, 0, 0);    // v0 = a  (row 0)
    expectInstr(p, 11, Op::Vld,    1, 4, 0, 0, 32);   // v1 = b  (row 1)
    expectInstr(p, 12, Op::Vshuf,  2, 6, 0, 0, 0);    // v2 = density(a)
    expectInstr(p, 13, Op::Vshuf,  3, 6, 1, 0, 0);    // v3 = density(b)
    expectInstr(p, 14, Op::Vcmpgt, 4, 2, 3, 0, 0);    // v4 = d(a) > d(b)
    expectInstr(p, 15, Op::Vcmpeq, 5, 0, 7, 0, 0);    // v5 = (a == Wall)
    expectInstr(p, 16, Op::Vandn,  4, 5, 4, 0, 0);    // v4 = ~v5 & v4
    expectInstr(p, 17, Op::Vblend, 2, 0, 1, 4, 0);    // v2 = mask ? b : a
    expectInstr(p, 18, Op::Vst,    2, 7, 0, 0, 0);    // store row 0

    // parity 0, ly 1 -> this row is the LOWER half of the SAME pair, so the
    // blend operands are reversed (it receives `a` when the pair swaps).
    expectInstr(p, 19, Op::Vld,    0, 4, 0, 0, 0);
    expectInstr(p, 20, Op::Vld,    1, 4, 0, 0, 32);
    expectInstr(p, 26, Op::Vblend, 2, 1, 0, 4, 0);    // v2 = mask ? a : b
    expectInstr(p, 27, Op::Vst,    2, 7, 0, 0, 32);   // store row 1
}

TEST_CASE("cross-tile rows address the neighbouring tile registers",
          "[toyisa][compiler]") {
    // 64x96 -> tilesY = 3, so there IS an interior tile group with a tile both
    // above and below. parity 0: row 31 is the LOWER half of pair (30,31) and
    // row 0 is the UPPER half of pair (0,1) -- neither crosses a tile seam.
    // parity 1 is the interesting one: row 0 pairs with row 31 of the tile
    // ABOVE, and row 31 pairs with row 0 of the tile BELOW.
    const Program p = br::toy::compileVerticalPass(64, 96, 1);

    bool sawAboveLoad = false;   // VLD from r5 at offset 31*32
    bool sawBelowLoad = false;   // VLD from r6 at offset 0
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Instr in = p.at(i);
        if (in.op == Op::Vld && in.rs1 == 5 && in.imm == 31 * 32) sawAboveLoad = true;
        if (in.op == Op::Vld && in.rs1 == 6 && in.imm == 0)       sawBelowLoad = true;
    }
    REQUIRE(sawAboveLoad);
    REQUIRE(sawBelowLoad);
}

TEST_CASE("program length matches the boundary-specialisation model",
          "[toyisa][compiler]") {
    // 64x64: 2 tile groups (top, bottom), no interior group.
    // Per group: 2 (LI,LI) + 4 (addressing) + body + 3 (ADDI,CMPLT,BNZ).
    // parity 0 -> every row of every group is a real pair: body = 32*9 = 288.
    {
        const Program p = br::toy::compileVerticalPass(64, 64, 0);
        REQUIRE(p.size() == 4u + 2u * (2u + 4u + 288u + 3u) + 1u);   // 599
        REQUIRE(p.at(p.size() - 1).op == Op::Halt);
    }
    // parity 1 -> one row per group degenerates to a 2-instruction copy:
    // body = 31*9 + 2 = 281.
    {
        const Program p = br::toy::compileVerticalPass(64, 64, 1);
        REQUIRE(p.size() == 4u + 2u * (2u + 4u + 281u + 3u) + 1u);   // 585
        REQUIRE(p.at(p.size() - 1).op == Op::Halt);
    }
    // 64x96 -> 3 groups (top, interior, bottom).
    {
        const Program p = br::toy::compileVerticalPass(64, 96, 0);
        REQUIRE(p.size() == 4u + 3u * (2u + 4u + 288u + 3u) + 1u);
    }
}

TEST_CASE("every tile group stores all 32 rows exactly once",
          "[toyisa][compiler]") {
    const Program p = br::toy::compileVerticalPass(64, 96, 0);
    // 3 groups x 32 stores
    REQUIRE(countOp(p, Op::Vst) == 3u * 32u);
    // One back-edge per group.
    REQUIRE(countOp(p, Op::Bnz) == 3u);
    // The density ROM and the Wall broadcast are loaded once, in the prologue.
    REQUIRE(countOp(p, Op::Vdens)   == 1u);
    REQUIRE(countOp(p, Op::Vbcastb) == 1u);
}

TEST_CASE("disassembly is non-empty and mentions the kernel name",
          "[toyisa][compiler]") {
    const Program p = br::toy::compileVerticalPass(64, 64, 0);
    const std::string text = p.disassemble();
    REQUIRE(text.find("vertical_pass_64x64_parity0") != std::string::npos);
    REQUIRE(text.find("VBLEND") != std::string::npos);
    REQUIRE(text.find("HALT")   != std::string::npos);
}

TEST_CASE("patchImm rewrites only the immediate field", "[toyisa][compiler]") {
    Program p;
    p.emit(Op::Bnz, 0, 8, 0, 0, 111);
    p.patchImm(0, 222);
    const Instr in = p.at(0);
    REQUIRE(in.op  == Op::Bnz);
    REQUIRE(in.rs1 == 8);
    REQUIRE(in.imm == 222);
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

TEST_CASE("interpreter executes a hand-written program", "[toyisa][interp]") {
    std::vector<std::uint8_t> src(64, 0u), dst(64, 0u);
    br::toy::MemoryImage mem{src.data(), dst.data(), 64};

    Program p;
    p.name = "store_broadcast";
    p.emit(Op::Li,      1, 0, 0, 0, 64);    // r1 = start of the DST region
    p.emit(Op::Vbcastb, 0, 0, 0, 0, 7);     // v0 = {7}*32
    p.emit(Op::Vst,     0, 1, 0, 0, 0);     // dst[0..32) = v0
    p.emit(Op::Vbcastb, 1, 0, 0, 0, 9);     // v1 = {9}*32
    p.emit(Op::Vst,     1, 1, 0, 0, 32);    // dst[32..64) = v1
    p.emit(Op::Halt);

    const br::toy::RunResult r = br::toy::execute(p, mem);
    INFO(r.error);
    REQUIRE(r.ok);
    REQUIRE(r.instructions == 6u);
    REQUIRE(r.vectorMemOps == 2u);
    for (int i = 0; i < 32; ++i) REQUIRE(dst[static_cast<std::size_t>(i)] == 7u);
    for (int i = 32; i < 64; ++i) REQUIRE(dst[static_cast<std::size_t>(i)] == 9u);
}

TEST_CASE("scalar ALU and the loop back-edge work", "[toyisa][interp]") {
    std::vector<std::uint8_t> src(64, 0u), dst(64, 0u);
    br::toy::MemoryImage mem{src.data(), dst.data(), 64};

    // for (r0 = 0; r0 < 5; ++r0) {}   then store r0 as a broadcast byte.
    Program p;
    p.emit(Op::Li, 0, 0, 0, 0, 0);          // 0: r0 = 0
    p.emit(Op::Li, 1, 0, 0, 0, 5);          // 1: r1 = 5
    p.emit(Op::Addi,  0, 0, 0, 0, 1);       // 2: r0 += 1     <- loop head
    p.emit(Op::CmpLt, 2, 0, 1, 0, 0);       // 3: r2 = r0 < r1
    p.emit(Op::Bnz,   0, 2, 0, 0, 2);       // 4: if (r2) goto 2
    p.emit(Op::Halt);                       // 5

    const br::toy::RunResult r = br::toy::execute(p, mem);
    REQUIRE(r.ok);
    REQUIRE(r.branches == 5u);              // 4 taken + 1 not taken
    REQUIRE(r.instructions == 2u + 5u * 3u + 1u);
}

TEST_CASE("VSHUF + VCMPGT + VANDN implement swapVertical exactly",
          "[toyisa][interp][rules]") {
    std::vector<std::uint8_t> src(64, 0u), dst(64, 0u);
    for (int i = 0; i < 32; ++i) {
        src[static_cast<std::size_t>(i)]      =
            static_cast<std::uint8_t>(i % br::kCellKinds);              // a
        src[static_cast<std::size_t>(32 + i)] =
            static_cast<std::uint8_t>((i / 5 + i) % br::kCellKinds);    // b
    }
    br::toy::MemoryImage mem{src.data(), dst.data(), 64};

    Program p;
    p.name = "swapVertical_mask";
    p.emit(Op::Li,      0, 0, 0, 0, 0);     // r0 = SRC base
    p.emit(Op::Li,      1, 0, 0, 0, 64);    // r1 = DST base
    p.emit(Op::Vdens,   6);                 // v6 = density ROM
    p.emit(Op::Vbcastb, 7, 0, 0, 0,
           static_cast<std::int32_t>(br::cid(br::Cell::Wall)));
    p.emit(Op::Vld,     0, 0, 0, 0, 0);     // v0 = a
    p.emit(Op::Vld,     1, 0, 0, 0, 32);    // v1 = b
    p.emit(Op::Vshuf,   2, 6, 0);
    p.emit(Op::Vshuf,   3, 6, 1);
    p.emit(Op::Vcmpgt,  4, 2, 3);
    p.emit(Op::Vcmpeq,  5, 0, 7);
    p.emit(Op::Vandn,   4, 5, 4);
    p.emit(Op::Vst,     4, 1, 0, 0, 0);
    p.emit(Op::Halt);

    const br::toy::RunResult r = br::toy::execute(p, mem);
    REQUIRE(r.ok);

    for (int i = 0; i < 32; ++i) {
        const std::uint8_t a = src[static_cast<std::size_t>(i)];
        const std::uint8_t b = src[static_cast<std::size_t>(32 + i)];
        const std::uint8_t want = br::swapVertical(a, b) ? 0xFFu : 0x00u;
        INFO("lane " << i << " a=" << int(a) << " b=" << int(b));
        REQUIRE(dst[static_cast<std::size_t>(i)] == want);
    }
}

TEST_CASE("VBLEND selects on the high bit of the mask", "[toyisa][interp]") {
    std::vector<std::uint8_t> src(64, 0u), dst(64, 0u);
    for (int i = 0; i < 32; ++i) {
        src[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((i % 2) ? 0xFF : 0x00);   // mask pattern
    }
    br::toy::MemoryImage mem{src.data(), dst.data(), 64};

    Program p;
    p.emit(Op::Li,      0, 0, 0, 0, 0);
    p.emit(Op::Li,      1, 0, 0, 0, 64);
    p.emit(Op::Vld,     3, 0, 0, 0, 0);      // v3 = mask
    p.emit(Op::Vbcastb, 0, 0, 0, 0, 11);     // v0 = 11 (false branch)
    p.emit(Op::Vbcastb, 1, 0, 0, 0, 22);     // v1 = 22 (true branch)
    p.emit(Op::Vblend,  2, 0, 1, 3, 0);
    p.emit(Op::Vst,     2, 1, 0, 0, 0);
    p.emit(Op::Halt);

    REQUIRE(br::toy::execute(p, mem).ok);
    for (int i = 0; i < 32; ++i) {
        REQUIRE(dst[static_cast<std::size_t>(i)] == ((i % 2) ? 22u : 11u));
    }
}

TEST_CASE("the scoreboard charges the documented load latency",
          "[toyisa][interp][cycles]") {
    std::vector<std::uint8_t> src(64, 1u), dst(64, 0u);
    br::toy::MemoryImage mem{src.data(), dst.data(), 64};

    Program p;
    p.emit(Op::Li,    0, 0, 0, 0, 0);    // cycle 0, r0 ready at 1
    p.emit(Op::Vld,   0, 0, 0, 0, 0);    // issues at 1, v0 ready at 5
    p.emit(Op::Vshuf, 1, 0, 0);          // needs v0 -> issues at 5 (3 stalls)
    p.emit(Op::Halt);

    const br::toy::RunResult r = br::toy::execute(p, mem);
    REQUIRE(r.ok);
    REQUIRE(r.stallCycles == 3u);
    REQUIRE(r.cycles > r.instructions);
    REQUIRE(r.ipc() < 1.0);
}

TEST_CASE("independent instructions do not stall", "[toyisa][interp][cycles]") {
    std::vector<std::uint8_t> src(64, 1u), dst(64, 0u);
    br::toy::MemoryImage mem{src.data(), dst.data(), 64};

    Program p;
    p.emit(Op::Vbcastb, 0, 0, 0, 0, 1);
    p.emit(Op::Vbcastb, 1, 0, 0, 0, 2);
    p.emit(Op::Vbcastb, 2, 0, 0, 0, 3);
    p.emit(Op::Halt);

    const br::toy::RunResult r = br::toy::execute(p, mem);
    REQUIRE(r.ok);
    REQUIRE(r.stallCycles == 0u);
    REQUIRE(r.cycles == r.instructions);
    REQUIRE(r.ipc() == 1.0);
}

TEST_CASE("interpreter faults on illegal memory behaviour", "[toyisa][faults]") {
    std::vector<std::uint8_t> src(64, 0u), dst(64, 0u);

    SECTION("store into the read-only SRC region") {
        br::toy::MemoryImage mem{src.data(), dst.data(), 64};
        Program p;
        p.emit(Op::Li,      0, 0, 0, 0, 0);
        p.emit(Op::Vbcastb, 0, 0, 0, 0, 1);
        p.emit(Op::Vst,     0, 0, 0, 0, 0);   // addr 0 -> SRC
        p.emit(Op::Halt);
        const br::toy::RunResult r = br::toy::execute(p, mem);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.fault == br::toy::Fault::StoreProtection);
        REQUIRE_FALSE(r.error.empty());
    }

    SECTION("misaligned vector access") {
        br::toy::MemoryImage mem{src.data(), dst.data(), 64};
        Program p;
        p.emit(Op::Li,      0, 0, 0, 0, 65);  // 65 % 32 != 0
        p.emit(Op::Vbcastb, 0, 0, 0, 0, 1);
        p.emit(Op::Vst,     0, 0, 0, 0, 0);
        p.emit(Op::Halt);
        const br::toy::RunResult r = br::toy::execute(p, mem);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.fault == br::toy::Fault::Alignment);
    }

    SECTION("access past the end of the image") {
        br::toy::MemoryImage mem{src.data(), dst.data(), 64};
        Program p;
        p.emit(Op::Li,  0, 0, 0, 0, 128);     // == 2 * size, nothing mapped
        p.emit(Op::Vld, 0, 0, 0, 0, 0);
        p.emit(Op::Halt);
        const br::toy::RunResult r = br::toy::execute(p, mem);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.fault == br::toy::Fault::OutOfRange);
    }

    SECTION("branch outside the program") {
        br::toy::MemoryImage mem{src.data(), dst.data(), 64};
        Program p;
        p.emit(Op::Jmp, 0, 0, 0, 0, 9999);
        p.emit(Op::Halt);
        const br::toy::RunResult r = br::toy::execute(p, mem);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.fault == br::toy::Fault::BadBranch);
    }

    SECTION("illegal opcode") {
        br::toy::MemoryImage mem{src.data(), dst.data(), 64};
        Program p;
        Instr bad;
        bad.op = static_cast<Op>(0xEE);
        p.emit(bad);
        p.emit(Op::Halt);
        const br::toy::RunResult r = br::toy::execute(p, mem);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.fault == br::toy::Fault::IllegalOpcode);
    }

    SECTION("null memory image") {
        br::toy::MemoryImage mem{nullptr, nullptr, 0};
        Program p;
        p.emit(Op::Halt);
        const br::toy::RunResult r = br::toy::execute(p, mem);
        REQUIRE_FALSE(r.ok);
    }

    SECTION("runaway program hits the instruction budget") {
        br::toy::MemoryImage mem{src.data(), dst.data(), 64};
        Program p;
        p.emit(Op::Jmp, 0, 0, 0, 0, 0);   // infinite loop
        const br::toy::RunResult r = br::toy::execute(p, mem, /*maxInstr=*/1000);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.fault == br::toy::Fault::InstructionLimit);
        REQUIRE(r.instructions == 1000u);
    }
}

TEST_CASE("a compiled kernel runs to completion on a real grid",
          "[toyisa][integration]") {
    br::Grid g(64, 64);
    g.clear();
    g.seedRandom(0x4242u, 0.3f, 0.2f);

    const Program p = br::toy::compileVerticalPass(g.width(), g.height(), 0);
    br::toy::MemoryImage mem{g.src(), g.dst(), g.bytes()};
    const br::toy::RunResult r = br::toy::execute(p, mem);

    INFO(r.error);
    REQUIRE(r.ok);
    REQUIRE(r.vectorMemOps == 4u * 32u * 2u - 0u);  // 4 tiles x 32 rows, ld+st
    REQUIRE(r.vectorAluOps > 0u);
    REQUIRE(r.scalarOps    > 0u);
    REQUIRE(r.ipc() > 0.0);
    REQUIRE(r.ipc() <= 1.0);
}