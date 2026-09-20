// ---------------------------------------------------------------------------
// backends/toy_isa/compiler.cpp
//
// Role in BackendRace: the "compiler" half of the toy accelerator.  It performs
// instruction selection for the vertical (gravity) update rule, specialises the
// loop nest on the concrete grid geometry, fully unrolls the 32 rows of a tile,
// emits three boundary-specialised loop bodies (top / interior / bottom tile
// row) so the steady state has no boundary checks, and encodes everything into
// the 64-bit TSA-32 instruction words defined in isa_spec.md.
//
// It also implements encode/decode/disassemble for the whole ISA.
// ---------------------------------------------------------------------------
#include "backends/toy_isa/isa.hpp"

#include <cstdio>
#include <sstream>

namespace br::toy {

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
std::uint64_t encode(const Instr& in) noexcept {
    const std::uint64_t immBits =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(in.imm));
    return (static_cast<std::uint64_t>(in.op)  << 56) |
           (static_cast<std::uint64_t>(in.rd  & 0xF) << 52) |
           (static_cast<std::uint64_t>(in.rs1 & 0xF) << 48) |
           (static_cast<std::uint64_t>(in.rs2 & 0xF) << 44) |
           (static_cast<std::uint64_t>(in.rs3 & 0xF) << 40) |
           (immBits << 8);
}

Instr decode(std::uint64_t w) noexcept {
    Instr in;
    in.op  = static_cast<Op>((w >> 56) & 0xFF);
    in.rd  = static_cast<std::uint8_t>((w >> 52) & 0xF);
    in.rs1 = static_cast<std::uint8_t>((w >> 48) & 0xF);
    in.rs2 = static_cast<std::uint8_t>((w >> 44) & 0xF);
    in.rs3 = static_cast<std::uint8_t>((w >> 40) & 0xF);
    in.imm = static_cast<std::int32_t>(
        static_cast<std::uint32_t>((w >> 8) & 0xFFFFFFFFull));
    return in;
}

const char* opName(Op op) noexcept {
    switch (op) {
        case Op::Halt:    return "HALT";
        case Op::Nop:     return "NOP";
        case Op::Li:      return "LI";
        case Op::Add:     return "ADD";
        case Op::Addi:    return "ADDI";
        case Op::Sub:     return "SUB";
        case Op::Shli:    return "SHLI";
        case Op::Andi:    return "ANDI";
        case Op::Mul:     return "MUL";
        case Op::CmpLt:   return "CMPLT";
        case Op::CmpEq:   return "CMPEQ";
        case Op::Bnz:     return "BNZ";
        case Op::Bz:      return "BZ";
        case Op::Jmp:     return "JMP";
        case Op::Vld:     return "VLD";
        case Op::Vst:     return "VST";
        case Op::Vshuf:   return "VSHUF";
        case Op::Vcmpgt:  return "VCMPGT";
        case Op::Vcmpeq:  return "VCMPEQ";
        case Op::Vand:    return "VAND";
        case Op::Vandn:   return "VANDN";
        case Op::Vor:     return "VOR";
        case Op::Vxor:    return "VXOR";
        case Op::Vblend:  return "VBLEND";
        case Op::Vbcastb: return "VBCASTB";
        case Op::Vmov:    return "VMOV";
        case Op::Vdens:   return "VDENS";
        default:          return "???";
    }
}

bool isVectorOp(Op op) noexcept {
    return static_cast<std::uint8_t>(op) >= 0x30;
}

int latencyOf(Op op) noexcept {
    switch (op) {
        case Op::Mul:    return 3;
        case Op::Vld:    return 4;
        case Op::Vshuf:  return 2;
        case Op::Vblend: return 2;
        default:         return 1;
    }
}

void Program::patchImm(std::size_t i, std::int32_t imm) {
    Instr in = decode(words_[i]);
    in.imm = imm;
    words_[i] = encode(in);
}

std::string Program::disassemble() const {
    std::ostringstream os;
    os << "; TSA-32 program: " << (name.empty() ? "<unnamed>" : name)
       << "  (" << words_.size() << " instructions)\n";
    for (std::size_t i = 0; i < words_.size(); ++i) {
        const Instr in = decode(words_[i]);
        char line[160];
        const char* nm = opName(in.op);
        switch (in.op) {
            case Op::Vld:
                std::snprintf(line, sizeof(line), "%04zu  %-7s v%u, r%u, %+d",
                              i, nm, in.rd, in.rs1, in.imm); break;
            case Op::Vst:
                std::snprintf(line, sizeof(line), "%04zu  %-7s v%u -> r%u, %+d",
                              i, nm, in.rd, in.rs1, in.imm); break;
            case Op::Vshuf: case Op::Vcmpgt: case Op::Vcmpeq:
            case Op::Vand:  case Op::Vandn:  case Op::Vor: case Op::Vxor:
                std::snprintf(line, sizeof(line), "%04zu  %-7s v%u, v%u, v%u",
                              i, nm, in.rd, in.rs1, in.rs2); break;
            case Op::Vblend:
                std::snprintf(line, sizeof(line),
                              "%04zu  %-7s v%u, v%u, v%u, v%u",
                              i, nm, in.rd, in.rs1, in.rs2, in.rs3); break;
            case Op::Vbcastb:
                std::snprintf(line, sizeof(line), "%04zu  %-7s v%u, #%d",
                              i, nm, in.rd, in.imm); break;
            case Op::Vmov:
                std::snprintf(line, sizeof(line), "%04zu  %-7s v%u, v%u",
                              i, nm, in.rd, in.rs1); break;
            case Op::Vdens:
                std::snprintf(line, sizeof(line), "%04zu  %-7s v%u",
                              i, nm, in.rd); break;
            case Op::Li:
                std::snprintf(line, sizeof(line), "%04zu  %-7s r%u, #%d",
                              i, nm, in.rd, in.imm); break;
            case Op::Addi: case Op::Shli: case Op::Andi:
                std::snprintf(line, sizeof(line), "%04zu  %-7s r%u, r%u, #%d",
                              i, nm, in.rd, in.rs1, in.imm); break;
            case Op::Add: case Op::Sub: case Op::Mul:
            case Op::CmpLt: case Op::CmpEq:
                std::snprintf(line, sizeof(line), "%04zu  %-7s r%u, r%u, r%u",
                              i, nm, in.rd, in.rs1, in.rs2); break;
            case Op::Bnz: case Op::Bz:
                std::snprintf(line, sizeof(line), "%04zu  %-7s r%u, @%d",
                              i, nm, in.rs1, in.imm); break;
            case Op::Jmp:
                std::snprintf(line, sizeof(line), "%04zu  %-7s @%d",
                              i, nm, in.imm); break;
            default:
                std::snprintf(line, sizeof(line), "%04zu  %-7s", i, nm); break;
        }
        os << line << '\n';
    }
    return os.str();
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------
namespace {

// Scalar register allocation (fixed; see isa_spec.md section 5).
constexpr int R_T     = 0;   // tile index
constexpr int R_END   = 1;   // end tile index
constexpr int R_TSTR  = 2;   // 1024
constexpr int R_DISP  = 3;   // src->dst displacement
constexpr int R_SRC   = 4;   // src base of current tile
constexpr int R_ABOVE = 5;
constexpr int R_BELOW = 6;
constexpr int R_DST   = 7;
constexpr int R_COND  = 8;

// Vector register allocation.
constexpr int V_A    = 0;
constexpr int V_B    = 1;
constexpr int V_T0   = 2;
constexpr int V_T1   = 3;
constexpr int V_MASK = 4;
constexpr int V_WALL_CMP = 5;
constexpr int V_LUT  = 6;
constexpr int V_WALL = 7;

// Nine-instruction core of the gravity rule for one 32-cell row.
// `baseA/offA` address the upper row, `baseB/offB` the lower row, and
// `keepIsLower` tells us which of the two is the row we are writing.
void emitPairRow(Program& p,
                 int baseA, std::int32_t offA,
                 int baseB, std::int32_t offB,
                 std::int32_t outOff,
                 bool writingUpperRow) {
    p.emit(Op::Vld,    V_A,  baseA, 0, 0, offA);
    p.emit(Op::Vld,    V_B,  baseB, 0, 0, offB);
    p.emit(Op::Vshuf,  V_T0, V_LUT, V_A);              // density(a)
    p.emit(Op::Vshuf,  V_T1, V_LUT, V_B);              // density(b)
    p.emit(Op::Vcmpgt, V_MASK, V_T0, V_T1);            // density(a) > density(b)
    p.emit(Op::Vcmpeq, V_WALL_CMP, V_A, V_WALL);       // a == Wall
    p.emit(Op::Vandn,  V_MASK, V_WALL_CMP, V_MASK);    // mask = ~wall(a) & gt
    if (writingUpperRow) {
        // This row is `a`; if the pair swaps it receives `b`.
        p.emit(Op::Vblend, V_T0, V_A, V_B, V_MASK);
    } else {
        // This row is `b`; if the pair swaps it receives `a`.
        p.emit(Op::Vblend, V_T0, V_B, V_A, V_MASK);
    }
    p.emit(Op::Vst, V_T0, R_DST, 0, 0, outOff);
}

void emitCopyRow(Program& p, std::int32_t off) {
    p.emit(Op::Vld, V_A, R_SRC, 0, 0, off);
    p.emit(Op::Vst, V_A, R_DST, 0, 0, off);
}

// One fully unrolled tile body, specialised on whether neighbour tile rows
// exist above/below.
void emitTileBody(Program& p, int parity, bool hasAbove, bool hasBelow) {
    for (int ly = 0; ly < 32; ++ly) {
        const std::int32_t off = static_cast<std::int32_t>(ly) * 32;
        if ((ly & 1) == parity) {
            // UPPER half of the pair (partner is ly+1).
            if (ly == 31 && !hasBelow) { emitCopyRow(p, off); continue; }
            const int          baseB = (ly < 31) ? R_SRC : R_BELOW;
            const std::int32_t offB  = (ly < 31) ? (off + 32) : 0;
            emitPairRow(p, R_SRC, off, baseB, offB, off, /*writingUpper=*/true);
        } else {
            // LOWER half of the pair (partner is ly-1).
            if (ly == 0 && !hasAbove) { emitCopyRow(p, off); continue; }
            const int          baseA = (ly > 0) ? R_SRC : R_ABOVE;
            const std::int32_t offA  = (ly > 0) ? (off - 32) : (31 * 32);
            emitPairRow(p, baseA, offA, R_SRC, off, off, /*writingUpper=*/false);
        }
    }
}

struct TileGroup { int start, end; bool hasAbove, hasBelow; };

}  // namespace

Program compileVerticalPass(int width, int height, int parity) {
    Program p;
    {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "vertical_pass_%dx%d_parity%d",
                      width, height, parity & 1);
        p.name = buf;
    }
    parity &= 1;

    const int tilesX = width  / 32;
    const int tilesY = height / 32;
    const std::int32_t tileRowStride =
        static_cast<std::int32_t>(tilesX) * kTileBytes;
    const std::int32_t gridBytes =
        static_cast<std::int32_t>(tilesX) *
        static_cast<std::int32_t>(tilesY) * kTileBytes;

    // ---- prologue: loop-invariant constants hoisted out of every loop ------
    p.emit(Op::Li,      R_TSTR, 0, 0, 0, kTileBytes);
    p.emit(Op::Li,      R_DISP, 0, 0, 0, gridBytes);
    p.emit(Op::Vdens,   V_LUT);
    p.emit(Op::Vbcastb, V_WALL, 0, 0, 0, 4 /* Cell::Wall */);

    // ---- boundary-specialised tile groups ---------------------------------
    std::vector<TileGroup> groups;
    if (tilesY == 1) {
        groups.push_back({0, tilesX, false, false});
    } else {
        groups.push_back({0, tilesX, false, true});
        if (tilesY > 2) {
            groups.push_back({tilesX, tilesX * (tilesY - 1), true, true});
        }
        groups.push_back({tilesX * (tilesY - 1), tilesX * tilesY, true, false});
    }

    for (const TileGroup& g : groups) {
        p.emit(Op::Li, R_T,   0, 0, 0, g.start);
        p.emit(Op::Li, R_END, 0, 0, 0, g.end);

        const auto loopHead = static_cast<std::int32_t>(p.size());
        p.emit(Op::Mul,  R_SRC,   R_T,   R_TSTR);          // tileBase = t * 1024
        p.emit(Op::Add,  R_DST,   R_SRC, R_DISP);          // dst = src + size
        p.emit(Op::Addi, R_ABOVE, R_SRC, 0, 0, -tileRowStride);
        p.emit(Op::Addi, R_BELOW, R_SRC, 0, 0,  tileRowStride);

        emitTileBody(p, parity, g.hasAbove, g.hasBelow);

        p.emit(Op::Addi,  R_T,    R_T, 0, 0, 1);
        p.emit(Op::CmpLt, R_COND, R_T, R_END);
        p.emit(Op::Bnz,   0,      R_COND, 0, 0, loopHead);
    }

    p.emit(Op::Halt);
    return p;
}

}  // namespace br::toy