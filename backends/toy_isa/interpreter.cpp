// ---------------------------------------------------------------------------
// backends/toy_isa/interpreter.cpp
//
// Role in BackendRace: the cycle-model simulator for TSA-32.  It really decodes
// the 64-bit instruction words produced by compiler.cpp, really executes them
// against the grid's memory image (including alignment and store-protection
// faults), and models a single-issue in-order scoreboarded pipeline so that the
// benchmark can report estimated accelerator cycles, stall cycles, IPC and
// cells/s at a nominal clock.
// ---------------------------------------------------------------------------
#include "backends/toy_isa/isa.hpp"

#include <algorithm>
#include <cstring>

namespace br::toy {

namespace {

// The hard-wired density ROM (biased by XOR 0x80); see isa_spec.md.
const std::uint8_t kDensityRom[16] = {
    0x81, 0x83, 0x82, 0x80, 0x7F, 0x7F, 0x7F, 0x7F,
    0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F
};

struct Machine {
    MemoryImage& mem;
    std::uint32_t s[kNumSregs]{};
    std::uint8_t  v[kNumVregs][kVecBytes]{};
    std::uint64_t sReady[kNumSregs]{};
    std::uint64_t vReady[kNumVregs]{};

    explicit Machine(MemoryImage& m) : mem(m) {}

    bool checkAccess(std::uint64_t addr, bool store, Fault& f) const {
        if ((addr & 31u) != 0) { f = Fault::Alignment; return false; }
        const std::uint64_t total = 2ull * mem.size;
        if (addr + kVecBytes > total) { f = Fault::OutOfRange; return false; }
        if (store && addr < mem.size) { f = Fault::StoreProtection; return false; }
        return true;
    }

    const std::uint8_t* readPtr(std::uint64_t addr) const {
        return (addr < mem.size) ? (mem.src + addr)
                                 : (mem.dst + (addr - mem.size));
    }
    std::uint8_t* writePtr(std::uint64_t addr) const {
        return mem.dst + (addr - mem.size);
    }
};

const char* faultText(Fault f) {
    switch (f) {
        case Fault::IllegalOpcode:    return "illegal opcode";
        case Fault::Alignment:        return "misaligned 32-byte vector access";
        case Fault::OutOfRange:       return "address outside the memory image";
        case Fault::StoreProtection:  return "store into the read-only SRC region";
        case Fault::InstructionLimit: return "instruction budget exhausted";
        case Fault::BadBranch:        return "branch target outside the program";
        default:                      return "none";
    }
}

}  // namespace

RunResult execute(const Program& prog, MemoryImage& mem,
                  std::uint64_t maxInstructions) {
    RunResult res;
    if (mem.src == nullptr || mem.dst == nullptr || mem.size == 0) {
        res.ok = false;
        res.fault = Fault::OutOfRange;
        res.error = "null or empty memory image";
        return res;
    }

    Machine M(mem);
    const std::size_t n = prog.size();
    std::uint64_t pc = 0;
    std::uint64_t now = 0;          // current issue cycle
    std::uint64_t stalls = 0;
    Fault fault = Fault::None;
    bool halted = false;

    while (!halted) {
        if (pc >= n) { fault = Fault::BadBranch; break; }
        if (res.instructions >= maxInstructions) {
            fault = Fault::InstructionLimit; break;
        }

        const Instr in = decode(prog.word(static_cast<std::size_t>(pc)));
        ++pc;

        // ---- phase 1: scoreboard -- compute the earliest issue cycle -------
        std::uint64_t issue = now;
        auto needS = [&](int r) { issue = std::max(issue, M.sReady[r & 15]); };
        auto needV = [&](int r) { issue = std::max(issue, M.vReady[r & 7]);  };

        switch (in.op) {
            case Op::Add: case Op::Sub: case Op::Mul:
            case Op::CmpLt: case Op::CmpEq:
                needS(in.rs1); needS(in.rs2); break;
            case Op::Addi: case Op::Shli: case Op::Andi:
                needS(in.rs1); break;
            case Op::Bnz: case Op::Bz:
                needS(in.rs1); break;
            case Op::Vld:
                needS(in.rs1); break;
            case Op::Vst:
                needS(in.rs1); needV(in.rd); break;
            case Op::Vshuf: case Op::Vcmpgt: case Op::Vcmpeq:
            case Op::Vand:  case Op::Vandn:  case Op::Vor: case Op::Vxor:
                needV(in.rs1); needV(in.rs2); break;
            case Op::Vblend:
                needV(in.rs1); needV(in.rs2); needV(in.rs3); break;
            case Op::Vmov:
                needV(in.rs1); break;
            default: break;
        }
        stalls += (issue - now);
        now = issue + 1;                               // single-issue pipeline
        const std::uint64_t ready =
            issue + static_cast<std::uint64_t>(latencyOf(in.op));

        // ---- phase 2: execute ---------------------------------------------
        const int rd  = in.rd  & 15, vd  = in.rd  & 7;
        const int rs1 = in.rs1 & 15, vs1 = in.rs1 & 7;
        const int rs2 = in.rs2 & 15, vs2 = in.rs2 & 7;
        const int vs3 = in.rs3 & 7;

        switch (in.op) {
            case Op::Halt: halted = true; ++res.scalarOps; break;
            case Op::Nop:  ++res.scalarOps; break;

            case Op::Li:
                M.s[rd] = static_cast<std::uint32_t>(in.imm);
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::Add:
                M.s[rd] = M.s[rs1] + M.s[rs2];
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::Addi:
                M.s[rd] = M.s[rs1] + static_cast<std::uint32_t>(in.imm);
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::Sub:
                M.s[rd] = M.s[rs1] - M.s[rs2];
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::Shli:
                M.s[rd] = M.s[rs1] << (in.imm & 31);
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::Andi:
                M.s[rd] = M.s[rs1] & static_cast<std::uint32_t>(in.imm);
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::Mul:
                M.s[rd] = M.s[rs1] * M.s[rs2];
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::CmpLt:
                M.s[rd] = (static_cast<std::int32_t>(M.s[rs1]) <
                           static_cast<std::int32_t>(M.s[rs2])) ? 1u : 0u;
                M.sReady[rd] = ready; ++res.scalarOps; break;
            case Op::CmpEq:
                M.s[rd] = (M.s[rs1] == M.s[rs2]) ? 1u : 0u;
                M.sReady[rd] = ready; ++res.scalarOps; break;

            case Op::Bnz: case Op::Bz: case Op::Jmp: {
                ++res.branches;
                bool taken = (in.op == Op::Jmp);
                if (in.op == Op::Bnz) taken = (M.s[rs1] != 0);
                if (in.op == Op::Bz)  taken = (M.s[rs1] == 0);
                if (taken) {
                    if (in.imm < 0 ||
                        static_cast<std::size_t>(in.imm) >= n) {
                        fault = Fault::BadBranch; halted = true; break;
                    }
                    pc = static_cast<std::uint64_t>(in.imm);
                    now += 2;                          // front-end bubble
                }
                break;
            }

            case Op::Vld: {
                const std::uint64_t addr =
                    static_cast<std::uint64_t>(M.s[rs1]) +
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(in.imm));
                if (!M.checkAccess(addr, false, fault)) { halted = true; break; }
                std::memcpy(M.v[vd], M.readPtr(addr), kVecBytes);
                M.vReady[vd] = ready; ++res.vectorMemOps; break;
            }
            case Op::Vst: {
                const std::uint64_t addr =
                    static_cast<std::uint64_t>(M.s[rs1]) +
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(in.imm));
                if (!M.checkAccess(addr, true, fault)) { halted = true; break; }
                std::memcpy(M.writePtr(addr), M.v[vd], kVecBytes);
                ++res.vectorMemOps; break;
            }

            case Op::Vshuf:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = M.v[vs1][M.v[vs2][i] & 15];
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vcmpgt:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = (static_cast<std::int8_t>(M.v[vs1][i]) >
                                  static_cast<std::int8_t>(M.v[vs2][i]))
                                 ? 0xFFu : 0x00u;
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vcmpeq:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = (M.v[vs1][i] == M.v[vs2][i]) ? 0xFFu : 0x00u;
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vand:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = static_cast<std::uint8_t>(M.v[vs1][i] & M.v[vs2][i]);
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vandn:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = static_cast<std::uint8_t>(
                        static_cast<std::uint8_t>(~M.v[vs1][i]) & M.v[vs2][i]);
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vor:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = static_cast<std::uint8_t>(M.v[vs1][i] | M.v[vs2][i]);
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vxor:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = static_cast<std::uint8_t>(M.v[vs1][i] ^ M.v[vs2][i]);
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vblend:
                for (int i = 0; i < kVecBytes; ++i)
                    M.v[vd][i] = (M.v[vs3][i] & 0x80u) ? M.v[vs2][i] : M.v[vs1][i];
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vbcastb:
                std::memset(M.v[vd], static_cast<int>(in.imm & 0xFF), kVecBytes);
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vmov:
                std::memcpy(M.v[vd], M.v[vs1], kVecBytes);
                M.vReady[vd] = ready; ++res.vectorAluOps; break;
            case Op::Vdens:
                std::memcpy(M.v[vd] + 0,  kDensityRom, 16);
                std::memcpy(M.v[vd] + 16, kDensityRom, 16);
                M.vReady[vd] = ready; ++res.vectorAluOps; break;

            default:
                fault = Fault::IllegalOpcode; halted = true; break;
        }

        ++res.instructions;
        if (fault != Fault::None) break;
    }

    res.cycles      = now;
    res.stallCycles = stalls;
    res.fault       = fault;
    res.ok          = (fault == Fault::None);
    if (!res.ok) res.error = faultText(fault);
    return res;
}

}  // namespace br::toy