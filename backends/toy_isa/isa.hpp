// ---------------------------------------------------------------------------
// backends/toy_isa/isa.hpp
//
// Role in BackendRace: public interface of the TSA-32 toy accelerator --
// opcode enum, 64-bit instruction encoding, program container, the memory
// image the interpreter executes against, the cycle-model result struct, and
// the compiler entry point.  See isa_spec.md for the normative definition.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace br::toy {

inline constexpr int kVecBytes = 32;
inline constexpr int kNumVregs = 8;
inline constexpr int kNumSregs = 16;
inline constexpr int kTileBytes = 1024;

enum class Op : std::uint8_t {
    Halt    = 0x00, Nop   = 0x01,
    Li      = 0x10, Add   = 0x11, Addi  = 0x12, Sub   = 0x13,
    Shli    = 0x14, Andi  = 0x15, Mul   = 0x16,
    CmpLt   = 0x20, CmpEq = 0x21, Bnz   = 0x22, Bz    = 0x23, Jmp = 0x24,
    Vld     = 0x30, Vst   = 0x31, Vshuf = 0x32, Vcmpgt= 0x33, Vcmpeq = 0x34,
    Vand    = 0x35, Vandn = 0x36, Vor   = 0x37, Vxor  = 0x38, Vblend = 0x39,
    Vbcastb = 0x3A, Vmov  = 0x3B, Vdens = 0x3C,
};

struct Instr {
    Op            op  = Op::Nop;
    std::uint8_t  rd  = 0;
    std::uint8_t  rs1 = 0;
    std::uint8_t  rs2 = 0;
    std::uint8_t  rs3 = 0;
    std::int32_t  imm = 0;
};

std::uint64_t encode(const Instr& in) noexcept;
Instr         decode(std::uint64_t word) noexcept;
const char*   opName(Op op) noexcept;
bool          isVectorOp(Op op) noexcept;
int           latencyOf(Op op) noexcept;

class Program {
public:
    std::string name;

    void emit(const Instr& in) { words_.push_back(encode(in)); }
    void emit(Op op, int rd = 0, int rs1 = 0, int rs2 = 0,
              int rs3 = 0, std::int32_t imm = 0) {
        Instr in;
        in.op  = op;
        in.rd  = static_cast<std::uint8_t>(rd);
        in.rs1 = static_cast<std::uint8_t>(rs1);
        in.rs2 = static_cast<std::uint8_t>(rs2);
        in.rs3 = static_cast<std::uint8_t>(rs3);
        in.imm = imm;
        words_.push_back(encode(in));
    }

    std::size_t   size() const noexcept { return words_.size(); }
    std::uint64_t word(std::size_t i) const { return words_[i]; }
    Instr         at(std::size_t i)  const { return decode(words_[i]); }
    void          patchImm(std::size_t i, std::int32_t imm);

    const std::vector<std::uint64_t>& words() const noexcept { return words_; }
    std::string disassemble() const;

private:
    std::vector<std::uint64_t> words_;
};

// Flat address space: [0,size) -> src (read-only), [size,2*size) -> dst.
struct MemoryImage {
    std::uint8_t*       src  = nullptr;
    std::uint8_t*       dst  = nullptr;
    std::size_t         size = 0;
};

enum class Fault : int {
    None = 0, IllegalOpcode, Alignment, OutOfRange, StoreProtection,
    InstructionLimit, BadBranch,
};

struct RunResult {
    bool          ok            = false;
    Fault         fault         = Fault::None;
    std::string   error;
    std::uint64_t instructions  = 0;
    std::uint64_t cycles        = 0;
    std::uint64_t stallCycles   = 0;
    std::uint64_t scalarOps     = 0;
    std::uint64_t vectorAluOps  = 0;
    std::uint64_t vectorMemOps  = 0;
    std::uint64_t branches      = 0;

    double ipc() const noexcept {
        return cycles ? static_cast<double>(instructions) /
                        static_cast<double>(cycles) : 0.0;
    }
};

// Cycle-model interpreter (interpreter.cpp).
RunResult execute(const Program& prog, MemoryImage& mem,
                  std::uint64_t maxInstructions = (1ull << 34));

// Compiler (compiler.cpp): lowers the vertical/gravity rule for a given grid
// geometry and row parity.  `width`/`height` must be multiples of 32.
Program compileVerticalPass(int width, int height, int parity);

}  // namespace br::toy