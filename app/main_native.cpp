// ---------------------------------------------------------------------------
// app/main_native.cpp
//
// Role in BackendRace: the native interactive front end.  Zero external
// dependencies: it renders the grid into the terminal with 24-bit ANSI colour
// and the U+2580 half-block glyph (two grid rows per text row) and reads
// keystrokes in raw mode, so the same "switch backend live and watch the
// overlay" experience works over SSH.
//
// Keys: 1-4 backend | Q W E R material | arrows move cursor | space emit
//       p pause | c clear | s re-seed | x quit
// ---------------------------------------------------------------------------
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/backend.hpp"
#include "core/grid.hpp"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <conio.h>
#else
#  include <sys/select.h>
#  include <termios.h>
#  include <unistd.h>
#endif

namespace {

// --------------------------------------------------------------------------
// Raw terminal handling
// --------------------------------------------------------------------------
class RawTerminal {
public:
    RawTerminal() {
#if defined(_WIN32)
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(hOut, &mode)) {
            savedOut_ = mode;
            SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            haveSaved_ = true;
        }
        SetConsoleOutputCP(CP_UTF8);
#else
        if (tcgetattr(STDIN_FILENO, &saved_) == 0) {
            termios raw = saved_;
            raw.c_lflag &= ~(static_cast<tcflag_t>(ICANON | ECHO));
            raw.c_cc[VMIN]  = 0;
            raw.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw);
            haveSaved_ = true;
        }
#endif
        std::fputs("\x1b[?25l", stdout);   // hide cursor
        std::fputs("\x1b[2J",  stdout);    // clear
    }

    ~RawTerminal() {
        std::fputs("\x1b[0m\x1b[?25h\n", stdout);
#if defined(_WIN32)
        if (haveSaved_) SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), savedOut_);
#else
        if (haveSaved_) tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
#endif
        std::fflush(stdout);
    }

    // Returns -1 when no byte is pending.
    int pollByte() {
#if defined(_WIN32)
        return _kbhit() ? _getch() : -1;
#else
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        timeval tv{0, 0};
        if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) <= 0) return -1;
        unsigned char c = 0;
        const ssize_t n = read(STDIN_FILENO, &c, 1);
        return (n == 1) ? static_cast<int>(c) : -1;
#endif
    }

private:
#if defined(_WIN32)
    DWORD savedOut_ = 0;
#else
    termios saved_{};
#endif
    bool haveSaved_ = false;
};

// --------------------------------------------------------------------------
// Renderer: two grid rows per terminal row using the upper half block.
// --------------------------------------------------------------------------
class TerminalRenderer {
public:
    explicit TerminalRenderer(const br::Grid& g)
        : rgba_(static_cast<std::size_t>(g.width()) *
                static_cast<std::size_t>(g.height()) * 4u) {
        out_.reserve(rgba_.size() * 4);
    }

    void draw(const br::Grid& g, const std::string& overlay) {
        g.renderRGBA(rgba_.data());
        const int W = g.width(), H = g.height();

        out_.clear();
        out_ += "\x1b[H";                        // cursor home

        char buf[64];
        for (int y = 0; y + 1 < H; y += 2) {
            for (int x = 0; x < W; ++x) {
                const std::size_t oTop =
                    (static_cast<std::size_t>(y) * W + x) * 4u;
                const std::size_t oBot =
                    (static_cast<std::size_t>(y + 1) * W + x) * 4u;
                std::snprintf(buf, sizeof(buf),
                              "\x1b[38;2;%u;%u;%um\x1b[48;2;%u;%u;%um",
                              rgba_[oTop], rgba_[oTop + 1], rgba_[oTop + 2],
                              rgba_[oBot], rgba_[oBot + 1], rgba_[oBot + 2]);
                out_ += buf;
                out_ += "\xE2\x96\x80";          // U+2580 UPPER HALF BLOCK
            }
            out_ += "\x1b[0m\r\n";
        }
        out_ += overlay;
        std::fwrite(out_.data(), 1, out_.size(), stdout);
        std::fflush(stdout);
    }

private:
    std::vector<std::uint8_t> rgba_;
    std::string               out_;
};

const char* materialName(br::Cell c) {
    switch (c) {
        case br::Cell::Empty: return "Erase";
        case br::Cell::Sand:  return "Sand";
        case br::Cell::Water: return "Water";
        case br::Cell::Fire:  return "Fire";
        case br::Cell::Wall:  return "Wall";
        default:              return "?";
    }
}

}  // namespace

int main(int argc, char** argv) {
    int W = 128, H = 96;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--width")  && i + 1 < argc) W = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "--height") && i + 1 < argc) H = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "--help")) {
            std::printf("usage: backendrace [--width N] [--height N]\n");
            return 0;
        }
    }

    br::Grid grid(W, H);
    grid.seedRandom(0xC0FFEEu, 0.15f, 0.10f);

    std::unique_ptr<br::IBackend> probes[4];
    for (int i = 0; i < 4; ++i) {
        probes[i] = br::createBackend(static_cast<br::BackendId>(i));
    }
    int backendId = probes[1]->available() ? 1 : 0;
    auto backend = br::createBackend(static_cast<br::BackendId>(backendId));

    RawTerminal term;
    TerminalRenderer renderer(grid);

    br::Cell material = br::Cell::Sand;
    int cx = grid.width() / 2, cy = 6, brush = 5;
    bool paused = false, running = true, emitting = true;

    double fps = 0.0, throughput = 0.0;
    auto last = std::chrono::steady_clock::now();
    std::string note = "keys: 1-4 backend | QWER material | arrows cursor | "
                       "space emit | p pause | c clear | s seed | x quit";

    while (running) {
        // ------------------------------ input ------------------------------
        int key;
        while ((key = term.pollByte()) >= 0) {
            if (key == 27) {                       // possible arrow escape
                const int a = term.pollByte();
                const int b = term.pollByte();
                if (a == '[') {
                    if (b == 'A') cy -= 3;
                    if (b == 'B') cy += 3;
                    if (b == 'D') cx -= 3;
                    if (b == 'C') cx += 3;
                }
                continue;
            }
            switch (key) {
                case '1': case '2': case '3': case '4': {
                    const int id = key - '1';
                    if (probes[id]->available()) {
                        backendId = id;
                        backend = br::createBackend(static_cast<br::BackendId>(id));
                        note = std::string("switched to ") + backend->name();
                        throughput = 0.0;
                    } else {
                        note = std::string(probes[id]->name()) + ": " +
                               probes[id]->description();
                    }
                    break;
                }
                case 'q': case 'Q': material = br::Cell::Sand;  break;
                case 'w': case 'W': material = br::Cell::Water; break;
                case 'e': case 'E': material = br::Cell::Fire;  break;
                case 'r': case 'R': material = br::Cell::Wall;  break;
                case 'z': case 'Z': material = br::Cell::Empty; break;
                case ' ': emitting = !emitting; break;
                case 'p': case 'P': paused = !paused; break;
                case 'c': case 'C': grid.clear(); break;
                case 's': case 'S': grid.clear();
                                    grid.seedRandom(0x1234u, 0.18f, 0.12f); break;
                case '+': case '=': if (brush < 24) ++brush; break;
                case '-': case '_': if (brush > 1)  --brush; break;
                case 'x': case 'X': running = false; break;
                default: break;
            }
        }
        if (cx < 1) cx = 1;
        if (cy < 1) cy = 1;
        if (cx > grid.width()  - 2) cx = grid.width()  - 2;
        if (cy > grid.height() - 2) cy = grid.height() - 2;

        if (emitting) grid.paintDisc(cx, cy, brush, material);

        // ------------------------------ step -------------------------------
        if (!paused) backend->step(grid);

        const double stepMs = backend->stats().lastStepMs;
        const double inst = (stepMs > 0.0)
            ? (static_cast<double>(grid.cells()) / (stepMs / 1000.0)) : 0.0;
        throughput = throughput * 0.85 + inst * 0.15;

        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        if (dt > 0.0) fps = fps * 0.9 + (1.0 / dt) * 0.1;

        // ---------------------------- overlay ------------------------------
        char ov[768];
        std::snprintf(ov, sizeof(ov),
            "\x1b[0m"
            " backend [%d] %-10s  %-52s\r\n"
            " fps %6.1f | step %7.3f ms | %8.2f Mcell/s | frame %llu%s\r\n"
            " material %-6s | brush %2d | cursor (%3d,%3d) | emit %-3s\r\n"
            " %s\x1b[K\r\n",
            backendId + 1, backend->name(), backend->description(),
            fps, stepMs, throughput / 1.0e6,
            static_cast<unsigned long long>(grid.frame()),
            paused ? "  [PAUSED]" : "",
            materialName(material), brush, cx, cy, emitting ? "on" : "off",
            note.c_str());

        renderer.draw(grid, ov);

        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }

    return 0;
}